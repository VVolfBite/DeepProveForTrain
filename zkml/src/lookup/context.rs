//! File containg code for lookup witness generation.

use std::collections::{BTreeSet, HashMap};

use ff::Field;
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tracing::debug;
use transcript::Transcript;

use crate::{
    Element,
    commit::precommit::Context,
    iop::ChallengeStorage,
    layers::{Layer, activation::Relu},
    lookup::logup_gkr::structs::LogUpInput,
    model::InferenceTrace,
    quantization::{self, Fieldizer},
};

use super::logup_gkr::error::LogUpError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TableType {
    Relu,
    Range,
}

impl TableType {
    fn get_merged_table_column<E: ExtensionField>(
        &self,
        column_separator: Element,
    ) -> (Vec<Element>, Vec<Vec<E::BaseField>>) {
        match self {
            TableType::Relu => {
                let (comb, field): (Vec<Element>, Vec<(E::BaseField, E::BaseField)>) =
                    (*quantization::MIN..=*quantization::MAX)
                        .map(|i| {
                            let out = Relu::apply(i);
                            let i_field: E = i.to_field();
                            let out_field: E = out.to_field();
                            (
                                i + out * column_separator,
                                (i_field.as_bases()[0], out_field.as_bases()[0]),
                            )
                        })
                        .unzip();
                let (col_one, col_two): (Vec<E::BaseField>, Vec<E::BaseField>) =
                    field.into_iter().unzip();
                (comb, vec![col_one, col_two])
            }
            TableType::Range => {
                let (element_out, field): (Vec<Element>, Vec<E::BaseField>) = (0..1
                    << *quantization::BIT_LEN)
                    .map(|i| {
                        let i_field: E = i.to_field();
                        (i, i_field.as_bases()[0])
                    })
                    .unzip();
                (element_out, vec![field])
            }
        }
    }

    pub fn name(&self) -> String {
        match self {
            TableType::Relu => "Relu".to_string(),
            TableType::Range => "Range".to_string(),
        }
    }

    pub fn evaluate_table_columns<E: ExtensionField>(
        &self,
        point: &[E],
    ) -> Result<Vec<E>, LogUpError> {
        match self {
            TableType::Range => {
                if point.len() != *quantization::BIT_LEN {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a range table evaluation, point size: {}, expected: {}",
                        point.len(),
                        *quantization::BIT_LEN
                    )));
                }

                Ok(vec![
                    point
                        .iter()
                        .enumerate()
                        .fold(E::ZERO, |acc, (index, p)| acc + *p * E::from(1u64 << index)),
                ])
            }
            TableType::Relu => {
                if point.len() != *quantization::BIT_LEN {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a relu table evaluation, point size: {}, expected: {}",
                        point.len(),
                        *quantization::BIT_LEN
                    )));
                }

                let first_column = point
                    .iter()
                    .enumerate()
                    .fold(E::ZERO, |acc, (index, p)| acc + *p * E::from(1u64 << index))
                    - E::from(1u64 << (*quantization::BIT_LEN - 1));

                let second_column = point
                    .iter()
                    .enumerate()
                    .take(point.len() - 1)
                    .fold(E::ZERO, |acc, (index, p)| {
                        acc + *p * E::from(1u64 << index) * point[point.len() - 1]
                    });
                Ok(vec![first_column, second_column])
            }
        }
    }

    pub fn generate_challenge<E: ExtensionField, T: Transcript<E>>(&self, transcript: &mut T) -> E {
        match self {
            TableType::Relu => transcript.get_and_append_challenge(b"Relu").elements,
            TableType::Range => {
                // Theres only one column for a range check so we don't need to generate a challenge
                E::ONE
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookupContext {
    tables: Vec<TableType>,
}

impl LookupContext {
    pub fn new(set: &BTreeSet<TableType>) -> LookupContext {
        LookupContext {
            tables: set.iter().copied().collect(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &TableType> {
        self.tables.iter()
    }
}

pub fn generate_lookup_witnesses<E: ExtensionField, T: Transcript<E>>(
    trace: &InferenceTrace<Element, E>,
    transcript: &mut T,
) -> Result<
    (
        Context<E>,
        ChallengeStorage<E>,
        Vec<LogUpInput<E>>,
        Vec<LogUpInput<E>>,
    ),
    LogUpError,
>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    let mut table_lookups = HashMap::<TableType, HashMap<Element, u64>>::new();
    let mut tables = BTreeSet::new();
    let mut polys_with_id = Vec::<(usize, Vec<E>)>::new();
    let mut lookups_no_challenges = Vec::<(Vec<Vec<E::BaseField>>, usize, TableType)>::new();
    let column_separator = 1i128 << 32;
    let mut total_steps = 0;
    debug!("Lookup witness generation: generating poly fields...");
    trace.iter().for_each(|(step_input, step)| {
        total_steps += 1;
        match step.layer {
            Layer::Activation(..) => {
                tables.insert(TableType::Relu);

                // Calculate the column_evals and also the merged lookups
                let (merged_lookups, field): (Vec<Element>, Vec<(E::BaseField, E::BaseField)>) =
                    step_input
                        .get_data()
                        .iter()
                        .zip(step.output.get_data().iter())
                        .map(|(a, b)| {
                            let a_field: E = a.to_field();
                            let b_field: E = b.to_field();
                            (
                                a + column_separator * b,
                                (a_field.as_bases()[0], b_field.as_bases()[0]),
                            )
                        })
                        .unzip();

                let (col_one, col_two): (Vec<E::BaseField>, Vec<E::BaseField>) =
                    field.into_iter().unzip();
                let table_lookup_map = table_lookups
                    .entry(TableType::Relu)
                    .or_insert_with(|| HashMap::default());

                merged_lookups
                    .into_iter()
                    .for_each(|lookup| *table_lookup_map.entry(lookup).or_insert(0u64) += 1);

                polys_with_id.push((
                    step.id,
                    step.output
                        .get_data()
                        .iter()
                        .map(Fieldizer::<E>::to_field)
                        .collect(),
                ));
                lookups_no_challenges.push((vec![col_one, col_two], 2, TableType::Relu));
            }

            Layer::Requant(requant) => {
                tables.insert(TableType::Range);
                let table_lookup_map = table_lookups
                    .entry(TableType::Range)
                    .or_insert_with(|| HashMap::default());

                let (merged_lookups, column_evals) =
                    requant.gen_lookup_witness::<E>(step_input.get_data());
                merged_lookups
                    .into_iter()
                    .for_each(|val| *table_lookup_map.entry(val).or_insert(0u64) += 1);

                polys_with_id.push((
                    step.id,
                    step.output
                        .get_data()
                        .iter()
                        .map(Fieldizer::<E>::to_field)
                        .collect(),
                ));

                lookups_no_challenges.push((column_evals, 1, TableType::Range));
            }
            Layer::Pooling(pooling) => {
                tables.insert(TableType::Range);
                let table_lookup_map = table_lookups
                    .entry(TableType::Range)
                    .or_insert_with(|| HashMap::default());

                let (merged_lookups, column_evals) = pooling.gen_lookup_witness::<E>(step_input);

                merged_lookups
                    .into_iter()
                    .for_each(|val| *table_lookup_map.entry(val).or_insert(0u64) += 1);

                polys_with_id.push((
                    step.id,
                    step.output
                        .get_data()
                        .iter()
                        .map(Fieldizer::<E>::to_field)
                        .collect(),
                ));
                lookups_no_challenges.push((column_evals, 1, TableType::Range));
            }
            _ => (),
        }
    });

    debug!("Lookup witness generation: generating table multiplicities...");
    // calculate the table multiplicities
    let tables_no_challenges = tables.iter().map(|table_type| {
        let (table_column, column_evals) = table_type.get_merged_table_column::<E>(column_separator);

        let table_lookup_data = table_lookups.get(table_type).ok_or(LogUpError::ParamterError(format!("Tried to retrieve lookups for a table of type: {:?}, but no table of that type exists", table_type)))?;

        let (multiplicities, mults_ext)  = table_column.iter().map(|table_val| {
            if let Some(lookup_count) = table_lookup_data.get(table_val) {
                (E::BaseField::from(*lookup_count), E::from(*lookup_count))
            } else {
                (E::BaseField::ZERO, E::ZERO)
            }
        }).unzip();

        polys_with_id.push((total_steps, mults_ext));
        total_steps += 1;
        Ok((column_evals, multiplicities, *table_type))
    }).collect::<Result<Vec<(Vec<Vec<E::BaseField>>, Vec<E::BaseField>, TableType)>, LogUpError>>()?;

    debug!("Lookup witness generation: commit context generation...");
    let ctx = Context::generate(polys_with_id).map_err(|e| {
        LogUpError::ParamterError(format!(
            "Could not generate Lookup witness commit context {{ inner: {:?}}}",
            e
        ))
    })?;

    ctx.write_to_transcript(transcript).map_err(|e| {
        LogUpError::ParamterError(format!(
            "Unable to write lookup witness commit context to transcript, {{ inner: {:?}}}",
            e
        ))
    })?;

    debug!("Lookup witness generation: challenge storage...");
    let challenge_storage = initialise_from_table_set::<E, T>(&tables, transcript);

    let lookup_inputs = lookups_no_challenges
        .into_iter()
        .map(|(column_evals, columns_per_instance, table_type)| {
            let (constant_challenge, column_challenge) = challenge_storage
                .get_challenges_by_name(&table_type.name())
                .ok_or(LogUpError::ParamterError(format!(
                    "No challegnes found for table type: {} when generating lookup witness",
                    table_type.name()
                )))?;

            LogUpInput::<E>::new_lookup(
                column_evals,
                constant_challenge,
                column_challenge,
                columns_per_instance,
            )
        })
        .collect::<Result<Vec<LogUpInput<E>>, LogUpError>>()?;

    let table_inputs = tables_no_challenges
        .into_iter()
        .map(|(column_evals, multiplicities, table_type)| {
            let (constant_challenge, column_challenge) = challenge_storage
                .get_challenges_by_name(&table_type.name())
                .ok_or(LogUpError::ParamterError(format!(
                    "No challegnes found for table type: {} when generating table witness",
                    table_type.name()
                )))?;

            LogUpInput::<E>::new_table(
                column_evals,
                multiplicities,
                constant_challenge,
                column_challenge,
            )
        })
        .collect::<Result<Vec<LogUpInput<E>>, LogUpError>>()?;
    Ok((ctx, challenge_storage, lookup_inputs, table_inputs))
}

fn initialise_from_table_set<E: ExtensionField, T: Transcript<E>>(
    set: &BTreeSet<TableType>,
    transcript: &mut T,
) -> ChallengeStorage<E> {
    let constant_challenge = transcript
        .get_and_append_challenge(b"table_constant")
        .elements;
    let challenge_map = set
        .iter()
        .map(|table_type| {
            let challenge = table_type.generate_challenge(transcript);

            (table_type.name(), challenge)
        })
        .collect::<HashMap<String, E>>();
    ChallengeStorage::<E> {
        constant_challenge,
        challenge_map,
    }
}

/*
# context.rs 文件分析

## 1. 文件概述
这是一个实现查找表证明生成和验证的核心文件。主要包含查找表的上下文管理和见证生成逻辑。

## 2. 核心数据结构

### TableType 枚举
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TableType {
    Relu,       // ReLU激活函数查找表
    Range,      // 范围检查查找表
}
```

### LookupContext
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookupContext {
    tables: Vec<TableType>,    // 保存所有使用的查找表类型
}
```

## 3. 主要功能实现

### 查找表操作
```rust
impl TableType {
    // 获取合并后的表列
    fn get_merged_table_column<E: ExtensionField>(
        &self,
        column_separator: Element,
    ) -> (Vec<Element>, Vec<Vec<E::BaseField>>)
    
    // 表列求值
    pub fn evaluate_table_columns<E: ExtensionField>(
        &self,
        point: &[E],
    ) -> Result<Vec<E>, LogUpError>
}
```

### 见证生成
```rust
pub fn generate_lookup_witnesses<E: ExtensionField, T: Transcript<E>>(
    trace: &InferenceTrace<Element, E>,
    transcript: &mut T,
) -> Result<(...), LogUpError>
```

## 4. 优化特点

1. **并行处理**
- 使用 Iterator 进行数据处理
- 采用 HashMap 优化查找性能

2. **内存优化**
```rust
// 使用引用避免不必要的复制
fn get_merged_table_column<E: ExtensionField>(&self, ...)
```

## 5. 错误处理

使用自定义错误类型:
```rust
pub enum LogUpError {
    VerifierError(String),
    ParamterError(String),
}
```

## 6. 关键算法

### ReLU表生成
```rust
match TableType::Relu => {
    let (comb, field): (Vec<Element>, Vec<(E::BaseField, E::BaseField)>) =
        (*quantization::MIN..=*quantization::MAX)
            .map(|i| {
                let out = Relu::apply(i);
                let i_field: E = i.to_field();
                let out_field: E = out.to_field();
                (
                    i + out * column_separator,
                    (i_field.as_bases()[0], out_field.as_bases()[0]),
                )
            })
            .unzip();
    // ...
}
```

### 范围检查表生成
```rust
match TableType::Range => {
    let (element_out, field): (Vec<Element>, Vec<E::BaseField>) = 
        (0..1 << *quantization::BIT_LEN)
            .map(|i| {
                let i_field: E = i.to_field();
                (i, i_field.as_bases()[0])
            })
            .unzip();
    // ...
}
```

## 7. 调试支持

使用 tracing 进行日志记录:
```rust
debug!("Lookup witness generation: generating poly fields...");
```

## 8. 建议优化

1. **性能优化**:
```rust
// 添加并行处理
use rayon::prelude::*;

fn get_merged_table_column<E: ExtensionField>(
    &self,
    column_separator: Element,
) -> (Vec<Element>, Vec<Vec<E::BaseField>>) {
    match self {
        TableType::Relu => {
            let (comb, field): (Vec<Element>, Vec<(E::BaseField, E::BaseField)>) =
                (*quantization::MIN..=*quantization::MAX)
                    .into_par_iter()  // 使用并行迭代器
                    .map(...)
                    .unzip();
            // ...
        }
        // ...
    }
}
```

2. **错误处理优化**:
```rust
// 使用 thiserror 简化错误定义
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LookupError {
    #[error("Verifier error: {0}")]
    VerifierError(String),
    
    #[error("Parameter error: {0}")]
    ParameterError(String),
}
```

## 9. 总结

context.rs 实现了完整的查找表管理和见证生成系统:

1. **功能完备**
- ReLU 和范围检查表支持
- 完整的见证生成
- 灵活的表类型扩展

2. **性能考虑**
- 高效的数据结构
- 优化的算法实现
- 可扩展的设计

3. **可维护性**
- 清晰的代码结构
- 完善的错误处理
- 良好的调试支持

该实现为零知识证明系统中的查找表操作提供了坚实基础。
 */
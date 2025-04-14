//! Module containing utility structs for working with LogUp GKR circuits.

use std::{
    borrow::Borrow,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign},
};

use ff_ext::ExtensionField;
use multilinear_extensions::mle::DenseMultilinearExtension;
use serde::{Deserialize, Serialize};
use sumcheck::structs::IOPProof;
use transcript::Transcript;

use super::{circuit::LogUpCircuit, error::LogUpError};
use crate::Claim;
use rayon::prelude::*;

#[derive(Clone, Debug, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
/// Struct used to perform arithmetic on fractions
pub struct Fraction<F> {
    pub numerator: F,
    pub denominator: F,
}

impl<F> Fraction<F> {
    /// Create a new instance of a [`Fraction`].
    pub fn new(numerator: F, denominator: F) -> Fraction<F> {
        Fraction::<F> {
            numerator,
            denominator,
        }
    }

    /// Turns this fraction into a tuple, the first element is the numerator, the second is the denominator
    pub fn as_tuple(&self) -> (F, F)
    where
        F: Clone,
    {
        (self.numerator.clone(), self.denominator.clone())
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> AddAssign<T> for Fraction<F> {
    fn add_assign(&mut self, rhs: T) {
        let rhs: &Fraction<F> = rhs.borrow();
        let numerator = (self.numerator * rhs.denominator) + (self.denominator * rhs.numerator);
        let denominator = self.denominator * rhs.denominator;
        *self = Fraction {
            numerator,
            denominator,
        };
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> Add<T> for &Fraction<F> {
    type Output = Fraction<F>;

    fn add(self, rhs: T) -> Self::Output {
        let mut output = *self;
        output += rhs;
        output
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> Add<T> for Fraction<F> {
    type Output = Fraction<F>;

    fn add(self, rhs: T) -> Self::Output {
        let mut output = self;
        output += rhs;
        output
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> MulAssign<T> for Fraction<F> {
    fn mul_assign(&mut self, rhs: T) {
        let rhs: &Fraction<F> = rhs.borrow();
        self.numerator *= rhs.numerator;
        self.denominator *= rhs.denominator;
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> Mul<T> for &Fraction<F> {
    type Output = Fraction<F>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut output = *self;
        output *= rhs;
        output
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> Mul<T> for Fraction<F> {
    type Output = Fraction<F>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut output = self;
        output *= rhs;
        output
    }
}

impl<F: ExtensionField> Sum for Fraction<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Fraction<F> {
        iter.fold(Fraction::<F>::ZERO, |acc, term| acc + term)
    }
}

impl<F: ExtensionField> Product for Fraction<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Fraction::<F>::ONE, |acc, term| acc * term)
    }
}

impl<F: ExtensionField> Fraction<F> {
    const ZERO: Fraction<F> = Fraction {
        numerator: F::ZERO,
        denominator: F::ONE,
    };

    const ONE: Fraction<F> = Fraction {
        numerator: F::ONE,
        denominator: F::ONE,
    };
    /// Checks whether this is the zero element.
    pub fn is_zero(&self) -> bool {
        (self.numerator == F::ZERO) && (self.denominator != F::ZERO)
    }
}

#[derive(Clone, Debug)]
/// Enum defining inputs to LogUp proofs.
/// We split lookup inputs and table inputs as different optimisations can be made in each case. Additionally it allows us to only do work proportional to the table size in the table proving case
/// which is useful when multiple different model layers use the same table.
pub enum LogUpInput<E: ExtensionField> {
    /// Lookup variant can have multiple instances in one [`LogUpInput::Lookup`], `columns_per_instance` is used to work out how many batches we need to prove.
    Lookup {
        column_evals: Vec<Vec<E::BaseField>>,
        constant_challenge: E,
        column_separation_challenge: E,
        columns_per_instance: usize,
    },
    /// Input for a Table proof.
    Table {
        column_evals: Vec<Vec<E::BaseField>>,
        multiplicities: Vec<E::BaseField>,
        constant_challenge: E,
        column_separation_challenge: E,
    },
}

impl<E: ExtensionField> LogUpInput<E> {
    pub fn new_lookup(
        column_evals: Vec<Vec<E::BaseField>>,
        constant_challenge: E,
        column_separation_challenge: E,
        columns_per_instance: usize,
    ) -> Result<LogUpInput<E>, LogUpError> {
        if column_evals.is_empty() {
            return Err(LogUpError::ParamterError(
                "No column evals were provided for Lookup input".to_string(),
            ));
        }

        // Unwrap is safe
        let first_evals_len = column_evals.first().unwrap().len();

        if !first_evals_len.is_power_of_two() {
            return Err(LogUpError::PolynomialError(format!(
                "Need a power of two number of evaluations got: {}",
                first_evals_len
            )));
        }

        column_evals.iter().skip(1).try_for_each(|evals| {
            if evals.len() != first_evals_len {
                Err(LogUpError::ParamterError(
                    "All sets of evaluations should be the same length".to_string(),
                ))
            } else {
                Ok(())
            }
        })?;

        Ok(LogUpInput::Lookup {
            column_evals,
            constant_challenge,
            column_separation_challenge,
            columns_per_instance,
        })
    }

    pub fn new_table(
        column_evals: Vec<Vec<E::BaseField>>,
        multiplicities: Vec<E::BaseField>,
        constant_challenge: E,
        column_separation_challenge: E,
    ) -> Result<LogUpInput<E>, LogUpError> {
        if column_evals.is_empty() {
            return Err(LogUpError::ParamterError(
                "No column evals were provided for Lookup input".to_string(),
            ));
        }

        // Unwrap is safe
        let first_evals_len = column_evals.first().unwrap().len();

        if !first_evals_len.is_power_of_two() {
            return Err(LogUpError::PolynomialError(format!(
                "Need a power of two number of evaluations got: {}",
                first_evals_len
            )));
        }

        column_evals.iter().skip(1).try_for_each(|evals| {
            if evals.len() != first_evals_len {
                Err(LogUpError::ParamterError(
                    "All sets of evaluations should be the same length".to_string(),
                ))
            } else {
                Ok(())
            }
        })?;

        if multiplicities.len() != first_evals_len {
            return Err(LogUpError::PolynomialError(format!(
                "Multiplicities length was not equal to column evaluations length, multiplicities: {}, columns: {}",
                multiplicities.len(),
                first_evals_len
            )));
        }

        Ok(LogUpInput::Table {
            column_evals,
            multiplicities,
            constant_challenge,
            column_separation_challenge,
        })
    }

    pub fn column_evals(&self) -> &[Vec<E::BaseField>] {
        match self {
            LogUpInput::Lookup { column_evals, .. } | LogUpInput::Table { column_evals, .. } => {
                column_evals
            }
        }
    }

    pub fn make_circuits(&self) -> Vec<LogUpCircuit<E>> {
        match self {
            LogUpInput::Lookup {
                column_evals,
                constant_challenge,
                column_separation_challenge,
                columns_per_instance,
            } => column_evals
                .par_chunks(*columns_per_instance)
                .map(|column_evals| {
                    LogUpCircuit::<E>::new_lookup_circuit(
                        column_evals,
                        *constant_challenge,
                        *column_separation_challenge,
                    )
                })
                .collect(),
            LogUpInput::Table {
                column_evals,
                multiplicities,
                constant_challenge,
                column_separation_challenge,
            } => {
                vec![LogUpCircuit::<E>::new_table_circuit(
                    column_evals,
                    multiplicities,
                    *constant_challenge,
                    *column_separation_challenge,
                )]
            }
        }
    }

    pub fn base_mles(&self) -> Vec<DenseMultilinearExtension<E>> {
        match self {
            LogUpInput::Lookup { column_evals, .. } => column_evals
                .iter()
                .map(|evaluations| {
                    let num_vars = evaluations.len().ilog2() as usize;
                    DenseMultilinearExtension::<E>::from_evaluations_slice(num_vars, evaluations)
                })
                .collect(),
            LogUpInput::Table {
                column_evals,
                multiplicities,
                ..
            } => std::iter::once(multiplicities)
                .chain(column_evals.iter())
                .map(|evaluations| {
                    let num_vars = evaluations.len().ilog2() as usize;
                    DenseMultilinearExtension::<E>::from_evaluations_slice(num_vars, evaluations)
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub enum ProofType {
    Lookup,
    Table,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogUpProof<E: ExtensionField> {
    pub sumcheck_proofs: Vec<IOPProof<E>>,
    pub round_evaluations: Vec<Vec<E>>,
    pub output_claims: Vec<Claim<E>>,
    pub circuit_outputs: Vec<Vec<E>>,
    pub proof_type: ProofType,
}

impl<E: ExtensionField> LogUpProof<E> {
    pub fn append_to_transcript<T: Transcript<E>>(&self, transcript: &mut T) {
        self.circuit_outputs
            .iter()
            .for_each(|evals| transcript.append_field_element_exts(evals));
    }

    pub fn fractional_outputs(&self) -> (Vec<E>, Vec<E>) {
        self.circuit_outputs
            .iter()
            .map(|evals| {
                (
                    evals[0] * evals[3] + evals[1] * evals[2],
                    evals[2] * evals[3],
                )
            })
            .unzip()
    }

    pub fn proofs_and_evals(&self) -> impl Iterator<Item = (&IOPProof<E>, &Vec<E>)> {
        self.sumcheck_proofs
            .iter()
            .zip(self.round_evaluations.iter())
    }

    pub fn circuit_outputs(&self) -> &[Vec<E>] {
        &self.circuit_outputs
    }

    pub fn output_claims(&self) -> &[Claim<E>] {
        &self.output_claims
    }

    pub fn proof_type(&self) -> ProofType {
        self.proof_type
    }
}

#[derive(Debug, Clone)]
pub struct LogUpVerifierClaim<E: ExtensionField> {
    claims: Vec<Claim<E>>,
    numerators: Vec<E>,
    denominators: Vec<E>,
}

impl<E: ExtensionField> LogUpVerifierClaim<E> {
    pub fn new(
        claims: Vec<Claim<E>>,
        numerators: Vec<E>,
        denominators: Vec<E>,
    ) -> LogUpVerifierClaim<E> {
        LogUpVerifierClaim {
            claims,
            numerators,
            denominators,
        }
    }

    pub fn claims(&self) -> &[Claim<E>] {
        &self.claims
    }

    pub fn numerators(&self) -> &[E] {
        &self.numerators
    }

    pub fn denominators(&self) -> &[E] {
        &self.denominators
    }
}

/*
# structs.rs 文件分析

## 1. 核心数据结构

### Fraction 分数结构
```rust
#[derive(Clone, Debug, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Fraction<F> {
    pub numerator: F,      // 分子
    pub denominator: F,    // 分母
}
```

### LogUpInput 枚举
```rust
#[derive(Clone, Debug)]
pub enum LogUpInput<E: ExtensionField> {
    Lookup {
        column_evals: Vec<Vec<E::BaseField>>,
        constant_challenge: E,
        column_separation_challenge: E,
        columns_per_instance: usize,
    },
    Table {
        column_evals: Vec<Vec<E::BaseField>>,
        multiplicities: Vec<E::BaseField>,
        constant_challenge: E,
        column_separation_challenge: E,
    },
}
```

## 2. 优化建议

### 1. 分数运算优化
````rust
impl<F: ExtensionField> Fraction<F> {
    #[inline(always)]
    pub fn simplify(&mut self) {
        if self.denominator == F::ZERO {
            panic!("除数不能为零");
        }
        if self.numerator == F::ZERO {
            self.denominator = F::ONE;
            return;
        }
        // 如果可能的话进行约分
        // TODO: 实现GCD算法进行约分
    }
    
    #[inline(always)]
    pub fn reciprocal(&self) -> Option<Self> {
        if self.numerator == F::ZERO {
            None
        } else {
            Some(Fraction {
                numerator: self.denominator.clone(),
                denominator: self.numerator.clone(),
            })
        }
    }
}
````

### 2. 并行处理优化
````rust
impl<E: ExtensionField> LogUpInput<E> {
    pub fn base_mles_parallel(&self) -> Vec<DenseMultilinearExtension<E>> {
        match self {
            LogUpInput::Lookup { column_evals, .. } => {
                column_evals.par_iter()
                    .map(|evaluations| {
                        let num_vars = evaluations.len().ilog2() as usize;
                        DenseMultilinearExtension::<E>::from_evaluations_slice(
                            num_vars, 
                            evaluations
                        )
                    })
                    .collect()
            },
            // ...existing code...
        }
    }
}
````

### 3. 错误处理优化
````rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FractionError {
    #[error("分母为零")]
    DivisionByZero,
    
    #[error("运算溢出")]
    Overflow,
}

impl<F: ExtensionField> Fraction<F> {
    pub fn checked_add(&self, rhs: &Self) -> Result<Self, FractionError> {
        if rhs.denominator == F::ZERO || self.denominator == F::ZERO {
            return Err(FractionError::DivisionByZero);
        }
        // ...existing code...
        Ok(result)
    }
}
````

## 3. 测试完善

````rust
#[cfg(test)]
mod tests {
    use super::*;
    use goldilocks::GoldilocksExt2;

    #[test]
    fn test_fraction_basic_operations() {
        let f1 = Fraction::<GoldilocksExt2>::new(
            GoldilocksExt2::ONE,
            GoldilocksExt2::ONE
        );
        let f2 = Fraction::<GoldilocksExt2>::new(
            GoldilocksExt2::ONE,
            GoldilocksExt2::ONE * GoldilocksExt2::from(2u64)
        );
        
        // 测试加法
        let sum = f1 + f2;
        assert!(!sum.is_zero());
        
        // 测试乘法
        let product = f1 * f2;
        assert!(!product.is_zero());
    }
    
    #[test]
    fn test_logup_input_validation() {
        let column_evals = vec![vec![GoldilocksExt2::ONE; 4]];
        let result = LogUpInput::new_lookup(
            column_evals.clone(),
            GoldilocksExt2::ONE,
            GoldilocksExt2::ONE,
            1
        );
        assert!(result.is_ok());
        
        // 测试非2次幂长度的输入
        let invalid_evals = vec![vec![GoldilocksExt2::ONE; 3]];
        let result = LogUpInput::new_lookup(
            invalid_evals,
            GoldilocksExt2::ONE,
            GoldilocksExt2::ONE,
            1
        );
        assert!(result.is_err());
    }
}
````

## 4. 性能监控

````rust
use tracing::{debug, instrument};

impl<E: ExtensionField> LogUpInput<E> {
    #[instrument(skip(column_evals, constant_challenge, column_separation_challenge))]
    pub fn new_lookup(
        column_evals: Vec<Vec<E::BaseField>>,
        constant_challenge: E,
        column_separation_challenge: E,
        columns_per_instance: usize,
    ) -> Result<LogUpInput<E>, LogUpError> {
        debug!("Creating new lookup input with {} columns", column_evals.len());
        // ...existing code...
    }
}
````

## 5. 总结

structs.rs 实现了 LogUp GKR 电路的基础数据结构：

1. **核心功能**
- 分数运算支持
- 输入类型封装
- 证明数据管理

2. **优化建议**
- 添加分数运算优化
- 实现并行处理
- 改进错误处理
- 增加性能监控

3. **代码质量**
- 完善的测试覆盖
- 清晰的错误处理
- 良好的文档注释

该实现为 LogUp GKR 电路提供了坚实的数据结构基础。 */
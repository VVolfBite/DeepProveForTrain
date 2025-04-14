use crate::{
    iop::precommit::{self, PolyID},
    layers::LayerCtx,
    lookup::context::{LookupContext, TableType},
    model::Model,
};
use anyhow::Context as CC;
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::BasefoldCommitment;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::BTreeSet;
use tracing::{debug, trace};
use transcript::Transcript;

/// Info related to the lookup protocol tables.
/// Here `poly_id` is the multiplicity poly for this table.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct TableCtx<E>
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    pub poly_id: PolyID,
    pub num_vars: usize,
    pub table_commitment: BasefoldCommitment<E>,
}

/// Common information between prover and verifier
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct Context<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// Information about each steps of the model. That's the information that the verifier
    /// needs to know from the setup to avoid the prover being able to cheat.
    /// in REVERSED order already since proving goes from last layer to first layer.
    pub steps_info: Vec<LayerCtx<E>>,
    /// Context related to the commitment and accumulation of claims related to the weights of model.
    /// This part contains the commitment of the weights.
    pub weights: precommit::Context<E>,
    /// Context holding all the different table types we use in lookups
    pub lookup: LookupContext,
}

/// Auxiliary information for the context creation
#[derive(Clone, Debug)]
pub(crate) struct ContextAux {
    pub tables: BTreeSet<TableType>,
    pub last_output_shape: Vec<usize>,
}

impl<E: ExtensionField> Context<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// Generates a context to give to the verifier that contains informations about the polynomials
    /// to prove at each step.
    /// INFO: it _assumes_ the model is already well padded to power of twos.
    pub fn generate(model: &Model, input_shape: Option<Vec<usize>>) -> anyhow::Result<Self> {
        let tables = BTreeSet::new();
        let last_output_shape = if let Some(shape) = input_shape {
            shape
        } else {
            model.input_shape()
        };
        let mut ctx_aux = ContextAux {
            tables,
            last_output_shape,
        };
        let mut step_infos = Vec::with_capacity(model.layer_count());
        debug!("Context : layer info generation ...");
        for (id, layer) in model.layers() {
            trace!(
                "Context : {}-th layer {}info generation ...",
                id,
                layer.describe()
            );
            let (info, new_aux) = layer.step_info(id, ctx_aux);
            step_infos.push(info);
            ctx_aux = new_aux;
        }
        debug!("Context : commitment generating ...");
        let commit_ctx = precommit::Context::generate_from_model(model)
            .context("can't generate context for commitment part")?;
        debug!("Context : lookup generation ...");
        let lookup_ctx = LookupContext::new(&ctx_aux.tables);
        Ok(Self {
            steps_info: step_infos.into_iter().rev().collect_vec(),
            weights: commit_ctx,
            lookup: lookup_ctx,
        })
    }

    pub fn write_to_transcript<T: Transcript<E>>(&self, t: &mut T) -> anyhow::Result<()> {
        for steps in self.steps_info.iter() {
            match steps {
                LayerCtx::Dense(info) => {
                    t.append_field_element(&E::BaseField::from(info.matrix_poly_id as u64));
                    info.matrix_poly_aux.write_to_transcript(t);
                }
                LayerCtx::Requant(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    t.append_field_element(&E::BaseField::from(info.num_vars as u64));
                }
                LayerCtx::Activation(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    t.append_field_element(&E::BaseField::from(info.num_vars as u64));
                }
                LayerCtx::Pooling(info) => {
                    t.append_field_element(&E::BaseField::from(info.poolinfo.kernel_size as u64));
                    t.append_field_element(&E::BaseField::from(info.poolinfo.stride as u64));
                }
                LayerCtx::Table(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    t.append_field_element(&E::BaseField::from(info.num_vars as u64));
                    t.append_field_elements(info.table_commitment.root().0.as_slice());
                }
                LayerCtx::Convolution(info) => {
                    t.append_field_element(&E::BaseField::from(info.poly_id as u64));
                    t.append_field_element(&E::BaseField::from(info.bias_poly_id as u64));

                    for i in 0..info.delegation_fft.len() {
                        info.delegation_fft[i].write_to_transcript(t);
                    }
                    for i in 0..info.delegation_ifft.len() {
                        info.delegation_ifft[i].write_to_transcript(t);
                    }
                    info.fft_aux.write_to_transcript(t);
                    info.ifft_aux.write_to_transcript(t);
                    info.hadamard.write_to_transcript(t);
                }
                LayerCtx::SchoolBookConvolution(_info) => {}
            }
        }
        self.weights.write_to_transcript(t)?;
        Ok(())
    }
}


/*
以下是对 context.rs 文件的整体结构和关键部分的总结：

---

### **1. 文件功能**
该文件实现了零知识证明系统的上下文管理功能,主要包括：
- 模型各层级信息的维护和管理
- 权重承诺(Commitment)的处理
- 查找表(Lookup Table)协议的上下文维护
- 证明系统状态的追踪与转录

---

### **2. 关键部分**

#### **核心数据结构**
- **`TableCtx<E>`**
  - 功能：存储查找表相关信息
  - 字段：
    - `poly_id: PolyID` - 多重性多项式ID
    - `num_vars: usize` - 变量数量
    - `table_commitment: BasefoldCommitment<E>` - 表承诺

- **`Context<E: ExtensionField>`**
  - 功能：全局上下文环境
  - 字段：
    - `steps_info: Vec<LayerCtx<E>>` - 模型层信息(反序)
    - `weights: precommit::Context<E>` - 权重承诺上下文
    - `lookup: LookupContext` - 查找表上下文

- **`ContextAux`**
  - 功能：上下文辅助信息
  - 字段：
    - `tables: BTreeSet<TableType>` - 表类型集合
    - `last_output_shape: Vec<usize>` - 最后输出形状

---

#### **主要方法实现**
- **`Context::generate`**
  - 功能：生成证明系统上下文
  - 参数：
    - `model: &Model` - 输入模型
    - `input_shape: Option<Vec<usize>>` - 可选输入形状
  - 流程：
    1. 初始化辅助上下文
    2. 生成各层信息
    3. 创建承诺上下文
    4. 构建查找表上下文

- **`Context::write_to_transcript`**
  - 功能：将上下文信息写入转录器
  - 实现：根据不同层类型写入对应信息
  - 支持层类型：
    - Dense
    - Requant
    - Activation
    - Pooling
    - Table
    - Convolution
    - SchoolBookConvolution

---

### **3. 优化特点**

1. **内存管理**
- 预分配向量容量
- 使用引用避免不必要的复制
```rust
let mut step_infos = Vec::with_capacity(model.layer_count());
```

2. **日志追踪**
- 分层级的日志记录
- 详细的调试信息
```rust
debug!("Context : layer info generation ...");
trace!("Context : {}-th layer {}info generation ...", id, layer.describe());
```

3. **错误处理**
- 使用 Result 类型返回错误
- 详细的错误信息和上下文

---

### **4. 安全性保证**

1. **类型安全**
- 泛型约束确保类型安全
- 序列化/反序列化边界清晰

2. **状态完整性**
- 严格的层级顺序管理
- 完整的上下文信息验证

---

### **5. 总结**
context.rs 实现了一个完整的零知识证明系统上下文管理框架。它通过模块化的设计和严格的类型系统,确保了证明系统的可靠性和可维护性。该实现特别注重性能优化和调试便利性,为整个零知识证明系统提供了坚实的基础设施支持。
*/
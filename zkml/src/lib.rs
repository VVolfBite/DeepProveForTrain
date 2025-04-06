#![feature(iter_next_chunk)]

use ff_ext::ExtensionField;
use gkr::structs::PointAndEval;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use transcript::{BasicTranscript, Transcript};
mod commit;
pub mod iop;
pub mod quantization;
pub use iop::{
    Context, Proof,
    prover::Prover,
    verifier::{IO, verify},
};
pub mod layers;
pub mod lookup;
pub mod model;
mod onnx_parse;
pub use onnx_parse::{ModelType, load_model};

pub mod tensor;
#[cfg(test)]
mod testing;

/// We allow higher range to account for overflow. Since we do a requant after each layer, we
/// can support with i128 with 8 bits quant:
/// 16 + log(c) = 64 => c = 2^48 columns in a dense layer
pub type Element = i128;

/// Claim type to accumulate in this protocol, for a certain polynomial, known in the context.
/// f(point) = eval
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Claim<E> {
    point: Vec<E>,
    eval: E,
}

impl<E> Claim<E> {
    pub fn new(point: Vec<E>, eval: E) -> Self {
        Self { point, eval }
    }
}

impl<E: ExtensionField> From<PointAndEval<E>> for Claim<E> {
    fn from(value: PointAndEval<E>) -> Self {
        Claim {
            point: value.point.clone(),
            eval: value.eval,
        }
    }
}

impl<E: ExtensionField> From<&PointAndEval<E>> for Claim<E> {
    fn from(value: &PointAndEval<E>) -> Self {
        Claim {
            point: value.point.clone(),
            eval: value.eval,
        }
    }
}

impl<E: ExtensionField> Claim<E> {
    /// Pad the point to the new size given
    /// This is necessary for passing from output of padded lookups to next dense layer proving for example.
    /// NOTE: you can use it to pad or reduce size
    pub fn pad(&self, new_num_vars: usize) -> Claim<E> {
        Self {
            eval: self.eval,
            point: self
                .point
                .iter()
                .chain(std::iter::repeat(&E::ZERO))
                .take(new_num_vars)
                .cloned()
                .collect_vec(),
        }
    }
}

/// Returns the default transcript the prover and verifier must instantiate to validate a proof.
pub fn default_transcript<E: ExtensionField>() -> BasicTranscript<E> {
    BasicTranscript::new(b"m2vec")
}

pub fn pad_vector<E: ExtensionField>(mut v: Vec<E>) -> Vec<E> {
    if !v.len().is_power_of_two() {
        v.resize(v.len().next_power_of_two(), E::ZERO);
    }
    v
}
/// Returns the bit sequence of num of bit_length length.
pub(crate) fn to_bit_sequence_le(
    num: usize,
    bit_length: usize,
) -> impl DoubleEndedIterator<Item = usize> {
    assert!(
        bit_length as u32 <= usize::BITS,
        "bit_length cannot exceed usize::BITS"
    );
    (0..bit_length).map(move |i| ((num >> i) & 1) as usize)
}

pub trait VectorTranscript<E: ExtensionField> {
    fn read_challenges(&mut self, n: usize) -> Vec<E>;
}

#[cfg(not(test))]
impl<T: Transcript<E>, E: ExtensionField> VectorTranscript<E> for T {
    fn read_challenges(&mut self, n: usize) -> Vec<E> {
        (0..n).map(|_| self.read_challenge().elements).collect_vec()
    }
}

#[cfg(test)]
impl<T: Transcript<E>, E: ExtensionField> VectorTranscript<E> for T {
    fn read_challenges(&mut self, n: usize) -> Vec<E> {
        (0..n).map(|_| E::ONE).collect_vec()
    }
}

pub fn argmax<T: PartialOrd>(v: &[T]) -> Option<usize> {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) // Unwrap is safe if T implements PartialOrd properly
        .map(|(idx, _)| idx)
}

#[cfg(test)]
mod test {
    use ark_std::rand::{Rng, thread_rng};
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::mle::{IntoMLE, MultilinearExtension};

    use crate::{
        Element, default_transcript,
        iop::{
            Context,
            prover::Prover,
            verifier::{IO, verify},
        },
        load_model,
        onnx_parse::ModelType,
        quantization::TensorFielder,
        tensor::Tensor,
        to_bit_sequence_le,
    };
    use ff_ext::ff::Field;

    type E = GoldilocksExt2;

    #[test]
    fn test_model_run() -> anyhow::Result<()> {
        test_model_run_helper()?;
        Ok(())
    }

    use std::path::PathBuf;

    fn workspace_root() -> PathBuf {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        PathBuf::from(manifest_dir).parent().unwrap().to_path_buf()
    }

    fn test_model_run_helper() -> anyhow::Result<()> {
        let filepath = workspace_root().join("zkml/assets/model.onnx");
        ModelType::MLP
            .validate(&filepath.to_string_lossy())
            .unwrap();
        let model = load_model::<Element>(&filepath.to_string_lossy()).unwrap();
        println!("[+] Loaded onnx file");
        let ctx = Context::<E>::generate(&model, None).expect("unable to generate context");
        println!("[+] Setup parameters");

        let shape = model.input_shape();
        assert_eq!(shape.len(), 1);
        let input = Tensor::random(vec![shape[0] - 1]);
        let input = model.prepare_input(input);

        let trace = model.run_feedforward(input.clone());
        let output = trace.final_output().clone();
        println!("[+] Run inference. Result: {:?}", output);

        let mut prover_transcript = default_transcript();
        let prover = Prover::<_, _>::new(&ctx, &mut prover_transcript);
        println!("[+] Run prover");
        let proof = prover.prove(trace).expect("unable to generate proof");

        let mut verifier_transcript = default_transcript();
        let io = IO::new(input.to_fields(), output.to_fields());
        verify::<_, _>(ctx, proof, io, &mut verifier_transcript).expect("invalid proof");
        println!("[+] Verify proof: valid");
        Ok(())
    }

    // TODO: move below code to a vector module

    #[test]
    fn test_vector_mle() {
        let n = (10 as usize).next_power_of_two();
        let v = (0..n).map(|_| E::random(&mut thread_rng())).collect_vec();
        let mle = v.clone().into_mle();
        let random_index = thread_rng().gen_range(0..v.len());
        let eval = to_bit_sequence_le(random_index, v.len().next_power_of_two().ilog2() as usize)
            .map(|b| E::from(b as u64))
            .collect_vec();
        let output = mle.evaluate(&eval);
        assert_eq!(output, v[random_index]);
    }
}

#[cfg(test)]
use std::sync::Once;

#[cfg(test)]
static INIT: Once = Once::new();

#[cfg(test)]
pub fn init_test_logging() {
    INIT.call_once(|| {
        // Initialize your logger only once
        env_logger::try_init().ok(); // The .ok() ignores if it's already been initialized
    });
}


/*
以下是对 lib.rs 文件的整体结构和关键部分的总结：

---

### **1. 文件功能**
该文件是项目的主入口模块,主要提供:
- 核心数据类型定义
- 模块导出和组织
- 基础工具函数
- 全局测试设施

---

### **2. 关键部分**

#### **模块组织**
- **核心模块**
  ```rust
  mod commit;
  pub mod iop;
  pub mod quantization;
  pub mod layers;
  pub mod lookup;
  pub mod model;
  mod onnx_parse;
  pub mod tensor;
  ```

#### **核心类型定义**
- **`Element`**
  - 功能: 项目的基础数值类型
  - 类型: `i128`
  - 说明: 支持较大范围以处理量化溢出

- **`Claim<E>`**
  - 功能: 协议中的声明类型
  - 字段:
    - `point: Vec<E>` - 多项式评估点
    - `eval: E` - 评估结果
  - 方法:
    - `pad()` - 填充评估点到指定大小
    - `new()` - 创建新实例

#### **工具 Trait 实现**
- **`VectorTranscript<E>`**
  - 功能: 处理向量形式的加密证明转录
  - 方法:
    - `read_challenges()` - 读取多个挑战值
  - 实现:
    - 生产环境: 生成真实随机挑战
    - 测试环境: 返回全1向量

#### **辅助函数**
- **`default_transcript()`**
  - 功能: 创建默认证明转录器
  - 返回: `BasicTranscript<E>`

- **`pad_vector<E>()`**
  - 功能: 将向量填充至2的幂长度
  - 实现: 使用零元素填充

- **`to_bit_sequence_le()`**
  - 功能: 生成数字的小端比特序列
  - 参数: 
    - `num: usize` - 输入数字
    - `bit_length: usize` - 期望位长度

- **`argmax()`**
  - 功能: 获取向量中最大值的索引
  - 泛型约束: `T: PartialOrd`

---

### **3. 测试设施**

#### **测试初始化**
- **`init_test_logging()`**
  - 功能: 单例模式初始化测试日志
  - 实现: 使用 `Once` 保证只初始化一次

#### **核心测试用例**
- **`test_model_run()`**
  - 功能: 端到端模型运行测试
  - 步骤:
    1. 加载 ONNX 模型
    2. 生成随机输入
    3. 运行推理
    4. 生成证明
    5. 验证证明

- **`test_vector_mle()`**
  - 功能: 测试向量多线性扩展
  - 验证: 随机点评估正确性

---

### **4. 特点与优化**
1. **模块化设计**
   - 清晰的模块划分
   - 灵活的公共接口暴露

2. **类型安全**
   - 泛型约束保证类型安全
   - 完善的错误处理

3. **测试覆盖**
   - 端到端测试
   - 单元测试
   - 测试辅助设施

---

### **5. 总结**
lib.rs 作为项目的根模块,通过合理的模块组织和类型定义,为整个项目提供了坚实的基础架构。它结合了零知识证明系统的核心功能与机器学习模型的运行环境,同时提供了完善的测试设施,确保了项目的可靠性和可维护性。
 */
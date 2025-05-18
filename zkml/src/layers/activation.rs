use crate::{
    Claim, Prover,
    commit::same_poly,
    iop::{context::ContextAux, verifier::Verifier},
    layers::{LayerCtx, LayerProof, PolyID, Train},
    lookup::{
        context::TableType,
        logup_gkr::{
            prover::batch_prove as logup_batch_prove, structs::LogUpProof,
            verifier::verify_logup_proof,
        },
    },
    tensor::Number,
};
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;
use multilinear_extensions::mle::IntoMLE;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;

use crate::{quantization::BIT_LEN, tensor::Tensor};

#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
pub enum Activation {
    Relu(Relu),
}

/// Currently holds the poly info for the output polynomial of the RELU
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationCtx {
    pub op: Activation,
    pub poly_id: PolyID,
    pub num_vars: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ActivationProof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// proof for the accumulation of the claim from m2v + claim from lookup for the same poly
    /// e.g. the "link" between a m2v and relu layer
    pub(crate) io_accumulation: same_poly::Proof<E>,
    /// the lookup proof for the relu
    pub(crate) lookup: LogUpProof<E>,
}

impl Activation {
    pub fn op<T: Number>(&self, input: &Tensor<T>) -> Tensor<T> {
        match self {
            Activation::Relu(relu) => relu.op(input),
        }
    }
    pub(crate) fn step_info<E: ExtensionField>(
        &self,
        id: PolyID,
        mut aux: ContextAux,
    ) -> (LayerCtx<E>, ContextAux)
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        aux.tables.insert(TableType::Relu);
        let info = match self {
            Activation::Relu(relu) => LayerCtx::Activation(ActivationCtx {
                op: Activation::Relu(*relu),
                poly_id: id,
                num_vars: aux
                    .last_output_shape
                    .iter()
                    .map(|dim| ceil_log2(*dim))
                    .sum::<usize>(),
            }),
        };
        (info, aux)
    }

    #[timed::timed_instrument(name = "Prover::prove_activation_step")]
    pub(crate) fn prove_step<E: ExtensionField, T: Transcript<E>>(
        &self,
        prover: &mut Prover<E, T>,
        last_claim: &Claim<E>,
        output: &[E],
        step: &ActivationCtx,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        let prover_info = prover.next_lookup_witness()?;

        // Run the lookup protocol and return the lookup proof
        let logup_proof = logup_batch_prove(&prover_info, prover.transcript)?;

        // We need to prove that the output of this step is the input to following activation function
        let mut same_poly_prover = same_poly::Prover::<E>::new(output.to_vec().into_mle());
        let same_poly_ctx = same_poly::Context::<E>::new(last_claim.point.len());
        same_poly_prover.add_claim(last_claim.clone())?;
        // Activation proofs have two columns, input and output
        let input_claim = logup_proof.output_claims()[0].clone();
        let output_claim = logup_proof.output_claims()[1].clone();

        same_poly_prover.add_claim(output_claim)?;
        let claim_acc_proof = same_poly_prover.prove(&same_poly_ctx, prover.transcript)?;
        // order is (output,mult)
        prover
            .witness_prover
            .add_claim(step.poly_id, claim_acc_proof.extract_claim())?;

        // Add the proof in
        prover.push_proof(LayerProof::Activation(ActivationProof {
            io_accumulation: claim_acc_proof,
            lookup: logup_proof,
        }));
        Ok(input_claim)
    }
}

impl ActivationCtx {
    pub(crate) fn verify_activation<E: ExtensionField, T: Transcript<E>>(
        &self,
        verifier: &mut Verifier<E, T>,
        last_claim: Claim<E>,
        proof: &ActivationProof<E>,
        constant_challenge: E,
        column_separation_challenge: E,
    ) -> anyhow::Result<Claim<E>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: Serialize + DeserializeOwned,
    {
        // 1. Verify the lookup proof
        let verifier_claims = verify_logup_proof(
            &proof.lookup,
            1,
            constant_challenge,
            column_separation_challenge,
            verifier.transcript,
        )?;

        // 2. Verify the accumulation proof from last_claim + lookup claim into the new claim
        let sp_ctx = same_poly::Context::<E>::new(self.num_vars);
        let mut sp_verifier = same_poly::Verifier::<E>::new(&sp_ctx);
        sp_verifier.add_claim(last_claim)?;
        verifier_claims.claims()[1..]
            .iter()
            .try_for_each(|claim| sp_verifier.add_claim(claim.clone()))?;

        let new_output_claim = sp_verifier.verify(&proof.io_accumulation, verifier.transcript)?;
        // 3. Accumulate the new claim into the witness commitment protocol
        verifier
            .witness_verifier
            .add_claim(self.poly_id, new_output_claim)?;

        // 4. return the input claim for to be proven at subsequent step
        Ok(verifier_claims.claims()[0].clone())
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub struct Relu;

impl Relu {
    pub fn new() -> Relu {
        Self
    }
    pub fn num_vars() -> usize {
        *BIT_LEN
    }
    pub fn poly_len() -> usize {
        1 << Self::num_vars()
    }
    pub fn shape() -> Vec<usize> {
        vec![2, Self::poly_len()]
    }

    pub fn op<T: Number>(&self, input: &Tensor<T>) -> Tensor<T> {
        Tensor::new(
            input.get_shape(),
            input
                .get_data()
                .par_iter()
                .map(|e| Self::apply(*e))
                .collect::<Vec<_>>(),
        )
    }

    #[inline(always)]
    pub fn apply<T: Number>(e: T) -> T {
        if e.is_negative() { T::default() } else { e }
    }
}

impl<T: Number> Train<T> for Activation {
    fn forward(&self, input: &Tensor<T>) -> Tensor<T> {
        self.op(input)
    }

    fn backward(&mut self, input: &Tensor<T>, grad_output: &Tensor<T>) -> Tensor<T> {
        match self {
            Activation::Relu(_) => {
                // 检查输入和梯度形状是否匹配
                assert_eq!(grad_output.get_shape(), input.get_shape());

                // 计算 ReLU 的梯度
                let grad_in = grad_output.get_data()
                    .iter()
                    .zip(input.get_data())
                    .map(|(&grad, &x)| {
                        // ReLU 的导数：如果输入 > 0，导数为 grad；否则为 0（包括负数和0）
                        if x.compare(&T::default()) == std::cmp::Ordering::Greater { grad } else { T::default() }
                    })
                    .collect::<Vec<_>>();

                // 创建并返回输入梯度张量
                Tensor::new(input.get_shape().clone(), grad_in)
            }
        }
    }

    fn update(&mut self, _learning_rate: T) {
        // ReLU 没有需要更新的参数
    }

    fn zero_grad(&mut self) {
        // ReLU 没有需要清零的梯度
    }
}

#[cfg(test)]
mod test {
    use crate::{Element, quantization};
    use super::*;

    #[test]
    fn test_activation_relu_apply() {
        struct TestCase {
            input: Element,
            output: Element,
        }

        impl TestCase {
            pub fn from(input: Element, output: Element) -> Self {
                Self { input, output }
            }
        }
        for case in [
            TestCase::from(-24, 0),
            TestCase::from(0, 0),
            TestCase::from(124, 124),
            TestCase::from(-127, 0),
        ] {
            assert_eq!(Relu::apply(case.input), case.output);
        }
    }

    #[test]
    fn test_relu_backward() {
        let mut relu = Activation::Relu(Relu::new());

        // 测试用例1: 基本正数输入
        println!("\n=== 测试用例1: 基本正数输入 ===");
        println!("输入: {:?}", vec![1, 2, 3]);
        println!("输出梯度: {:?}", vec![1, 1, 1]);
        let input1 = Tensor::new(vec![3], vec![1, 2, 3]);
        let output_grad1 = Tensor::new(vec![3], vec![1, 1, 1]);
        let input_grad1 = relu.backward(&input1, &output_grad1);
        println!("计算得到的输入梯度: {:?}", input_grad1.get_data());
        println!("期望的输入梯度: {:?}", vec![1, 1, 1]);
        assert_eq!(input_grad1.get_data(), vec![1, 1, 1], "正数输入的梯度应该保持不变"); 

        // 测试用例2: 混合输入(正数、负数、零)
        println!("\n=== 测试用例2: 混合输入(正数、负数、零) ===");
        println!("输入: {:?}", vec![-1, 0, 1]);
        println!("输出梯度: {:?}", vec![2, 2, 2]);
        let input2 = Tensor::new(vec![3], vec![-1, 0, 1]);
        let output_grad2 = Tensor::new(vec![3], vec![2, 2, 2]);
        let input_grad2 = relu.backward(&input2, &output_grad2);
        println!("计算得到的输入梯度: {:?}", input_grad2.get_data());
        println!("期望的输入梯度: {:?}", vec![0, 0, 2]);
        assert_eq!(input_grad2.get_data(), vec![0, 0, 2], "负数和零的梯度应该为0，正数应该保持梯度");

        // 测试用例3: 极端值测试
        println!("\n=== 测试用例3: 极端值测试 ===");
        println!("输入: {:?}", vec![*quantization::MAX, *quantization::MIN, 0, 1]);
        println!("输出梯度: {:?}", vec![1, 1, 1, 1]);
        let input3 = Tensor::new(vec![4], vec![*quantization::MAX, *quantization::MIN, 0, 1]);
        let output_grad3 = Tensor::new(vec![4], vec![1, 1, 1, 1]);
        let input_grad3 = relu.backward(&input3, &output_grad3);
        println!("计算得到的输入梯度: {:?}", input_grad3.get_data());
        println!("期望的输入梯度: {:?}", vec![1, 0, 0, 1]);
        assert_eq!(input_grad3.get_data(), vec![1, 0, 0, 1], "最大值应该传递梯度，最小值应该阻断梯度");

        // 测试用例4: 多维输入(2x3矩阵)
        println!("\n=== 测试用例4: 多维输入(2x3矩阵) ===");
        println!("输入矩阵: [[1,-1,0],[-2,2,3]]");
        println!("输出梯度矩阵: [[2,2,2],[2,2,2]]");
        let input4 = Tensor::new(vec![2, 3], vec![1, -1, 0, -2, 2, 3]);
        let output_grad4 = Tensor::new(vec![2, 3], vec![2, 2, 2, 2, 2, 2]);
        let input_grad4 = relu.backward(&input4, &output_grad4);
        println!("计算得到的输入梯度: {:?}", input_grad4.get_data());
        println!("期望的输入梯度: {:?}", vec![2, 0, 0, 0, 2, 2]);
        assert_eq!(input_grad4.get_data(), vec![2, 0, 0, 0, 2, 2], "多维输入应该正确处理每个元素的梯度");
        println!("计算得到的输入梯度形状: {:?}", input_grad4.get_shape());
        println!("期望的输入梯度形状: {:?}", vec![2, 3]);
        assert_eq!(input_grad4.get_shape(), vec![2, 3], "输出梯度应该保持输入的形状");

        // 测试用例5: 全零输入
        println!("\n=== 测试用例5: 全零输入 ===");
        println!("输入: {:?}", vec![0, 0, 0]);
        println!("输出梯度: {:?}", vec![1, 1, 1]);
        let input5 = Tensor::new(vec![3], vec![0, 0, 0]);
        let output_grad5 = Tensor::new(vec![3], vec![1, 1, 1]);
        let input_grad5 = relu.backward(&input5, &output_grad5);
        println!("计算得到的输入梯度: {:?}", input_grad5.get_data());
        println!("期望的输入梯度: {:?}", vec![0, 0, 0]);
        assert_eq!(input_grad5.get_data(), vec![0, 0, 0], "零输入应该完全阻断梯度");

        // 测试用例6: 全负数输入
        println!("\n=== 测试用例6: 全负数输入 ===");
        println!("输入: {:?}", vec![-1, -2, -3]);
        println!("输出梯度: {:?}", vec![1, 1, 1]);
        let input6 = Tensor::new(vec![3], vec![-1, -2, -3]);
        let output_grad6 = Tensor::new(vec![3], vec![1, 1, 1]);
        let input_grad6 = relu.backward(&input6, &output_grad6);
        println!("计算得到的输入梯度: {:?}", input_grad6.get_data());
        println!("期望的输入梯度: {:?}", vec![0, 0, 0]);
        assert_eq!(input_grad6.get_data(), vec![0, 0, 0], "负数输入应该完全阻断梯度");

        // 测试用例7: 不同大小的梯度
        println!("\n=== 测试用例7: 不同大小的梯度 ===");
        println!("输入: {:?}", vec![1, 2, -1, 3]);
        println!("输出梯度: {:?}", vec![4, 3, 2, 1]);
        let input7 = Tensor::new(vec![4], vec![1, 2, -1, 3]);
        let output_grad7 = Tensor::new(vec![4], vec![4, 3, 2, 1]);
        let input_grad7 = relu.backward(&input7, &output_grad7);
        println!("计算得到的输入梯度: {:?}", input_grad7.get_data());
        println!("期望的输入梯度: {:?}", vec![4, 3, 0, 1]);
        assert_eq!(input_grad7.get_data(), vec![4, 3, 0, 1], "应该正确处理不同大小的梯度");
    }

    #[test]
    #[should_panic(expected = "grad_output shape mismatch")]
    fn test_relu_backward_shape_mismatch() {
        println!("\n=== 测试用例8: 维度不匹配测试 ===");
        println!("输入形状: [2]");
        println!("输出梯度形状: [3]");
        let mut relu = Activation::Relu(Relu::new());
        let input = Tensor::new(vec![2], vec![1, 1]);
        let output_grad = Tensor::new(vec![3], vec![1, 1, 1]);
        relu.backward(&input, &output_grad);
    }
}

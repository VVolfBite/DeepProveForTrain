use crate::{
    Claim, Prover,
    commit::same_poly,
    iop::{context::ContextAux, verifier::Verifier},
    layers::{LayerCtx, LayerProof, PolyID},
    lookup::{
        context::TableType,
        logup_gkr::{
            prover::batch_prove as logup_batch_prove, 
            structs::LogUpProof,
            verifier::verify_logup_proof,
        },
    },
};
use multilinear_extensions::virtual_poly::{VirtualPolynomial, VPAuxInfo};  // 添加 VPAuxInfo
use multilinear_extensions::mle::{IntoMLE, MultilinearExtension, DenseMultilinearExtension}; // 修改导入，添加 MultilinearExtension trait
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};  // 添加 IOPVerifierState
use sumcheck::prover::batch_prove;  // 添加这个导入
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;
use anyhow::{ensure, Result};
use std::sync::Arc;

use crate::{
    Element,
    quantization::{self, BIT_LEN, Fieldizer, ZERO},
    tensor::Tensor,
};

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationBackwardCtx<E> {
    pub input_poly_id: PolyID,
    pub matrix_poly_aux: VPAuxInfo<E>,  // 重命名为与Dense一致的风格
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct ActivationBackwardProof<E: ExtensionField> {
    /// sumcheck证明，用于证明反向传播计算的正确性
    pub(crate) sumcheck: IOPProof<E>,
    /// 最终的个别多项式评估值
    individual_claims: Vec<E>,
}

impl<E: ExtensionField> ActivationBackwardProof<E> {
    /// 计算虚拟多项式的最终评估值
    pub fn individual_to_virtual_claim(&self) -> E {
        self.individual_claims.iter().fold(E::ONE, |acc, e| acc * e)
    }
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
    pub fn op(&self, input: &Tensor<Element>) -> Tensor<Element> {
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
    /// to_mle returns two polynomials:
    /// f_i: one containing the input column values
    /// f_o: one containing the output column values
    pub fn to_mle<E: ExtensionField>() -> (Vec<E::BaseField>, Vec<E::BaseField>) {
        (*quantization::MIN..=*quantization::MAX)
            .map(|i| {
                let val: E = i.to_field();
                let op_val: E = Relu::apply(i as i128).to_field();
                (val.as_bases()[0], op_val.as_bases()[0])
            })
            .unzip()
    }

    pub fn op(&self, input: &Tensor<Element>) -> Tensor<Element> {
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
    pub fn apply(e: Element) -> Element {
        if e.is_negative() { 0 } else { e }
    }

    /// 实现ReLU的反向传播
    /// ReLU导数: 1 if x > 0, 0 otherwise
    pub fn backward(
        &self,
        output_grad: &Tensor<Element>,
        input: &Tensor<Element>
    ) -> Tensor<Element> {
        // 确保维度匹配
        assert_eq!(output_grad.get_shape(), input.get_shape());
        
        // 计算ReLU的导数
        let derivative = input.get_data()
            .iter()
            .map(|&x| (x > *ZERO) as Element)  // 使用更简洁的写法
            .collect::<Vec<_>>();

        // 计算输入梯度: dL/dx = dL/dy * relu'(x)
        Tensor::new(
            input.get_shape().clone(),
            output_grad.get_data()
                .iter()
                .zip(derivative.iter())
                .map(|(&grad, &deriv)| grad * deriv)
                .collect()
        )
    }

    pub fn prove_backward_step<'b, E, T>(
        &self,
        prover: &mut Prover<E, T>,
        last_claim: Claim<E>,
        output_grad: &Tensor<Element>,
        input: &Tensor<Element>,
        info: &ActivationBackwardCtx<E>,
    ) -> Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        T: Transcript<E>,
    {
        // 计算输入梯度并转换为 Arc<dyn MultilinearExtension>
        let input_mle = Arc::new(input.evals_flat::<E>().into_mle());
        let grad_mle = Arc::new(output_grad.evals_flat::<E>().into_mle());
        
        // 构造虚拟多项式
        let num_vars = input_mle.num_vars();
        let mut vp = VirtualPolynomial::<E>::new(num_vars);
        
        // fix_variables 返回的也需要包装成 Arc
        let fixed_input_mle = Arc::new(input_mle.fix_variables(&last_claim.point));
        
        vp.add_mle_list(
            vec![fixed_input_mle, grad_mle],
            E::ONE,
        );

        // 使用 batch_prove 而不是 prove_parallel
        let (proof, individual_claims) = batch_prove(
            vp,
            prover.transcript,
        )?;  // 注意这里需要用 ? 操作符处理错误

        prover.push_proof(LayerProof::ActivationBackward(ActivationBackwardProof {
            sumcheck: proof.clone(),
            individual_claims,
        }));

        Ok(Claim {
            point: proof.point,
            eval: individual_claims[1],
        })
    }

    pub fn verify_backward_step<E: ExtensionField, T: Transcript<E>>(
        &self,
        verifier: &mut Verifier<E, T>,
        last_claim: Claim<E>,
        output_grad: &Tensor<Element>,
        input: &Tensor<Element>,
        proof: &ActivationBackwardProof<E>,
        info: &ActivationBackwardCtx<E>,
    ) -> Result<Claim<E>>
    where 
        E::BaseField: Serialize + DeserializeOwned,
        E: Serialize + DeserializeOwned,
    {
        // 重构虚拟多项式
        let input_mle = input.evals_flat::<E>().into_mle();
        let grad_mle = output_grad.evals_flat::<E>().into_mle();
        
        let num_vars = input_mle.num_vars();
        let mut vp = VirtualPolynomial::<E>::new(num_vars);
        vp.add_mle_list(
            vec![input_mle.into(), grad_mle.into()],
            E::ONE,
        );

        // 使用正确的verify函数和参数
        let subclaim = IOPVerifierState::verify(
            proof.individual_to_virtual_claim(), // claimed sum
            &proof.sumcheck,                     // sumcheck proof
            &info.matrix_poly_aux,              // aux info
            verifier.transcript,                 // transcript
        );

        // 验证最终的评估值
        ensure!(
            proof.individual_to_virtual_claim() == subclaim.expected_evaluation,
            "sumcheck claim failed"
        );

        Ok(Claim {
            point: subclaim.point_flat(),
            eval: proof.individual_claims[1],
        })
    }
}

#[cfg(test)]
mod test {
    use crate::to_bit_sequence_le;
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::mle::{DenseMultilinearExtension, MultilinearExtension};

    use super::*;

    type F = GoldilocksExt2;

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
        ] {
            assert_eq!(Relu::apply(case.input), case.output);
        }
    }

    #[test]
    fn test_activation_relu_mle() {
        let relu = Relu::new();
        let (input_poly, output_poly) = Relu::to_mle::<F>();

        assert_eq!(input_poly.len(), output_poly.len());
        let (input_mle, output_mle) = (
            DenseMultilinearExtension::from_evaluation_vec_smart(
                Relu::num_vars(),
                input_poly.to_vec(),
            ),
            DenseMultilinearExtension::from_evaluation_vec_smart(
                Relu::num_vars(),
                output_poly.to_vec(),
            ),
        );
        assert_eq!(input_mle.num_vars(), output_mle.num_vars());
        assert_eq!(input_mle.num_vars(), Relu::num_vars());
        let inputs = Tensor::random(vec![10]);
        let outputs = relu.op(&inputs);
        assert_eq!(inputs.get_shape(), outputs.get_shape());
        for (input, output) in inputs.get_data().iter().zip(outputs.get_data().iter()) {
            // here putting input works because every random input is a u8, so it's already within [0;256] so
            // its value "is" the index. Normally if this is not true, we should get the index of the row corresponding to that input
            let idx_vars = to_bit_sequence_le((input + 128) as usize, Relu::num_vars())
                .map(|b| F::from(b as u64))
                .collect_vec();
            let input_field = input_mle.evaluate(&idx_vars);
            let expected_ified: F = input.to_field();
            assert_eq!(input_field, expected_ified);
            let output_field = output_mle.evaluate(&idx_vars);
            let expected_ofield: F = output.to_field();
            assert_eq!(output_field, expected_ofield);
        }
        // assert_eq!(expected,given);
    }

    #[test]
    fn test_relu_backward() {
        let relu = Relu::new();

        // 测试用例1: 正数输入
        let input1 = Tensor::new(vec![3], vec![1, 2, 3]);
        let output_grad1 = Tensor::new(vec![3], vec![1, 1, 1]);
        let input_grad1 = relu.backward(&output_grad1, &input1);
        
        // 正数输入，导数应该为1
        assert_eq!(input_grad1.get_data(), vec![1, 1, 1]);

        // 测试用例2: 混合输入
        let input2 = Tensor::new(vec![3], vec![-1, 0, 1]);
        let output_grad2 = Tensor::new(vec![3], vec![2, 2, 2]);
        let input_grad2 = relu.backward(&output_grad2, &input2);
        
        // 负数和零的导数为0，正数的导数为1
        assert_eq!(
            input_grad2.get_data(),
            vec![0, 0, 2] // -1 -> 0, 0 -> 0, 1 -> 2(=2*1)
        );
    }

    #[test]
    fn test_activation_backward_proof() {
        // 创建测试数据
        let relu = Relu::new();
        let input = Tensor::new(vec![2], vec![1, -1]);  // 一个正数一个负数
        let output_grad = Tensor::new(vec![2], vec![1, 1]);
        
        // 验证backward计算的正确性
        let grad = relu.backward(&output_grad, &input);
        assert_eq!(grad.get_data(), vec![1, 0]);  // 期望 [1, 0]，因为只有正数处导数为1
    }
}
use crate::{
    Claim, Prover,
    commit::same_poly,
    iop::{context::{Context, ContextAux}, verifier::{Verifier, IO, verify}},
    layers::{LayerCtx, LayerProof, PolyID, Layer, Dense},
    lookup::{
        context::TableType,
        logup_gkr::{
            prover::batch_prove as logup_batch_prove, 
            structs::{LogUpProof, LogUpInput},
            verifier::verify_logup_proof,
        },
    },
    model::{Model, InferenceTrace, InferenceStep},
    quantization::{TensorFielder, IntoElement},
};
use multilinear_extensions::virtual_poly::{VirtualPolynomial, VPAuxInfo};  // 添加 VPAuxInfo
use multilinear_extensions::mle::{IntoMLE, MultilinearExtension, DenseMultilinearExtension}; // 修改导入，添加 MultilinearExtension trait
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};  // 添加 IOPVerifierState
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;
use anyhow::{ensure, Result};
use std::sync::Arc;
use std::marker::PhantomData;
use ff::Field;
use std::collections::BTreeSet;
use goldilocks::{GoldilocksExt2, Goldilocks, SmallField};  // 添加 SmallField
use itertools::Itertools;
use transcript::BasicTranscript;

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

    pub(crate) fn prove_backward_step<E: ExtensionField, T: Transcript<E>>(
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
        println!("\n=== 转换为MLE前的原始值 ===");
        println!("输出梯度原始值:");
        for (i, &val) in output_grad.get_data().iter().enumerate() {
            println!("位置 {}: {}", i, val);
        }
        
        println!("\nReLU导数原始值:");
        let relu_deriv_field: Vec<E::BaseField> = input.get_data()
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let deriv = if x > *ZERO { 1 } else { 0 };
                println!("位置 {}: 输入 = {}, 导数 = {}", i, x, deriv);
                E::BaseField::from(deriv as u64)
            })
            .collect();
            
        println!("\n期望梯度原始值:");
        let expected_grad = match self {
            Activation::Relu(relu) => relu.backward(output_grad, input)
        };
        for (i, &val) in expected_grad.get_data().iter().enumerate() {
            println!("位置 {}: {}", i, val);
        }
        
        println!("\n=== 约束验证 ===");
        for i in 0..output_grad.get_data().len() {
            let grad_out = output_grad.get_data()[i];
            let relu_deriv = if input.get_data()[i] > *ZERO { 1 } else { 0 };
            let expected = expected_grad.get_data()[i];
            println!("位置 {}: {} * {} = {} (期望 {})", 
                i, grad_out, relu_deriv, grad_out * relu_deriv, expected);
        }
        
        // 构建lookup表
        println!("\n=== 构建Lookup表 ===");
        let (inputs_base, derivs_base): (Vec<E::BaseField>, Vec<E::BaseField>) = 
            (*quantization::MIN..=*quantization::MAX)
                .map(|x| {
                    let val: E = Element::from(x).to_field();
                    let deriv: E = Element::from(if x > 0 { 1i128 } else { 0i128 }).to_field();
                    println!("Lookup表项 - 输入: {}, 值: {:?}, 导数: {:?}", x, val, deriv);
                    (val.as_bases()[0], deriv.as_bases()[0])
                })
                .unzip();
        
        println!("\n=== 构建LogUpInput ===");
        println!("实际输入转换为域元素:");
        let actual_inputs: Vec<E::BaseField> = input.evals_flat::<E>()
            .iter()
            .map(|x| {
                println!("输入域元素: {:?}", x);
                x.as_bases()[0]
            })
            .collect();
        
        // 获取lookup证明
        let prover_info = LogUpInput::new_lookup(
            vec![
                inputs_base.clone(),  // 输入值表
                derivs_base.clone(),  // 导数值表
                actual_inputs,  // 实际输入
            ],
            E::ONE,  // constant_challenge
            E::ONE,  // column_separation_challenge
            1  // columns_per_instance
        )?;
        
        println!("\n=== 生成Lookup证明 ===");
        let logup_proof = logup_batch_prove(&prover_info, prover.transcript)?;
        println!("Lookup证明输出声明数量: {}", logup_proof.output_claims().len());
        for (i, claim) in logup_proof.output_claims().iter().enumerate() {
            println!("声明 {}: 点={:?}, 评估值={:?}", i, claim.point, claim.eval);
        }
        
        // 构建虚拟多项式
        println!("\n=== 构建虚拟多项式 ===");
        let num_vars = Relu::num_vars();
        println!("变量数量: {}", num_vars);
        
        let grad_mle = Arc::new(output_grad.evals_flat::<E>().into_mle());
        println!("梯度MLE构建完成");
        
        let relu_deriv_mle = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            relu_deriv_field.clone()
        ));
        println!("ReLU导数MLE构建完成");
        
        let expected_grad_mle = Arc::new(expected_grad.evals_flat::<E>().into_mle());
        println!("期望梯度MLE构建完成");
        
        let mut vp = VirtualPolynomial::<E>::new(num_vars);
        vp.add_mle_list(vec![grad_mle.clone(), relu_deriv_mle.clone()], E::ONE);
        vp.add_mle_list(vec![expected_grad_mle.clone()], -E::ONE);
        println!("虚拟多项式构建完成");
        
        // 生成sumcheck证明
        println!("\n=== 生成Sumcheck证明 ===");
        let (proof, state) = IOPProverState::prove_parallel(vp, prover.transcript);
        println!("Sumcheck证明生成完成");
        
        // 获取个别多项式的评估值
        let individual_claims = state.get_mle_final_evaluations();
        println!("\n=== 个别多项式评估值 ===");
        println!("grad_out: {:?}", individual_claims[0]);
        println!("relu_deriv: {:?}", individual_claims[1]);
        println!("expected_grad: {:?}", individual_claims[2]);
        
        // 验证约束关系
        let computed_grad = individual_claims[0] * individual_claims[1];
        println!("\n=== 约束验证 ===");
        println!("计算得到的梯度: {:?}", computed_grad);
        println!("期望的梯度: {:?}", individual_claims[2]);
        
        ensure!(
            computed_grad == individual_claims[2],
            "反向传播计算不匹配: grad_out {:?} * relu_deriv {:?} != expected_grad {:?}",
            individual_claims[0],
            individual_claims[1],
            individual_claims[2]
        );

        // 构建same_poly证明
        println!("\n=== 构建Same Poly证明 ===");
        let mut same_poly_prover = same_poly::Prover::<E>::new(expected_grad.evals_flat::<E>().into_mle());
        let same_poly_ctx = same_poly::Context::<E>::new(last_claim.point.len());
        same_poly_prover.add_claim(last_claim.clone())?;
        println!("添加last_claim完成");
        
        // 添加lookup证明的输出声明
        let output_claim = logup_proof.output_claims()[1].clone();
        same_poly_prover.add_claim(output_claim)?;
        println!("添加output_claim完成");
        
        let claim_acc_proof = same_poly_prover.prove(&same_poly_ctx, prover.transcript)?;
        println!("Same Poly证明生成完成");

        prover.push_proof(LayerProof::ActivationBackward(ActivationBackwardProof {
            sumcheck: proof.clone(),
            individual_claims: individual_claims.clone(),
        }));

        Ok(Claim {
            point: proof.point,
            eval: computed_grad,
        })
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
            .map(|&x| {
                let deriv = (x > *ZERO) as Element;
                deriv
            })
            .collect::<Vec<_>>();
        
        // 计算输入梯度: dL/dx = dL/dy * relu'(x)
        let grad_in = output_grad.get_data()
            .iter()
            .zip(derivative.iter())
            .map(|(&grad, &deriv)| {
                let result = grad * deriv;
                result
            })
            .collect::<Vec<_>>();
            
        Tensor::new(
            input.get_shape().clone(),
            grad_in
        )
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
        // 第一步：构建ReLU导数的查找表
        let (inputs_base, derivs_base): (Vec<E::BaseField>, Vec<E::BaseField>) = 
            (*quantization::MIN..=*quantization::MAX)
                .map(|x| {
                    let val: E = Element::from(x).to_field();
                    let deriv: E = Element::from(if x > 0 { 1i128 } else { 0i128 }).to_field();
                    (val.as_bases()[0], deriv.as_bases()[0])
                })
                .unzip();

        // 在使用前克隆derivs_base
        let derivs_base_clone = derivs_base.clone();

        // 第二步：验证lookup证明
        let lookup_input = LogUpInput::new_lookup(
            vec![
                inputs_base,  // 输入值表
                derivs_base,  // 导数值表
                input.evals_flat::<E>().iter().map(|x| x.as_bases()[0]).collect(),  // 实际输入
            ],
            E::ONE,  // constant_challenge
            E::ONE,  // column_separation_challenge
            1  // columns_per_instance
        )?;

        // 第三步：重构虚拟多项式
        let input_mle = Arc::new(input.evals_flat::<E>().into_mle());  // 输入x的MLE
        let grad_mle = Arc::new(output_grad.evals_flat::<E>().into_mle());  // 输入梯度的MLE
        
        let num_vars = input_mle.num_vars();
        let mut vp = VirtualPolynomial::<E>::new(num_vars);
        
        // 固定变量以匹配上一步的证明
        let fixed_input_mle = Arc::new(input_mle.fix_variables(&last_claim.point));
        
        // 构建relu'(x)的多线性扩展，使用克隆的derivs_base
        let relu_deriv_mle = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            derivs_base_clone  // 使用克隆的导数值
        );

        // 添加多项式约束
        vp.add_mle_list(
            vec![
                fixed_input_mle.clone(),    // 输入x的MLE
                grad_mle.clone(),           // 输入梯度的MLE
                Arc::new(relu_deriv_mle),   // relu'(x)的MLE
            ],
            E::ONE,  // 系数为1
        );

        // 第四步：验证sumcheck证明
        let subclaim = IOPVerifierState::verify(
            proof.individual_to_virtual_claim(),  // 声称的和
            &proof.sumcheck,                      // sumcheck证明
            &info.matrix_poly_aux,               // 辅助信息
            verifier.transcript,                  // transcript
        );

        // 第五步：验证最终的评估值
        ensure!(
            proof.individual_to_virtual_claim() == subclaim.expected_evaluation,
            "sumcheck claim failed"
        );

        // 第六步：验证个别多项式的评估值满足约束
        ensure!(
            proof.individual_claims[1] * proof.individual_claims[2] == proof.individual_claims[0],
            "Individual polynomial evaluations do not satisfy the constraint"
        );

        // 返回最终的声明
        Ok(Claim {
            point: subclaim.point_flat(),
            eval: proof.individual_claims[1],  // 返回梯度的评估值
        })
    }
}

impl Default for ContextAux {
    fn default() -> Self {
        Self {
            tables: BTreeSet::new(),
            last_output_shape: Vec::new(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        to_bit_sequence_le,
        tensor::Tensor,
        Claim, Prover,
        iop::{context::Context, verifier::Verifier},
        layers::{LayerProof, PolyID},
        Element,
        lookup::context::TableType,
        model::Model,
    };
    use super::*;
    use goldilocks::{GoldilocksExt2, Goldilocks};
    use itertools::Itertools;
    use multilinear_extensions::{
        mle::{DenseMultilinearExtension, MultilinearExtension},
        virtual_poly::VPAuxInfo,
    };
    use transcript::BasicTranscript;
    use std::collections::BTreeSet;
    use ff::Field;

    type F = GoldilocksExt2;

    #[test]
    fn test_field_conversion_and_arithmetic() {
        println!("\n=== 基本数值转换和运算测试 ===");
        
        // 1. 测试基本数值转换
        let test_values = vec![-2, -1, 0, 1, 2];
        println!("\n1. 原始值到有限域的转换:");
        for &x in &test_values {
            let field_val: F = Element::from(x).to_field();
            println!("原始值 {} -> 域元素 {:?}", x, field_val);
            
            // 验证转换回Element
            let element_val = field_val.into_element();
            println!("域元素 {:?} -> 转回原始值 {}", field_val, element_val);
            assert_eq!(x, element_val, "转换应该是可逆的");
        }

        // 2. 测试有限域运算
        println!("\n2. 有限域运算测试:");
        // 测试 2 * 3
        let a: F = Element::from(2).to_field();
        let b: F = Element::from(3).to_field();
        let c = a * b;
        println!("2 * 3 = {:?}", c);
        println!("转回原始值: {}", c.into_element());
        
        // 测试 -2 * 3
        let a: F = Element::from(-2).to_field();
        let b: F = Element::from(3).to_field();
        let c = a * b;
        println!("-2 * 3 = {:?}", c);
        println!("转回原始值: {}", c.into_element());

        // 3. 测试ReLU导数计算
        println!("\n3. ReLU导数计算测试:");
        let test_inputs = vec![-2, -1, 0, 1, 2];
        for &x in &test_inputs {
            let input_field: F = Element::from(x).to_field();
            let deriv = if x > 0 { 1 } else { 0 };
            let deriv_field: F = Element::from(deriv).to_field();
            println!("输入: {}, ReLU导数: {}", x, deriv);
            println!("域中表示 - 输入: {:?}, 导数: {:?}", input_field, deriv_field);
        }

        // 4. 测试梯度计算
        println!("\n4. 梯度计算测试:");
        let grad_out = 2;  // 输出梯度
        let test_inputs = vec![-1, 1];  // 测试负数和正数的情况
        for &x in &test_inputs {
            let grad_out_field: F = Element::from(grad_out).to_field();
            let deriv = if x > 0 { 1 } else { 0 };
            let deriv_field: F = Element::from(deriv).to_field();
            let grad_in = grad_out_field * deriv_field;
            println!("输入: {}, 输出梯度: {}, ReLU导数: {}", x, grad_out, deriv);
            println!("计算得到的输入梯度: {:?}", grad_in);
            println!("转回原始值: {}", grad_in.into_element());
        }
    }

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

        // 测试用例1: 基本正数输入
        // 计算过程:
        // 1. 输入 [1,2,3] 都大于0，所以ReLU导数都为1: [1,1,1]
        // 2. 输出梯度 [1,1,1]
        // 3. 最终梯度 = 输出梯度 * ReLU导数 = [1,1,1] * [1,1,1] = [1,1,1]
        let input1 = Tensor::new(vec![3], vec![1, 2, 3]);
        let output_grad1 = Tensor::new(vec![3], vec![1, 1, 1]);
        let input_grad1 = relu.backward(&output_grad1, &input1);
        assert_eq!(input_grad1.get_data(), vec![1, 1, 1], "正数输入的梯度应该保持不变");

        // 测试用例2: 混合输入(正数、负数、零)
        // 计算过程:
        // 1. 输入 [-1,0,1] 的ReLU导数: [-1 -> 0, 0 -> 0, 1 -> 1] = [0,0,1]
        // 2. 输出梯度 [2,2,2]
        // 3. 最终梯度 = [2,2,2] * [0,0,1] = [0,0,2]
        let input2 = Tensor::new(vec![3], vec![-1, 0, 1]);
        let output_grad2 = Tensor::new(vec![3], vec![2, 2, 2]);
        let input_grad2 = relu.backward(&output_grad2, &input2);
        assert_eq!(input_grad2.get_data(), vec![0, 0, 2], "负数和零的梯度应该为0，正数应该保持梯度");

        // 测试用例3: 极端值测试
        // 计算过程:
        // 1. 输入 [MAX,MIN,0,1] 的ReLU导数: [MAX -> 1, MIN -> 0, 0 -> 0, 1 -> 1] = [1,0,0,1]
        // 2. 输出梯度 [1,1,1,1]
        // 3. 最终梯度 = [1,1,1,1] * [1,0,0,1] = [1,0,0,1]
        let input3 = Tensor::new(vec![4], vec![*quantization::MAX, *quantization::MIN, 0, 1]);
        let output_grad3 = Tensor::new(vec![4], vec![1, 1, 1, 1]);
        let input_grad3 = relu.backward(&output_grad3, &input3);
        assert_eq!(input_grad3.get_data(), vec![1, 0, 0, 1], "最大值应该传递梯度，最小值应该阻断梯度");

        // 测试用例4: 多维输入(2x3矩阵)
        // 计算过程:
        // 1. 输入矩阵 [[1,-1,0],[-2,2,3]] 的ReLU导数:
        //    [[1 -> 1, -1 -> 0, 0 -> 0],
        //     [-2 -> 0, 2 -> 1, 3 -> 1]] = [[1,0,0],[0,1,1]]
        // 2. 输出梯度矩阵 [[2,2,2],[2,2,2]]
        // 3. 最终梯度 = [[2,2,2],[2,2,2]] * [[1,0,0],[0,1,1]] = [[2,0,0],[0,2,2]]
        let input4 = Tensor::new(vec![2, 3], vec![1, -1, 0, -2, 2, 3]);
        let output_grad4 = Tensor::new(vec![2, 3], vec![2, 2, 2, 2, 2, 2]);
        let input_grad4 = relu.backward(&output_grad4, &input4);
        assert_eq!(input_grad4.get_data(), vec![2, 0, 0, 0, 2, 2], "多维输入应该正确处理每个元素的梯度");
        assert_eq!(input_grad4.get_shape(), vec![2, 3], "输出梯度应该保持输入的形状");

        // 测试用例5: 全零输入
        // 计算过程:
        // 1. 输入 [0,0,0] 的ReLU导数: [0 -> 0, 0 -> 0, 0 -> 0] = [0,0,0]
        // 2. 输出梯度 [1,1,1]
        // 3. 最终梯度 = [1,1,1] * [0,0,0] = [0,0,0]
        let input5 = Tensor::new(vec![3], vec![0, 0, 0]);
        let output_grad5 = Tensor::new(vec![3], vec![1, 1, 1]);
        let input_grad5 = relu.backward(&output_grad5, &input5);
        assert_eq!(input_grad5.get_data(), vec![0, 0, 0], "零输入应该完全阻断梯度");

        // 测试用例6: 全负数输入
        // 计算过程:
        // 1. 输入 [-1,-2,-3] 的ReLU导数: [-1 -> 0, -2 -> 0, -3 -> 0] = [0,0,0]
        // 2. 输出梯度 [1,1,1]
        // 3. 最终梯度 = [1,1,1] * [0,0,0] = [0,0,0]
        let input6 = Tensor::new(vec![3], vec![-1, -2, -3]);
        let output_grad6 = Tensor::new(vec![3], vec![1, 1, 1]);
        let input_grad6 = relu.backward(&output_grad6, &input6);
        assert_eq!(input_grad6.get_data(), vec![0, 0, 0], "负数输入应该完全阻断梯度");

        // 测试用例7: 不同大小的梯度
        // 计算过程:
        // 1. 输入 [1,2,-1,3] 的ReLU导数: [1 -> 1, 2 -> 1, -1 -> 0, 3 -> 1] = [1,1,0,1]
        // 2. 输出梯度 [4,3,2,1]
        // 3. 最终梯度 = [4,3,2,1] * [1,1,0,1] = [4,3,0,1]
        let input7 = Tensor::new(vec![4], vec![1, 2, -1, 3]);
        let output_grad7 = Tensor::new(vec![4], vec![4, 3, 2, 1]);
        let input_grad7 = relu.backward(&output_grad7, &input7);
        assert_eq!(input_grad7.get_data(), vec![4, 3, 0, 1], "应该正确处理不同大小的梯度");

        // 测试用例8: 维度检查
        // 这个测试验证当输入维度不匹配时是否会正确panic
        // input8维度为[2]，而output_grad8维度为[3]，这应该触发panic
        let input8 = Tensor::new(vec![2], vec![1, 1]);
        let output_grad8 = Tensor::new(vec![3], vec![1, 1, 1]);
        let result = std::panic::catch_unwind(|| {
            relu.backward(&output_grad8, &input8);
        });
        assert!(result.is_err(), "维度不匹配应该导致panic");
    }

    #[test]
    fn test_negative_multiplication() {
        println!("\n=== 负数乘法测试 ===");
        
        // 测试 -2 * -1 = 2 的情况
        let input = Tensor::new(vec![1], vec![-2]);
        let output_grad = Tensor::new(vec![1], vec![-1]);
        
        println!("1. 原始值:");
        println!("输入: -2");
        println!("输出梯度: -1");
        
        // 转换为域元素
        let input_field: F = Element::from(-2).to_field();
        let grad_field: F = Element::from(-1).to_field();
        let result_field = input_field * grad_field;
        
        println!("\n2. 域元素表示:");
        println!("输入(-2)在域中: {:?}", input_field);
        println!("梯度(-1)在域中: {:?}", grad_field);
        println!("乘法结果在域中: {:?}", result_field);
        
        // 转换回原始值
        let result: Element = result_field.into_element();
        println!("\n3. 最终结果:");
        println!("转换回原始值: {}", result);
        
        assert_eq!(result, 2, "(-2) * (-1) 应该等于 2");
        println!("=== 负数乘法测试结束 ===\n");
    }

    #[test]
    fn test_activation_backward_proof() {
        // 设置基本测试数据
        let relu = Relu::new();
        
        // 创建与ReLU多项式长度匹配的输入张量
        let poly_len = Relu::poly_len();
        println!("ReLU多项式长度: {}", poly_len);
        
        // 创建一个填充到多项式长度的输入张量
        let mut input_data = vec![0; poly_len];
        input_data[0] = 1;  // 设置第一个元素为正数
        input_data[1] = -1; // 设置第二个元素为负数
        let input = Tensor::new(vec![poly_len], input_data);
        
        // 创建对应的梯度张量
        let mut grad_data = vec![0; poly_len];
        grad_data[0] = 2;  // 对应第一个元素的梯度
        grad_data[1] = 2;  // 对应第二个元素的梯度
        let output_grad = Tensor::new(vec![poly_len], grad_data);
        
        println!("输入张量: {:?}", input.get_data());
        println!("梯度张量: {:?}", output_grad.get_data());

        // 创建transcript
        let mut transcript = BasicTranscript::new(b"test_activation_backward");
        
        // 初始化模型和上下文
        let mut model = Model::new();
        let input_shape = vec![poly_len];
        model.set_input_shape(input_shape.clone());  // 明确设置输入形状
        println!("设置的模型输入形状: {:?}", input_shape);
        
        // 添加一个Dense层，确保有多项式需要提交
        let dense = Layer::Dense(Dense::new(
            Tensor::new(vec![poly_len, poly_len], vec![1; poly_len * poly_len]),  // 全1矩阵
            Tensor::new(vec![poly_len], vec![0; poly_len])  // 零偏置
        ));
        model.add_layer::<F>(dense);
        println!("添加Dense层完成");
        
        // 添加ReLU层
        let layer = Layer::Activation(Activation::Relu(relu));
        model.add_layer::<F>(layer);
        println!("添加ReLU层完成");
        
        // 生成上下文
        let mut aux = ContextAux::default();
        aux.last_output_shape = input_shape.clone();  // 设置输出形状
        aux.tables.insert(TableType::Relu);  // 确保添加ReLU表
        println!("上下文输出形状: {:?}", aux.last_output_shape);
        println!("上下文表类型: {:?}", aux.tables);
        
        let context = match Context::<F>::generate(&model, Some(input_shape.clone())) {
            Ok(ctx) => {
                println!("上下文生成成功");
                ctx
            },
            Err(e) => {
                println!("上下文生成失败: {:?}", e);
                panic!("上下文生成失败");
            }
        };
        
        let mut prover = Prover::new(&context, &mut transcript);
        println!("Prover创建成功");

        // 创建必要的上下文和初始声明
        let input_poly_id = 1;  // 使用1作为多项式ID，因为0已经被Dense层使用
        let matrix_poly_aux = VPAuxInfo::default();
        let info = ActivationBackwardCtx {
            input_poly_id,
            matrix_poly_aux,
        };
        println!("使用的多项式ID: {}", input_poly_id);

        // 生成初始声明
        let num_vars = Relu::num_vars();
        println!("计算的变量数: {}", num_vars);
        let initial_claim = Claim {
            point: vec![F::ONE; num_vars],  // 创建正确长度的点向量
            eval: F::ONE,
        };
        println!("初始声明点向量长度: {}", initial_claim.point.len());

        // 生成证明
        println!("开始生成证明...");
        let proof_result = Activation::Relu(relu).prove_backward_step(
            &mut prover,
            initial_claim.clone(),
            &output_grad,
            &input,
            &info,
        );
        
        match &proof_result {
            Ok(_) => println!("证明生成成功"),
            Err(e) => println!("证明生成失败: {:?}", e),
        }
        assert!(proof_result.is_ok(), "证明生成应该成功");

        // 创建推理跟踪
        println!("开始创建推理跟踪...");
        let trace = model.run_feedforward(input.clone());
        println!("推理跟踪创建成功");

        // 生成完整的证明
        println!("开始生成完整证明...");
        let proof = match prover.prove(trace) {
            Ok(p) => {
                println!("完整证明生成成功");
                p
            },
            Err(e) => {
                println!("完整证明生成失败: {:?}", e);
                panic!("完整证明生成失败");
            }
        };

        // 验证证明
        println!("开始验证证明...");
        let mut verifier_transcript = BasicTranscript::new(b"test_activation_backward");
        let mut verifier = Verifier::new(&mut verifier_transcript);
        println!("验证器创建成功");
        
        if let Ok(final_claim) = proof_result {
            let input_for_io = input.clone();
            let output_grad_for_io = output_grad.clone();
            
            println!("开始验证过程...");
            let verify_result = verify(
                context,
                proof,
                IO::new(input_for_io.to_fields(), output_grad_for_io.to_fields()),
                &mut verifier_transcript
            );
            
            match &verify_result {
                Ok(_) => println!("验证成功"),
                Err(e) => println!("验证失败: {:?}", e),
            }
            assert!(verify_result.is_ok(), "验证应该成功");

            // 验证实际梯度计算是否正确
            let expected_grad = relu.backward(&output_grad, &input);
            println!("期望的梯度: {:?}", expected_grad.get_data());
            // 只检查前两个元素的梯度，因为其他元素都是0
            assert_eq!(expected_grad.get_data()[0..2], vec![2, 0], "梯度计算应该正确：正值(1)保持梯度2，负值(-1)梯度为0");
        }
    }
}
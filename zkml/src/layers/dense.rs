use std::cmp::Ordering;

use crate::{
    Claim, Prover,
    iop::{context::ContextAux, verifier::Verifier},
    layers::{LayerCtx, LayerProof, PolyID, Train},
    padding::PaddingMode,
    quantization::{self, ScalingFactor},
    tensor::Number,
};
use anyhow::{Context, ensure};
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use timed::timed_instrument;
use tracing::{trace, warn};
use transcript::Transcript;

use crate::{Element, tensor::Tensor};

/// Bias to compute the bias ID polynomials. Since originally we take the index of each
/// layer to be the index of the layer, we need to add a bias to avoid collision with other
/// layers poly id.
pub(crate) const BIAS_POLY_ID: PolyID = 100_000;

/// Description of the layer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dense<T> {
    pub matrix: Tensor<T>,
    pub bias: Tensor<T>,
    // set to matrix shape if the matrix is not padded
    pub unpadded_matrix_shape: Vec<usize>,
    // 训练相关的字段
    #[serde(skip)]
    pub grad_matrix: Option<Tensor<T>>,
    #[serde(skip)]
    pub grad_bias: Option<Tensor<T>>,
}

/// Information stored in the context (setup phase) for this layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DenseCtx<E> {
    pub matrix_poly_id: PolyID,
    pub matrix_poly_aux: VPAuxInfo<E>,
    pub bias_poly_id: PolyID,
    pub unpadded_matrix_shape: Vec<usize>,
    pub padded_matrix_shape: Vec<usize>,
}

/// Proof of the layer.
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct DenseProof<E: ExtensionField> {
    /// the actual sumcheck proof proving the mat2vec protocol
    pub(crate) sumcheck: IOPProof<E>,
    /// The evaluation of the bias at the previous claims in the proving flow.
    /// The verifier substracts this from the previous claim to end up with one claim only
    /// about the matrix, without the bias.
    bias_eval: E,
    /// The individual evaluations of the individual polynomial for the last random part of the
    /// sumcheck. One for each polynomial involved in the "virtual poly". Since we only support quadratic right now it's
    /// a flat list.
    individual_claims: Vec<E>,
}

impl<T: Number> Dense<T> {
    pub fn new(matrix: Tensor<T>, bias: Tensor<T>) -> Self {
        assert_eq!(matrix.nrows_2d(), bias.get_shape()[0]);
        let unpadded_matrix_shape = matrix.get_shape().to_vec();
        Self {
            matrix,
            bias,
            unpadded_matrix_shape,
            grad_matrix: None,
            grad_bias: None,
        }
    }
    pub fn ncols(&self) -> usize {
        self.matrix.ncols_2d()
    }
    pub fn nrows(&self) -> usize {
        self.matrix.nrows_2d()
    }

    pub fn op(&self, input: &Tensor<T>) -> Tensor<T> {
        if input.get_shape().len() != 1 {
            let flat_input = input.flatten();
            let matvec = self.matrix.matvec(&flat_input);
            matvec.add(&self.bias)
        } else {
            self.matrix.matvec(input).add(&self.bias)
        }
    }

    pub fn pad_next_power_of_two(self) -> Self {
        let matrix = self.matrix.pad_next_power_of_two();
        let bias = self.bias.pad_1d(matrix.nrows_2d());
        Self {
            matrix,
            bias,
            unpadded_matrix_shape: self.unpadded_matrix_shape.to_vec(),
            grad_matrix: self.grad_matrix,
            grad_bias: self.grad_bias,
        }
    }

    pub fn output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        assert_eq!(
            input_shape.iter().product::<usize>(),
            self.unpadded_matrix_shape[1],
            "unpadded_matrix_shape must be 2D: input_shape {:?} vs matrix {:?}",
            input_shape,
            self.unpadded_matrix_shape
        );
        vec![self.unpadded_matrix_shape[0]]
    }
    pub fn describe(&self) -> String {
        format!(
            "Dense: ({}x{}) + bias ({})",
            self.matrix.nrows_2d(),
            self.matrix.ncols_2d(),
            !self
                .bias
                .get_data()
                .iter()
                .all(|x| x.compare(&T::default()) == Ordering::Equal)
        )
    }
}

impl Dense<f32> {
    /// Quantize the parameters of the dense layer. It uses a custom scaling factor `bias_s` for
    /// the bias, if provided, otherwise the same scaling factor of the weights (i.e., `s`) is used
    pub fn quantize(self, s: &ScalingFactor, bias_s: &ScalingFactor) -> Dense<Element> {
        let matrix = self.matrix.quantize(s);
        let bias = self.bias.quantize(bias_s);
        let grad_matrix = self.grad_matrix.map(|g| g.quantize(s));
        let grad_bias = self.grad_bias.map(|g| g.quantize(bias_s));
        Dense::<Element> {
            matrix,
            bias,
            unpadded_matrix_shape: self.unpadded_matrix_shape.to_vec(),
            grad_matrix,
            grad_bias,
        }
    }

    pub fn new_from_weights(weights: Tensor<f32>, bias: Tensor<f32>) -> Self {
        let unpadded_matrix_shape = weights.get_shape().to_vec();
        Self {
            matrix: weights,
            bias,
            unpadded_matrix_shape,
            grad_matrix: None,
            grad_bias: None,
        }
    }

    /// TODO: compute two different scaling factors for weights and bias
    pub fn max_abs_weight(&self) -> f32 {
        let max_weight = self.matrix.max_abs_output();
        let max_bias = self.bias.max_abs_output();
        let distance = (max_weight - max_bias).abs() / max_weight;
        if distance > 0.1 {
            warn!(
                "max_abs_weight DENSE: distance between max_weight and max_bias is too large: {:.2}%",
                distance * 100.0
            );
        }
        self.matrix.max_abs_output().max(self.bias.max_abs_output())
    }
}

impl Dense<Element> {
    /// Returns the (min,max) output range of the dense layer for a given input range.
    pub fn output_range(&self, _min_input: Element, _max_input: Element) -> (Element, Element) {
        // formula is 2^{2 * BIT_LEN + log(c) + 1} where c is the number of columns and +1 because of the bias
        let ncols = self.matrix.ncols_2d() as u32;
        // - 1 because numbers are signed so only half of the range is used when doing multiplication
        let power = 2 * (*quantization::BIT_LEN as u32 - 1) + ncols.ilog2() + 1;
        let min = -(2u64.pow(power as u32) as Element);
        let max = 2u64.pow(power as u32) as Element;
        return (min, max);
    }
    #[timed::timed_instrument(name = "Prover::prove_dense")]
    pub fn prove_step<'b, E, T>(
        &self,
        prover: &mut Prover<E, T>,
        last_claim: Claim<E>,
        input: &Tensor<E>,
        output: &Tensor<E>,
        info: &DenseCtx<E>,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        T: Transcript<E>,
    {
        let matrix = &self.matrix;
        let (nrows, ncols) = (matrix.nrows_2d(), matrix.ncols_2d());
        assert_eq!(
            nrows,
            output.get_data().len(),
            "dense proving: nrows {} vs output {}",
            nrows,
            output.get_data().len()
        );
        assert_eq!(
            nrows.ilog2() as usize,
            last_claim.point.len(),
            "something's wrong with the randomness"
        );
        assert_eq!(
            ncols,
            input.get_data().len(),
            "something's wrong with the input"
        );
        // Evaluates the bias at the random point so verifier can substract the evaluation
        // from the sumcheck claim that is only about the matrix2vec product.
        assert_eq!(
            self.bias.get_data().len().ilog2() as usize,
            last_claim.point.len(),
            "something's wrong with the randomness"
        );
        let bias_eval = self
            .bias
            .evals_flat::<E>()
            .into_mle()
            .evaluate(&last_claim.point);
        // contruct the MLE combining the input and the matrix
        let mut mat_mle = matrix.to_mle_2d();
        // fix the variables from the random input
        // NOTE: here we must fix the HIGH variables because the MLE is addressing in little
        // endian so (rows,cols) is actually given in (cols, rows)
        // mat_mle.fix_variables_in_place_parallel(partial_point);
        mat_mle.fix_high_variables_in_place(&last_claim.point);
        let input_mle = input.get_data().to_vec().into_mle();

        assert_eq!(mat_mle.num_vars(), input_mle.num_vars());
        let num_vars = input_mle.num_vars();
        let mut vp = VirtualPolynomial::<E>::new(num_vars);
        // TODO: remove the clone once prover+verifier are working
        vp.add_mle_list(
            vec![mat_mle.clone().into(), input_mle.clone().into()],
            E::ONE,
        );
        let tmp_transcript = prover.transcript.clone();
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);

        debug_assert!({
            let mut t = tmp_transcript;
            // just construct manually here instead of cloning in the non debug code
            let mut vp = VirtualPolynomial::<E>::new(num_vars);
            vp.add_mle_list(vec![mat_mle.into(), input_mle.into()], E::ONE);
            // asserted_sum in this case is the output MLE evaluated at the random point
            let mle_output = output.get_data().to_vec().into_mle();
            let claimed_sum = mle_output.evaluate(&last_claim.point);
            let claimed_sum_no_bias = claimed_sum - bias_eval;
            debug_assert_eq!(claimed_sum, last_claim.eval, "sumcheck eval weird");
            debug_assert_eq!(
                claimed_sum_no_bias,
                proof.extract_sum(),
                "sumcheck output weird"
            );

            trace!("prover: claimed sum: {:?}", claimed_sum);
            let subclaim =
                IOPVerifierState::<E>::verify(claimed_sum_no_bias, &proof, &vp.aux_info, &mut t);
            // now assert that the polynomial evaluated at the random point of the sumcheck proof
            // is equal to last small poly sent by prover (`subclaim.expected_evaluation`). This
            // step can be done via PCS opening proofs for all steps but first (output of
            // inference) and last (input of inference)
            let computed_point = vp.evaluate(subclaim.point_flat().as_ref());

            let final_prover_point = state
                .get_mle_final_evaluations()
                .into_iter()
                .fold(E::ONE, |acc, eval| acc * eval);
            assert_eq!(computed_point, final_prover_point);

            // NOTE: this expected_evaluation is computed by the verifier on the "reduced"
            // last polynomial of the sumcheck protocol. It's easy to compute since it's a degree
            // one poly. However, it needs to be checked against the original polynomial and this
            // is done via PCS.
            computed_point == subclaim.expected_evaluation
        });

        // PCS part: here we need to create an opening proof for the final evaluation of the matrix poly
        // Note we need the _full_ input to the matrix since the matrix MLE has (row,column) vars space
        let point = [proof.point.as_slice(), last_claim.point.as_slice()].concat();
        let eval = state.get_mle_final_evaluations()[0];
        prover
            .commit_prover
            .add_claim(info.matrix_poly_id, Claim::new(point, eval))
            .context("unable to add matrix claim")?;
        // add the bias claim over the last claim input, since that is what is needed to "remove" the bias
        // to only verify the matrix2vec product via the sumcheck proof.
        prover
            .commit_prover
            .add_claim(info.bias_poly_id, Claim::new(last_claim.point, bias_eval))
            .context("unable to add bias claim")?;

        // the claim that this proving step outputs is the claim about not the matrix but the vector poly.
        // at next step, that claim will be proven over this vector poly (either by the next dense layer proving, or RELU etc).
        let claim = Claim {
            point: proof.point.clone(),
            eval: state.get_mle_final_evaluations()[1],
        };
        prover.push_proof(LayerProof::Dense(DenseProof {
            sumcheck: proof,
            bias_eval,
            individual_claims: state.get_mle_final_evaluations(),
        }));
        Ok(claim)
    }

    pub(crate) fn step_info<E: ExtensionField>(
        &self,
        id: PolyID,
        mut ctx_aux: ContextAux,
    ) -> (LayerCtx<E>, ContextAux)
    where
        E: ExtensionField + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        // construct dimension of the polynomial given to the sumcheck
        let ncols = self.matrix.ncols_2d();
        ctx_aux.last_output_shape = vec![self.matrix.nrows_2d()];
        // each poly is only two polynomial right now: matrix and vector
        // for matrix, each time we fix the variables related to rows so we are only left
        // with the variables related to columns
        let matrix_num_vars = ncols.ilog2() as usize;
        let vector_num_vars = matrix_num_vars;
        // there is only one product (i.e. quadratic sumcheck)
        let dense_info = LayerCtx::Dense(DenseCtx {
            matrix_poly_id: id,
            matrix_poly_aux: VPAuxInfo::<E>::from_mle_list_dimensions(&vec![vec![
                matrix_num_vars,
                vector_num_vars,
            ]]),
            bias_poly_id: BIAS_POLY_ID + id,
            unpadded_matrix_shape: self.unpadded_matrix_shape.clone(),
            padded_matrix_shape: self.matrix.get_shape().to_vec(),
        });
        (dense_info, ctx_aux)
    }
}

impl<E: ExtensionField> DenseCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    pub fn output_shape(&self, input_shape: &[usize], mode: PaddingMode) -> Vec<usize> {
        let mat_shape = match mode {
            PaddingMode::NoPadding => self.unpadded_matrix_shape.clone(),
            PaddingMode::Padding => self.padded_matrix_shape.clone(),
        };
        assert_eq!(
            input_shape.iter().product::<usize>(),
            mat_shape[1],
            "dense output shape (pad = {:?}) -> input_shape {:?} vs matrix {:?}",
            mode,
            input_shape,
            mat_shape
        );
        vec![mat_shape[0]]
    }
    pub(crate) fn verify_dense<T: Transcript<E>>(
        &self,
        verifier: &mut Verifier<E, T>,
        last_claim: Claim<E>,
        proof: &DenseProof<E>,
    ) -> anyhow::Result<Claim<E>> {
        let info = self;
        // Subtract the bias evaluation from the previous claim to remove the bias
        let eval_no_bias = last_claim.eval - proof.bias_eval;
        // TODO: currently that API can panic - should remove panic for error
        let subclaim = IOPVerifierState::<E>::verify(
            eval_no_bias,
            &proof.sumcheck,
            &info.matrix_poly_aux,
            verifier.transcript,
        );

        // MATRIX OPENING PART
        // pcs_eval means this evaluation should come from a PCS opening proof
        let pcs_eval_input = subclaim
            .point_flat()
            .iter()
            .chain(last_claim.point.iter())
            .cloned()
            .collect_vec();
        // 0 because Matrix comes first in Matrix x Vector
        // Note we don't care about verifying that for the vector since it's verified at the next
        // step.
        let pcs_eval_output = proof.individual_claims[0];
        verifier.commit_verifier.add_claim(
            info.matrix_poly_id,
            Claim::new(pcs_eval_input, pcs_eval_output),
        )?;
        verifier.commit_verifier.add_claim(
            info.bias_poly_id,
            Claim::new(last_claim.point, proof.bias_eval),
        )?;

        // SUMCHECK verification part
        // Instead of computing the polynomial at the random point requested like this
        // let computed_point = vp.evaluate(
        //     subclaim
        //         .point
        //         .iter()
        //         .map(|c| c.elements)
        //         .collect_vec()
        //         .as_ref(),
        //
        // We compute the evaluation directly from the individual final evaluations of each polynomial
        // involved in the sumcheck the prover's giving,e.g. y(res) = SUM f_i(res)
        ensure!(
            proof.individual_to_virtual_claim() == subclaim.expected_evaluation,
            "sumcheck claim failed",
        );

        // the output claim for this step that is going to be verified at next step
        Ok(Claim {
            // the new randomness to fix at next layer is the randomness from the sumcheck !
            point: subclaim.point_flat(),
            // the claimed sum for the next sumcheck is MLE of the current vector evaluated at the
            // random point. 1 because vector is secondary.
            eval: proof.individual_claims[1],
        })
    }
}

impl<E: ExtensionField> DenseProof<E> {
    /// Returns the individual claims f_1(r) f_2(r)  f_3(r) ... at the end of a sumcheck multiplied
    /// together
    pub fn individual_to_virtual_claim(&self) -> E {
        self.individual_claims.iter().fold(E::ONE, |acc, e| acc * e)
    }
}

impl<T: Number> Train<T> for Dense<T> {
    fn forward(&self, input: &Tensor<T>) -> Tensor<T> {
        self.op(input)
    }

    fn backward(&mut self, input_tensor: &Tensor<T>, grad_output: &Tensor<T>) -> Tensor<T> {
        // 1. 处理输入形状
        let input_2d = if input_tensor.get_shape().len() == 1 {
            Tensor::new(
                vec![1, input_tensor.get_data().len()],
                input_tensor.get_data().to_vec()
            )
        } else {
            Tensor::new(
                input_tensor.get_shape().to_vec(),
                input_tensor.get_data().to_vec()
            )
        };

        let grad_output_2d = if grad_output.get_shape().len() == 1 {
            Tensor::new(
                vec![1, grad_output.get_data().len()],
                grad_output.get_data().to_vec()
            )
        } else {
            Tensor::new(
                grad_output.get_shape().to_vec(),
                grad_output.get_data().to_vec()
            )
        };

        // 2. 检查维度匹配
        let batch_size = input_2d.get_shape()[0];
        let input_size = input_2d.get_shape()[1];
        let output_size = self.matrix.nrows_2d();

        assert_eq!(
            grad_output_2d.get_shape(),
            vec![batch_size, output_size],
            "梯度形状不匹配: 期望 [{}, {}], 实际 {:?}",
            batch_size,
            output_size,
            grad_output_2d.get_shape()
        );

        // 3. 计算权重的梯度: grad_weights = grad_output.T @ input
        let grad_weights = grad_output_2d.transpose().matmul(&input_2d);

        // 4. 计算偏置的梯度: grad_bias = 对grad_output按batch维度求和
        // 初始化第一个样本的梯度
        let mut grad_bias = Tensor::new(
            vec![output_size],
            grad_output_2d.get_data()[..output_size].to_vec()
        );
        // 累加其余样本的梯度
        for i in 1..batch_size {
            let start = i * output_size;
            let end = start + output_size;
            let sample_grad = Tensor::new(
                vec![output_size],
                grad_output_2d.get_data()[start..end].to_vec()
            );
            grad_bias = grad_bias.add(&sample_grad);
        }

        // 5. 计算输入的梯度: grad_input = grad_output @ weights
        let grad_input = if batch_size == 1 {
            self.matrix.transpose().matvec(&grad_output_2d.flatten())
        } else {
            grad_output_2d.matmul(&self.matrix)
        };

        // 6. 验证梯度形状
        assert_eq!(
            grad_weights.get_shape(),
            self.matrix.get_shape(),
            "权重梯度形状不匹配"
        );
        assert_eq!(
            grad_bias.get_shape(),
            self.bias.get_shape(),
            "偏置梯度形状不匹配"
        );

        // 7. 如果输入是1维的,输出也应该是1维的
        let grad_input_final = if input_tensor.get_shape().len() == 1 {
            grad_input.flatten()
        } else {
            grad_input
        };

        // 8. 存储梯度
        self.grad_matrix = Some(grad_weights);
        self.grad_bias = Some(grad_bias);

        grad_input_final
    }

    fn update(&mut self, learning_rate: T) {
        if let Some(grad_matrix) = &self.grad_matrix {
            // 更新权重矩阵: weights = weights - learning_rate * grad_weights
            let scaled_grad_weights = grad_matrix.scalar_mul(&learning_rate);
            self.matrix = self.matrix.sub(&scaled_grad_weights);
        }
        
        if let Some(grad_bias) = &self.grad_bias {
            // 更新偏置: bias = bias - learning_rate * grad_bias
            let scaled_grad_bias = grad_bias.scalar_mul(&learning_rate);
            self.bias = self.bias.sub(&scaled_grad_bias);
        }
    }

    fn zero_grad(&mut self) {
        self.grad_matrix = None;
        self.grad_bias = None;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    impl<T: Number> Dense<T> {
        pub fn random(shape: Vec<usize>) -> Self {
            assert_eq!(shape.len(), 2);
            let (nrows, ncols) = (shape[0], shape[1]);
            let matrix = Tensor::<T>::random(&vec![nrows, ncols]);
            // let bias = Tensor::random(vec![nrows]);
            let bias = Tensor::<T>::random(&vec![nrows]);
            Self::new(matrix, bias)
        }
    }

    #[test]
    fn test_dense_pad_next_power_of_two() {
        // Create a Dense layer with non-power-of-two dimensions
        let matrix =
            Tensor::<Element>::matix_from_coeffs(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]])
                .unwrap();

        let bias = Tensor::<Element>::new(vec![3], vec![10, 11, 12]);

        let dense = Dense::new(matrix, bias);

        // Pad to next power of two
        let padded = dense.pad_next_power_of_two();

        // Check padded dimensions are powers of two
        let padded_dims = padded.matrix.get_shape();
        assert_eq!(padded_dims[0], 4); // Next power of 2 after 3
        assert_eq!(padded_dims[1], 4); // Next power of 2 after 3

        // Check bias is padded
        let bias_dims = padded.bias.get_shape();
        assert_eq!(bias_dims[0], 4); // Next power of 2 after 3

        // Check original values are preserved
        assert_eq!(padded.matrix.get_data()[0], 1);
        assert_eq!(padded.matrix.get_data()[1], 2);
        assert_eq!(padded.matrix.get_data()[2], 3);
        assert_eq!(padded.matrix.get_data()[4], 4);
        assert_eq!(padded.matrix.get_data()[8], 7);

        // Check added values are zeros
        assert_eq!(padded.matrix.get_data()[3], 0);
        assert_eq!(padded.matrix.get_data()[7], 0);
        assert_eq!(padded.matrix.get_data()[15], 0);

        // Check bias values
        assert_eq!(padded.bias.get_data()[0], 10);
        assert_eq!(padded.bias.get_data()[1], 11);
        assert_eq!(padded.bias.get_data()[2], 12);
        assert_eq!(padded.bias.get_data()[3], 0); // Padding
    }

    #[test]
    fn test_dense_pad_already_power_of_two() {
        // Create a Dense layer with power-of-two dimensions
        let matrix = Tensor::<Element>::matix_from_coeffs(vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
        ])
        .unwrap();

        let bias = Tensor::<Element>::new(vec![4], vec![20, 21, 22, 23]);

        let dense = Dense::new(matrix, bias);

        // Pad to next power of two
        let padded = dense.clone().pad_next_power_of_two();

        // Check dimensions remain the same
        let padded_dims = padded.matrix.get_shape();
        assert_eq!(padded_dims[0], 4);
        assert_eq!(padded_dims[1], 4);

        // Check bias dimensions remain the same
        let bias_dims = padded.bias.get_shape();
        assert_eq!(bias_dims[0], 4);

        // Check values are preserved
        for i in 0..16 {
            assert_eq!(padded.matrix.get_data()[i], dense.matrix.get_data()[i]);
        }

        for i in 0..4 {
            assert_eq!(padded.bias.get_data()[i], dense.bias.get_data()[i]);
        }
    }

    #[test]
    fn test_dense_pad_mixed_dimensions() {
        // Create a Dense layer with one power-of-two dimension and one non-power-of-two
        let matrix =
            Tensor::<Element>::matix_from_coeffs(vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![
                9, 10, 11, 12,
            ]])
            .unwrap();

        let bias = Tensor::<Element>::new(vec![3], vec![20, 21, 22]);

        let dense = Dense::new(matrix, bias);

        // Pad to next power of two
        let padded = dense.pad_next_power_of_two();

        // Check dimensions are padded correctly
        let padded_dims = padded.matrix.get_shape();
        assert_eq!(padded_dims[0], 4); // Next power of 2 after 3
        assert_eq!(padded_dims[1], 4); // Already a power of 2

        // Check bias is padded
        let bias_dims = padded.bias.get_shape();
        assert_eq!(bias_dims[0], 4); // Next power of 2 after 3

        // Check original values are preserved and padding is zeros
        assert_eq!(padded.matrix.get_data()[0], 1);
        assert_eq!(padded.matrix.get_data()[4], 5);
        assert_eq!(padded.matrix.get_data()[8], 9);
        assert_eq!(padded.matrix.get_data()[12], 0); // Padding

        // Check bias values
        assert_eq!(padded.bias.get_data()[0], 20);
        assert_eq!(padded.bias.get_data()[1], 21);
        assert_eq!(padded.bias.get_data()[2], 22);
        assert_eq!(padded.bias.get_data()[3], 0); // Padding
    }

    #[test]
    fn test_quantization_with_padded_dense() {
        // Create input data that needs quantization
        let input_data = vec![0.5f32, -0.3f32, 0.1f32];

        // Quantize the input
        let quantized_input: Vec<Element> = input_data
            .iter()
            .map(|x| ScalingFactor::default().quantize(x))
            .collect();

        // Create a Dense layer
        let matrix =
            Tensor::<Element>::matix_from_coeffs(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();

        let bias = Tensor::<Element>::new(vec![2], vec![10, 11]);

        let dense = Dense::new(matrix, bias);

        // Pad the dense layer
        let padded = dense.clone().pad_next_power_of_two();

        // Create input tensor
        let input_tensor = Tensor::<Element>::new(vec![3], quantized_input);

        // Apply the dense operation on both original and padded
        let output = dense.op(&input_tensor);
        let padded_output = padded.op(&input_tensor.pad_1d(4));

        // Check that the result is correct (for the non-padded parts)
        for i in 0..2 {
            assert_eq!(output.get_data()[i], padded_output.get_data()[i]);
        }
    }

    #[test]
    fn test_dense_backward() {
        // 创建Dense层: 2x3
        let weights = Tensor::new(
            vec![2, 3],
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        let bias = Tensor::new(vec![2], vec![0.1f32, 0.2]);
        let mut dense = Dense::new(weights, bias);

        println!("\n=== Dense层参数 ===");
        println!("权重矩阵:\n{:?}", dense.matrix.get_data());
        println!("偏置向量: {:?}", dense.bias.get_data());

        // 创建batch输入: 2x3 (两个样本)
        let input = Tensor::new(
            vec![2, 3],
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        
        println!("\n=== 输入数据 ===");
        println!("输入形状: {:?}", input.get_shape());
        println!("输入数据:\n{:?}", input.get_data());
        
        // batch的梯度: 2x2
        let grad_output = Tensor::new(
            vec![2, 2],
            vec![1.0f32, 2.0, 3.0, 4.0]
        );

        println!("\n=== 输出梯度 ===");
        println!("梯度形状: {:?}", grad_output.get_shape());
        println!("梯度数据:\n{:?}", grad_output.get_data());

        // 计算反向传播
        let grad_input = dense.backward(&input, &grad_output);

        println!("\n=== 反向传播结果 ===");
        println!("输入梯度形状: {:?}", grad_input.get_shape());
        println!("输入梯度:\n{:?}", grad_input.get_data());
        println!("\n权重梯度形状: {:?}", dense.grad_matrix.as_ref().unwrap().get_shape());
        println!("权重梯度:\n{:?}", dense.grad_matrix.as_ref().unwrap().get_data());
        println!("\n偏置梯度形状: {:?}", dense.grad_bias.as_ref().unwrap().get_shape());
        println!("偏置梯度: {:?}", dense.grad_bias.as_ref().unwrap().get_data());

        // 验证梯度形状
        assert_eq!(grad_input.get_shape(), vec![2, 3]);  // batch的输入梯度
        assert_eq!(dense.grad_matrix.as_ref().unwrap().get_shape(), vec![2, 3]); // 权重梯度
        assert_eq!(dense.grad_bias.as_ref().unwrap().get_shape(), vec![2]);      // 偏置梯度

        // 验证梯度值
        // 1. 输入梯度 = grad_output @ weights
        let expected_grad_input = vec![9.0, 12.0, 15.0, 19.0, 26.0, 33.0];
        println!("\n=== 验证结果 ===");
        println!("期望的输入梯度: {:?}", expected_grad_input);
        for (actual, expected) in grad_input.get_data().iter().zip(expected_grad_input.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        // 2. 权重梯度 = grad_output.T @ input
        let expected_grad_weights = vec![13.0, 17.0, 21.0, 18.0, 24.0, 30.0];
        println!("期望的权重梯度: {:?}", expected_grad_weights);
        for (actual, expected) in dense.grad_matrix.as_ref().unwrap().get_data().iter().zip(expected_grad_weights.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        // 3. 偏置梯度 = sum(grad_output, axis=0)
        let expected_grad_bias = vec![4.0, 6.0];
        println!("期望的偏置梯度: {:?}", expected_grad_bias);
        for (actual, expected) in dense.grad_bias.as_ref().unwrap().get_data().iter().zip(expected_grad_bias.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_parameter_update() {
        // 设置学习率
        let learning_rate: f32 = 0.1;

        // 初始化权重矩阵和偏置
        let matrix = Tensor::new(
            vec![2, 2],
            vec![1.0f32, 2.0, 3.0, 4.0]
        );
        let bias = Tensor::new(
            vec![2],
            vec![1.0f32, 1.0]
        );

        // 初始化梯度
        let grad_matrix = Tensor::new(
            vec![2, 2],
            vec![0.5f32, 0.5, 0.5, 0.5]
        );
        let grad_bias = Tensor::new(
            vec![2],
            vec![0.2f32, 0.2]
        );

        // 构建 Dense 层
        let mut dense = Dense::new(matrix, bias);
        dense.grad_matrix = Some(grad_matrix);
        dense.grad_bias = Some(grad_bias);

        println!("\n=== 更新前参数 ===");
        println!("权重矩阵:\n{:?}", dense.matrix.get_data());
        println!("偏置向量: {:?}", dense.bias.get_data());

        // 执行参数更新
        dense.update(learning_rate);

        println!("\n=== 更新后参数 ===");
        println!("权重矩阵:\n{:?}", dense.matrix.get_data());
        println!("偏置向量: {:?}", dense.bias.get_data());

        // 验证更新后的权重矩阵
        let expected_matrix = vec![0.95f32, 1.95, 2.95, 3.95];
        for (actual, expected) in dense.matrix.get_data().iter().zip(expected_matrix.iter()) {
            assert!((actual - expected).abs() < 1e-5, 
                "权重矩阵更新失败: 期望 {}, 实际 {}", expected, actual);
        }

        // 验证更新后的偏置
        let expected_bias = vec![0.98f32, 0.98];
        for (actual, expected) in dense.bias.get_data().iter().zip(expected_bias.iter()) {
            assert!((actual - expected).abs() < 1e-5,
                "偏置更新失败: 期望 {}, 实际 {}", expected, actual);
        }

        println!("test_parameter_update 测试通过");
    }
}

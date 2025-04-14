use crate::{
    Claim, Element, Prover,
    commit::{compute_betas_eval, identity_eval, precommit::PolyID, same_poly},
    iop::verifier::Verifier,
    layers::{ContextAux, LayerProof},
    lookup::{
        context::TableType,
        logup_gkr::{
            prover::batch_prove as logup_batch_prove, structs::LogUpProof,
            verifier::verify_logup_proof,
        },
    },
    quantization::{Fieldizer, IntoElement, ZERO},
    tensor::Tensor,
};
use anyhow::{Context, ensure};
use ff_ext::ExtensionField;
use gkr::util::ceil_log2;
use itertools::{Itertools, izip};
use multilinear_extensions::{
    mle::{ArcDenseMultilinearExtension, DenseMultilinearExtension, IntoMLE, MultilinearExtension},
    virtual_poly::{ArcMultilinearExtension, VPAuxInfo, VirtualPolynomial},
};
use serde::de::DeserializeOwned;
use sumcheck::structs::{IOPProof, IOPVerifierState};
use transcript::Transcript;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sumcheck::structs::IOPProverState;

use super::LayerCtx;

pub const MAXPOOL2D_KERNEL_SIZE: usize = 2;

#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
pub enum Pooling {
    Maxpool2D(Maxpool2D),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolingCtx {
    pub poolinfo: Maxpool2D,
    pub poly_id: PolyID,
    pub num_vars: usize,
}

/// Contains proof material related to one step of the inference
#[derive(Clone, Serialize, Deserialize)]
pub struct PoolingProof<E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// the actual sumcheck proof proving that the product of correct terms is always zero
    pub(crate) sumcheck: IOPProof<E>,
    /// The lookup proof showing that the diff is always in the correct range
    pub(crate) lookup: LogUpProof<E>,
    /// proof for the accumulation of the claim from the zerocheck + claim from lookup for the same poly for both input and output
    pub(crate) io_accumulation: same_poly::Proof<E>,
    /// The claims that are accumulated for the output of this step
    pub(crate) output_claims: Vec<Claim<E>>,
    /// The output evaluations of the diff polys produced by the zerocheck
    pub(crate) zerocheck_evals: Vec<E>,
    /// This tells the verifier how far apart the variables get fixed on the input MLE
    pub(crate) variable_gap: usize,
}

impl Pooling {
    pub fn op(&self, input: &Tensor<Element>) -> Tensor<Element> {
        match self {
            Pooling::Maxpool2D(maxpool2d) => maxpool2d.op(input),
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
        let info = match self {
            Pooling::Maxpool2D(info) => {
                aux.tables.insert(TableType::Range);
                // Pooling only affects the last two dimensions
                let total_number_dims = aux.last_output_shape.len();

                aux.last_output_shape
                    .iter_mut()
                    .skip(total_number_dims - 2)
                    .for_each(|dim| *dim = (*dim - info.kernel_size) / info.stride + 1);
                LayerCtx::Pooling(PoolingCtx {
                    poolinfo: *info,
                    poly_id: id,
                    num_vars: aux
                        .last_output_shape
                        .iter()
                        .map(|dim| ceil_log2(*dim))
                        .sum::<usize>(),
                })
            }
        };
        (info, aux)
    }
    pub fn gen_lookup_witness<E: ExtensionField>(
        &self,
        input: &Tensor<Element>,
    ) -> (Vec<Element>, Vec<Vec<E::BaseField>>) {
        match self {
            Pooling::Maxpool2D(maxpool2d) => {
                let field_vecs = maxpool2d.compute_polys::<E>(input);

                let merged_lookups = field_vecs
                    .iter()
                    .flat_map(|vector| {
                        vector
                            .iter()
                            .map(|&a| E::from(a).into_element())
                            .collect::<Vec<Element>>()
                    })
                    .collect::<Vec<Element>>();

                (merged_lookups, field_vecs)
            }
        }
    }
    #[timed::timed_instrument(level = "debug")]
    pub fn prove_pooling<E: ExtensionField, T: Transcript<E>>(
        &self,
        prover: &mut Prover<E, T>,
        // last random claim made
        last_claim: Claim<E>,
        // input to the dense layer
        input: &Tensor<E>,
        // output of dense layer evaluation
        output: &Tensor<E>,
        info: &PoolingCtx,
    ) -> anyhow::Result<Claim<E>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: Serialize + DeserializeOwned,
    {
        assert_eq!(input.get_shape().len(), 3, "Maxpool needs 3D inputs.");
        // Create the range check proof for the diff
        let prover_info = prover.next_lookup_witness()?;

        let logup_proof = logup_batch_prove(&prover_info, prover.transcript)?;

        // These are the polys that get passed to the zero check make sure their product is zero at every evaluation point
        let mut diff_polys = prover_info
            .column_evals()
            .iter()
            .map(|diff| {
                DenseMultilinearExtension::<E>::from_evaluations_slice(info.num_vars, diff).into()
            })
            .collect::<Vec<ArcMultilinearExtension<E>>>();

        // Run the Zerocheck that checks enforces that output does contain the maximum value for the kernel
        let mut vp = VirtualPolynomial::<E>::new(info.num_vars);

        // We reuse the logup point here for the zerocheck challenge
        let lookup_point = &logup_proof.output_claims()[0].point;

        // Comput the identity poly
        let batch_challenge = prover
            .transcript
            .get_and_append_challenge(b"batch_pooling")
            .elements;

        let beta_eval = compute_betas_eval(&lookup_point);
        let beta_poly: ArcDenseMultilinearExtension<E> =
            DenseMultilinearExtension::<E>::from_evaluations_ext_vec(info.num_vars, beta_eval)
                .into();
        let mut challenge_combiner = batch_challenge;
        let lookup_parts = diff_polys
            .iter()
            .map(|diff| {
                let out = (vec![diff.clone(), beta_poly.clone()], challenge_combiner);
                challenge_combiner *= batch_challenge;
                out
            })
            .collect::<Vec<(Vec<_>, E)>>();

        diff_polys.push(beta_poly);
        vp.add_mle_list(diff_polys, E::ONE);
        lookup_parts
            .into_iter()
            .for_each(|(prod, coeff)| vp.add_mle_list(prod, coeff));

        #[allow(deprecated)]
        let (proof, sumcheck_state) = IOPProverState::<E>::prove_parallel(vp, prover.transcript);

        // We need to prove that the output of this step is the input to following activation function
        let output_mle = output.get_data().to_vec().into_mle();
        let mut same_poly_prover = same_poly::Prover::<E>::new(output_mle.clone());

        let zerocheck_point = &proof.point;
        let output_zerocheck_eval = output_mle.evaluate(zerocheck_point);

        // Accumulate claims about the output polynomial in each of the protocols we ran together with the final claim from the previous proof.
        let mut output_claims = Vec::<Claim<E>>::new();
        let same_poly_ctx = same_poly::Context::<E>::new(last_claim.point.len());
        same_poly_prover.add_claim(last_claim.clone())?;

        let zerocheck_claim = Claim {
            point: zerocheck_point.clone(),
            eval: output_zerocheck_eval,
        };
        same_poly_prover.add_claim(zerocheck_claim.clone())?;
        output_claims.push(zerocheck_claim);

        // This is the proof for the output poly
        let claim_acc_proof = same_poly_prover.prove(&same_poly_ctx, prover.transcript)?;

        let output_claim = claim_acc_proof.extract_claim();

        prover
            .witness_prover
            .add_claim(info.poly_id, output_claim)
            .context("unable to add claim")?;
        // Now we must do the samething accumulating evals for the input poly as we fix variables on the input poly.
        // The point length is 2 longer because for now we only support MaxPool2D.

        let padded_input_shape = input.get_shape();
        let padded_input_row_length_log = ceil_log2(padded_input_shape[2]);
        // We can batch all of the claims for the input poly with 00, 10, 01, 11 fixed into one with random challenges
        let [r1, r2] = [prover
            .transcript
            .get_and_append_challenge(b"input_batching")
            .elements; 2];

        let one_minus_r1 = E::ONE - r1;
        let one_minus_r2 = E::ONE - r2;
        // To the input claims we add evaluations at both the zerocheck point and lookup point
        // in the order 00, 01, 10, 11. These will be used in conjunction with r1 and r2 by the verifier to link the claims output by the sumcheck and lookup GKR
        // proofs with the claims fed to the same poly verifier.

        let multiplicands = [
            one_minus_r1 * one_minus_r2,
            one_minus_r1 * r2,
            r1 * one_minus_r2,
            r1 * r2,
        ];

        let zc_in_claim = izip!(
            multiplicands.iter(),
            sumcheck_state.get_mle_final_evaluations().iter(),
        )
        .fold(E::ZERO, |zc_acc, (m, zc)| {
            zc_acc + *m * (output_zerocheck_eval - *zc)
        });

        let point_1 = [
            &[r1],
            &zerocheck_point[..padded_input_row_length_log - 1],
            &[r2],
            &zerocheck_point[padded_input_row_length_log - 1..],
        ]
        .concat();

        let next_claim = Claim {
            point: point_1,
            eval: zc_in_claim,
        };

        // We don't need the last eval of the the sumcheck state as it is the beta poly

        let zerocheck_evals = sumcheck_state.get_mle_final_evaluations()[..4].to_vec();
        // Push the step proof to the list
        prover.push_proof(LayerProof::Pooling(PoolingProof {
            sumcheck: proof,
            lookup: logup_proof,
            io_accumulation: claim_acc_proof,
            output_claims,
            zerocheck_evals,
            variable_gap: padded_input_row_length_log - 1,
        }));
        Ok(next_claim)
    }
}

impl PoolingCtx {
    pub(crate) fn verify_pooling<E: ExtensionField, T: Transcript<E>>(
        &self,
        verifier: &mut Verifier<E, T>,
        last_claim: Claim<E>,
        proof: &PoolingProof<E>,
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
            4,
            constant_challenge,
            column_separation_challenge,
            verifier.transcript,
        )?;

        // 2. Verify the sumcheck proof
        let poly_aux = VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![self.num_vars; 5]]);
        let batching_challenge = verifier
            .transcript
            .get_and_append_challenge(b"batch_pooling")
            .elements;
        let initial_value = verifier_claims
            .claims()
            .iter()
            .fold((E::ZERO, batching_challenge), |(acc, comb), claim| {
                (acc + claim.eval * comb, comb * batching_challenge)
            })
            .0;
        let subclaim = IOPVerifierState::<E>::verify(
            initial_value,
            &proof.sumcheck,
            &poly_aux,
            verifier.transcript,
        );

        // Run the same poly verifier for the output claims
        let sp_ctx = same_poly::Context::<E>::new(self.num_vars);
        let mut sp_verifier = same_poly::Verifier::<E>::new(&sp_ctx);

        sp_verifier.add_claim(last_claim)?;

        let output_claims = &proof.output_claims;
        output_claims
            .iter()
            .try_for_each(|claim| sp_verifier.add_claim(claim.clone()))?;

        let output_proof = &proof.io_accumulation;

        let commit_claim = sp_verifier.verify(output_proof, verifier.transcript)?;

        // Add the result of the same poly verifier to the commitment verifier.
        verifier
            .witness_verifier
            .add_claim(self.poly_id, commit_claim)?;

        // Challenegs used to batch input poly claims together and link them with zerocheck and lookup verification output
        let [r1, r2] = [verifier
            .transcript
            .get_and_append_challenge(b"input_batching")
            .elements; 2];
        let one_minus_r1 = E::ONE - r1;
        let one_minus_r2 = E::ONE - r2;

        let eval_multiplicands = [
            one_minus_r1 * one_minus_r2,
            one_minus_r1 * r2,
            r1 * one_minus_r2,
            r1 * r2,
        ];
        let zc_point = subclaim
            .point
            .iter()
            .map(|chal| chal.elements)
            .collect::<Vec<E>>();
        let zerocheck_point = [
            &[r1],
            &zc_point[..proof.variable_gap],
            &[r2],
            &zc_point[proof.variable_gap..],
        ]
        .concat();

        let zerocheck_input_eval = izip!(proof.zerocheck_evals.iter(), eval_multiplicands.iter())
            .fold(E::ZERO, |zerocheck_acc, (&ze, &me)| {
                zerocheck_acc + (output_claims[0].eval - ze) * me
            });

        let out_claim = Claim {
            point: zerocheck_point,
            eval: zerocheck_input_eval,
        };

        // Now we check consistency between the lookup/sumcheck proof claims and the claims passed to the same poly verifiers.
        let beta_eval = identity_eval(&output_claims[0].point, &verifier_claims.claims()[0].point);

        let computed_zerocheck_claim = proof
            .zerocheck_evals
            .iter()
            .chain(std::iter::once(&beta_eval))
            .product::<E>()
            + proof
                .zerocheck_evals
                .iter()
                .fold((E::ZERO, batching_challenge), |(acc, comb), v| {
                    (acc + *v * beta_eval * comb, comb * batching_challenge)
                })
                .0;

        ensure!(
            computed_zerocheck_claim == subclaim.expected_evaluation,
            "Computed zerocheck claim did not line up with output of sumcheck verification"
        );

        Ok(out_claim)
    }
}

/// Information about a maxpool2d step
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Copy, PartialOrd, Ord, Hash)]
pub struct Maxpool2D {
    pub kernel_size: usize,
    pub stride: usize,
}

impl Default for Maxpool2D {
    fn default() -> Self {
        Maxpool2D {
            kernel_size: MAXPOOL2D_KERNEL_SIZE,
            stride: MAXPOOL2D_KERNEL_SIZE,
        }
    }
}

impl Maxpool2D {
    pub fn op(&self, input: &Tensor<Element>) -> Tensor<Element> {
        assert!(
            self.kernel_size == MAXPOOL2D_KERNEL_SIZE,
            "Maxpool2D works only for kernel size {}",
            MAXPOOL2D_KERNEL_SIZE
        );
        assert!(
            self.stride == MAXPOOL2D_KERNEL_SIZE,
            "Maxpool2D works only for stride size {}",
            MAXPOOL2D_KERNEL_SIZE
        );
        input.maxpool2d(self.kernel_size, self.stride)
    }

    /// Computes MLE evaluations related to proving Maxpool function.
    /// The outputs of this function are the four polynomials corresponding to the input to the Maxpool, each with two variables fixed
    /// so that PROD (Output - poly_i) == 0 at every evaluation point.
    pub fn compute_polys<E: ExtensionField>(
        &self,
        input: &Tensor<Element>,
    ) -> Vec<Vec<E::BaseField>> {
        let padded_input = input.pad_next_power_of_two();

        let padded_output = self.op(&input).pad_next_power_of_two();
        let padded_input_shape = padded_input.get_shape();

        let new_fixed = (0..padded_input_shape[2] << 1)
            .into_par_iter()
            .map(|i| {
                padded_input
                    .get_data()
                    .iter()
                    .skip(i)
                    .step_by(padded_input_shape[2] << 1)
                    .copied()
                    .collect::<Vec<Element>>()
            })
            .collect::<Vec<Vec<Element>>>();

        let new_even = new_fixed
            .iter()
            .step_by(2)
            .cloned()
            .collect::<Vec<Vec<Element>>>();

        let new_odd = new_fixed
            .iter()
            .skip(1)
            .step_by(2)
            .cloned()
            .collect::<Vec<Vec<Element>>>();

        let (even_diff, odd_diff): (Vec<Vec<E::BaseField>>, Vec<Vec<E::BaseField>>) = new_even
            .par_chunks(padded_input_shape[2] >> 1)
            .zip(new_odd.par_chunks(padded_input_shape[2] >> 1))
            .map(|(even_chunk, odd_chunk)| {
                let mut even_merged = even_chunk.to_vec();
                let mut odd_merged = odd_chunk.to_vec();
                for i in (0..ceil_log2(padded_input_shape[2]) - 1).rev() {
                    let mid_point = 1 << i;
                    let (even_low, even_high) = even_merged.split_at(mid_point);
                    let (odd_low, odd_high) = odd_merged.split_at(mid_point);
                    even_merged = even_low
                        .iter()
                        .zip(even_high.iter())
                        .map(|(l, h)| {
                            l.iter()
                                .interleave(h.iter())
                                .copied()
                                .collect::<Vec<Element>>()
                        })
                        .collect::<Vec<Vec<Element>>>();
                    odd_merged = odd_low
                        .iter()
                        .zip(odd_high.iter())
                        .map(|(l, h)| {
                            l.iter()
                                .interleave(h.iter())
                                .copied()
                                .collect::<Vec<Element>>()
                        })
                        .collect::<Vec<Vec<Element>>>();
                }

                izip!(
                    even_merged[0].iter(),
                    odd_merged[0].iter(),
                    padded_output.get_data()
                )
                .map(|(e, o, data)| {
                    let e_field: E = (data - e).to_field();
                    let o_field: E = (data - o).to_field();
                    (e_field.as_bases()[0], o_field.as_bases()[0])
                })
                .unzip::<_, _, Vec<E::BaseField>, Vec<E::BaseField>>()
            })
            .unzip();

        [even_diff, odd_diff].concat()
    }

    /// 实现池化层的反向传播
    /// 最大池化的反向传播规则:
    /// - 只有在前向传播时被选中的最大值位置会传递梯度
    /// - 其他位置的梯度为0
    pub fn backward(
        &self,
        output_grad: &Tensor<Element>,
        input: &Tensor<Element>,
        cache: Option<&Tensor<Element>>, // 可以存储前向传播时的索引信息
    ) -> Tensor<Element> {
        assert!(
            self.kernel_size == MAXPOOL2D_KERNEL_SIZE,
            "Maxpool2D works only for kernel size {}",
            MAXPOOL2D_KERNEL_SIZE
        );
        assert!(
            self.stride == MAXPOOL2D_KERNEL_SIZE,
            "Maxpool2D works only for stride size {}",
            MAXPOOL2D_KERNEL_SIZE
        );

        let input_shape = input.get_shape();
        let dims = input_shape.len();
        let (h, w) = (input_shape[dims - 2], input_shape[dims - 1]);
        
        // 使用 *ZERO 来初始化梯度张量
        let mut grad_input = vec![*ZERO; input.get_data().len()];

        let outer_dims: usize = input_shape[..dims - 2].iter().product();
        
        // 对每个池化窗口计算梯度
        for n in 0..outer_dims {
            for i in 0..(h/self.kernel_size) {
                for j in 0..(w/self.kernel_size) {
                    // 获取当前输出梯度
                    let out_idx = n * (h/self.kernel_size) * (w/self.kernel_size) + i * (w/self.kernel_size) + j;
                    let out_grad = output_grad.get_data()[out_idx];

                    // 找到最大值的位置
                    let mut max_val = Element::MIN; // 这里也需要处理
                    let mut max_i = 0;
                    let mut max_j = 0;
                    
                    // 在池化窗口中寻找最大值位置
                    for ki in 0..self.kernel_size {
                        for kj in 0..self.kernel_size {
                            let in_idx = n * h * w + (i * self.stride + ki) * w + (j * self.stride + kj);
                            let val = input.get_data()[in_idx];
                            if val > max_val {
                                max_val = val;
                                max_i = ki;
                                max_j = kj;
                            }
                        }
                    }

                    // 将梯度传递给最大值位置
                    let grad_idx = n * h * w + (i * self.stride + max_i) * w + (j * self.stride + max_j);
                    grad_input[grad_idx] = out_grad;
                }
            }
        }

        Tensor::new(input_shape.clone(), grad_input)
    }
}

#[cfg(test)]
mod tests {
    use crate::{commit::compute_betas_eval, default_transcript};

    use super::*;
    use crate::quantization::Fieldizer;
    use ark_std::rand::{Rng, thread_rng};
    use ff::Field;
    use gkr::util::ceil_log2;
    use goldilocks::{Goldilocks, GoldilocksExt2};
    use itertools::Itertools;
    use multilinear_extensions::{
        mle::{DenseMultilinearExtension, MultilinearExtension},
        virtual_poly::{ArcMultilinearExtension, VirtualPolynomial},
    };
    use sumcheck::structs::{IOPProverState, IOPVerifierState};

    type F = GoldilocksExt2;

    #[test]
    fn test_max_pool_zerocheck() {
        let mut rng = thread_rng();
        for _ in 0..50 {
            let random_shape = (0..4)
                .map(|i| {
                    if i < 2 {
                        rng.gen_range(2usize..6)
                    } else {
                        2 * rng.gen_range(2usize..5)
                    }
                })
                .collect::<Vec<usize>>();
            let input_data_size = random_shape.iter().product::<usize>();
            let data = (0..input_data_size)
                .map(|_| rng.gen_range(-128i128..128))
                .collect::<Vec<Element>>();
            let input = Tensor::<Element>::new(random_shape.clone(), data);

            let info = Maxpool2D {
                kernel_size: MAXPOOL2D_KERNEL_SIZE,
                stride: MAXPOOL2D_KERNEL_SIZE,
            };

            let output = info.op(&input);

            let padded_input = input.pad_next_power_of_two();

            let padded_output = output.pad_next_power_of_two();

            let padded_input_shape = padded_input.get_shape();

            let num_vars = padded_input.get_data().len().ilog2() as usize;
            let output_num_vars = padded_output.get_data().len().ilog2() as usize;

            let mle = DenseMultilinearExtension::<F>::from_evaluations_vec(
                num_vars,
                padded_input
                    .get_data()
                    .iter()
                    .map(|val_i128| {
                        let field: F = val_i128.to_field();
                        field.as_bases()[0]
                    })
                    .collect::<Vec<Goldilocks>>(),
            );

            // This should give all possible combinations of fixing the lowest three bits in ascending order

            let fixed_mles = (0..padded_input_shape[3] << 1)
                .map(|i| {
                    let point = (0..ceil_log2(padded_input_shape[3]) + 1)
                        .map(|n| F::from((i as u64 >> n) & 1))
                        .collect::<Vec<F>>();

                    mle.fix_variables(&point)
                })
                .collect::<Vec<DenseMultilinearExtension<F>>>();
            // f(0,x,0,..) = x * f(0,1,0,...) + (1 - x) * f(0,0,0,...)
            let even_mles = fixed_mles
                .iter()
                .cloned()
                .step_by(2)
                .collect::<Vec<DenseMultilinearExtension<F>>>();
            let odd_mles = fixed_mles
                .iter()
                .cloned()
                .skip(1)
                .step_by(2)
                .collect::<Vec<DenseMultilinearExtension<F>>>();

            let even_merged = even_mles
                .chunks(padded_input_shape[3] >> 1)
                .map(|mle_chunk| {
                    let mut mles_vec = mle_chunk
                        .iter()
                        .map(|m| m.get_ext_field_vec().to_vec())
                        .collect::<Vec<Vec<F>>>();
                    while mles_vec.len() > 1 {
                        let half = mles_vec.len() / 2;

                        mles_vec = mles_vec[..half]
                            .iter()
                            .zip(mles_vec[half..].iter())
                            .map(|(l, h)| {
                                l.iter().interleave(h.iter()).copied().collect::<Vec<F>>()
                            })
                            .collect::<Vec<Vec<F>>>();
                    }

                    DenseMultilinearExtension::<F>::from_evaluations_ext_slice(
                        output_num_vars,
                        &mles_vec[0],
                    )
                })
                .collect::<Vec<DenseMultilinearExtension<F>>>();

            let odd_merged = odd_mles
                .chunks(padded_input_shape[3] >> 1)
                .map(|mle_chunk| {
                    let mut mles_vec = mle_chunk
                        .iter()
                        .map(|m| m.get_ext_field_vec().to_vec())
                        .collect::<Vec<Vec<F>>>();
                    while mles_vec.len() > 1 {
                        let half = mles_vec.len() / 2;

                        mles_vec = mles_vec[..half]
                            .iter()
                            .zip(mles_vec[half..].iter())
                            .map(|(l, h)| {
                                l.iter().interleave(h.iter()).copied().collect::<Vec<F>>()
                            })
                            .collect::<Vec<Vec<F>>>();
                    }

                    DenseMultilinearExtension::<F>::from_evaluations_ext_slice(
                        output_num_vars,
                        &mles_vec[0],
                    )
                })
                .collect::<Vec<DenseMultilinearExtension<F>>>();

            let merged_input_mles = [even_merged, odd_merged].concat();

            let output_mle = DenseMultilinearExtension::<F>::from_evaluations_ext_vec(
                output_num_vars,
                padded_output
                    .get_data()
                    .iter()
                    .map(|val_i128| {
                        let field: F = val_i128.to_field();
                        field
                    })
                    .collect::<Vec<F>>(),
            );

            let mut vp = VirtualPolynomial::<F>::new(output_num_vars);

            let diff_mles = merged_input_mles
                .iter()
                .map(|in_mle| {
                    DenseMultilinearExtension::<F>::from_evaluations_ext_vec(
                        output_num_vars,
                        in_mle
                            .get_ext_field_vec()
                            .iter()
                            .zip(output_mle.get_ext_field_vec().iter())
                            .map(|(input, output)| *output - *input)
                            .collect::<Vec<F>>(),
                    )
                    .into()
                })
                .collect::<Vec<ArcMultilinearExtension<F>>>();

            (0..1 << output_num_vars).for_each(|j| {
                let values = diff_mles
                    .iter()
                    .map(|mle| mle.get_ext_field_vec()[j])
                    .collect::<Vec<F>>();
                assert_eq!(values.iter().product::<F>(), F::ZERO)
            });

            vp.add_mle_list(diff_mles, F::ONE);

            let random_point = (0..output_num_vars)
                .map(|_| F::random(&mut rng))
                .collect::<Vec<F>>();

            let beta_evals = compute_betas_eval(&random_point);

            let beta_mle: ArcMultilinearExtension<F> =
                DenseMultilinearExtension::<F>::from_evaluations_ext_vec(
                    output_num_vars,
                    beta_evals,
                )
                .into();
            vp.mul_by_mle(beta_mle.clone(), Goldilocks::ONE);

            let aux_info = vp.aux_info.clone();

            let mut prover_transcript = default_transcript::<F>();

            #[allow(deprecated)]
            let (proof, state) = IOPProverState::<F>::prove_parallel(vp, &mut prover_transcript);

            let mut verifier_transcript = default_transcript::<F>();

            let subclaim =
                IOPVerifierState::<F>::verify(F::ZERO, &proof, &aux_info, &mut verifier_transcript);

            let point = subclaim
                .point
                .iter()
                .map(|chal| chal.elements)
                .collect::<Vec<F>>();

            let fixed_points = [[F::ZERO, F::ZERO], [F::ZERO, F::ONE], [F::ONE, F::ZERO], [
                F::ONE,
                F::ONE,
            ]]
            .map(|pair| {
                [
                    &[pair[0]],
                    &point[..ceil_log2(padded_input_shape[3]) - 1],
                    &[pair[1]],
                    &point[ceil_log2(padded_input_shape[3]) - 1..],
                ]
                .concat()
            });

            let output_eval = output_mle.evaluate(&point);
            let input_evals = fixed_points
                .iter()
                .map(|p| mle.evaluate(p))
                .collect::<Vec<F>>();

            let eq_eval = beta_mle.evaluate(&point);

            let calc_eval = input_evals
                .iter()
                .map(|ie| output_eval - *ie)
                .product::<F>()
                * eq_eval;

            assert_eq!(calc_eval, subclaim.expected_evaluation);

            // in order output - 00, output - 10, output - 01, output - 11, eq I believe
            let final_mle_evals = state.get_mle_final_evaluations();

            let [r1, r2] = [F::random(&mut rng); 2];
            let one_minus_r1 = F::ONE - r1;
            let one_minus_r2 = F::ONE - r2;

            let maybe_eval = (output_eval - final_mle_evals[0]) * one_minus_r1 * one_minus_r2
                + (output_eval - final_mle_evals[2]) * one_minus_r1 * r2
                + (output_eval - final_mle_evals[1]) * r1 * one_minus_r2
                + (output_eval - final_mle_evals[3]) * r1 * r2;

            let mle_eval = mle.evaluate(
                &[
                    &[r1],
                    &point[..ceil_log2(padded_input_shape[3]) - 1],
                    &[r2],
                    &point[ceil_log2(padded_input_shape[3]) - 1..],
                ]
                .concat(),
            );

            assert_eq!(mle_eval, maybe_eval);
        }
    }

    #[test]
    fn test_maxpool_backward() {
        let pool = Maxpool2D::default();
        
        // 创建测试输入
        let input = Tensor::new(
            vec![1, 1, 4, 4],
            vec![
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
            ]
        );

        // 前向传播
        let output = pool.op(&input);

        // 创建输出梯度
        let output_grad = Tensor::new(
            vec![1, 1, 2, 2],
            vec![1, 1, 1, 1]
        );

        // 执行反向传播
        let input_grad = pool.backward(&output_grad, &input, None);

        // 验证梯度形状
        assert_eq!(input_grad.get_shape(), input.get_shape());

        // 验证梯度值 - 只有最大值位置有梯度
        let expected = vec![
            0, 0, 0, 0,
            0, 1, 0, 1,
            0, 0, 0, 0,
            0, 1, 0, 1
        ];
        
        assert_eq!(input_grad.get_data(), expected);
    }
}

//! Contains code for verifying a LogUpProof

use ff_ext::ExtensionField;
use multilinear_extensions::virtual_poly::VPAuxInfo;
use sumcheck::structs::IOPVerifierState;
use transcript::Transcript;

use crate::commit::identity_eval;

use super::{
    error::LogUpError,
    structs::{LogUpProof, LogUpVerifierClaim, ProofType},
};

pub fn verify_logup_proof<E: ExtensionField, T: Transcript<E>>(
    proof: &LogUpProof<E>,
    num_instances: usize,
    constant_challenge: E,
    column_separation_challenge: E,
    transcript: &mut T,
) -> Result<LogUpVerifierClaim<E>, LogUpError> {
    // Append the number of instances along with their output evals to the transcript and then squeeze our first alpha and lambda
    transcript.append_field_element(&E::BaseField::from(num_instances as u64));
    proof.append_to_transcript(transcript);

    let (numerators, denominators): (Vec<E>, Vec<E>) = proof.fractional_outputs();

    let batching_challenge = transcript
        .get_and_append_challenge(b"inital_batching")
        .elements;
    let mut alpha = transcript
        .get_and_append_challenge(b"inital_alpha")
        .elements;
    let mut lambda = transcript
        .get_and_append_challenge(b"initial_lambda")
        .elements;

    let (mut current_claim, _) =
        proof
            .circuit_outputs()
            .iter()
            .fold((E::ZERO, E::ONE), |(acc, alpha_comb), e| {
                // we have four evals and we batch them as alpha * (batching_challenge * (e[1] - e[0]) + e[0] + lambda * (batching_challenge * (e[3] - e[2]) + e[2]) )
                (
                    acc + alpha_comb
                        * (batching_challenge * (e[1] - e[0])
                            + e[0]
                            + lambda * (batching_challenge * (e[3] - e[2]) + e[2])),
                    alpha_comb * alpha,
                )
            });
    // The initial sumcheck point is just the batching challenge
    let mut sumcheck_point: Vec<E> = vec![batching_challenge];

    for (i, (sumcheck_proof, round_evaluations)) in proof.proofs_and_evals().enumerate() {
        // Append the current claim to the transcript
        transcript.append_field_element_ext(&current_claim);

        // Calculate the eq_poly evaluation for this round
        let eq_eval = identity_eval(&sumcheck_point, &sumcheck_proof.point);

        // Run this rounds sumcheck verification
        let current_num_vars = i + 1;
        let aux_info = VPAuxInfo::<E>::from_mle_list_dimensions(&[vec![current_num_vars; 3]]);
        let sumcheck_subclaim =
            IOPVerifierState::<E>::verify(current_claim, sumcheck_proof, &aux_info, transcript);

        // Squeeze the challenges to combine everything into a single sumcheck
        let batching_challenge = transcript
            .get_and_append_challenge(b"logup_batching")
            .elements;
        let next_alpha = transcript.get_and_append_challenge(b"logup_alpha").elements;
        let next_lambda = transcript
            .get_and_append_challenge(b"logup_lambda")
            .elements;

        // Now we tak the round evals and check their consistency with the sumcheck claim
        let evals_per_instance = round_evaluations.len() / num_instances;

        current_claim = if evals_per_instance == 4 {
            let (next_claim, _, sumcheck_claim, _) = round_evaluations.chunks(4).fold(
                (E::ZERO, E::ONE, E::ZERO, E::ONE),
                |(acc_next_claim, next_alpha_comb, acc_sumcheck_claim, prev_alpha), e| {
                    let next_claim_term = acc_next_claim
                        + next_alpha_comb
                            * (batching_challenge * (e[2] - e[0])
                                + e[0]
                                + next_lambda * (batching_challenge * (e[1] - e[3]) + e[3]));

                    let sumcheck_claim_term = acc_sumcheck_claim
                        + prev_alpha
                            * (eq_eval * (e[0] * e[1] + e[2] * e[3] + lambda * e[3] * e[1]));
                    (
                        next_claim_term,
                        next_alpha_comb * next_alpha,
                        sumcheck_claim_term,
                        prev_alpha * alpha,
                    )
                },
            );
            if sumcheck_claim != sumcheck_subclaim.expected_evaluation {
                return Err(LogUpError::VerifierError(format!(
                    "Calculated sumcheck claim: {:?} does not equal this rounds sumcheck output claim: {:?} at round: {}",
                    sumcheck_claim, sumcheck_subclaim.expected_evaluation, i
                )));
            }
            next_claim
        } else {
            let (next_claim, _, sumcheck_claim, _) = round_evaluations.chunks(2).fold(
                (E::ZERO, E::ONE, E::ZERO, E::ONE),
                |(acc_next_claim, alpha_comb, acc_sumcheck_claim, prev_alpha), e| {
                    let next_claim_term =
                        acc_next_claim + alpha_comb * (batching_challenge * (e[0] - e[1]) + e[1]);
                    let sumcheck_claim_term = acc_sumcheck_claim
                        + prev_alpha * eq_eval * (-e[1] - e[0] + lambda * e[0] * e[1]);
                    (
                        next_claim_term,
                        alpha_comb * next_alpha,
                        sumcheck_claim_term,
                        prev_alpha * alpha,
                    )
                },
            );
            if sumcheck_claim != sumcheck_subclaim.expected_evaluation {
                return Err(LogUpError::VerifierError(format!(
                    "Calculated sumcheck claim: {:?} does not equal this rounds sumcheck output claim: {:?} at round: {}",
                    sumcheck_claim, sumcheck_subclaim.expected_evaluation, i
                )));
            }
            next_claim
        };

        alpha = next_alpha;
        lambda = next_lambda;

        sumcheck_point = sumcheck_subclaim
            .point
            .iter()
            .map(|chal| chal.elements)
            .collect::<Vec<E>>();
        sumcheck_point.push(batching_challenge);
    }

    let calculated_eval = calculate_final_eval(
        proof,
        constant_challenge,
        column_separation_challenge,
        alpha,
        lambda,
        num_instances,
    );

    if calculated_eval != current_claim {
        return Err(LogUpError::VerifierError(format!(
            "Calculated final value: {:?} does not match final sumcheck output: {:?}",
            calculated_eval, current_claim
        )));
    }

    Ok(LogUpVerifierClaim::<E>::new(
        proof.output_claims().to_vec(),
        numerators,
        denominators,
    ))
}

fn calculate_final_eval<E: ExtensionField>(
    proof: &LogUpProof<E>,
    constant_challenge: E,
    column_separation_challenge: E,
    alpha: E,
    lambda: E,
    num_instances: usize,
) -> E {
    match proof.proof_type() {
        ProofType::Lookup => {
            let claims_per_instance = proof.output_claims().len() / num_instances;

            proof
                .output_claims()
                .chunks(claims_per_instance)
                .fold((E::ZERO, E::ONE), |(acc, alpha_comb), chunk| {
                    let chunk_eval = chunk
                        .iter()
                        .fold((constant_challenge, E::ONE), |(acc, csc_comb), cl| {
                            (
                                acc + cl.eval * csc_comb,
                                csc_comb * column_separation_challenge,
                            )
                        })
                        .0;
                    (acc + chunk_eval * alpha_comb, alpha_comb * alpha)
                })
                .0
        }
        ProofType::Table => {
            // The first output claim is the multiplicity poly which is the numerator
            let columns_eval = proof.output_claims()[1..]
                .iter()
                .fold((constant_challenge, E::ONE), |(acc, csc_comb), cl| {
                    (
                        acc + cl.eval * csc_comb,
                        csc_comb * column_separation_challenge,
                    )
                })
                .0;

            proof.output_claims()[0].eval + lambda * columns_eval
        }
    }
}
/*
# verifier.rs 文件分析

## 1. 核心功能

验证器实现了 LogUp GKR 证明系统的验证逻辑。主要包括:

1. 证明验证函数
2. 最终评估计算
3. 一致性检查

## 2. 主要结构

### LogUp验证函数
```rust
pub fn verify_logup_proof<E: ExtensionField, T: Transcript<E>>(
    proof: &LogUpProof<E>,
    num_instances: usize,
    constant_challenge: E,
    column_separation_challenge: E,
    transcript: &mut T,
) -> Result<LogUpVerifierClaim<E>, LogUpError>
```

### 最终评估计算
```rust
fn calculate_final_eval<E: ExtensionField>(
    proof: &LogUpProof<E>,
    constant_challenge: E,
    column_separation_challenge: E,
    alpha: E,
    lambda: E,
    num_instances: usize,
) -> E
```

## 3. 优化建议

### 1. 错误处理优化

````rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VerifierError {
    #[error("验证失败: {0}")]
    ValidationFailed(String),
    
    #[error("求和检查错误: {0}")]
    SumcheckError(String),
    
    #[error("Transcript错误: {0}")]
    TranscriptError(String),
}

impl From<LogUpError> for VerifierError {
    fn from(error: LogUpError) -> Self {
        match error {
            LogUpError::VerifierError(msg) => VerifierError::ValidationFailed(msg),
            _ => VerifierError::ValidationFailed("未知错误".to_string()),
        }
    }
}
````

### 2. 性能优化

````rust
use rayon::prelude::*;

fn calculate_final_eval<E: ExtensionField>(
    proof: &LogUpProof<E>,
    constant_challenge: E,
    column_separation_challenge: E,
    alpha: E,
    lambda: E,
    num_instances: usize,
) -> E {
    match proof.proof_type() {
        ProofType::Lookup => {
            let claims_per_instance = proof.output_claims().len() / num_instances;
            
            // 并行处理chunks
            proof.output_claims()
                .par_chunks(claims_per_instance)
                .map(|chunk| {
                    let chunk_eval = chunk.iter()
                        .fold((constant_challenge, E::ONE), 
                            |(acc, csc_comb), cl| {
                                (acc + cl.eval * csc_comb,
                                 csc_comb * column_separation_challenge)
                            }).0;
                    chunk_eval
                })
                .reduce(|| E::ZERO,
                       |acc, chunk_eval| acc + chunk_eval)
        }
        // ...existing code...
    }
}
````

### 3. 调试支持

````rust
use tracing::{debug, info, warn};

#[timed::timed_instrument(level = "debug")]
pub fn verify_logup_proof<E: ExtensionField, T: Transcript<E>>(
    proof: &LogUpProof<E>,
    num_instances: usize,
    constant_challenge: E,
    column_separation_challenge: E,
    transcript: &mut T,
) -> Result<LogUpVerifierClaim<E>, LogUpError> {
    debug!("开始验证 LogUp 证明，实例数量: {}", num_instances);
    
    let start = std::time::Instant::now();
    // ...existing code...
    
    info!("验证完成，耗时: {:?}", start.elapsed());
    Ok(claim)
}
````

## 4. 测试完善

````rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_transcript;
    use goldilocks::GoldilocksExt2;

    #[test]
    fn test_verify_valid_proof() {
        let mut transcript = default_transcript::<GoldilocksExt2>();
        let proof = create_valid_test_proof();
        let result = verify_logup_proof(
            &proof,
            1,
            GoldilocksExt2::ONE,
            GoldilocksExt2::ONE,
            &mut transcript
        );
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_verify_invalid_proof() {
        let mut transcript = default_transcript::<GoldilocksExt2>();
        let proof = create_invalid_test_proof();
        let result = verify_logup_proof(
            &proof,
            1,
            GoldilocksExt2::ONE,
            GoldilocksExt2::ONE,
            &mut transcript
        );
        assert!(result.is_err());
    }
    
    #[test]
    fn test_calculate_final_eval() {
        let proof = create_test_proof();
        let eval = calculate_final_eval(
            &proof,
            GoldilocksExt2::ONE,
            GoldilocksExt2::ONE,
            GoldilocksExt2::ONE,
            GoldilocksExt2::ONE,
            1
        );
        assert_ne!(eval, GoldilocksExt2::ZERO);
    }
}
````

## 5. 总结

verifier.rs 实现了 LogUp GKR 系统的验证功能：

1. **主要特点**
   - 完整的证明验证
   - 高效的评估计算
   - 严格的错误处理

2. **优化方向**
   - 添加并行处理
   - 改进错误处理
   - 增加调试支持
   - 完善测试覆盖

3. **代码健壮性**
   - 类型安全
   - 错误传播
   - 性能监控

该实现为零知识证明系统提供了可靠的验证支持。 */
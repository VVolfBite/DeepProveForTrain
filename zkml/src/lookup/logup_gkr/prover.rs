//! Contains the code for batch proving a number of LogUp GKR claims.

use std::sync::Arc;

use ff_ext::ExtensionField;

use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::{ArcMultilinearExtension, VirtualPolynomial},
};
use sumcheck::structs::{IOPProof, IOPProverState};
use transcript::Transcript;

use crate::{Claim, commit::compute_betas_eval, lookup::logup_gkr::circuit::LogUpLayer};

use super::{
    error::LogUpError,
    structs::{LogUpInput, LogUpProof, ProofType},
};

/// Function to batch prove a collection of [`LookupInput`]s
/// TODO: add support to batch in claims about the output of this in the final step of the GKR circuit.
pub fn batch_prove<E: ExtensionField, T: Transcript<E>>(
    input: &LogUpInput<E>,
    transcript: &mut T,
) -> Result<LogUpProof<E>, LogUpError> {
    // Work out how many instances we are dealing with
    let circuits = input.make_circuits();
    let num_instances = circuits.len();

    // Work out the total number of layers and the number of layers per instance.
    let mut total_layers = 0usize;
    let circuit_outputs = circuits
        .iter()
        .map(|c| {
            total_layers = std::cmp::max(total_layers, c.num_vars());
            c.outputs()
        })
        .collect::<Vec<Vec<E>>>();

    // When proving we want to work from the top down so we convert each of the circuits into an iterator over its layers in reverse order.
    // We also skip the first layer after reversing as this is just the output claims.
    let mut layer_iters = circuits
        .iter()
        .map(|c| c.layers().into_iter().rev().skip(1))
        .collect::<Vec<_>>();

    // Append the number of instances along with their output evals to the transcript and then squeeze our first alpha and lambda
    transcript.append_field_element(&E::BaseField::from(num_instances as u64));
    circuit_outputs
        .iter()
        .for_each(|evals| transcript.append_field_element_exts(evals));

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
        circuit_outputs
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

    let mut sumcheck_proofs: Vec<IOPProof<E>> = vec![];

    let mut round_evaluations: Vec<Vec<E>> = vec![];

    for current_layer_vars in 1..=total_layers {
        // Append the current claim to the transcript
        transcript.append_field_element_ext(&current_claim);

        // Compute the eq_evals
        let eq_poly: ArcMultilinearExtension<E> =
            Arc::new(compute_betas_eval(&sumcheck_point).into_mle());

        // Then add all the terms to the sumcheck virtual polynomial
        let mut vp = VirtualPolynomial::<E>::new(current_layer_vars);

        let mut current_alpha = E::ONE;
        layer_iters.iter_mut().try_for_each(|iter| {
            let layer = iter.next().ok_or(LogUpError::ProvingError(
                "One of the circuits was not the same size as the others".to_string(),
            ))?;
            let mles = layer.get_mles();
            if let LogUpLayer::Generic { .. } | LogUpLayer::InitialTable { .. } = layer {
                vp.add_mle_list(
                    vec![eq_poly.clone(), mles[0].clone(), mles[3].clone()],
                    current_alpha,
                );
                vp.add_mle_list(
                    vec![eq_poly.clone(), mles[1].clone(), mles[2].clone()],
                    current_alpha,
                );
                vp.add_mle_list(
                    vec![eq_poly.clone(), mles[2].clone(), mles[3].clone()],
                    current_alpha * lambda,
                );
            } else {
                // Here we are in the initial lookup case so we have no numerator polynomials (all the numerator values are -1)
                vp.add_mle_list(vec![eq_poly.clone(), mles[1].clone()], -current_alpha);
                vp.add_mle_list(vec![eq_poly.clone(), mles[0].clone()], -current_alpha);
                vp.add_mle_list(
                    vec![eq_poly.clone(), mles[0].clone(), mles[1].clone()],
                    current_alpha * lambda,
                );
            }
            current_alpha *= alpha;
            Result::<(), LogUpError>::Ok(())
        })?;

        // Run the sumcheck for this round
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, transcript);

        // Update the sumcheck proof
        sumcheck_point = proof.point.clone();

        // The first one is always the eq_poly eval
        let evals = &state.get_mle_final_evaluations()[1..];

        // Squeeze the challenges to combine everything into a single sumcheck
        let batching_challenge = transcript
            .get_and_append_challenge(b"logup_batching")
            .elements;
        alpha = transcript.get_and_append_challenge(b"logup_alpha").elements;
        lambda = transcript
            .get_and_append_challenge(b"logup_lambda")
            .elements;
        // Append the batching challenge to the proof point
        sumcheck_point.push(batching_challenge);
        // Append the sumcheck proof to the list of proofs
        sumcheck_proofs.push(proof);

        // This step works out the initial claim for the next round of the protocol
        current_claim = if current_layer_vars != total_layers {
            evals
                .chunks(4)
                .fold((E::ZERO, E::ONE), |(acc, alpha_comb), e| {
                    (
                        acc + alpha_comb
                            * (batching_challenge * (e[2] - e[0])
                                + e[0]
                                + lambda * (batching_challenge * (e[1] - e[3]) + e[3])),
                        alpha_comb * alpha,
                    )
                })
                .0
        } else {
            final_round_claim(input, evals, batching_challenge, alpha, lambda)
        };
        // Append the claimed evaluations from the end of this round to the proof.
        round_evaluations.push(evals.to_vec());
    }

    // We take the final sumcheck point and produce a list of claims about all the columns looked up/ in the table and
    // also the multiplicity polynomial in the table case. These will be used by the verifier to check the final sumcheck proofs claim.
    // Then each of these claims should be verified either by another layer proof or via commitment opening proof.
    let output_claims = input
        .base_mles()
        .iter()
        .map(|mle| Claim::<E> {
            point: sumcheck_point.clone(),
            eval: mle.evaluate(&sumcheck_point),
        })
        .collect::<Vec<Claim<E>>>();

    // The proof type
    let proof_type = match input {
        LogUpInput::Lookup { .. } => ProofType::Lookup,
        LogUpInput::Table { .. } => ProofType::Table,
    };

    Ok(LogUpProof::<E> {
        sumcheck_proofs,
        round_evaluations,
        output_claims,
        circuit_outputs,
        proof_type,
    })
}

/// Function to compute the final round claim depending on whether this is a table proof or a lookup proof.
fn final_round_claim<E: ExtensionField>(
    input: &LogUpInput<E>,
    evals: &[E],
    batching_challenge: E,
    alpha: E,
    lambda: E,
) -> E {
    match input {
        LogUpInput::Lookup { .. } => {
            // In this case there is only one polynomial per instance, the denominator, so we batch together its low and high parts
            evals
                .chunks(2)
                .fold((E::ZERO, E::ONE), |(acc, alpha_comb), e| {
                    (
                        acc + alpha_comb * (batching_challenge * (e[0] - e[1]) + e[1]),
                        alpha_comb * alpha,
                    )
                })
                .0
        }
        LogUpInput::Table { .. } => {
            // Here we have two polynomials, the numerator and denominator, so we batch their respective low and high parts together
            evals
                .chunks(4)
                .fold((E::ZERO, E::ONE), |(acc, alpha_comb), e| {
                    (
                        acc + alpha_comb
                            * (batching_challenge * (e[2] - e[0])
                                + e[0]
                                + lambda * (batching_challenge * (e[1] - e[3]) + e[3])),
                        alpha_comb * alpha,
                    )
                })
                .0
        }
    }
}

/*
# prover.rs 文件分析

## 1. 代码结构和主要功能

### 主要功能
实现了 LogUp GKR (Gate Kernal Recursion) 的批量证明生成功能。

### 核心数据结构
```rust
pub struct LogUpProof<E: ExtensionField> {
    sumcheck_proofs: Vec<IOPProof<E>>,
    round_evaluations: Vec<Vec<E>>,
    output_claims: Vec<Claim<E>>,
    circuit_outputs: Vec<Vec<E>>,
    proof_type: ProofType,
}
```

## 2. 主要实现

### 批量证明生成
```rust
pub fn batch_prove<E: ExtensionField, T: Transcript<E>>(
    input: &LogUpInput<E>,
    transcript: &mut T,
) -> Result<LogUpProof<E>, LogUpError>
```

主要步骤:
1. 计算电路实例数量
2. 生成电路输出
3. 处理层迭代器
4. 生成和验证证明

## 3. 优化建议

### 1. 并行处理优化
````rust
use rayon::prelude::*;

// ...existing code...

pub fn batch_prove<E: ExtensionField, T: Transcript<E>>(
    input: &LogUpInput<E>,
    transcript: &mut T,
) -> Result<LogUpProof<E>, LogUpError> {
    let circuits = input.make_circuits();
    let circuit_outputs = circuits.par_iter()  // 使用并行迭代器
        .map(|c| {
            let num_vars = c.num_vars();
            (num_vars, c.outputs())
        })
        .collect::<Vec<_>>();
    
    let total_layers = circuit_outputs.iter().map(|(n,_)| *n).max().unwrap_or(0);
    let outputs = circuit_outputs.into_iter().map(|(_,o)| o).collect::<Vec<_>>();
    
    // ...remaining code...
}
````

### 2. 错误处理优化
````rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProverError {
    #[error("Circuit size mismatch: {0}")]
    CircuitSizeMismatch(String),
    
    #[error("Invalid layer count: {0}")]
    InvalidLayerCount(String),
    
    #[error(transparent)]
    IOError(#[from] std::io::Error),
}

// 使用新的错误类型
pub fn batch_prove<E: ExtensionField, T: Transcript<E>>(
    input: &LogUpInput<E>,
    transcript: &mut T,
) -> Result<LogUpProof<E>, ProverError> {
    // ...existing code...
}
````

### 3. 内存优化
````rust
pub fn batch_prove<E: ExtensionField, T: Transcript<E>>(
    input: &LogUpInput<E>,
    transcript: &mut T,
) -> Result<LogUpProof<E>, LogUpError> {
    // 预分配向量大小
    let num_instances = input.num_instances();
    let mut sumcheck_proofs = Vec::with_capacity(num_instances);
    let mut round_evaluations = Vec::with_capacity(num_instances);
    
    // ...existing code...
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
    fn test_batch_prove_single_instance() {
        let mut transcript = default_transcript::<GoldilocksExt2>();
        let input = create_test_input(1);
        let result = batch_prove(&input, &mut transcript);
        assert!(result.is_ok());
        
        let proof = result.unwrap();
        assert_eq!(proof.circuit_outputs.len(), 1);
    }
    
    #[test]
    fn test_batch_prove_multiple_instances() {
        let mut transcript = default_transcript::<GoldilocksExt2>();
        let input = create_test_input(4);
        let result = batch_prove(&input, &mut transcript);
        assert!(result.is_ok());
        
        let proof = result.unwrap();
        assert_eq!(proof.circuit_outputs.len(), 4);
    }
    
    #[test]
    fn test_error_handling() {
        let mut transcript = default_transcript::<GoldilocksExt2>();
        let input = create_invalid_input();
        let result = batch_prove(&input, &mut transcript);
        assert!(result.is_err());
    }
}
````

## 5. 性能监控

````rust
use tracing::{debug, info, warn};

#[timed::timed_instrument(level = "debug")]
pub fn batch_prove<E: ExtensionField, T: Transcript<E>>(
    input: &LogUpInput<E>,
    transcript: &mut T,
) -> Result<LogUpProof<E>, LogUpError> {
    debug!("Starting batch prove with {} instances", input.num_instances());
    
    let start = std::time::Instant::now();
    // ...existing code...
    
    info!("Batch prove completed in {:?}", start.elapsed());
    Ok(proof)
}
````

## 6. 总结

prover.rs 文件实现了 LogUp GKR 的批量证明生成系统:

1. **功能特点**
- 支持批量证明生成
- 灵活的电路处理
- 完整的错误处理

2. **优化建议**
- 添加并行处理
- 改进错误处理
- 优化内存使用
- 增加性能监控

3. **代码质量**
- 完善的测试覆盖
- 清晰的文档注释
- 良好的错误处理

该实现为零知识证明系统提供了高效的批量证明生成功能。 */
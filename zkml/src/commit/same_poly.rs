//! This module contains logic to prove the opening of several claims related to the _same_ polynomial.
//! e.g. a set of (r_i,y_i) such that f(r_i) = y_i for all i's.
//! a_i = randomness() for i:0 -> |r_i|
//! for r_i, compute Beta_{r_i} = [beta_{r_i}(0),(1),...(2^|r_i|)]
//! then Beta_j = SUM_j a_i * Beta_{r_i}
//!
//! Note the output of the verifier is a claim that needs to be verified outside of this protocol.
//! It could be via an opening directly OR via an accumulation scheme.

use crate::{Claim, VectorTranscript, commit::identity_eval};
use anyhow::{Ok, ensure};
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::{
    mle::{DenseMultilinearExtension, IntoMLE, MultilinearExtension},
    virtual_poly::{VPAuxInfo, VirtualPolynomial},
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};
use transcript::Transcript;

use super::{aggregated_rlc, compute_betas_eval};

pub struct Context<E: ExtensionField> {
    vp_info: VPAuxInfo<E>,
}

impl<E: ExtensionField> Context<E> {
    /// number of variables of the poly in question
    pub fn new(num_vars: usize) -> Self {
        Self {
            vp_info: VPAuxInfo::from_mle_list_dimensions(&[vec![num_vars, num_vars]]),
        }
    }
}
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Proof<E: ExtensionField> {
    sumcheck: IOPProof<E>,
    // [0] about the betas, [1] about the poly
    evals: Vec<E>,
}

impl<E: ExtensionField> Proof<E> {
    pub fn extract_claim(&self) -> Claim<E> {
        Claim {
            point: self.sumcheck.point.clone(),
            eval: self.evals[1],
        }
    }
}

pub struct Prover<E: ExtensionField> {
    claims: Vec<Claim<E>>,
    poly: DenseMultilinearExtension<E>,
}

impl<E> Prover<E>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// The polynomial over which the claims are to be accumulated and proven
    /// Note the prover also _commits_ to this polynomial.
    pub fn new(poly: DenseMultilinearExtension<E>) -> Self {
        Self {
            claims: Default::default(),
            poly,
        }
    }
    pub fn add_claim(&mut self, claim: Claim<E>) -> anyhow::Result<()> {
        ensure!(
            claim.point.len() == self.poly.num_vars(),
            format!(
                "Invalid claim length: input.len() = {} vs poly.num_vars = {} ",
                claim.point.len(),
                self.poly.num_vars()
            )
        );
        self.claims.push(claim);
        Ok(())
    }
    pub fn prove<T: Transcript<E>>(self, ctx: &Context<E>, t: &mut T) -> anyhow::Result<Proof<E>> {
        let challenges = t.read_challenges(self.claims.len());

        let beta_evals = challenges
            .into_par_iter()
            .zip(self.claims.into_par_iter())
            .map(|(a_i, c_i)| {
                // c_i.input = r_i
                compute_betas_eval(&c_i.point)
                    .into_iter()
                    .map(|b_i| a_i * b_i)
                    .collect_vec()
            })
            .collect::<Vec<_>>();
        let final_beta = (0..1 << ctx.vp_info.max_num_variables)
            .into_par_iter()
            .map(|i| {
                beta_evals
                    .iter()
                    .map(|beta_for_r_i| beta_for_r_i[i])
                    .fold(E::ZERO, |acc, b| acc + b)
            })
            .collect::<Vec<_>>();

        // then run the sumcheck on it
        let mut vp = VirtualPolynomial::new(self.poly.num_vars());
        vp.add_mle_list(vec![final_beta.into_mle().into(), self.poly.into()], E::ONE);
        #[allow(deprecated)]
        let (sumcheck_proof, state) = IOPProverState::<E>::prove_parallel(vp, t);

        Ok(Proof {
            sumcheck: sumcheck_proof,
            evals: state.get_mle_final_evaluations(),
        })
    }
}

pub struct Verifier<'a, E: ExtensionField>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    claims: Vec<Claim<E>>,
    ctx: &'a Context<E>,
}

impl<'a, E: ExtensionField> Verifier<'a, E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    pub fn new(ctx: &'a Context<E>) -> Self {
        Self {
            claims: Default::default(),
            ctx,
        }
    }

    pub fn add_claim(&mut self, claim: Claim<E>) -> anyhow::Result<()> {
        ensure!(
            claim.point.len() == self.ctx.vp_info.max_num_variables,
            "invalid input len wrt to poly in ctx, claim point length: {}, expected point length: {}",
            claim.point.len(),
            self.ctx.vp_info.max_num_variables
        );
        self.claims.push(claim);
        Ok(())
    }

    pub fn verify<T: Transcript<E>>(self, proof: &Proof<E>, t: &mut T) -> anyhow::Result<Claim<E>> {
        let fs_challenges = t.read_challenges(self.claims.len());
        let (rs, ys): (Vec<_>, Vec<_>) = self.claims.into_iter().map(|c| (c.point, c.eval)).unzip();
        let y_res = aggregated_rlc(&ys, &fs_challenges);
        // check sumcheck proof
        let subclaim = IOPVerifierState::<E>::verify(y_res, &proof.sumcheck, &self.ctx.vp_info, t);
        // check sumcheck output: first check for the betas we can compute
        // for(int i = 0; i < a.size(); i++){y += a[i]*identity_eval(claims[i].first,P.randomness[0]);}
        let computed_y = fs_challenges
            .into_iter()
            .zip(rs)
            .fold(E::ZERO, |acc, (a_i, r_i)| {
                acc + a_i * identity_eval(&r_i, &proof.sumcheck.point)
            });
        let given_y = proof.evals[0];
        ensure!(computed_y == given_y, "beta evaluation do not match");
        // here instead of checking this claim via PCS, we actually put it in the output of the verify function.
        // That claims will be accumulated and verified elsewhere in the protocol.
        // Note the claim is only about the actual poly, not the betas since it has been verified just ^
        let claim = proof.extract_claim();

        // then check that both betas and poly evaluation lead to the outcome of the sumcheck, e.g. the sum
        let expected = proof.evals[0] * proof.evals[1];
        let computed = subclaim.expected_evaluation;
        ensure!(expected == computed, "final evals of sumcheck is not valid");
        Ok(claim)
    }
}

#[cfg(test)]
mod test {
    use goldilocks::GoldilocksExt2;
    use mpcs::PolynomialCommitmentScheme;
    use multilinear_extensions::mle::{IntoMLE, MultilinearExtension};

    use crate::{Claim, commit::Pcs, default_transcript, testing::random_field_vector};
    use itertools::Itertools;

    use super::{Context, Prover, Verifier};

    type F = GoldilocksExt2;

    #[test]
    fn test_pcs() {
        let num_vars = 10;
        let len = 1 << num_vars;
        let _param = Pcs::<F>::setup(len).expect("unable to setup");
    }

    #[test]
    fn test_same_poly_proof() -> anyhow::Result<()> {
        // number of vars
        let num_vars = 10 as usize;
        let poly_len = 1 << num_vars;
        let poly = random_field_vector::<F>(poly_len);
        let poly_mle = poly.clone().into_mle();
        // number of clains
        let m = 14;
        let claims = (0..m)
            .map(|_| {
                let r_i = random_field_vector::<F>(num_vars);
                let y_i = poly_mle.evaluate(&r_i);
                (r_i, y_i)
            })
            .collect_vec();
        // COMMON PART
        assert_eq!(poly.len(), 1 << num_vars);
        let ctx = Context::new(num_vars);
        // PROVER PART
        let mut t = default_transcript();
        let mut prover = Prover::new(poly_mle.clone());
        for (r_i, y_i) in claims.clone().into_iter() {
            prover.add_claim(Claim::new(r_i, y_i))?;
        }
        let proof = prover.prove(&ctx, &mut t)?;
        // VERIFIER PART
        let mut t = default_transcript();
        let mut verifier = Verifier::new(&ctx);
        for (r_i, y_i) in claims.into_iter() {
            verifier.add_claim(Claim::new(r_i, y_i))?;
        }
        let claim = verifier.verify(&proof, &mut t)?;
        let expected = poly_mle.evaluate(&claim.point);
        assert_eq!(claim.eval, expected);
        Ok(())
    }
}

/*
让我为您分析 same_poly.rs 文件：

# same_poly.rs 文件分析

### **1. 文件功能**
该文件实现了针对同一多项式的多个声明的证明系统，主要功能包括：
- 对同一多项式的多个评估点进行批量证明
- 使用 Sumcheck 协议进行验证
- Beta 向量的计算和验证

### **2. 关键部分**

#### **核心数据结构**

##### **Context**
```rust
pub struct Context<E: ExtensionField> {
    vp_info: VPAuxInfo<E>,
}
```
- 功能：存储验证所需的上下文信息
- 主要包含虚拟多项式的辅助信息

##### **Proof**
```rust
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Proof<E: ExtensionField> {
    sumcheck: IOPProof<E>,
    evals: Vec<E>,  // [0]为beta评估，[1]为多项式评估
}
```

#### **核心组件**

##### **Prover（证明者）**
```rust
pub struct Prover<E: ExtensionField> {
    claims: Vec<Claim<E>>,
    poly: DenseMultilinearExtension<E>,
}
```
主要方法：
- `new()`: 创建新的证明者实例
- `add_claim()`: 添加待证明的声明
- `prove()`: 生成证明

##### **Verifier（验证者）**
```rust
pub struct Verifier<'a, E: ExtensionField> {
    claims: Vec<Claim<E>>,
    ctx: &'a Context<E>,
}
```
主要方法：
- `new()`: 创建新的验证者实例
- `add_claim()`: 添加待验证的声明
- `verify()`: 验证证明

### **3. 关键算法流程**

#### **证明生成流程**
1. 读取随机挑战值
2. 计算 Beta 评估值
3. 合并 Beta 向量
4. 运行 Sumcheck 协议
5. 生成最终证明

```rust
pub fn prove<T: Transcript<E>>(self, ctx: &Context<E>, t: &mut T) -> anyhow::Result<Proof<E>> {
    let challenges = t.read_challenges(self.claims.len());
    // ...计算beta评估值
    // ...运行sumcheck
    Ok(Proof {
        sumcheck: sumcheck_proof,
        evals: state.get_mle_final_evaluations(),
    })
}
```

#### **验证流程**
1. 验证 Sumcheck 证明
2. 验证 Beta 评估值
3. 验证最终评估结果

### **4. 优化特点**

1. **并行计算**
```rust
let beta_evals = challenges
    .into_par_iter()
    .zip(self.claims.into_par_iter())
    .map(|(a_i, c_i)| {
        // ...并行计算beta评估值
    })
    .collect::<Vec<_>>();
```

2. **内存优化**
- 使用迭代器避免不必要的内存分配
- 合理使用所有权系统

### **5. 测试用例**

```rust
#[test]
fn test_same_poly_proof() -> anyhow::Result<()> {
    // 测试完整的证明-验证流程
}

#[test]
fn test_pcs() {
    // 测试多项式承诺方案的基本功能
}
```

### **6. 总结**
same_poly.rs 实现了一个高效的批量证明系统，允许对同一多项式的多个评估点进行批量证明和验证。该实现结合了 Sumcheck 协议和 Beta 向量技术，通过并行计算优化性能，为零知识证明系统提供了重要的基础设施。
 */
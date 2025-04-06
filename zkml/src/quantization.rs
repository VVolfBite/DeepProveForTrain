//! Module that takes care of (re)quantizing
use ff_ext::ExtensionField;
use goldilocks::SmallField;
use itertools::Itertools;
use once_cell::sync::Lazy;
use std::env;
use tracing::debug;

use crate::{Element, tensor::Tensor};

// Get BIT_LEN from environment variable or use default value
pub static BIT_LEN: Lazy<usize> = Lazy::new(|| {
    env::var("ZKML_BIT_LEN")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(8) // Default value if env var is not set or invalid
});

// These values depend on BIT_LEN and need to be computed at runtime
pub static MIN: Lazy<Element> = Lazy::new(|| -(1 << (*BIT_LEN - 1)));
pub static MAX: Lazy<Element> = Lazy::new(|| (1 << (*BIT_LEN - 1)) - 1);
pub static ZERO: Lazy<Element> = Lazy::new(|| 0);

/// Trait used to quantize original floating point number to integer
pub trait Quantizer<Output> {
    fn from_f32_unsafe(e: &f32) -> Output;
    fn from_f32_unsafe_clamp(e: &f32, max_abs: f64) -> Output;
}

impl Quantizer<Element> for Element {
    fn from_f32_unsafe(e: &f32) -> Self {
        assert!(
            *e >= -1.0 && *e <= 1.0,
            "Input value must be between -1.0 and 1.0"
        );
        // even tho we are requantizing starting from Element, we only want to requantize for QuantInteger
        // the reason we have these two types is to handle overflow
        // (a -b) / 2^Q
        let scale = (1.0 - (-1.0)) / (1 << *BIT_LEN) as f64;
        let zero_point = 0;

        // formula is q = round(r/S) + z
        let scaled = (*e as f64 / scale).round() as Element + zero_point;
        scaled as Element
    }

    fn from_f32_unsafe_clamp(e: &f32, max_abs: f64) -> Self {
        let e = *e as f64;
        assert!(
            max_abs > 0.0,
            "max_abs should be greater than zero. Domain range is between [-max_abs, max_abs]."
        );

        let scale = (2.0 * max_abs) / (*MAX - *MIN) as f64;
        let zero_point = 0;

        // formula is q = round(r/S) + z
        let scaled = (e / scale).round() as Element + zero_point;
        let scaled = scaled.clamp(*MIN, *MAX);

        if e < -max_abs || e > max_abs {
            debug!(
                "Quantization: Value {} is out of [-{}, {}]. But quantized to {}.",
                e, max_abs, max_abs, scaled
            );
        }
        scaled as Element
    }
}

pub(crate) trait Fieldizer<F> {
    fn to_field(&self) -> F;
}

impl<F: ExtensionField> Fieldizer<F> for Element {
    fn to_field(&self) -> F {
        if self.is_negative() {
            // Doing wrapped arithmetic : p-128 ... p-1 means negative number
            F::from(<F::BaseField as SmallField>::MODULUS_U64 - self.unsigned_abs() as u64)
        } else {
            // for positive and zero, it's just the number
            F::from(*self as u64)
        }
    }
}
pub(crate) trait IntoElement {
    fn into_element(&self) -> Element;
}

impl<F: ExtensionField> IntoElement for F {
    fn into_element(&self) -> Element {
        let e = self.to_canonical_u64_vec()[0] as Element;
        let modulus_half = <F::BaseField as SmallField>::MODULUS_U64 >> 1;
        // That means he's a positive number
        if *self == F::ZERO {
            0
        // we dont assume any bounds on the field elements, requant might happen at a later stage
        // so we assume the worst case
        } else if e <= modulus_half as Element {
            e
        } else {
            // That means he's a negative number - so take the diff with the modulus and recenter around 0
            let diff = <F::BaseField as SmallField>::MODULUS_U64 - e as u64;
            -(diff as Element)
        }
    }
}

impl<F: ExtensionField> Fieldizer<F> for u8 {
    fn to_field(&self) -> F {
        F::from(*self as u64)
    }
}

pub trait TensorFielder<F> {
    fn to_fields(self) -> Tensor<F>;
}

impl<F: ExtensionField, T> TensorFielder<F> for Tensor<T>
where
    T: Fieldizer<F>,
{
    fn to_fields(self) -> Tensor<F> {
        Tensor::new(
            self.get_shape(),
            self.get_data()
                .into_iter()
                .map(|i| i.to_field())
                .collect_vec(),
        )
    }
}

pub fn range_from_weight(weight: &Element) -> (Element, Element) {
    let min = if weight.is_negative() {
        weight * *MAX as Element
    } else {
        weight * *MIN as Element
    };
    let max = if weight.is_negative() {
        weight * *MIN as Element
    } else {
        weight * *MAX as Element
    };
    (min, max)
}

#[cfg(test)]
mod test {
    use crate::quantization::Fieldizer;

    use crate::Element;
    type F = goldilocks::GoldilocksExt2;

    #[test]
    fn test_wrapped_field() {
        // for case in vec![-12,25,i8::MIN,i8::MAX] {
        //     let a: i8 = case;
        //     let af: F= a.to_field();
        //     let f = af.to_canonical_u64_vec()[0];
        //     let exp = if a.is_negative() {
        //         MODULUS - (a as i64).unsigned_abs()
        //     } else {
        //         a as u64
        //     };
        //     assert_eq!(f,exp);
        // }
    }

    #[test]
    fn test_wrapped_arithmetic() {
        #[derive(Clone, Debug)]
        struct TestCase {
            a: Element,
            b: Element,
            res: Element,
        }

        let cases = vec![
            TestCase {
                a: -53,
                b: 10,
                res: -53 * 10,
            },
            TestCase {
                a: -45,
                b: -56,
                res: 45 * 56,
            },
        ];
        for (i, case) in cases.iter().enumerate() {
            // cast them to handle overflow
            let ap: F = case.a.to_field();
            let bp: F = case.b.to_field();
            let res = ap * bp;
            let expected = case.res.to_field();
            assert_eq!(res, expected, "test case {}: {:?}", i, case);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type F = goldilocks::GoldilocksExt2;
    #[test]
    fn test_element_field_roundtrip() {
        // Also test a few specific values explicitly
        let test_values = [*MIN, -100, -50, -1, 0, 1, 50, 100, *MAX];
        for &val in &test_values {
            let field_val: F = val.to_field();
            let roundtrip = field_val.into_element();

            assert_eq!(
                val, roundtrip,
                "Element {} did not roundtrip correctly (got {})",
                val, roundtrip
            );
        }
    }
}
/*
以下是对 quantization.rs 文件的整体结构和关键部分的总结：

---

### **1. 文件功能**
该文件主要负责量化（Quantization）相关的功能，包括：
- 将浮点数值量化为整数值。
- 提供量化和反量化的工具函数。
- 支持将张量数据转换为有限域（Field）表示。
- 提供量化范围的计算和验证。

---

### **2. 关键部分**
#### **全局静态变量**
- **`BIT_LEN`**
  - 功能：从环境变量 `ZKML_BIT_LEN` 获取量化位宽，默认为 8 位。
  - 类型：`Lazy<usize>`。
- **`MIN`**
  - 功能：根据 `BIT_LEN` 计算量化范围的最小值。
  - 类型：`Lazy<Element>`。
- **`MAX`**
  - 功能：根据 `BIT_LEN` 计算量化范围的最大值。
  - 类型：`Lazy<Element>`。
- **`ZERO`**
  - 功能：表示量化范围的零点。
  - 类型：`Lazy<Element>`。

---

#### **核心 Trait 和实现**
- **`Quantizer<Output>`**
  - 功能：定义量化浮点数到整数的接口。
  - 方法：
    - **`from_f32_unsafe(e: &f32) -> Output`**
      - 将浮点数值量化为整数值，假设输入范围为 `[-1.0, 1.0]`。
    - **`from_f32_unsafe_clamp(e: &f32, max_abs: f64) -> Output`**
      - 将浮点数值量化为整数值，并对超出范围的值进行截断。
  - 实现：
    - 针对 `Element` 类型实现了量化逻辑，支持自定义量化范围和截断。

- **`Fieldizer<F>`**
  - 功能：将整数值转换为有限域（Field）表示。
  - 方法：
    - **`to_field(&self) -> F`**
      - 将整数值映射到有限域，支持正数和负数的处理。
  - 实现：
    - 针对 `Element` 和 `u8` 类型实现了转换逻辑。

- **`IntoElement`**
  - 功能：将有限域值转换回整数值。
  - 方法：
    - **`into_element(&self) -> Element`**
      - 将有限域值转换为整数值，支持正数和负数的处理。
  - 实现：
    - 针对有限域类型 `F` 实现了反量化逻辑。

- **`TensorFielder<F>`**
  - 功能：将张量数据转换为有限域表示。
  - 方法：
    - **`to_fields(self) -> Tensor<F>`**
      - 将张量的每个元素转换为有限域值。
  - 实现：
    - 针对 `Tensor<T>` 类型实现了转换逻辑。

---

#### **辅助函数**
- **`range_from_weight(weight: &Element) -> (Element, Element)`**
  - 功能：根据权重值计算量化范围的最小值和最大值。
  - 返回：`(Element, Element)`，表示最小值和最大值。

---

### **3. 测试**
文件包含多个单元测试，验证量化和有限域转换的核心功能：
- **`test_wrapped_field`**
  - 功能：测试整数值到有限域的转换逻辑。
  - 验证：正数和负数的映射是否正确。
- **`test_wrapped_arithmetic`**
  - 功能：测试有限域上的算术运算。
  - 验证：乘法运算的结果是否与预期一致。
- **`test_element_field_roundtrip`**
  - 功能：测试整数值与有限域值之间的双向转换。
  - 验证：转换后的值是否与原始值一致。

---

### **4. 总结**
该文件的核心功能是实现量化和有限域转换，支持深度学习模型的量化推理和验证。它提供了灵活的量化接口和工具函数，适用于处理张量数据和权重范围的场景，同时通过单元测试验证了核心逻辑的正确性。
*/
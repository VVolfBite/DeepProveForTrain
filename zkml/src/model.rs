use crate::{
    Element,
    layers::{Layer, LayerOutput, Train},
    padding::PaddingMode,
    quantization::{ModelMetadata, TensorFielder, AbsoluteMax, ScalingStrategy},
    tensor::{ConvData, Number, Tensor},
};
use anyhow::Result;
use ff_ext::ExtensionField;
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tracing::info;

// The index of the step, starting from the input layer. (proving is done in the opposite flow)
pub type StepIdx = usize;

/// 损失函数类型
#[derive(Debug, Clone, Copy)]
pub enum LossFunction<T> {
    /// 平方误差损失
    SquaredError,
    /// 交叉熵损失（仅适用于概率输出，如 Softmax 后）
    CrossEntropy,
    /// 自定义损失函数
    Custom(fn(&Tensor<T>, &Tensor<T>) -> T),
}

/// 每一层反向传播后的梯度信息
pub struct LayerGradient<T> {
    /// 当前层输入的梯度（用于往前传）
    pub input_grad: Tensor<T>,

    /// 当前层参数的梯度（如权重/偏置），无则为 None
    pub param_grad: Option<LayerParamGradient<T>>,
}

/// 用于描述各种层参数的梯度（不同层可能结构不同）
pub enum LayerParamGradient<T> {
    Dense {
        weights_grad: Tensor<T>,
        bias_grad: Tensor<T>,
    },
    Convolution {
        kernel_grad: Tensor<T>,
        bias_grad: Tensor<T>,
    },
    SchoolBookConvolution {
        kernel_grad: Tensor<T>,
        bias_grad: Tensor<T>,
    },
    // 其他层可能不需要梯度，比如 Activation, Requant, Reshape
}

/// NOTE: this doesn't handle dynamism in the model with loops for example for LLMs where it
/// produces each token one by one.
#[derive(Clone, Debug)]


pub struct Model<T> {
    pub unpadded_input: Vec<usize>,
    pub(crate) padded_input: Vec<usize>,
    pub(crate) layers: Vec<Layer<T>>,
}

impl<T: Number> Model<T> {
    pub fn new_from(
        layers: Vec<Layer<T>>,
        input_not_padded_shape: Vec<usize>,
        input_padded_shape: Vec<usize>,
    ) -> Self {
        Self {
            unpadded_input: input_not_padded_shape,
            padded_input: input_padded_shape,
            layers,
        }
    }
    pub fn new(unpadded_input_shape: &[usize]) -> Self {
        info!(
            "Creating model with {} BIT_LEN quantization",
            *crate::quantization::BIT_LEN
        );
        let mut model = Self {
            unpadded_input: Vec::new(),
            padded_input: Vec::new(),
            layers: Default::default(),
        };
        model.set_input_shape(unpadded_input_shape.to_vec());
        model
    }

    /// Adds a layer to the model. The model may add additional layers by itself, e.g. requantization
    /// layers.
    pub fn add_layer(&mut self, l: Layer<T>) {
        self.layers.push(l);
    }

    pub fn set_input_shape(&mut self, not_padded: Vec<usize>) {
        self.padded_input = not_padded
            .iter()
            .map(|dim| dim.next_power_of_two())
            .collect_vec();
        self.unpadded_input = not_padded;
    }
    pub fn load_input_flat(&self, input: Vec<T>) -> Tensor<T> {
        let input_tensor = Tensor::<T>::new(self.unpadded_input.clone(), input);
        self.prepare_input(input_tensor)
    }

    pub fn prepare_input(&self, input: Tensor<T>) -> Tensor<T> {
        match self.layers[0] {
            Layer::Dense(ref dense) => input.pad_1d(dense.ncols()),
            Layer::Convolution(_) | Layer::SchoolBookConvolution(_) => {
                assert!(
                    self.padded_input.len() > 0,
                    "Set the input shape using `set_input_shape`"
                );
                let mut input = input;
                input.pad_to_shape(self.padded_input.clone());
                input
            }
            _ => {
                panic!("unable to deal with non-vector input yet");
            }
        }
    }

    pub fn layers(&self) -> impl DoubleEndedIterator<Item = (StepIdx, &Layer<T>)> {
        self.layers.iter().enumerate()
    }
    pub fn provable_layers(&self) -> impl DoubleEndedIterator<Item = (StepIdx, &Layer<T>)> {
        self.layers
            .iter()
            .enumerate()
            .filter(|(_, l)| (*l).is_provable())
    }

    pub fn unpadded_input_shape(&self) -> Vec<usize> {
        self.unpadded_input.clone()
    }
    pub fn input_shape(&self) -> Vec<usize> {
        if let Layer::Dense(mat) = &self.layers[0] {
            vec![mat.matrix.ncols_2d()]
        } else if matches!(
            &self.layers[0],
            Layer::Convolution(_) | Layer::SchoolBookConvolution(_)
        ) {
            assert!(
                self.padded_input.len() > 0,
                "Set the input shape using `set_input_shape`"
            );
            self.padded_input.clone()
        } else {
            panic!("layer is not starting with a dense or conv layer?")
        }
    }

    pub fn first_output_shape(&self) -> Vec<usize> {
        if let Layer::Dense(mat) = &self.layers[0] {
            vec![mat.matrix.nrows_2d()]
        } else if let Layer::Convolution(filter) = &self.layers[0] {
            vec![filter.nrows_2d()]
        } else {
            panic!("layer is not starting with a dense layer?")
        }
    }
    /// Prints to stdout
    pub fn describe(&self) {
        info!("Model description:");
        info!("Unpadded input shape: {:?}", self.unpadded_input);
        info!("Padded input shape: {:?}", self.padded_input);
        for (idx, layer) in self.layers() {
            info!("\t- {}: {}", idx, layer.describe());
        }
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// 计算模型输出与目标值之间的损失
    pub fn compute_loss(&self, prediction: &Tensor<T>, target: &Tensor<T>, loss_fn: LossFunction<T>) -> T {
        match loss_fn {
            LossFunction::SquaredError => {
                assert_eq!(prediction.get_shape(), target.get_shape(), "预测值与目标值形状不一致");
                let mut sum = T::default();
                for (p, t) in prediction.get_data().iter().zip(target.get_data().iter()) {
                    let diff = *p - *t;
                    sum = sum + diff * diff;
                }
                sum
            }
            LossFunction::CrossEntropy => {
                unimplemented!("CrossEntropy 损失函数尚未实现")
            }
            LossFunction::Custom(f) => f(prediction, target),
        }
    }

    // 计算损失函数的梯度
    fn compute_loss_gradient(&self, prediction: &Tensor<T>, target: &Tensor<T>, loss_fn: LossFunction<T>) -> Tensor<T> {
        match loss_fn {
            LossFunction::SquaredError => {
                let mut grad_data = Vec::with_capacity(prediction.get_data().len());
                for (p, t) in prediction.get_data().iter().zip(target.get_data().iter()) {
                    let diff = *p - *t;
                    grad_data.push(diff + diff);
                }
                Tensor::new(prediction.get_shape(), grad_data)
            }
            LossFunction::CrossEntropy => {
                unimplemented!("CrossEntropy 反向传播尚未实现")
            }
            LossFunction::Custom(_) => {
                unimplemented!("自定义损失函数的反向传播尚未实现")
            }
        }
    }

    // 前向传播并保存中间结果
    pub fn forward_with_intermediates(&self, input: &Tensor<T>) -> (Tensor<T>, Vec<Tensor<T>>) {
        let mut intermediate_outputs = Vec::new();
        let mut current_input = self.prepare_input(input.clone());
        intermediate_outputs.push(current_input.clone());
        
        for layer in &self.layers {
            current_input = layer.forward(&current_input);
            intermediate_outputs.push(current_input.clone());
        }
        
        (current_input, intermediate_outputs)
    }

    // 计算损失和梯度
    pub fn compute_loss_and_gradient(&self, output: &Tensor<T>, target: &Tensor<T>, loss_fn: LossFunction<T>) -> (T, Tensor<T>) {
        let loss = self.compute_loss(output, target, loss_fn);
        let grad = self.compute_loss_gradient(output, target, loss_fn);
        (loss, grad)
    }

    // 反向传播
    pub fn backward(&mut self, intermediate_outputs: &[Tensor<T>], grad: &Tensor<T>) -> Tensor<T> {
        let mut current_grad = grad.clone();
        
        // 反向传播
        for (layer, input) in self.layers.iter_mut().rev().zip(intermediate_outputs.iter().rev().skip(1)) {
            current_grad = layer.backward(input, &current_grad);
        }
        
        current_grad
    }

    /// 训练模型
    pub fn train(
        &mut self,
        data: &[(Tensor<T>, Tensor<T>)], // 输入和标签对
        loss_fn: LossFunction<T>,
        optimizer: &mut dyn Optimizer<T>,
        epochs: usize,
    ) {
        for epoch in 0..epochs {
            let mut total_loss = T::default();
            
            // 遍历每个训练样本
            for (input, target) in data.iter() {
                // 前向传播并保存中间结果
                let (output, intermediate_outputs) = self.forward_with_intermediates(input);
                
                // 计算损失和梯度
                let (loss, grad) = self.compute_loss_and_gradient(&output, target, loss_fn.clone());
                
                // 反向传播
                self.backward(&intermediate_outputs, &grad);
                
                // 更新参数
                optimizer.step(&mut self.layers);
                
                total_loss = total_loss + loss;
            }

            println!("Epoch {}, loss: {:?}", epoch, total_loss);
        }
    }
}

impl Model<Element> {
    pub fn run<'a, E: ExtensionField>(
        &'a self,
        input: Tensor<Element>,
    ) -> Result<InferenceTrace<'a, Element, E>> {
        #[cfg(test)]
        let unpadded_input_shape = {
            if self.unpadded_input.len() == 0 {
                input.get_shape()
            } else {
                self.unpadded_input.clone()
            }
        };
        #[cfg(not(test))]
        let unpadded_input_shape = self.unpadded_input.clone();
        let mut trace = InferenceTrace::<Element, E>::new(input, unpadded_input_shape.clone());
        let mut unpadded_input_shape = unpadded_input_shape;
        for (id, layer) in self.layers() {
            let input = trace.last_input();
            let output = layer.op(input, &unpadded_input_shape)?;
            unpadded_input_shape =
                layer.output_shape(&unpadded_input_shape, PaddingMode::NoPadding);
            match output {
                LayerOutput::NormalOut(output) => {
                    let conv_data = ConvData::default();
                    let step = InferenceStep {
                        layer,
                        output,
                        id,
                        conv_data,
                        unpadded_shape: unpadded_input_shape.clone(),
                    };
                    trace.push_step(step);
                }
                LayerOutput::ConvOut((output, conv_data)) => {
                    let step = InferenceStep {
                        layer,
                        output,
                        id,
                        conv_data,
                        unpadded_shape: unpadded_input_shape.clone(),
                    };
                    trace.push_step(step);
                }
            }
        }
        Ok(trace)
    }
}

/// Keeps track of all input and outputs of each layer, with a reference to the layer.
pub struct InferenceTrace<'a, E, F: ExtensionField> {
    pub steps: Vec<InferenceStep<'a, E, F>>,
    /// The initial input to the model
    input: Tensor<E>,
    unpadded_shape: Vec<usize>,
}

impl<'a, F: ExtensionField> InferenceTrace<'a, Element, F> {
    pub fn provable_steps(&self) -> Self {
        let mut filtered_steps = Vec::new();
        for step in self.steps.iter() {
            if step.layer.is_provable() {
                filtered_steps.push(step.clone());
            } else {
                // we want the output of this step to be the output of the previous step
                let last_idx = filtered_steps.len() - 1;
                filtered_steps[last_idx].output = step.output.clone();
            }
        }
        InferenceTrace {
            steps: filtered_steps,
            input: self.input.clone(),
            unpadded_shape: self.unpadded_shape.clone(),
        }
    }
    pub fn dequantized(&self, md: &ModelMetadata) -> InferenceTrace<'a, f32, F> {
        let input = self.input.dequantize(&md.input);
        let mut last_layer_output_scaling = None;
        let steps = self
            .steps
            .iter()
            .map(|step| {
                if step.layer.needs_requant() {
                    last_layer_output_scaling = Some(md.layer_output_scaling_factor(step.id));
                }
                let output = step.output.dequantize(
                    last_layer_output_scaling
                        .as_ref()
                        .expect("Model must start with a 'need-requant' layer"),
                );
                InferenceStep {
                    id: step.id,
                    layer: step.layer,
                    output,
                    conv_data: step.conv_data.clone(),
                    unpadded_shape: step.unpadded_shape.clone(),
                }
            })
            .collect();
        InferenceTrace {
            steps,
            input,
            unpadded_shape: self.unpadded_shape.clone(),
        }
    }
    pub fn to_field(self) -> InferenceTrace<'a, F, F> {
        let input = self.input.to_fields();
        let field_steps = self
            .steps
            .into_par_iter()
            .map(|step| InferenceStep {
                id: step.id,
                layer: step.layer,
                output: step.output.to_fields(),
                conv_data: step.conv_data.clone(),
                unpadded_shape: step.unpadded_shape.clone(),
            })
            .collect::<Vec<_>>();
        InferenceTrace {
            steps: field_steps,
            input,
            unpadded_shape: self.unpadded_shape.clone(),
        }
    }
}

impl<'a, E, F: ExtensionField> InferenceTrace<'a, E, F> {
    /// The input must be the already padded input tensor via `Model::prepare_input`
    fn new(input: Tensor<E>, unpadded_shape: Vec<usize>) -> Self {
        Self {
            steps: Default::default(),
            input,
            unpadded_shape,
        }
    }

    pub fn last_step(&self) -> &InferenceStep<'a, E, F> {
        self.steps
            .last()
            .expect("can't call last_step on empty inferece")
    }

    /// Useful when building the trace. The next input is either the first input or the last
    /// output.
    fn last_input(&self) -> &Tensor<E> {
        if self.steps.is_empty() {
            &self.input
        } else {
            // safe unwrap since it's not empty
            &self.steps.last().unwrap().output
        }
    }

    /// Returns the final output of the whole trace
    pub fn final_output(&self) -> &Tensor<E> {
        &self
            .steps
            .last()
            .expect("can't call final_output on empty trace")
            .output
    }

    fn push_step(&mut self, step: InferenceStep<'a, E, F>) {
        self.steps.push(step);
    }

    /// Returns an iterator over (input, step) pairs
    pub fn iter(&self) -> InferenceTraceIterator<'_, 'a, E, F> {
        InferenceTraceIterator {
            trace: self,
            current_idx: 0,
            end_idx: self.steps.len(),
        }
    }
}

/// Iterator that yields (input, step) pairs for each inference step
pub struct InferenceTraceIterator<'t, 'a, E, F: ExtensionField> {
    trace: &'t InferenceTrace<'a, E, F>,
    current_idx: usize,
    /// For double-ended iteration
    end_idx: usize,
}

impl<'t, 'a, E, F: ExtensionField> Iterator for InferenceTraceIterator<'t, 'a, E, F> {
    type Item = (&'t Tensor<E>, &'t InferenceStep<'a, E, F>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.end_idx {
            return None;
        }

        let step = &self.trace.steps[self.current_idx];
        let input = if self.current_idx == 0 {
            &self.trace.input
        } else {
            &self.trace.steps[self.current_idx - 1].output
        };

        self.current_idx += 1;
        Some((input, step))
    }
}

impl<'t, 'a, E, F: ExtensionField> DoubleEndedIterator for InferenceTraceIterator<'t, 'a, E, F> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.end_idx {
            return None;
        }

        self.end_idx -= 1;
        let step = &self.trace.steps[self.end_idx];
        let input = if self.end_idx == 0 {
            &self.trace.input
        } else {
            &self.trace.steps[self.end_idx - 1].output
        };

        Some((input, step))
    }
}

#[derive(Clone)]
pub struct InferenceStep<'a, E, F: ExtensionField> {
    pub id: StepIdx,
    /// Reference to the layer that produced this step
    /// Note the layer is of type `Element` since we only run the trace
    /// in the quantized domain.
    pub layer: &'a Layer<Element>,
    /// Output produced by this layer
    pub output: Tensor<E>,
    /// Shape of the output in the unpadded domain. This is useful for proving
    /// and eliminating some side effects of padding during proving.
    pub unpadded_shape: Vec<usize>,
    /// Convolution data - is set to default if not a convolution layer
    /// TODO: move that to an Option
    pub conv_data: ConvData<F>,
}

impl<'a, E, F: ExtensionField> InferenceStep<'a, E, F> {
    pub fn is_provable(&self) -> bool {
        self.layer.is_provable()
    }
}

// Add a specific implementation for f32 models
impl Model<f32> {
    /// Runs the model in float format and returns the output tensor
    pub fn run_float(&self, input: Vec<f32>) -> Tensor<f32> {
        let mut last_output = Tensor::new(self.unpadded_input.clone(), input);
        for layer in self.layers.iter() {
            last_output = layer.run(&last_output);
        }
        last_output
    }
}

/// 优化器 trait
pub trait Optimizer<T: Number> {
    /// 执行一步优化
    fn step(&mut self, layers: &mut [Layer<T>]);
}

/// 随机梯度下降优化器
pub struct SGD<T> {
    /// 学习率
    pub learning_rate: T,
}

impl<T: Number> SGD<T> {
    pub fn new(learning_rate: T) -> Self {
        Self { learning_rate }
    }
}

impl<T: Number> Optimizer<T> for SGD<T> {
    fn step(&mut self, layers: &mut [Layer<T>]) {
        for layer in layers.iter_mut() {
            layer.update(self.learning_rate);
        }
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        ScalingFactor,
        layers::{
            Layer,
            activation::{Activation, Relu},
            convolution::Convolution,
            dense::Dense,
            pooling::{MAXPOOL2D_KERNEL_SIZE, Maxpool2D, Pooling},
            requant::Requant,
        },
        quantization,
        testing::{random_bool_vector, random_vector},
    };
    use ark_std::rand::{Rng, RngCore, thread_rng};
    use ff_ext::ExtensionField;
    use goldilocks::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::{
        mle::{IntoMLE, MultilinearExtension},
        virtual_poly::VirtualPolynomial,
    };
    use sumcheck::structs::{IOPProverState, IOPVerifierState};
    use tract_onnx::tract_core::ops::matmul::quant;

    use crate::{Element, default_transcript, quantization::TensorFielder, tensor::Tensor};

    use super::Model;

    type F = GoldilocksExt2;
    const SELECTOR_DENSE: usize = 0;
    const SELECTOR_RELU: usize = 1;
    const SELECTOR_POOLING: usize = 2;
    const MOD_SELECTOR: usize = 2;

    impl Model<Element> {
        pub fn random(num_dense_layers: usize) -> (Self, Tensor<Element>) {
            let mut rng = thread_rng();
            Model::random_with_rng(num_dense_layers, &mut rng)
        }
        /// Returns a random model with specified number of dense layers and a matching input.
        /// Note that currently everything is considered padded, e.g. unpadded_shape = padded_shape
        pub fn random_with_rng<R: RngCore>(
            num_dense_layers: usize,
            rng: &mut R,
        ) -> (Self, Tensor<Element>) {
            let mut last_row: usize = rng.gen_range(3..15);
            let mut model = Model::new(&vec![last_row.next_power_of_two()]);
            for selector in 0..num_dense_layers {
                if selector % MOD_SELECTOR == SELECTOR_DENSE {
                    // if true {
                    // last row becomes new column
                    let (nrows, ncols): (usize, usize) = (rng.gen_range(3..15), last_row);
                    last_row = nrows;
                    let dense =
                        Dense::random(vec![nrows.next_power_of_two(), ncols.next_power_of_two()]);
                    // Figure out the requant information such that output is still within range
                    let (min_output_range, max_output_range) =
                        dense.output_range(*quantization::MIN, *quantization::MAX);
                    let output_scaling_factor = ScalingFactor::from_scale(
                        ((max_output_range - min_output_range) as f64
                            / (*quantization::MAX - *quantization::MIN) as f64)
                            as f32,
                        None,
                    );
                    let input_scaling_factor = ScalingFactor::from_scale(1.0, None);
                    let max_model = dense.matrix.max_value().max(dense.bias.max_value()) as f32;
                    let model_scaling_factor = ScalingFactor::from_absolute_max(max_model, None);
                    let shift =
                        input_scaling_factor.shift(&model_scaling_factor, &output_scaling_factor);
                    let requant = Requant::new(min_output_range as usize, shift);
                    model.add_layer(Layer::Dense(dense));
                    model.add_layer(Layer::Requant(requant));
                } else if selector % MOD_SELECTOR == SELECTOR_RELU {
                    model.add_layer(Layer::Activation(Activation::Relu(Relu::new())));
                    // no need to change the `last_row` since RELU layer keeps the same shape
                    // of outputs
                } else if selector % MOD_SELECTOR == SELECTOR_POOLING {
                    // Currently unreachable until Model is updated to work with higher dimensional tensors
                    // TODO: Implement higher dimensional tensor functionality.
                    model.add_layer(Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default())));
                    last_row -= MAXPOOL2D_KERNEL_SIZE - 1;
                } else {
                    panic!("random selection shouldn't be in that case");
                }
            }
            let Some(model_shape) = model.layers.first().unwrap().model_shape() else {
                panic!("Model must start with a dense layer");
            };
            // ncols since matrix2vector is summing over the columns
            let input = Tensor::random(&vec![model_shape[1]]);
            (model, input)
        }

        /// Returns a model that only contains pooling and relu layers.
        /// The output [`Model`] will contain `num_layers` [`Maxpool2D`] layers and a [`Dense`] layer as well.
        pub fn random_pooling(num_layers: usize) -> (Model<Element>, Tensor<Element>) {
            let mut rng = thread_rng();
            // Since Maxpool reduces the size of the output based on the kernel size and the stride we need to ensure that
            // Our starting input size is large enough for the number of layers.

            // If maxpool input matrix has dimensions w x h then output has width and height
            // out_w = (w - kernel_size) / stride + 1
            // out_h = (h - kenrel_size) / stride + 1
            // Hence to make sure we have a large enough tensor for the last step
            // we need to have that w_first > 2^{num_layers + 1} + 2^{num_layers}
            // and likewise for h_first.

            let minimum_initial_size = (1 << num_layers) * (3usize);

            let mut input_shape = (0..3)
                .map(|i| {
                    if i < 1 {
                        rng.gen_range(1..5usize).next_power_of_two()
                    } else {
                        (minimum_initial_size + rng.gen_range(1..4usize)).next_power_of_two()
                    }
                })
                .collect::<Vec<usize>>();

            let mut model = Model::new(&input_shape);

            let input = Tensor::<Element>::random(&input_shape);

            let info = Maxpool2D::default();
            for _ in 0..num_layers {
                input_shape
                    .iter_mut()
                    .skip(1)
                    .for_each(|dim| *dim = (*dim - info.kernel_size) / info.stride + 1);
                model.add_layer(Layer::Pooling(Pooling::Maxpool2D(info)));
            }

            let (nrows, ncols): (usize, usize) =
                (rng.gen_range(3..15), input_shape.iter().product::<usize>());

            model.add_layer(Layer::Dense(Dense::random(vec![
                nrows.next_power_of_two(),
                ncols.next_power_of_two(),
            ])));

            (model, input)
        }
    }

    #[test]
    fn test_model_long() {
        let (model, input) = Model::random(3);
        model.run::<F>(input).unwrap();
    }

    pub fn check_tensor_consistency_field<E: ExtensionField>(
        real_tensor: Tensor<E>,
        padded_tensor: Tensor<E>,
    ) {
        let n_x = padded_tensor.shape[1];
        for i in 0..real_tensor.shape[0] {
            for j in 0..real_tensor.shape[1] {
                for k in 0..real_tensor.shape[1] {
                    // if(real_tensor.data[i*real_tensor.shape[1]*real_tensor.shape[1]+j*real_tensor.shape[1]+k] > 0){
                    assert!(
                        real_tensor.data[i * real_tensor.shape[1] * real_tensor.shape[1]
                            + j * real_tensor.shape[1]
                            + k]
                            == padded_tensor.data[i * n_x * n_x + j * n_x + k],
                        "Error in tensor consistency"
                    );
                    //}else{
                    //   assert!(-E::from(-real_tensor.data[i*real_tensor.shape[1]*real_tensor.shape[1]+j*real_tensor.shape[1]+k] as u64) == E::from(padded_tensor.data[i*n_x*n_x + j*n_x + k] as u64) ,"Error in tensor consistency");
                    //}
                }

                // assert!(real_tensor.data[i*real_tensor.shape[1]*real_tensor.shape[1]+j ] == padded_tensor.data[i*n_x*n_x + j],"Error in tensor consistency");
            }
        }
    }

    fn random_vector_quant(n: usize) -> Vec<Element> {
        // vec![thread_rng().gen_range(-128..128); n]
        random_vector(n)
    }

    #[test]
    fn test_cnn() {
        let mut in_dimensions: Vec<Vec<usize>> =
            vec![vec![1, 32, 32], vec![16, 29, 29], vec![4, 26, 26]];

        for i in 0..in_dimensions.len() {
            for j in 0..in_dimensions[0].len() {
                in_dimensions[i][j] = (in_dimensions[i][j]).next_power_of_two();
            }
        }
        // println!("in_dimensions: {:?}", in_dimensions);
        let w1 = random_vector_quant(16 * 16);
        let w2 = random_vector_quant(16 * 4 * 16);
        let w3 = random_vector_quant(16 * 8);

        let shape1 = vec![1 << 4, 1 << 0, 1 << 2, 1 << 2]; // [16, 1, 4, 4]
        let shape2 = vec![1 << 2, 1 << 4, 1 << 2, 1 << 2]; // [4, 16, 4, 4]
        let shape3 = vec![1 << 1, 1 << 2, 1 << 2, 1 << 2]; // [2, 4, 4, 4]
        let bias1: Tensor<Element> = Tensor::zeros(vec![shape1[0]]);
        let bias2: Tensor<Element> = Tensor::zeros(vec![shape2[0]]);
        let bias3: Tensor<Element> = Tensor::zeros(vec![shape3[0]]);

        let trad_conv1: Tensor<Element> = Tensor::new(shape1.clone(), w1.clone());
        let trad_conv2: Tensor<i128> = Tensor::new(shape2.clone(), w2.clone());
        let trad_conv3: Tensor<i128> = Tensor::new(shape3.clone(), w3.clone());

        let input_shape = vec![1, 32, 32];
        let input = Tensor::random(&input_shape);

        let mut model = Model::new(&input_shape);
        model.add_layer(Layer::Convolution(
            Convolution::new(trad_conv1.clone(), bias1.clone())
                .into_padded_and_ffted(&in_dimensions[0]),
        ));
        model.add_layer(Layer::Convolution(
            Convolution::new(trad_conv2.clone(), bias2.clone())
                .into_padded_and_ffted(&in_dimensions[1]),
        ));
        model.add_layer(Layer::Convolution(
            Convolution::new(trad_conv3.clone(), bias3.clone())
                .into_padded_and_ffted(&in_dimensions[2]),
        ));

        // END TEST
        let trace: crate::model::InferenceTrace<'_, _, GoldilocksExt2> =
            model.run::<F>(input.clone()).unwrap();

        let mut model2 = Model::new(&input_shape);
        model2.add_layer(Layer::SchoolBookConvolution(Convolution::new(
            trad_conv1, bias1,
        )));
        model2.add_layer(Layer::SchoolBookConvolution(Convolution::new(
            trad_conv2, bias2,
        )));
        model2.add_layer(Layer::SchoolBookConvolution(Convolution::new(
            trad_conv3, bias3,
        )));
        let trace2 = model.run::<F>(input.clone()).unwrap();

        check_tensor_consistency_field::<GoldilocksExt2>(
            trace2.final_output().clone().to_fields(),
            trace.final_output().clone().to_fields(),
        );

        let _out1: &Tensor<i128> = trace.final_output();
    }

    #[test]
    fn test_conv_maxpool() {
        let input_shape = vec![3usize, 32, 32];
        let shape1 = vec![6, 3, 5, 5];
        let filter = Tensor::random(&shape1);
        let bias1 = Tensor::random(&vec![shape1[0]]);

        let mut model = Model::new(&input_shape);
        model.add_layer(Layer::Convolution(
            Convolution::new(filter.clone(), bias1.clone()).into_padded_and_ffted(&input_shape),
        ));
        model.add_layer(Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default())));

        // TODO: have a "builder" for the model that automatically tracks the shape after each layer such that
        // we can just do model.prepare_input(&input).
        // Here is not possible since we didnt run through the onnx loader
        let input_padded = Tensor::random(&input_shape).pad_next_power_of_two();
        let _: crate::model::InferenceTrace<'_, _, GoldilocksExt2> =
            model.run::<F>(input_padded).unwrap();
    }

    #[test]
    fn test_model_manual_run() {
        let dense1 = Dense::<Element>::random(vec![
            10usize.next_power_of_two(),
            11usize.next_power_of_two(),
        ]);
        let dense2 = Dense::<Element>::random(vec![
            7usize.next_power_of_two(),
            dense1.ncols().next_power_of_two(),
        ]);
        let input_shape = vec![dense1.ncols()];
        let input = Tensor::<Element>::random(&input_shape);
        let output1 = dense1.op(&input);
        let final_output = dense2.op(&output1);

        let mut model = Model::<Element>::new(&input_shape);
        model.add_layer(Layer::Dense(dense1.clone()));
        model.add_layer(Layer::Dense(dense2.clone()));

        let trace = model.run::<F>(input.clone()).unwrap();
        assert_eq!(trace.steps.len(), 2);
        // Verify first step
        assert_eq!(trace.steps[0].output, output1);

        // Verify second step
        assert_eq!(trace.steps[1].output, final_output.clone());
        let (nrow, _) = (dense2.nrows(), dense2.ncols());
        assert_eq!(final_output.get_data().len(), nrow);
    }

    #[test]
    fn test_inference_trace_iterator() {
        let dense1 = Dense::random(vec![
            10usize.next_power_of_two(),
            11usize.next_power_of_two(),
        ]);
        let relu1 = Activation::Relu(Relu);
        let dense2 = Dense::random(vec![
            7usize.next_power_of_two(),
            dense1.ncols().next_power_of_two(),
        ]);
        let relu2 = Activation::Relu(Relu);
        let input_shape = vec![dense1.ncols()];
        let mut model = Model::new(&input_shape);
        let input = Tensor::random(&input_shape);
        model.add_layer(Layer::Dense(dense1));
        model.add_layer(Layer::Activation(relu1));
        model.add_layer(Layer::Dense(dense2));
        model.add_layer(Layer::Activation(relu2));

        let trace = model.run::<F>(input.clone()).unwrap();

        // Verify iterator yields correct input/output pairs
        let mut iter = trace.iter();

        // First step should have original input
        let (first_input, first_step) = iter.next().unwrap();
        assert_eq!(*first_input, trace.input);
        assert_eq!(first_step.output, trace.steps[0].output);

        // Second step should have first step's output as input
        let (second_input, second_step) = iter.next().unwrap();
        assert_eq!(*second_input, trace.steps[0].output);
        assert_eq!(second_step.output, trace.steps[1].output);

        // Third step should have second step's output as input
        let (third_input, third_step) = iter.next().unwrap();
        assert_eq!(*third_input, trace.steps[1].output);
        assert_eq!(third_step.output, trace.steps[2].output);

        // Fourth step should have third step's output as input
        let (fourth_input, fourth_step) = iter.next().unwrap();
        assert_eq!(*fourth_input, trace.steps[2].output);
        assert_eq!(fourth_step.output, trace.steps[3].output);

        // Iterator should be exhausted
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_inference_trace_reverse_iterator() {
        let dense1 = Dense::random(vec![
            10usize.next_power_of_two(),
            11usize.next_power_of_two(),
        ]);
        let dense2 = Dense::random(vec![10usize.next_power_of_two(), dense1.nrows()]);
        let input_shape = vec![dense1.ncols()];
        let input = Tensor::random(&input_shape);

        let mut model = Model::new(&input_shape);
        model.add_layer(Layer::Dense(dense1));
        model.add_layer(Layer::Dense(dense2));
        let trace = model.run::<F>(input.clone()).unwrap();

        // Test reverse iteration
        let mut rev_iter = trace.iter().rev();

        // Last step should come first in reverse
        let (last_input, last_step) = rev_iter.next().unwrap();
        assert_eq!(*last_input, trace.steps[0].output);
        assert_eq!(last_step.output, trace.steps[1].output);

        // First step should come last in reverse
        let (first_input, first_step) = rev_iter.next().unwrap();
        assert_eq!(*first_input, trace.input);
        assert_eq!(first_step.output, trace.steps[0].output);

        // Iterator should be exhausted
        assert!(rev_iter.next().is_none());
    }

    use ff::Field;
    #[test]
    fn test_model_sequential() {
        let (mut model, input) = Model::random(1);
        // remove the requant layer just for this specific test we dont need it.
        model.layers.remove(model.layers.len() - 1);
        model.describe();
        let bb = model.clone();
        let trace = bb.run::<F>(input.clone()).unwrap().to_field();
        let dense_layers = model
            .layers()
            .flat_map(|(_id, l)| match l {
                Layer::Dense(ref dense) => Some(dense.clone()),
                _ => None,
            })
            .collect_vec();
        let matrices_mle = dense_layers
            .iter()
            .map(|d| d.matrix.to_mle_2d::<F>())
            .collect_vec();
        let point1 = random_bool_vector(dense_layers[0].matrix.nrows_2d().ilog2() as usize);
        let computed_eval1 = trace.steps[trace.steps.len() - 1]
            .output
            .get_data()
            .to_vec()
            .into_mle()
            .evaluate(&point1);
        let flatten_mat1 = matrices_mle[0].fix_high_variables(&point1);
        let bias_eval = dense_layers[0]
            .bias
            .evals_flat::<F>()
            .into_mle()
            .evaluate(&point1);
        let computed_eval1_no_bias = computed_eval1 - bias_eval;
        let input_vector = trace.input.clone();
        // since y = SUM M(j,i) x(i) + B(j)
        // then
        // y(r) - B(r) = SUM_i m(r,i) x(i)
        let full_poly = vec![
            flatten_mat1.clone().into(),
            input_vector.get_data().to_vec().into_mle().into(),
        ];
        let mut vp = VirtualPolynomial::new(flatten_mat1.num_vars());
        vp.add_mle_list(full_poly, F::ONE);
        #[allow(deprecated)]
        let (proof, _state) =
            IOPProverState::<F>::prove_parallel(vp.clone(), &mut default_transcript());
        let (p2, _s2) =
            IOPProverState::prove_batch_polys(1, vec![vp.clone()], &mut default_transcript());
        let given_eval1 = proof.extract_sum();
        assert_eq!(p2.extract_sum(), proof.extract_sum());
        assert_eq!(computed_eval1_no_bias, given_eval1);

        let _subclaim = IOPVerifierState::<F>::verify(
            computed_eval1_no_bias,
            &proof,
            &vp.aux_info,
            &mut default_transcript(),
        );
    }

    use crate::{Context, IO, Prover, verify};
    use transcript::BasicTranscript;

    #[test]
    #[ignore = "This test should be deleted since there is no requant and it is not testing much"]
    fn test_single_matvec_prover() {
        let w1 = random_vector_quant(1024 * 1024);
        let conv1 = Tensor::new(vec![1024, 1024], w1.clone());
        let w2 = random_vector_quant(1024);
        let conv2 = Tensor::new(vec![1024], w2.clone());
        let input_shape = vec![1024];
        let input = Tensor::random(&input_shape);

        let mut model = Model::new(&input_shape);
        model.add_layer(Layer::Dense(Dense::new(conv1, conv2)));
        model.describe();
        let trace: crate::model::InferenceTrace<'_, _, GoldilocksExt2> =
            model.run::<F>(input.clone()).unwrap();
        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"m2vec");
        let ctx =
            Context::<GoldilocksExt2>::generate(&model, None).expect("Unable to generate context");
        let output = trace.final_output().clone();
        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>> =
            Prover::new(&ctx, &mut tr);
        let proof = prover.prove(trace).expect("unable to generate proof");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");
        let io = IO::new(input.to_fields(), output.to_fields());
        verify::<_, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_single_cnn_prover() {
        let n_w = 1 << 2;
        let k_w = 1 << 4;
        let n_x = 1 << 5;
        let k_x = 1 << 1;

        let in_dimensions: Vec<Vec<usize>> =
            vec![vec![k_x, n_x, n_x], vec![16, 29, 29], vec![4, 26, 26]];

        let conv1 = Tensor::random(&vec![k_w, k_x, n_w, n_w]);
        let input_shape = vec![k_x, n_x, n_x];
        let input = Tensor::random(&input_shape);

        let mut model = Model::new(&input_shape);
        model.add_layer(Layer::Convolution(
            Convolution::new(conv1.clone(), Tensor::random(&vec![conv1.kw()]))
                .into_padded_and_ffted(&in_dimensions[0]),
        ));
        model.describe();
        let trace: crate::model::InferenceTrace<'_, _, GoldilocksExt2> =
            model.run::<F>(input.clone()).unwrap();
        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"m2vec");
        let ctx = Context::<GoldilocksExt2>::generate(&model, Some(input.get_shape()))
            .expect("Unable to generate context");
        let output = trace.final_output().clone();

        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>> =
            Prover::new(&ctx, &mut tr);
        let proof = prover.prove(trace).expect("unable to generate proof");

        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");
        let io = IO::new(input.to_fields(), output.to_fields());
        verify::<_, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_cnn_prover() {
        for i in 0..3 {
            for j in 2..5 {
                for l in 0..4 {
                    for n in 1..(j - 1) {
                        let n_w = 1 << n;
                        let k_w = 1 << l;
                        let n_x = 1 << j;
                        let k_x = 1 << i;

                        let in_dimensions: Vec<Vec<usize>> =
                            vec![vec![k_x, n_x, n_x], vec![16, 29, 29], vec![4, 26, 26]];
                        let input_shape = vec![k_x, n_x, n_x];
                        let conv1 = Tensor::random(&vec![k_w, k_x, n_w, n_w]);
                        let mut model = Model::<Element>::new(&input_shape);
                        let input = Tensor::random(&input_shape);
                        model.add_layer(Layer::Convolution(
                            Convolution::new(conv1.clone(), Tensor::random(&vec![conv1.kw()]))
                                .into_padded_and_ffted(&in_dimensions[0]),
                        ));
                        model.describe();
                        let trace: crate::model::InferenceTrace<'_, _, GoldilocksExt2> =
                            model.run::<F>(input.clone()).unwrap();
                        let mut tr: BasicTranscript<GoldilocksExt2> =
                            BasicTranscript::new(b"m2vec");
                        let ctx =
                            Context::<GoldilocksExt2>::generate(&model, Some(input.get_shape()))
                                .expect("Unable to generate context");
                        let output = trace.final_output().clone();
                        let prover: Prover<'_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>> =
                            Prover::new(&ctx, &mut tr);
                        let proof = prover.prove(trace).expect("unable to generate proof");
                        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
                            BasicTranscript::new(b"m2vec");
                        let io = IO::new(input.to_fields(), output.to_fields());
                        verify::<_, _>(ctx, proof, io, &mut verifier_transcript).unwrap();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Layer, activation::{Activation, Relu}, dense::Dense};
    use crate::tensor::Tensor;
    use crate::quantization::ScalingFactor;

    #[test]
    fn test_forward_simple_f32() {
        println!("\n=== 测试简单网络前向传播(f32) ===");
        
        // 1. 模型参数设置
        let input_dim = 4;
        let hidden_dim = 3;
        let output_dim = 2;
        
        let weights1_data = vec![
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2
        ];
        let bias1_data = vec![0.1, 0.2, 0.3];
        
        let weights2_data = vec![
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6
        ];
        let bias2_data = vec![0.1, 0.2];
        
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        
        // 2. 模型构建
        let mut model = Model::<f32>::new(&vec![input_dim]);
        
        // 第一层：Dense
        let weights1 = Tensor::<f32>::new(vec![hidden_dim, input_dim], weights1_data);
        let bias1 = Tensor::<f32>::new(vec![hidden_dim], bias1_data);
        let dense1 = Dense::<f32>::new(weights1, bias1);
        model.add_layer(Layer::Dense(dense1));
        
        // ReLU层
        model.add_layer(Layer::Activation(Activation::Relu(Relu::new())));
        
        // 第二层：Dense
        let weights2 = Tensor::<f32>::new(vec![output_dim, hidden_dim], weights2_data);
        let bias2 = Tensor::<f32>::new(vec![output_dim], bias2_data);
        let dense2 = Dense::<f32>::new(weights2, bias2);
        model.add_layer(Layer::Dense(dense2));
        
        // 3. 模型运行
        let input = Tensor::<f32>::new(vec![input_dim], input_data);
        let (output, intermediate_outputs) = model.forward_with_intermediates(&input);
        
        println!("\n=== 浮点模型各层输出 ===");
        println!("输入: {:?}", input.get_data());
        
        // 打印每一层的输出
        for (i, layer_output) in intermediate_outputs.iter().enumerate() {
            println!("\n第{}层输出:", i + 1);
            println!("输出形状: {:?}", layer_output.get_shape());
            println!("输出值: {:?}", layer_output.get_data());
        }
        
        println!("\n=== 最终输出 ===");
        println!("输出形状: {:?}", output.get_shape());
        println!("输出值: {:?}", output.get_data());
        
        // 验证输出
        assert_eq!(output.get_shape(), vec![output_dim]);
        
        let expected_output = vec![5.24, 11.82];
        let actual_output = output.get_data();
        
        // 使用近似比较，因为浮点数计算可能有微小误差
        for (expected, actual) in expected_output.iter().zip(actual_output.iter()) {
            assert!((expected - actual).abs() < 1e-5);
        }
    }

    #[test]
    fn test_squared_error_loss_f32() {
        println!("\n=== 测试平方误差损失(f32) ===");
        
        // 1. 模型参数设置
        let input_dim = 4;
        let hidden_dim = 3;
        let output_dim = 2;
        
        let weights1_data = vec![
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2
        ];
        let bias1_data = vec![0.1, 0.2, 0.3];
        
        let weights2_data = vec![
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6
        ];
        let bias2_data = vec![0.1, 0.2];
        
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let target_data = vec![5.0, 10.0];
        
        // 2. 模型构建
        let mut model = Model::<f32>::new(&vec![input_dim]);
        
        // 第一层：Dense
        let weights1 = Tensor::<f32>::new(vec![hidden_dim, input_dim], weights1_data);
        let bias1 = Tensor::<f32>::new(vec![hidden_dim], bias1_data);
        let dense1 = Dense::<f32>::new(weights1, bias1);
        model.add_layer(Layer::Dense(dense1));
        
        // ReLU层
        model.add_layer(Layer::Activation(Activation::Relu(Relu::new())));
        
        // 第二层：Dense
        let weights2 = Tensor::<f32>::new(vec![output_dim, hidden_dim], weights2_data);
        let bias2 = Tensor::<f32>::new(vec![output_dim], bias2_data);
        let dense2 = Dense::<f32>::new(weights2, bias2);
        model.add_layer(Layer::Dense(dense2));
        
        // 3. 模型运行
        let input = Tensor::<f32>::new(vec![input_dim], input_data);
        let target = Tensor::<f32>::new(vec![output_dim], target_data);
        
        let (output, _) = model.forward_with_intermediates(&input);
        let loss = model.compute_loss(&output, &target, LossFunction::<f32>::SquaredError);
        
        // 验证损失值
        let expected_loss = (5.24f32 - 5.0f32).powi(2) + (11.82f32 - 10.0f32).powi(2);
        assert!((loss - expected_loss).abs() < 1e-5);
    }

    #[test]
    fn test_backward_simple_f32() {
        println!("\n=== 测试反向传播(f32) ===");
        
        // 1. 模型参数设置
        let input_dim = 4;
        let hidden_dim = 3;
        let output_dim = 2;
        let learning_rate = 0.01f32;
        
        let weights1_data = vec![
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2
        ];
        let bias1_data = vec![0.1, 0.2, 0.3];
        
        let weights2_data = vec![
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6
        ];
        let bias2_data = vec![0.1, 0.2];
        
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let target_data = vec![5.0, 10.0];
        
        // 2. 模型构建
        let mut model = Model::<f32>::new(&vec![input_dim]);
        
        // 第一层：Dense
        let weights1 = Tensor::<f32>::new(vec![hidden_dim, input_dim], weights1_data);
        let bias1 = Tensor::<f32>::new(vec![hidden_dim], bias1_data);
        let dense1 = Dense::<f32>::new(weights1, bias1);
        model.add_layer(Layer::Dense(dense1));
        
        // ReLU层
        model.add_layer(Layer::Activation(Activation::Relu(Relu::new())));
        
        // 第二层：Dense
        let weights2 = Tensor::<f32>::new(vec![output_dim, hidden_dim], weights2_data);
        let bias2 = Tensor::<f32>::new(vec![output_dim], bias2_data);
        let dense2 = Dense::<f32>::new(weights2, bias2);
        model.add_layer(Layer::Dense(dense2));
        
        // 3. 模型运行
        let input = Tensor::<f32>::new(vec![input_dim], input_data);
        let target = Tensor::<f32>::new(vec![output_dim], target_data);
        
        // 前向传播
        let (output, intermediate_outputs) = model.forward_with_intermediates(&input);
        
        // 计算损失和梯度
        let (loss, grad) = model.compute_loss_and_gradient(&output, &target, LossFunction::<f32>::SquaredError);
        
        // 反向传播
        model.backward(&intermediate_outputs, &grad);
        
        // 参数更新
        let mut optimizer = SGD::new(learning_rate);
        optimizer.step(&mut model.layers);
        
        // 验证更新后的参数
        if let Layer::Dense(dense) = &model.layers[0] {
            let expected_weights = vec![
                0.084960006, 0.16992001, 0.25488, 0.33984002,
                0.48084, 0.56168, 0.64252, 0.72336,
                0.87671995, 0.95344, 1.0301601, 1.1068801
            ];
            for (expected, actual) in expected_weights.iter().zip(dense.matrix.get_data().iter()) {
                assert!((expected - actual).abs() < 1e-5);
            }
            
            let expected_bias = vec![0.084960006, 0.18084, 0.27672002];
            for (expected, actual) in expected_bias.iter().zip(dense.bias.get_data().iter()) {
                assert!((expected - actual).abs() < 1e-5);
            }
        }
        
        if let Layer::Dense(dense) = &model.layers[2] {
            let expected_weights = vec![
                0.085120015, 0.16544004, 0.24576007,
                0.28716004, 0.23792008, 0.18868011
            ];
            for (expected, actual) in expected_weights.iter().zip(dense.matrix.get_data().iter()) {
                assert!((expected - actual).abs() < 1e-5);
            }
            
            let expected_bias = vec![0.09520001, 0.16360001];
            for (expected, actual) in expected_bias.iter().zip(dense.bias.get_data().iter()) {
                assert!((expected - actual).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_forward_simple_element() {
        println!("\n=== 测试简单网络(Element类型) ===");
        
        // 1. 模型参数设置
        let input_dim = 4;
        let hidden_dim = 3;
        let output_dim = 2;
        
        let weights1_data = vec![
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2
        ];
        let bias1_data = vec![0.1, 0.2, 0.3];
        
        let weights2_data = vec![
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6
        ];
        let bias2_data = vec![0.1, 0.2];
        
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        
        // 2. 模型构建
        let mut model = Model::<f32>::new(&vec![input_dim]);
        
        // 第一层：Dense
        let weights1 = Tensor::<f32>::new(vec![hidden_dim, input_dim], weights1_data);
        let bias1 = Tensor::<f32>::new(vec![hidden_dim], bias1_data);
        let dense1 = Dense::<f32>::new(weights1, bias1);
        model.add_layer(Layer::Dense(dense1));
        
        // ReLU层
        model.add_layer(Layer::Activation(Activation::Relu(Relu::new())));
        
        // 第二层：Dense
        let weights2 = Tensor::<f32>::new(vec![output_dim, hidden_dim], weights2_data);
        let bias2 = Tensor::<f32>::new(vec![output_dim], bias2_data);
        let dense2 = Dense::<f32>::new(weights2, bias2);
        model.add_layer(Layer::Dense(dense2));

        // 3. 使用AbsoluteMax策略进行量化
        let strategy = AbsoluteMax::new();
        let (quantized_model, metadata) = strategy.quantize(model).unwrap();
        
        println!("\n=== 量化后的模型信息 ===");
        println!("输入量化因子: {:?}", metadata.input);
        
        // 打印每一层的量化信息
        for (i, layer) in quantized_model.layers() {
            println!("\n第{}层量化信息:", i + 1);
            println!("层类型: {:?}", layer);
            match layer {
                Layer::Dense(dense) => {
                    if let Some(scaling_factor) = metadata.output_layers_scaling.get(&i) {
                        println!("输出量化因子: {:?}", scaling_factor);
                    }
                    println!("\n量化后的权重矩阵:");
                    println!("{:?}", dense.matrix.get_data());
                    println!("\n量化后的偏置向量:");
                    println!("{:?}", dense.bias.get_data());
                }
                Layer::Activation(_) => {
                    println!("激活层不需要量化");
                }
                _ => {
                    println!("其他类型的层: {:?}", layer);
                }
            }
        }

        // 4. 运行量化后的模型
        let input = Tensor::<Element>::new(vec![input_dim], input_data.iter().map(|&x| x as i128).collect());
        let (output, intermediate_outputs) = quantized_model.forward_with_intermediates(&input);
        
        println!("\n=== 量化模型各层输出 ===");
        println!("输入: {:?}", input.get_data());
        
        // 打印每一层的输出
        for (i, layer_output) in intermediate_outputs.iter().enumerate() {
            println!("\n第{}层输出:", i + 1);
            println!("输出形状: {:?}", layer_output.get_shape());
            println!("输出值: {:?}", layer_output.get_data());
        }
        
        println!("\n=== 最终输出 ===");
        println!("输出形状: {:?}", output.get_shape());
        println!("输出值: {:?}", output.get_data());
        
        // 暂时注释掉反量化部分
        /*
        // 5. 反量化输出
        let dequantized_output = output.dequantize(&metadata.output_layers_scaling.get(&2).unwrap());
        println!("\n=== 反量化后的输出 ===");
        println!("反量化输出: {:?}", dequantized_output.get_data());
        
        // 验证输出
        let expected_output = vec![5.24, 11.82];
        for (expected, actual) in expected_output.iter().zip(dequantized_output.get_data().iter()) {
            assert!((expected - actual).abs() < 1e-2); // 由于量化可能引入误差，放宽误差范围
        }
        */
    }
    // ... existing code ...
    #[test]
    fn test_forward_simple_compare() {
        println!("\n=== 测试简单网络f32与Element对比 ===");
        // 1. 模型参数设置
        let input_dim = 4;
        let hidden_dim = 3;
        let output_dim = 2;
        let weights1_data = vec![
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2
        ];
        let bias1_data = vec![0.1, 0.2, 0.3];
        let weights2_data = vec![
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6
        ];
        let bias2_data = vec![0.1, 0.2];
        let input_data = vec![3.0, 2.0, 3.0, 4.0];

        // 2. 构建f32模型
        let mut model = Model::<f32>::new(&vec![input_dim]);
        let weights1 = Tensor::<f32>::new(vec![hidden_dim, input_dim], weights1_data.clone());
        let bias1 = Tensor::<f32>::new(vec![hidden_dim], bias1_data.clone());
        model.add_layer(Layer::Dense(Dense::<f32>::new(weights1, bias1)));
        model.add_layer(Layer::Activation(Activation::Relu(Relu::new())));
        let weights2 = Tensor::<f32>::new(vec![output_dim, hidden_dim], weights2_data.clone());
        let bias2 = Tensor::<f32>::new(vec![output_dim], bias2_data.clone());
        model.add_layer(Layer::Dense(Dense::<f32>::new(weights2, bias2)));

        // 3. f32模型推理
        let input_f32 = Tensor::<f32>::new(vec![input_dim], input_data.clone());
        let (f32_output, _) = model.forward_with_intermediates(&input_f32);
        println!("\n[f32模型最终输出]: {:?}", f32_output.get_data());

        // 4. 量化为Element模型
        let strategy = AbsoluteMax::new();
        let (quantized_model, metadata) = strategy.quantize(model).unwrap();
        let input_element = Tensor::<Element>::new(vec![input_dim], input_data.iter().map(|&x| x as i128).collect());
        let (element_output, _) = quantized_model.forward_with_intermediates(&input_element);
        println!("[Element模型最终输出(整数)]: {:?}", element_output.get_data());
        let deq_output = element_output.dequantize(&metadata.output_layers_scaling.get(&2).unwrap());
        println!("[Element模型最终输出(反量化f32)]: {:?}", deq_output.get_data());

        // 比例关系与损失分析
        let f32_vals = f32_output.get_data();
        let elem_vals = element_output.get_data();
        if f32_vals.len() == elem_vals.len() && !f32_vals.is_empty() {
            for i in 0..f32_vals.len() {
                let ratio = elem_vals[i] as f64 / f32_vals[i] as f64;
                let diff = elem_vals[i] as f64 - f32_vals[i] as f64;
                println!("[对比] 输出{}: f32 = {:.6}, element = {}, 比例 = {:.6}, 差值 = {:.6}", i, f32_vals[i], elem_vals[i], ratio, diff);
            }
        }
        // 打印metadata信息
        println!("[Meta] input scaling: {:?}", metadata.input);
        println!("[Meta] output_layers_scaling: {:?}", metadata.output_layers_scaling);
    }
}

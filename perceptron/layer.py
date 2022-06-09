import random
from typing import List, Union

from perceptron.activations import Activation
from perceptron import activations
from perceptron import weight_inits


class Layer:
    __slots__ = [
        "input_size",
        "layer_size",
        "activation",
        "weights",
        "biases",
        "outputs",
        "inputs",
        "inputs_gradients",
        "weights_gradients",
        "bias_gradients",
        "velocities",
        "bias_velocities",
        "weights_cache",
        "biases_cache",
        "accumulators",
        "bias_accumulators",
        "first_moment_accumulators",
        "second_moment_accumulators",
        "first_moment_bias_accumulators",
        "second_moment_bias_accumulators",
    ]

    def __init__(
        self,
        input_size: int,
        layer_size: int,
        activation: Union[str, Activation],
        init_method: str = "he",
    ):
        self.input_size = input_size
        self.layer_size = layer_size

        if init_method not in ("uniform", "gauss", "zeros", "he", "xavier"):
            raise ValueError("Invalid weight initialization method")
        init_method = getattr(weight_inits, init_method)

        if isinstance(activation, Activation):
            self.activation = activation
        else:
            if not isinstance(activation, str):
                raise ValueError(
                    f"activation must be a string or inherit from Activation class, not {type(optimizer)}"
                )
            activation = activation.lower()
            if activation == "heavyside":
                self.activation = activations.Heavyside()
            elif activation == "linear":
                self.activation = activations.Linear()
            elif activation == "relu":
                self.activation = activations.Relu()
            elif activation == "leaky_relu":
                self.activation = activations.LeakyRelu()
            elif activation == "sigmoid":
                self.activation = activations.Sigmoid()
            elif activation == "tanh":
                self.activation = activations.Tanh()
            elif activation == "softmax":
                self.activation = activations.Softmax()
            else:
                raise ValueError(f"Invalid activation {activation}")

        self.weights: List[List[float]] = init_method(input_size, layer_size)
        self.biases: List[float] = [0.0] * layer_size
        self.outputs: List[float] = None

        # Reference to the most recent input
        self.inputs: List[float] = None
        # Store reference to the layer input gradients 
        # (gradients of next layer outputs)
        self.inputs_gradients: List[float] = None
        # Accumulated gradients of the weigts
        self.weights_gradients: List[List[float]] = None
        self.bias_gradients: List[float] = None
        # Velocities acumulators for momentum optimizer
        self.velocities: List[List[float]] = None
        self.bias_velocities: List[float] = None
        # Cache for nesterov optimizer
        self.weights_cache: List[List[float]] = None
        self.biases_cache: List[List[float]] = None
        # Learning rate scale accumulators for rmsprop and adagrad optimizers
        self.accumulators: List[List[float]] = None
        self.bias_accumulators: List[float] = None
        # Accumulators for adam optimizer
        self.first_moment_accumulators: List[List[float]] = None
        self.second_moment_accumulators: List[List[float]] = None
        self.first_moment_bias_accumulators: List[float] = None
        self.second_moment_bias_accumulators: List[float] = None

    def forward_pass(self, inputs: List[float]) -> List[float]:

        self.inputs = inputs
        self.outputs = self.biases.copy()

        for neuron_idx in range(self.layer_size):
            for w, i in zip(self.weights[neuron_idx], inputs):
                self.outputs[neuron_idx] += w * i

        self.outputs = self.activation.activate(self.outputs)

        return self.outputs

    def backprop(self, outputs_gradients: List[float]) -> List[float]:

        # First step is to accumulate gradients to later update the weights
        for neuron_index in range(self.layer_size):
            for weight_index in range(self.input_size):
                self.weights_gradients[neuron_index][weight_index] += (
                    outputs_gradients[neuron_index] * self.inputs[weight_index]
                )
            self.bias_gradients[neuron_index] += outputs_gradients[neuron_index]

        # Next, output_gradients need to be propagated backwards
        # through the layer weights to calculate previous layer output gradients
        self.inputs_gradients = [0.0] * self.input_size

        for input_index in range(self.input_size):
            for neuron_index in range(self.layer_size):
                self.inputs_gradients[input_index] += (
                    self.weights[neuron_index][input_index]
                    * outputs_gradients[neuron_index]
                )

        return self.inputs_gradients

    def reset_gradients(self):
        self.weights_gradients = [[0.0] * self.input_size for _ in range(self.layer_size)]
        self.bias_gradients = [0.0] * self.layer_size

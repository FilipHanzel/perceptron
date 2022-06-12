import random
from typing import List, Union

from perceptron import weight_init


class Layer:
    """Dense layer.

    Stores weights and biases, all parameters required for training
    and initialization functions for those parameters.
    Implements forward_pass and backprop methods."""

    __slots__ = [
        "input_size",
        "layer_size",
        "weights",
        "biases",
        "inputs",
        "weights_gradients",
        "biases_gradients",
        "weights_velocities",
        "biases_velocities",
        "weights_cache",
        "biases_cache",
        "weights_accumulators",
        "biases_accumulators",
        "step",
        "first_moment_weights_accumulators",
        "second_moment_weights_accumulators",
        "first_moment_biases_accumulators",
        "second_moment_biases_accumulators",
    ]

    def __init__(
        self,
        input_size: int,
        layer_size: int,
        init_method: str = "he",
    ):
        self.input_size = input_size
        self.layer_size = layer_size

        if init_method not in ("uniform", "gauss", "zeros", "he", "xavier"):
            raise ValueError("Invalid weight initialization method")
        init_method = getattr(weight_init, init_method)

        self.weights: List[List[float]] = init_method(input_size, layer_size)
        self.biases: List[float] = [0.0] * layer_size

        # Reference to the most recent inputs and outputs
        self.inputs: List[float] = None

        # Accumulated gradients of the weigts and biases
        self.weights_gradients: List[List[float]] = None
        self.biases_gradients: List[float] = None

        # Velocities for momentum optimizer
        self.weights_velocities: List[List[float]] = None
        self.biases_velocities: List[float] = None

        # Cache for nesterov optimizer
        self.weights_cache: List[List[float]] = None
        self.biases_cache: List[List[float]] = None

        # Learning rate scale accumulators for rmsprop and adagrad optimizers
        self.weights_accumulators: List[List[float]] = None
        self.biases_accumulators: List[float] = None

        # Accumulators and step for adam optimizer
        self.step: int = None
        self.first_moment_weights_accumulators: List[List[float]] = None
        self.second_moment_weights_accumulators: List[List[float]] = None
        self.first_moment_biases_accumulators: List[float] = None
        self.second_moment_biases_accumulators: List[float] = None

    def forward_pass(self, inputs: List[float]) -> List[float]:
        """Push inputs through the layer. Store reference to inputs and outputs."""
        self.inputs = inputs
        outputs = self.biases.copy()

        for neuron_idx in range(self.layer_size):
            for w, i in zip(self.weights[neuron_idx], inputs):
                outputs[neuron_idx] += w * i

        return outputs

    def backprop(self, outputs_gradients: List[float]) -> List[float]:
        """Push outputs gradients backwards through the layer. Accumulate weights gradients."""

        # First step is to accumulate gradients to later update the weights
        for neuron_index in range(self.layer_size):
            for weight_index in range(self.input_size):
                self.weights_gradients[neuron_index][weight_index] += (
                    outputs_gradients[neuron_index] * self.inputs[weight_index]
                )
            self.biases_gradients[neuron_index] += outputs_gradients[neuron_index]

        # Next, output_gradients need to be propagated backwards
        # through the layer weights to calculate previous layer output gradients
        inputs_gradients = [0.0] * self.input_size

        for input_index in range(self.input_size):
            for neuron_index in range(self.layer_size):
                inputs_gradients[input_index] += (
                    self.weights[neuron_index][input_index]
                    * outputs_gradients[neuron_index]
                )

        return inputs_gradients

    def init_gradients(self) -> None:
        """Initialize gradients."""
        self.weights_gradients = [
            [0.0] * self.input_size for _ in range(self.layer_size)
        ]
        self.biases_gradients = [0.0] * self.layer_size

    def init_velocities(self) -> None:
        """Initialize velocities for Momentum and Nesterov optimizers."""
        self.weights_velocities = [
            [0.0] * self.input_size for _ in range(self.layer_size)
        ]
        self.biases_velocities = [0.0] * self.layer_size

    def init_accumulators(self, initial_value: float = 0.1) -> None:
        """Initialize accumulators for Adagrad and RMSprop optimizers."""
        self.weights_accumulators = [
            [initial_value] * self.input_size for _ in range(self.layer_size)
        ]
        self.biases_accumulators = [initial_value] * self.layer_size

    def init_step(self) -> None:
        """Initialize step counter for Adam optimizer."""
        self.step = 1

    def init_first_and_second_moment_accumulators(self) -> None:
        """Initialize first and second momentum accumulators for Adam optimizer."""
        self.first_moment_weights_accumulators = [
            [0.0] * self.input_size for _ in range(self.layer_size)
        ]
        self.second_moment_weights_accumulators = [
            [0.0] * self.input_size for _ in range(self.layer_size)
        ]
        self.first_moment_biases_accumulators = [0.0] * self.layer_size
        self.second_moment_biases_accumulators = [0.0] * self.layer_size

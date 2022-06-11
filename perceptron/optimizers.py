from typing import List, Union
from abc import ABC, abstractmethod

from perceptron.layer import Layer
from perceptron.loss import Loss
import perceptron.loss


class Optimizer(ABC):
    def __init__(self, *args, **kwargs):
        """Initialize all necessary optimizer parameters."""

    def init(self, layers: List[Layer], loss_function: Union[Loss, str]) -> None:
        """Store a reference to a list of layers and a loss function to minimize.

        Subclasses should initialize all necessary layers parameters in init method.
        Initializing optimizer is necessary in order to use it."""

        if isinstance(loss_function, Loss):
            pass
        else:
            if not isinstance(loss_function, str):
                raise ValueError(
                    f"loss_function must be a string or inherit from Loss class, not {type(loss_function)}"
                )
            loss_function = loss_function.lower()
            if loss_function == "mse":
                loss_function = perceptron.loss.MSE()
            elif loss_function == "msle":
                loss_function = perceptron.loss.MSLE()
            elif loss_function == "mae":
                loss_function = perceptron.loss.MAE()
            elif loss_function == "binary_crossentropy":
                loss_function = perceptron.loss.BinaryCrossentropy()
            elif loss_function == "categorical_crossentropy":
                loss_function = perceptron.loss.CategoricalCrossentropy()
            else:
                raise ValueError(f"Invalid loss function {loss_function}")

        self.layers = layers
        self.loss_function = loss_function
        self.batch_size = 0

        # Initialize gradients with zeros
        self.reset_gradients()

    def forward_pass(self, inputs: List[float]) -> List[float]:
        """Pass inputs through all the layers."""

        for layer in self.layers:
            outputs = layer.forward_pass(inputs)
            inputs = outputs

        return outputs

    def backprop(self, targets: List[float]) -> None:
        """Calculate error and propagate it backwards."""

        *hidden_layers, output_layer = self.layers

        dstate = self.loss_function.derivative(output_layer.outputs, targets)
        loss = dstate

        dstate = output_layer.activation.backprop(dstate)
        dstate = output_layer.backprop(dstate)

        for layer in reversed(hidden_layers):
            dstate = layer.activation.backprop(dstate)
            dstate = layer.backprop(dstate)

        self.batch_size += 1

    def reset_gradients(self):
        for layer in self.layers:
            layer.reset_gradients()
        self.batch_size = 0

    def update(self, learning_rate: float) -> List[float]:
        """Update weights based on calculated gradient. Synonym to __call__ method."""
        return self(learning_rate)

    @abstractmethod
    def __call__(self, learning_rate: float) -> List[float]:
        """Update weights based on calculated gradient. Synonym to update method."""


class GD(Optimizer):
    def __call__(self, learning_rate: float) -> None:
        for layer in self.layers:
            for n in range(layer.layer_size):

                for w in range(layer.input_size):
                    layer.weights[n][w] -= (
                        learning_rate * layer.weights_gradients[n][w] / self.batch_size
                    )

                layer.biases[n] -= (
                    learning_rate * layer.bias_gradients[n] / self.batch_size
                )


class Momentum(Optimizer):
    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma

    def init(self, layers: List[Layer], loss_function: Loss) -> None:
        super().init(layers, loss_function)

        for layer in self.layers:
            layer.velocities = [
                [0.0] * layer.input_size for _ in range(layer.layer_size)
            ]
            layer.bias_velocities = [0.0] * layer.layer_size

    def __call__(self, learning_rate: float) -> None:
        for layer in self.layers:
            for n in range(layer.layer_size):

                for w in range(layer.input_size):
                    layer.velocities[n][w] *= self.gamma
                    layer.velocities[n][w] += (
                        learning_rate * layer.weights_gradients[n][w] / self.batch_size
                    )
                    layer.weights[n][w] -= layer.velocities[n][w]

                layer.bias_velocities[n] *= self.gamma
                layer.bias_velocities[n] += (
                    learning_rate * layer.bias_gradients[n] / self.batch_size
                )
                layer.biases[n] -= layer.bias_velocities[n]


class Nesterov(Momentum):
    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma

    def forward_pass(self, inputs: List[float]) -> List[float]:
        for layer in self.layers:
            layer.weights_cache = layer.weights
            layer.biases_cache = layer.biases

            layer.weights = [
                [w - v * self.gamma for w, v in zip(weights, velocities)]
                for weights, velocities in zip(layer.weights, layer.velocities)
            ]
            layer.biases = [
                b - v * self.gamma for b, v in zip(layer.biases, layer.bias_velocities)
            ]

        output = super().forward_pass(inputs)

        for layer in self.layers:
            layer.weights = layer.weights_cache
            layer.biases = layer.biases_cache

        return output


class Adagrad(Optimizer):
    def __init__(self, epsilon: float = 1e-8, initial_accumulator_value: float = 0.1):
        self.epsilon = epsilon
        self.init_acc = initial_accumulator_value

    def init(self, layers: List[Layer], loss_function: Loss) -> None:
        super().init(layers, loss_function)

        for layer in self.layers:
            layer.accumulators = [
                [self.init_acc] * layer.input_size for _ in range(layer.layer_size)
            ]
            layer.bias_accumulators = [self.init_acc] * layer.layer_size

    def __call__(self, learning_rate: float) -> None:
        for layer in self.layers:
            for n in range(layer.layer_size):

                for w in range(layer.input_size):
                    grad = layer.weights_gradients[n][w] / self.batch_size

                    layer.accumulators[n][w] += grad**2
                    scale = self.epsilon + layer.accumulators[n][w] ** 0.5
                    layer.weights[n][w] -= learning_rate * grad / scale

                bias_grad = layer.bias_gradients[n] / self.batch_size

                layer.bias_accumulators[n] += bias_grad**2
                bias_scale = self.epsilon + layer.bias_accumulators[n] ** 0.5
                layer.biases[n] -= learning_rate * bias_grad / bias_scale


class RMSprop(Optimizer):
    def __init__(
        self,
        epsilon: float = 1e-8,
        initial_accumulator_value: float = 0.0,
        decay_rate: float = 0.9,
    ):
        self.epsilon = epsilon
        self.init_acc = initial_accumulator_value
        self.decay_rate = decay_rate

    def init(self, layers: List[Layer], loss_function: Loss) -> None:
        super().init(layers, loss_function)

        for layer in self.layers:
            layer.accumulators = [
                [self.init_acc] * layer.input_size for _ in range(layer.layer_size)
            ]
            layer.bias_accumulators = [self.init_acc] * layer.layer_size

    def __call__(self, learning_rate: float) -> None:
        for layer in self.layers:
            for n in range(layer.layer_size):

                for w in range(layer.input_size):
                    grad = layer.weights_gradients[n][w] / self.batch_size

                    layer.accumulators[n][w] *= self.decay_rate
                    layer.accumulators[n][w] += (1 - self.decay_rate) * grad**2

                    scale = self.epsilon + layer.accumulators[n][w] ** 0.5
                    layer.weights[n][w] -= learning_rate * grad / scale

                bias_grad = layer.bias_gradients[n] / self.batch_size

                layer.bias_accumulators[n] *= self.decay_rate
                layer.bias_accumulators[n] += (1 - self.decay_rate) * bias_grad**2

                bias_scale = self.epsilon + layer.bias_accumulators[n] ** 0.5
                layer.biases[n] -= learning_rate * bias_grad / bias_scale


class Adam(Optimizer):
    def __init__(
        self, epsilon: float = 1e-8, beta_1: float = 0.9, beta_2: float = 0.999
    ):
        self.epsilon = epsilon
        self.b1 = beta_1
        self.b2 = beta_2

        self.step = 1

    def init(self, layers: List[Layer], loss_function: Loss) -> None:
        super().init(layers, loss_function)

        for layer in self.layers:
            layer.first_moment_accumulators = [
                [0.0] * layer.input_size for _ in range(layer.layer_size)
            ]
            layer.second_moment_accumulators = [
                [0.0] * layer.input_size for _ in range(layer.layer_size)
            ]
            layer.first_moment_bias_accumulators = [0.0] * layer.layer_size
            layer.second_moment_bias_accumulators = [0.0] * layer.layer_size

    def __call__(self, learning_rate: float) -> None:
        for layer in self.layers:
            for n in range(layer.layer_size):

                for w in range(layer.input_size):
                    grad = layer.weights_gradients[n][w] / self.batch_size

                    layer.first_moment_accumulators[n][w] *= self.b1
                    layer.first_moment_accumulators[n][w] += (1 - self.b1) * grad

                    layer.second_moment_accumulators[n][w] *= self.b2
                    layer.second_moment_accumulators[n][w] += (1 - self.b2) * grad**2

                    f_corrected = layer.first_moment_accumulators[n][w] / (
                        1 - self.b1**self.step
                    )
                    s_corrected = layer.second_moment_accumulators[n][w] / (
                        1 - self.b2**self.step
                    )

                    layer.weights[n][w] -= (
                        learning_rate
                        * f_corrected
                        / (self.epsilon + s_corrected**0.5)
                    )

                bias_grad = layer.bias_gradients[n] / self.batch_size

                layer.first_moment_bias_accumulators[n] *= self.b1
                layer.first_moment_bias_accumulators[n] += (1 - self.b1) * bias_grad

                layer.second_moment_bias_accumulators[n] *= self.b2
                layer.second_moment_bias_accumulators[n] += (
                    1 - self.b2
                ) * bias_grad**2

                f_corrected = layer.first_moment_bias_accumulators[n] / (
                    1 - self.b1**self.step
                )
                s_corrected = layer.second_moment_bias_accumulators[n] / (
                    1 - self.b2**self.step
                )

                layer.biases[n] -= (
                    learning_rate * f_corrected / (self.epsilon + s_corrected**0.5)
                )

        self.step += 1

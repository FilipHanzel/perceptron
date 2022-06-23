from typing import List, Union
from abc import ABC, abstractmethod

from perceptron.layer import Layer


def optimizer_from_string(name: str) -> "Optimizer":
    """Get optimizer object with default values, based on string. Convenience function."""

    name = name.lower()
    if name == "gd":
        optimizer = GD()
    elif name == "momentum":
        optimizer = Momentum()
    elif name == "nesterov":
        optimizer = Nesterov()
    elif name == "adagrad":
        optimizer = Adagrad()
    elif name == "rmsprop":
        optimizer = RMSprop()
    elif name == "adam":
        optimizer = Adam()
    else:
        raise ValueError(f"Invalid optimization method {name}")

    return optimizer


class Optimizer(ABC):
    def __init__(self, *args, **kwargs):
        """Initialize all necessary optimizer parameters."""

    def init(self, layer: Layer) -> None:
        """Prepare layer for training (initialize all necessary parameters)."""

        layer.init_gradients()

    def forward_pass(self, layer: Layer, inputs: List[float]) -> List[float]:
        """Forward pass used in training. Wraps layer forward pass.

        Can be overridden if optimizer needs to change standard forward pass of a layer (Nesterov)."""

        return layer.forward_pass(inputs)

    def update(self, layer: Layer, learning_rate: float, batch_size: int) -> None:
        """Update weights based on calculated gradient. Synonym to __call__ method."""

        self(layer, learning_rate, batch_size)

    @abstractmethod
    def __call__(self, layer: Layer, learning_rate: float, batch_size: int) -> None:
        """Update weights based on calculated gradient. Synonym to update method."""


class GD(Optimizer):
    def __call__(self, layer: Layer, learning_rate: float, batch_size: int) -> None:
        for n in range(layer.layer_size):

            for w in range(layer.input_size):
                layer.weights[n][w] -= (
                    learning_rate * layer.weights_gradients[n][w] / batch_size
                )

            layer.biases[n] -= learning_rate * layer.biases_gradients[n] / batch_size

        layer.init_gradients()


class Momentum(Optimizer):
    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma

    def init(self, layer: Layer) -> None:
        super().init(layer)
        layer.init_velocities()

    def __call__(self, layer: Layer, learning_rate: float, batch_size: int) -> None:
        for n in range(layer.layer_size):

            for w in range(layer.input_size):
                layer.weights_velocities[n][w] *= self.gamma
                layer.weights_velocities[n][w] += (
                    learning_rate * layer.weights_gradients[n][w] / batch_size
                )
                layer.weights[n][w] -= layer.weights_velocities[n][w]

            layer.biases_velocities[n] *= self.gamma
            layer.biases_velocities[n] += (
                learning_rate * layer.biases_gradients[n] / batch_size
            )
            layer.biases[n] -= layer.biases_velocities[n]

        layer.init_gradients()


class Nesterov(Momentum):
    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma

    def forward_pass(self, layer: Layer, inputs: List[float]) -> List[float]:
        layer.weights_cache = layer.weights
        layer.biases_cache = layer.biases

        layer.weights = [
            [w - v * self.gamma for w, v in zip(weights, velocities)]
            for weights, velocities in zip(layer.weights, layer.weights_velocities)
        ]
        layer.biases = [
            b - v * self.gamma for b, v in zip(layer.biases, layer.biases_velocities)
        ]

        output = super().forward_pass(layer, inputs)

        layer.weights = layer.weights_cache
        layer.biases = layer.biases_cache

        return output

        layer.init_gradients()


class Adagrad(Optimizer):
    def __init__(self, epsilon: float = 1e-8, initial_accumulator_value: float = 0.1):
        self.epsilon = epsilon
        self.init_acc = initial_accumulator_value

    def init(self, layer: Layer) -> None:
        super().init(layer)
        layer.init_accumulators(self.init_acc)

    def __call__(self, layer: Layer, learning_rate: float, batch_size: int) -> None:
        for n in range(layer.layer_size):

            for w in range(layer.input_size):
                grad = layer.weights_gradients[n][w] / batch_size

                layer.weights_accumulators[n][w] += grad**2
                scale = self.epsilon + layer.weights_accumulators[n][w] ** 0.5
                layer.weights[n][w] -= learning_rate * grad / scale

            biases_grad = layer.biases_gradients[n] / batch_size

            layer.biases_accumulators[n] += biases_grad**2
            biases_scale = self.epsilon + layer.biases_accumulators[n] ** 0.5
            layer.biases[n] -= learning_rate * biases_grad / biases_scale

        layer.init_gradients()


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

    def init(self, layer: Layer) -> None:
        super().init(layer)
        layer.init_accumulators(self.init_acc)

    def __call__(self, layer: Layer, learning_rate: float, batch_size: int) -> None:
        for n in range(layer.layer_size):

            for w in range(layer.input_size):
                grad = layer.weights_gradients[n][w] / batch_size

                layer.weights_accumulators[n][w] *= self.decay_rate
                layer.weights_accumulators[n][w] += (1 - self.decay_rate) * grad**2

                scale = self.epsilon + layer.weights_accumulators[n][w] ** 0.5
                layer.weights[n][w] -= learning_rate * grad / scale

            biases_grad = layer.biases_gradients[n] / batch_size

            layer.biases_accumulators[n] *= self.decay_rate
            layer.biases_accumulators[n] += (1 - self.decay_rate) * biases_grad**2

            biases_scale = self.epsilon + layer.biases_accumulators[n] ** 0.5
            layer.biases[n] -= learning_rate * biases_grad / biases_scale

        layer.init_gradients()


class Adam(Optimizer):
    def __init__(
        self, epsilon: float = 1e-8, beta_1: float = 0.9, beta_2: float = 0.999
    ):
        self.epsilon = epsilon
        self.b1 = beta_1
        self.b2 = beta_2

    def init(self, layer: Layer) -> None:
        super().init(layer)
        layer.init_step()
        layer.init_first_and_second_moment_accumulators()

    def __call__(self, layer: Layer, learning_rate: float, batch_size: int) -> None:
        for n in range(layer.layer_size):

            for w in range(layer.input_size):
                grad = layer.weights_gradients[n][w] / batch_size

                fmwa = layer.first_moment_weights_accumulators
                smwa = layer.second_moment_weights_accumulators

                fmwa[n][w] *= self.b1
                fmwa[n][w] += (1 - self.b1) * grad

                smwa[n][w] *= self.b2
                smwa[n][w] += (1 - self.b2) * grad**2

                f_corrected = fmwa[n][w] / (1 - self.b1**layer.step)
                s_corrected = smwa[n][w] / (1 - self.b2**layer.step)

                layer.weights[n][w] -= (
                    learning_rate * f_corrected / (self.epsilon + s_corrected**0.5)
                )

            biases_grad = layer.biases_gradients[n] / batch_size

            fmba = layer.first_moment_biases_accumulators
            smba = layer.second_moment_biases_accumulators

            fmba[n] *= self.b1
            fmba[n] += (1 - self.b1) * biases_grad

            smba[n] *= self.b2
            smba[n] += (1 - self.b2) * biases_grad**2

            f_corrected = fmba[n] / (1 - self.b1**layer.step)
            s_corrected = smba[n] / (1 - self.b2**layer.step)

            layer.biases[n] -= (
                learning_rate * f_corrected / (self.epsilon + s_corrected**0.5)
            )

        layer.step += 1

        layer.init_gradients()

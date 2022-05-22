from typing import List
from abc import ABC, abstractmethod

from perceptron import Neuron


class Optimizer(ABC):
    def __init__(self, *args, **kwargs):
        """Initialize all necessary parameters."""

    def init(self, layers: List[List[Neuron]]) -> None:
        """Initialize parameters in neurons."""

    @abstractmethod
    def __call__(self, layers: List[List[Neuron]], learning_rate: float) -> None:
        """Update weights based on state of neurons."""


class SGD(Optimizer):
    def __call__(self, layers: List[List[Neuron]], learning_rate: float) -> None:
        for layer in layers:
            for neuron in layer:
                for weight_index, inp in enumerate(neuron.inputs):
                    neuron.weights[weight_index] += learning_rate * neuron.error * inp
                neuron.bias += learning_rate * neuron.error


class Momentum(Optimizer):
    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma

    def init(self, layers: List[List[Neuron]]) -> None:
        """Initializes neurons velocities with zeros. Has to be invoked to use momentum."""
        for layer in layers:
            for neuron in layer:
                neuron.velocities = [0] * len(neuron.weights)
                neuron.bias_velocity = 0

    def __call__(self, layers: List[List[Neuron]], learning_rate: float) -> None:
        for layer in layers:
            for neuron in layer:
                lre = learning_rate * neuron.error

                for weight_index, inp in enumerate(neuron.inputs):
                    neuron.velocities[weight_index] = (
                        neuron.velocities[weight_index] * self.gamma + lre * inp
                    )
                    neuron.weights[weight_index] += neuron.velocities[weight_index]

                neuron.bias_velocity = neuron.bias_velocity * self.gamma + lre
                neuron.bias += neuron.bias_velocity


class Nesterov(Optimizer):
    pass


class Adagrad(Optimizer):
    def __init__(self, epsilon: float = 1e-8, initial_accumulator_value: float = 0.1):
        self.epsilon = epsilon
        self.initial_accumulator_value = 0.1

    def init(self, layers: List[List[Neuron]]) -> None:
        """Initializes neurons accumulators with zeros. Has to be invoked to use adagrad."""
        for layer in layers:
            for neuron in layer:
                neuron.accumulator = [self.initial_accumulator_value] * len(
                    neuron.weights
                )
                neuron.bias_accumulator = self.initial_accumulator_value

    def __call__(self, layers: List[List[Neuron]], learning_rate: float) -> None:
        for layer in layers:
            for neuron in layer:
                for weight_index, inp in enumerate(neuron.inputs):
                    gradient = neuron.error * inp
                    neuron.accumulator[weight_index] += gradient**2

                    scale = self.epsilon + neuron.accumulator[weight_index] ** 0.5
                    neuron.weights[weight_index] += learning_rate * gradient / scale

                bias_gradient = neuron.error
                neuron.bias_accumulator += bias_gradient**2

                bias_scale = self.epsilon + neuron.bias_accumulator**0.5
                neuron.bias += learning_rate * bias_gradient / bias_scale


class RMSprop(Optimizer):
    def __init__(
        self,
        epsilon: float = 1e-8,
        initial_accumulator_value: float = 0.0,
        decay_rate: float = 0.9,
    ):
        self.epsilon = epsilon
        self.initial_accumulator_value = initial_accumulator_value
        self.decay_rate = decay_rate

    def init(self, layers: List[List[Neuron]]) -> None:
        """Initializes neurons accumulators with zeros. Has to be invoked to use rmsprop."""
        for layer in layers:
            for neuron in layer:
                neuron.accumulator = [self.initial_accumulator_value] * len(
                    neuron.weights
                )
                neuron.bias_accumulator = self.initial_accumulator_value

    def __call__(self, layers: List[List[Neuron]], learning_rate: float) -> None:
        for layer in layers:
            for neuron in layer:
                for weight_index, inp in enumerate(neuron.inputs):
                    gradient = neuron.error * inp
                    neuron.accumulator[weight_index] *= self.decay_rate
                    neuron.accumulator[weight_index] += (
                        1 - self.decay_rate
                    ) * gradient**2

                    scale = self.epsilon + neuron.accumulator[weight_index] ** 0.5
                    neuron.weights[weight_index] += learning_rate * gradient / scale

                bias_gradient = neuron.error
                neuron.bias_accumulator *= self.decay_rate
                neuron.bias_accumulator += (1 - self.decay_rate) * bias_gradient**2

                bias_scale = self.epsilon + neuron.bias_accumulator**0.5
                neuron.bias += learning_rate * bias_gradient / bias_scale


class Adam(Optimizer):
    def __init__(
        self, epsilon: float = 1e-8, beta_1: float = 0.9, beta_2: float = 0.999
    ):
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.step = 1

    def init(self, layers: List[List[Neuron]]) -> None:
        """Initializes neurons accumulators with zeros. Has to be invoked to use rmsprop."""
        for layer in layers:
            for neuron in layer:
                neuron.first_moment_accumulator = [0.0] * len(neuron.weights)
                neuron.second_moment_accumulator = [0.0] * len(neuron.weights)
                neuron.first_moment_bias_accumulator = 0.0
                neuron.second_moment_bias_accumulator = 0.0

    def __call__(self, layers: List[List[Neuron]], learning_rate: float) -> None:
        for layer in layers:
            for neuron in layer:
                for weight_index, inp in enumerate(neuron.inputs):
                    gradient = neuron.error * inp

                    neuron.first_moment_accumulator[weight_index] *= self.beta_1
                    neuron.first_moment_accumulator[weight_index] += (
                        1 - self.beta_1
                    ) * gradient

                    neuron.second_moment_accumulator[weight_index] *= self.beta_2
                    neuron.second_moment_accumulator[weight_index] += (
                        1 - self.beta_2
                    ) * gradient**2

                    f_corrected = neuron.first_moment_accumulator[weight_index] / (
                        1 - self.beta_1**self.step
                    )
                    s_corrected = neuron.second_moment_accumulator[weight_index] / (
                        1 - self.beta_2**self.step
                    )

                    neuron.weights[weight_index] += (
                        learning_rate
                        * f_corrected
                        / (self.epsilon + s_corrected**0.5)
                    )

                bias_gradient = neuron.error
                neuron.first_moment_bias_accumulator *= self.beta_1
                neuron.first_moment_bias_accumulator += (
                    1 - self.beta_1
                ) * bias_gradient

                neuron.second_moment_bias_accumulator *= self.beta_2
                neuron.second_moment_bias_accumulator += (
                    1 - self.beta_2
                ) * bias_gradient**2

                neuron.bias += (
                    learning_rate * f_corrected / (self.epsilon + s_corrected**0.5)
                )

        self.step += 1

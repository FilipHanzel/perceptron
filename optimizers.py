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
    def __init__(self):
        self.epsilon = 1e-8

    def init(self, layers: List[List[Neuron]]) -> None:
        """Initializes neurons velocities with zeros. Has to be invoked to use adagrad."""
        for layer in layers:
            for neuron in layer:
                neuron.scale = [0.1] * len(neuron.weights)
                neuron.bias_scale = 0.1

    def __call__(self, layers: List[List[Neuron]], learning_rate: float) -> None:
        for layer in layers:
            for neuron in layer:
                for weight_index, inp in enumerate(neuron.inputs):
                    neuron.scale[weight_index] += (neuron.error * inp) ** 2
                    neuron.weights[weight_index] += (
                        learning_rate
                        * (neuron.error * inp)
                        / (self.epsilon + neuron.scale[weight_index] ** 0.5)
                    )
                neuron.bias_scale += (neuron.error) ** 2
                neuron.bias += (
                    learning_rate
                    * (neuron.error)
                    / (self.epsilon + neuron.bias_scale**0.5)
                )


class RMSprop(Optimizer):
    pass


class Adam(Optimizer):
    pass

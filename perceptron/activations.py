from math import exp
from abc import ABC, abstractmethod
from typing import List


class Activation:
    @abstractmethod
    def activate(self, values: List[float]) -> List[float]:
        """Calculate the activation value."""

    @abstractmethod
    def derivative(self, values: List[float]) -> List[float]:
        """Calculate derivative of the activation function."""


class Heavyside(Activation):
    def activate(self, values: List[float]) -> List[float]:
        return [1.0 if value >= 0.0 else 0.0 for value in values]

    def derivative(self, values: List[float]) -> List[float]:
        """Return input value.

        Heavyside activation is non-differentiable and thus
        does not have a derivative. This function returns
        input value for compatibility with weight updates."""
        return values


class Linear(Activation):
    def activate(self, values: List[float]) -> List[float]:
        return values

    def derivative(self, values: List[float]) -> List[float]:
        return [1.0] * len(values)


class Relu(Activation):
    def activate(self, values: List[float]) -> List[float]:
        return [max(0.0, value) for value in values]

    def derivative(self, values: List[float]) -> List[float]:
        return [1.0 if value > 0.0 else 0.0 for value in values]


class LeakyRelu(Activation):
    def __init__(self, leak_coefficient: List[float] = 0.1):
        self.leak_coefficient = leak_coefficient

    def activate(self, values: List[float]) -> List[float]:
        return [
            self.leak_coefficient * value if value < 0.0 else value for value in values
        ]

    def derivative(self, values: List[float]) -> List[float]:
        return [self.leak_coefficient if value < 0.0 else 1 for value in values]


class Sigmoid(Activation):
    def activate(self, values: List[float]) -> List[float]:
        return [1.0 / (1.0 + exp(-value)) for value in values]

    def derivative(self, values: List[float]) -> List[float]:
        values = [1.0 / (1.0 + exp(-value)) for value in values]
        return [value * (1.0 - value) for value in values]


class Softmax(Activation):
    def activate(self, values: List[float]) -> List[float]:
        exps = [exp(value) for value in values]
        sum_ = sum(exps)
        return [exp_ / sum_ for exp_ in exps]

    def derivative(self, values: List[float]) -> List[float]:
        return [value * (1 - value) for value in values]

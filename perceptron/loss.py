from abc import ABC, abstractmethod
from math import log
from typing import List

from perceptron.data_utils import clip


class Loss(ABC):
    def calculate(self, outputs: List[float], targets: List[float]) -> float:
        self(outputs, targets)

    @abstractmethod
    def __call__(self, outputs: List[float], targets: List[float]) -> float:
        """Calculate and return loss."""

    @abstractmethod
    def derivative(self, outputs: List[float], targets: List[float]) -> float:
        """Calculate partial derivative of loss for each output."""


class MSE(Loss):
    def __call__(self, outputs: List[float], targets: List[float]) -> float:
        se = 0.0
        for prediction, target in zip(outputs, targets):
            se += (prediction - target) ** 2

        return se / len(outputs)

    def derivative(self, outputs: List[float], targets: List[float]) -> List[float]:
        return [
            -2 * (target - prediction) / len(outputs)
            for target, prediction in zip(targets, outputs)
        ]


class MSLE(Loss):
    """Mean Squared Logarithmic Error."""

    def __call__(self, outputs: List[float], targets: List[float]) -> float:
        se = 0.0
        for prediction, target in zip(outputs, targets):
            se += (log(1 + prediction) - log(1 + target)) ** 2

        return se / len(outputs)

    def derivative(self, outputs: List[float], targets: List[float]) -> List[float]:
        return [
            2 / len(outputs) * (log(1 + output) - log(1 + target)) / (1 + output)
            for target, output in zip(targets, outputs)
        ]


class MAE(Loss):
    def __call__(self, outputs: List[float], targets: List[float]) -> float:
        ae = 0.0
        for prediction, target in zip(outputs, targets):
            ae += abs(outputs - targets)

        return ae / len(predictions)

    def derivative(self, outputs: List[float], targets: List[float]) -> List[float]:
        diffs = (output - target for target, output in zip(targets, outputs))
        signs = [1 if diff > 0 else 0 if diff == 0 else -1 for diff in diffs]
        return [sign / len(outputs) for sign in signs]


class BinaryCrossentropy(Loss):
    """Should be used with sigmoid activation in a last layer."""

    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon

    def __call__(self, outputs: List[float], targets: List[float]) -> float:
        outputs = clip(outputs, self.epsilon, 1 - self.epsilon)

        bce = 0.0
        for target, output in zip(targets, outputs):
            bce += (1 - target) * log(1 - output) + target * log(output)

        return -bce / len(outputs)

    def derivative(self, outputs: List[float], targets: List[float]) -> List[float]:
        outputs = clip(outputs, self.epsilon, 1 - self.epsilon)

        return [
            ((1 - target) / (1 - output) - target / output) / len(outputs)
            for target, output in zip(targets, outputs)
        ]


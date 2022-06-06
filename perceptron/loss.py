from abc import ABC, abstractmethod
from typing import List


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


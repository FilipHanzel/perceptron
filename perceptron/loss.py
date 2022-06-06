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

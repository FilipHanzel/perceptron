from abc import ABC, abstractmethod
from typing import List


class Loss(ABC):
    def calculate(self, predictions: List[float], targets: List[float]) -> float:
        self(predictions, targets)

    @abstractmethod
    def __call__(self, predictions: List[float], targets: List[float]) -> float:
        """Calculate and return loss."""


class MSE(Loss):
    def __call__(self, predictions: List[float], targets: List[float]) -> float:

        se = 0.0
        for prediction, target in zip(predictions, targets):
            se += (prediction - target) ** 2

        return se / len(predictions)

    def partial_derivatives(
        self, predictions: List[float], targets: List[float]
    ) -> List[float]:

        return [
            -2 * (target - prediction) / len(predictions)
            for target, prediction in zip(targets, predictions)
        ]

from abc import ABC, abstractmethod
from math import log
from typing import List

from perceptron.data_util import clip


def loss_from_string(name: str) -> "Loss":
    """Get loss object with default values, based on string. Convenience function."""

    name = name.lower()
    if name == "mse":
        loss_function = MSE()
    elif name == "msle":
        loss_function = MSLE()
    elif name == "mae":
        loss_function = MAE()
    elif name == "binary_crossentropy":
        loss_function = BinaryCrossentropy()
    elif name == "categorical_crossentropy":
        loss_function = CategoricalCrossentropy()
    else:
        raise ValueError(f"Invalid loss function {name}")

    return loss_function


class Loss(ABC):
    def calculate(self, outputs: List[float], targets: List[float]) -> float:
        """Calculate and return loss. Synonym to __call__ method."""
        self(outputs, targets)

    @abstractmethod
    def __call__(self, outputs: List[float], targets: List[float]) -> float:
        """Calculate and return loss. Synonym to calculate method."""

    @abstractmethod
    def derivative(self, outputs: List[float], targets: List[float]) -> float:
        """Calculate partial derivative of loss for each output."""


class MSE(Loss):
    """Mean Squared Error."""

    def __call__(self, outputs: List[float], targets: List[float]) -> float:
        se = 0.0
        for output, target in zip(outputs, targets):
            se += (output - target) ** 2

        return se / len(outputs)

    def derivative(self, outputs: List[float], targets: List[float]) -> List[float]:
        return [
            -2 * (target - output) / len(outputs)
            for target, output in zip(targets, outputs)
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
    """Mean Absolute Error."""

    def __call__(self, outputs: List[float], targets: List[float]) -> float:
        ae = 0.0
        for output, target in zip(outputs, targets):
            ae += abs(output - target)

        return ae / len(outputs)

    def derivative(self, outputs: List[float], targets: List[float]) -> List[float]:
        diffs = (output - target for target, output in zip(targets, outputs))
        signs = [1 if diff > 0 else 0 if diff == 0 else -1 for diff in diffs]
        return [sign / len(outputs) for sign in signs]


class BinaryCrossentropy(Loss):
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


class CategoricalCrossentropy(Loss):
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon

    def __call__(self, outputs: List[float], targets: List[float]) -> float:
        outputs = clip(outputs, self.epsilon, 1 - self.epsilon)

        cce = 0.0
        for target, output in zip(targets, outputs):
            cce += target * log(output)

        return -cce / len(outputs)

    def derivative(self, outputs: List[float], targets: List[float]) -> List[float]:
        outputs = clip(outputs, self.epsilon, 1 - self.epsilon)

        return [-target / output for target, output in zip(targets, outputs)]

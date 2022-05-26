from math import exp
from abc import ABC, abstractmethod


class Decay(ABC):
    def __init__(self, *args, **kwargs):
        """Initialize all necessary parameters."""

    @abstractmethod
    def __call__(self, current_epoch: int) -> float:
        """Calculate learning rate for provided epoch."""


class LinearDecay(Decay):
    def __init__(self, base_learning_rate: float, epochs: int):
        self.base_rate = base_learning_rate
        self.epochs = epochs

    def __call__(self, current_epoch: int) -> float:
        return self.base_rate * (1.0 - (current_epoch / self.epochs))


class PolynomialDecay(Decay):
    def __init__(self, base_learning_rate: float, epochs: int, power: float):
        self.base_rate = base_learning_rate
        self.epochs = epochs
        self.power = power

    def __call__(self, current_epoch: int) -> float:
        return self.base_rate * (1.0 - (current_epoch / self.epochs)) ** self.power


class TimeBasedDecay(Decay):
    def __init__(self, base_learning_rate: float, epochs: int):
        self.base_rate = base_learning_rate
        self.epochs = epochs

    def __call__(self, current_epoch: int) -> float:
        rate = self.base_rate
        for epoch in range(current_epoch):
            rate /= 1.0 + (epoch * self.base_rate / self.epochs)
        return rate


class ExpDecay(Decay):
    def __init__(self, base_learning_rate: float, decay_rate: float):
        self.base_rate = base_learning_rate
        self.decay_rate = decay_rate

    def __call__(self, current_epoch: int) -> float:
        return self.base_rate * exp(-self.decay_rate * current_epoch)


class StepDecay(Decay):
    def __init__(self, base_learning_rate: float, drop: float, interval: int):
        self.base_rate = base_learning_rate
        self.drop = drop
        self.interval = interval

    def __call__(self, current_epoch: int) -> float:
        return self.base_rate * self.drop ** (current_epoch // self.interval)

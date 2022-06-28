from abc import ABC, abstractmethod
from math import exp


def decay_from_string(name: str, base_learning_rate: float, epochs: int) -> "Decay":
    """Get decay object with default values, based on string. Convenience function."""

    name = name.lower()
    if name == "linear":
        decay = LinearDecay(base_learning_rate, epochs)
    elif name == "polynomial":
        decay = PolynomialDecay(base_learning_rate, epochs)
    elif name == "timebased":
        decay = TimeBasedDecay(base_learning_rate, epochs)
    elif name == "exponential":
        decay = ExponentialDecay(base_learning_rate)
    elif name == "step":
        decay = StepDecay(base_learning_rate, epochs // 10)
    else:
        raise ValueError(f"Invalid learning rate decay {name}")

    return decay


class Decay(ABC):
    """Base class for learning rate decays."""

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
    def __init__(self, base_learning_rate: float, epochs: int, power: float = 2):
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


class ExponentialDecay(Decay):
    def __init__(self, base_learning_rate: float, decay_rate: float = 0.1):
        self.base_rate = base_learning_rate
        self.decay_rate = decay_rate

    def __call__(self, current_epoch: int) -> float:
        return self.base_rate * exp(-self.decay_rate * current_epoch)


class StepDecay(Decay):
    def __init__(self, base_learning_rate: float, interval: int, drop: float = 0.5):
        self.base_rate = base_learning_rate
        self.drop = drop
        self.interval = interval

    def __call__(self, current_epoch: int) -> float:
        return self.base_rate * self.drop ** (current_epoch // self.interval)

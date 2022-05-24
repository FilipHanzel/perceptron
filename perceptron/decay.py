from math import exp


class LinearDecay:
    def __init__(self, base_learning_rate: float, epochs: int):
        self.base_rate = base_learning_rate
        self.epochs = epochs

    def __call__(self, current_epoch: int) -> float:
        return self.base_rate * (1.0 - (current_epoch / self.epochs))


class PolynomialDecay:
    def __init__(self, base_learning_rate: float, epochs: int, power: float):
        self.base_rate = base_learning_rate
        self.epochs = epochs
        self.power = power

    def __call__(self, current_epoch: int) -> float:
        return self.base_rate * (1.0 - (current_epoch / self.epochs)) ** self.power


class TimeBasedDecay:
    def __init__(self, base_learning_rate: float, epochs: int):
        self.base_rate = base_learning_rate
        self.epochs = epochs

    def __call__(self, current_epoch: int) -> float:
        rate = self.base_rate
        for epoch in range(current_epoch):
            rate /= 1.0 + (epoch * self.base_rate / self.epochs)
        return rate


class ExpDecay:
    def __init__(self, base_learning_rate: float, decay_rate: float):
        self.base_rate = base_learning_rate
        self.decay_rate = decay_rate

    def __call__(self, current_epoch: int) -> float:
        return self.base_rate * exp(-self.decay_rate * current_epoch)


class StepDecay:
    def __init__(self, base_learning_rate: float, drop: float, interval: int):
        self.base_rate = base_learning_rate
        self.drop = drop
        self.interval = interval

    def __call__(self, current_epoch: int) -> float:
        return self.base_rate * self.drop ** (current_epoch // self.interval)

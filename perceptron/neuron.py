import random
from typing import List


class Neuron:
    __slots__ = [
        "weights",
        "bias",
        "inputs",
        "output",
        "error",
        "gradients",
        "bias_gradient",
        "velocities",
        "bias_velocity",
        "weights_cache",
        "accumulator",
        "bias_accumulator",
        "first_moment_accumulator",
        "second_moment_accumulator",
        "first_moment_bias_accumulator",
        "second_moment_bias_accumulator",
    ]

    def __init__(self, weights: List[float], bias: float):
        self.weights = weights
        self.bias = bias

        self.inputs: List[float] = None
        self.output: float = None
        self.error: float = None
        self.velocities: List[float] = None
        self.bias_velocity: float = None
        self.weights_cache: List[float] = None
        self.accumulator: float = None
        self.bias_accumulator: float = None
        self.first_moment_accumulator: float = None
        self.second_moment_accumulator: float = None
        self.first_moment_bias_accumulator: float = None
        self.second_moment_bias_accumulator: float = None


class WeightInitialization:
    @staticmethod
    def uniform(inputs: int) -> List[float]:
        return [random.uniform(-1, 1) for _ in range(inputs)]

    @staticmethod
    def gauss(inputs: int) -> List[float]:
        return [random.gauss(0, 1) for _ in range(inputs)]

    @staticmethod
    def zeros(inputs: int) -> List[float]:
        return [0.0] * inputs

    @staticmethod
    def he(inputs: int) -> List[float]:
        scale = (2 / inputs) ** 0.5
        return [random.gauss(0, 1) * scale for _ in range(inputs)]

    @staticmethod
    def xavier(inputs: int) -> List[float]:
        scale = (1 / inputs) ** 0.5
        return [random.gauss(0, 1) * scale for _ in range(inputs)]

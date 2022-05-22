from typing import List


class Neuron:
    __slots__ = [
        "weights",
        "bias",
        "inputs",
        "output",
        "error",
        "velocities",
        "bias_velocity",
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
        self.accumulator: float = None
        self.bias_accumulator: float = None
        self.first_moment_accumulator: float = None
        self.second_moment_accumulator: float = None
        self.first_moment_bias_accumulator: float = None
        self.second_moment_bias_accumulator: float = None

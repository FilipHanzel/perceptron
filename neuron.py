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
    ]

    def __init__(self, weights: List[float], bias: float):
        self.weights = weights
        self.bias = bias

        self.inputs: List[float] = None
        self.output: float = None
        self.error: float = None
        self.velocities: List[float] = None
        self.bias_velocity: float = None

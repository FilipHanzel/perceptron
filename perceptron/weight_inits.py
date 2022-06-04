import random
from typing import List


def uniform(input_size: int, layer_size: int) -> List[List[float]]:
    return [
        [random.uniform(-1, 1) for _ in range(input_size)] for _ in range(layer_size)
    ]


def gauss(input_size: int, layer_size: int) -> List[List[float]]:
    return [[random.gauss(0, 1) for _ in range(input_size)] for _ in range(layer_size)]


def zeros(input_size: int, layer_size: int) -> List[List[float]]:
    return [[0.0] * input_size for _ in range(layer_size)]


def he(input_size: int, layer_size: int) -> List[List[float]]:
    scale = (2 / input_size) ** 0.5
    return [
        [random.gauss(0, 1) * scale for _ in range(input_size)]
        for _ in range(layer_size)
    ]


def xavier(input_size: int, layer_size: int) -> List[List[float]]:
    scale = (1 / input_size) ** 0.5
    return [
        [random.gauss(0, 1) * scale for _ in range(input_size)]
        for _ in range(layer_size)
    ]

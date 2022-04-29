import random
from math import exp
from typing import List, Tuple, Callable


class Activation:
    @staticmethod
    def heavyside(x):
        return 1 if x >= 0 else 0

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def relu(x):
        return max(0, x)

    @staticmethod
    def leaky_relu(x):
        return 0.3 * x if x < 0 else x

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + exp(-x))


class Perceptron:
    __slots__ = ["weights", "bias", "activation"]

    def __init__(self, inputs: int, activation: Callable[[float], float]):
        self.weights = [random.uniform(0, 1) for _ in range(inputs)]
        self.bias = 0
        self.activation = activation

    def predict(self, input_features: List[float]) -> float:
        state = 0
        for (w, s) in zip(self.weights, input_features):
            state += w * s
        state += self.bias
        return self.activation(state)

    def update(
        self, train_vector: List[float], learning_rate: float
    ) -> Tuple[float, float]:
        *features, target = train_vector
        prediction = self.predict(features)
        error = target - prediction

        self.bias += learning_rate * error
        for idx, feature in enumerate(features):
            self.weights[idx] = self.weights[idx] + learning_rate * error * feature

        return (prediction, error**2)

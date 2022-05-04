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


class Derivative:
    @staticmethod
    def relu(x):
        return 0.0 if x < 0.0 else 1.0

    @staticmethod
    def leaky_relu(x):
        return 0.3 if x < 0.0 else 1.0

    @staticmethod
    def sigmoid(x):
        return x * (1.0 - x)

    @staticmethod
    def linear(x):
        return 1.0


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


class MultilayerPerceptron:
    __slots__ = ["activation", "derivative", "layers"]

    class Neuron:
        __slots__ = ["weights", "bias", "inputs", "output", "error"]

        def __init__(self, weights: List[float], bias: float):
            self.weights = weights
            self.bias = bias

            self.inputs: List[float] = None
            self.output: float = None
            self.error: float = None

    def __init__(
        self,
        inputs: int,
        layer_sizes: List[int],
        activation: Callable[[float], float],
        derivative: Callable[[float], float],
    ):
        self.activation = activation
        self.derivative = derivative

        input_sizes = [inputs, *layer_sizes]

        self.layers = [
            [
                self.Neuron(weights=[0.0 for _ in range(input_size)], bias=0.0)
                for _ in range(layer_size)
            ]
            for (input_size, layer_size) in zip(input_sizes, layer_sizes)
        ]

    def predict(self, inputs: List[float]):
        state = inputs
        for layer in self.layers:
            state = [
                self.activation(
                    sum([weight * inp for weight, inp in zip(neuron.weights, state)])
                )
                for neuron in layer
            ]
        return state

    def update(self, inputs: List[float], targets: List[float], learning_rate: float):
        # Forward pass
        for layer in self.layers:
            state = []
            for neuron in layer:
                neuron.inputs = inputs

                neuron.output = 0.0
                for w, i in zip(neuron.weights, neuron.inputs):
                    neuron.output += w * i
                neuron.output = self.activation(neuron.output)

                state.append(neuron.output)
            inputs = state

        # Error backpropagation
        *hidden_layers, output_layer = self.layers

        for neuron, target in zip(output_layer, targets):
            neuron.error = (target - neuron.output) * self.derivative(neuron.output)

        for index in reversed(range(len(hidden_layers))):
            for neuron_index, neuron in enumerate(self.layers[index]):

                neuron.error = 0.0
                for front_neuron in self.layers[index + 1]:
                    neuron.error += (
                        front_neuron.weights[neuron_index] * front_neuron.error
                    )
                neuron.error *= self.derivative(neuron.output)

        # Weight update
        for layer in self.layers:
            for neuron in layer:
                for weight_index, inp in enumerate(neuron.inputs):
                    neuron.weights[weight_index] += learning_rate * neuron.error * inp

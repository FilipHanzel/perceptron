import random
from math import exp
from typing import List, Tuple

from tqdm import tqdm


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

    def __init__(self, inputs: int, activation: str):
        assert activation in (
            "heavyside",
            "linear",
            "relu",
            "leaky_relu",
            "sigmoid",
        ), "Invalid activation"

        self.weights = [random.uniform(0, 1) for _ in range(inputs)]
        self.bias = 0

        self.activation = getattr(Activation, activation)

    def predict(self, inputs: List[float]) -> float:
        state = 0
        for (w, s) in zip(self.weights, inputs):
            state += w * s
        state += self.bias
        return self.activation(state)

    def update(self, inputs: List[float], target: float, learning_rate: float) -> float:
        prediction = self.predict(inputs)

        error = target - prediction
        self.bias += learning_rate * error
        for idx, feature in enumerate(inputs):
            self.weights[idx] = self.weights[idx] + learning_rate * error * feature

        return prediction

    def train(
        self,
        list_of_inputs: List[List[float]],
        list_of_targets: List[float],
        epochs: int,
        learning_rate: float,
    ) -> None:

        progress = tqdm(
            range(epochs),
            unit="epochs",
            ncols=100,
            bar_format="Training: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}",
        )

        for epoch in progress:
            sse = 0.0

            for inputs, target in zip(list_of_inputs, list_of_targets):
                prediction = self.update(inputs, target, learning_rate)

                sse += (prediction - target) ** 2
            progress.set_postfix(sse=round(sse, 3))


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

        def __str__(self):
            return f"Neuron <weights: {self.weights}, bias: {self.bias}>"

        def __repr__(self):
            return self.__str__()

    def __init__(self, inputs: int, layer_sizes: List[int], activation: str):
        assert activation in (
            "linear",
            "relu",
            "leaky_relu",
            "sigmoid",
        ), "Invalid activation"

        self.activation = getattr(Activation, activation)
        self.derivative = getattr(Derivative, activation)

        input_sizes = [inputs, *layer_sizes]

        self.layers = [
            [
                self.Neuron(
                    weights=[random.uniform(0, 1) for _ in range(input_size)], bias=0.0
                )
                for _ in range(layer_size)
            ]
            for (input_size, layer_size) in zip(input_sizes, layer_sizes)
        ]

    def predict(self, inputs: List[float]) -> List[float]:
        state = inputs
        for layer in self.layers:
            state = [
                self.activation(
                    sum([weight * inp for weight, inp in zip(neuron.weights, state)])
                    + neuron.bias
                )
                for neuron in layer
            ]
        return state

    def update(
        self, inputs: List[float], targets: List[float], learning_rate: float
    ) -> Tuple[List[float], float]:
        # Forward pass
        for layer in self.layers:
            state = []
            for neuron in layer:
                neuron.inputs = inputs

                neuron.output = neuron.bias
                for w, i in zip(neuron.weights, neuron.inputs):
                    neuron.output += w * i
                neuron.output = self.activation(neuron.output)

                state.append(neuron.output)
            inputs = state
        output = state

        # Error backpropagation
        *hidden_layers, output_layer = self.layers

        for neuron, target in zip(output_layer, targets):
            neuron.error = target - neuron.output

        for index in reversed(range(len(hidden_layers))):
            for neuron_index, neuron in enumerate(self.layers[index]):

                neuron.error = 0.0
                for front_neuron in self.layers[index + 1]:
                    neuron.error += (
                        front_neuron.weights[neuron_index] * front_neuron.error
                    )
                neuron.error

        # Weight update
        for layer in self.layers:
            for neuron in layer:
                update = self.derivative(neuron.output) * learning_rate * neuron.error
                for weight_index, inp in enumerate(neuron.inputs):
                    neuron.weights[weight_index] += update * inp
                neuron.bias += update

        return output

    def train(
        self,
        list_of_inputs: List[List[float]],
        list_of_targets: List[List[float]],
        epochs: int,
        learning_rate: float,
    ) -> None:

        progress = tqdm(
            range(epochs),
            unit="epochs",
            ncols=100,
            bar_format="Training: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}",
        )

        for epoch in progress:
            sse = 0.0

            for inputs, target in zip(list_of_inputs, list_of_targets):
                prediction = self.update(inputs, target, learning_rate)

                sse += sum([(p - t) ** 2 for p, t in zip(prediction, target)])
            progress.set_postfix(sse=round(sse, 3))

import random
from math import exp
from typing import List, Tuple, Union

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


class WeightInit:
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


def linear_decay(base_rate: float, current_epoch: int, total_epochs: int) -> float:
    return base_rate * (1.0 - (current_epoch / total_epochs))


class Metric:
    @staticmethod
    def acc(predictions: List, targets: List) -> float:
        correct = 0

        if isinstance(predictions[0], float) or isinstance(predictions[0], int):
            for prediction, target in zip(predictions, targets):
                correct += int(prediction) == target
        else:
            for prediction_list, target_list in zip(predictions, targets):
                prediction = prediction_list.index(max(prediction_list))
                target = target_list.index(max(target_list))

                correct += prediction == target

        return correct / len(predictions)

    @staticmethod
    def sse(predictions: List, targets: List) -> float:
        sse = 0.0

        if isinstance(predictions[0], float) or isinstance(predictions[0], int):
            for prediction, target in zip(predictions, targets):
                sse += (prediction - target) ** 2
        else:
            for prediction_list, target_list in zip(predictions, targets):
                sse += sum(
                    [
                        (prediction - target) ** 2
                        for prediction, target in zip(prediction_list, target_list)
                    ]
                )

        return sse


class Perceptron:
    __slots__ = ["weights", "bias", "activation"]

    def __init__(self, inputs: int, activation: str, init_method: str = "uniform"):
        assert activation in (
            "heavyside",
            "linear",
            "relu",
            "leaky_relu",
            "sigmoid",
        ), "Invalid activation"

        assert init_method in (
            "uniform",
            "gauss",
            "zeros",
            "he",
            "xavier",
        ), "Invalid weight initialization method"

        self.weights = getattr(WeightInit, init_method)(inputs)
        self.bias = 0

        self.activation = getattr(Activation, activation)

    def predict(self, inputs: List[float]) -> float:
        state = 0
        for w, s in zip(self.weights, inputs):
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
        training_inputs: List[List[float]],
        training_targets: List[float],
        epochs: int,
        base_learning_rate: float,
        learning_rate_decay: Union[str, None] = "linear",
        metrics: List[str] = ["sse"],
        validation_inputs: List[List[float]] = [],
        validation_targets: List[float] = [],
    ) -> List:
        assert learning_rate_decay in [
            None,
            "linear",
        ], "Unsupported learning rate decay"

        for metric in metrics:
            assert hasattr(Metric, metric), "Unsupported metric"

        progress = tqdm(
            range(epochs),
            unit="epochs",
            bar_format="Training: {percentage:3.0f}% |{bar:40}| {n_fmt}/{total_fmt}{postfix}",
        )

        validate = len(validation_inputs) > 0

        learning_rate = base_learning_rate
        for epoch in progress:

            if learning_rate_decay == "linear":
                learning_rate = linear_decay(base_learning_rate, epoch, epochs)

            for inputs, target in zip(training_inputs, training_targets):
                prediction = self.update(inputs, target, learning_rate)

            predictions = [self.predict(inputs) for inputs in training_inputs]
            calculated_metrics = {
                metric: getattr(Metric, metric)(predictions, training_targets)
                for metric in metrics
            }
            if validate:
                predictions = [self.predict(inputs) for inputs in validation_inputs]
                for metric in metrics:
                    calculated_metrics["val_" + metric] = getattr(Metric, metric)(
                        predictions, validation_targets
                    )

            progress.set_postfix(**calculated_metrics)


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

    def __init__(
        self,
        inputs: int,
        layer_sizes: List[int],
        activation: str,
        init_method: str = "gauss",
    ):
        assert activation in (
            "linear",
            "relu",
            "leaky_relu",
            "sigmoid",
        ), "Invalid activation"

        assert init_method in (
            "uniform",
            "gauss",
            "zeros",
            "he",
            "xavier",
        ), "Invalid weight initialization method"

        self.activation = getattr(Activation, activation)
        self.derivative = getattr(Derivative, activation)

        input_sizes = [inputs, *layer_sizes]

        init_method = getattr(WeightInit, init_method)
        self.layers = [
            [
                self.Neuron(weights=init_method(input_size), bias=0.0)
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
                neuron.bias += learning_rate * neuron.error

        return output

    def train(
        self,
        training_inputs: List[List[float]],
        training_targets: List[List[float]],
        epochs: int,
        base_learning_rate: float,
        learning_rate_decay: Union[str, None] = "linear",
        metrics: List[str] = ["sse"],
        validation_inputs: List[List[float]] = [],
        validation_targets: List[List[float]] = [],
    ) -> List:
        assert learning_rate_decay in [
            None,
            "linear",
        ], "Unsupported learning rate decay"

        for metric in metrics:
            assert hasattr(Metric, metric), "Unsupported metric"

        progress = tqdm(
            range(epochs),
            unit="epochs",
            bar_format="Training: {percentage:3.0f}% |{bar:40}| {n_fmt}/{total_fmt}{postfix}",
        )

        validate = len(validation_inputs) > 0

        learning_rate = base_learning_rate
        for epoch in progress:

            if learning_rate_decay == "linear":
                learning_rate = linear_decay(base_learning_rate, epoch, epochs)

            for inputs, target in zip(training_inputs, training_targets):
                prediction = self.update(inputs, target, learning_rate)

            predictions = [self.predict(inputs) for inputs in training_inputs]
            calculated_metrics = {
                metric: getattr(Metric, metric)(predictions, training_targets)
                for metric in metrics
            }

            if validate:
                predictions = [self.predict(inputs) for inputs in validation_inputs]
                for metric in metrics:
                    calculated_metrics["val_" + metric] = getattr(Metric, metric)(
                        predictions, validation_targets
                    )

            progress.set_postfix(**calculated_metrics)

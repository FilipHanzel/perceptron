import random
from math import exp, ceil
from typing import List, Tuple, Union, Type, Dict

from tqdm import tqdm

from neuron import Neuron
import data_utils
import normalizers
import optimizers


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
    def linear(x):
        return 1.0

    @staticmethod
    def relu(x):
        return 0.0 if x < 0.0 else 1.0

    @staticmethod
    def leaky_relu(x):
        return 0.3 if x < 0.0 else 1.0

    @staticmethod
    def sigmoid(x):
        return x * (1.0 - x)


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
    def binary_accuracy(predictions: List[List], targets: List[List]) -> float:
        threshold = 0.5

        correct = 0
        total = 0

        for predictions_row, target_row in zip(predictions, targets):
            total += len(predictions_row)
            for prediction, target in zip(predictions_row, target_row):
                prediction = 1 if prediction > threshold else 0
                correct += prediction == target

        return correct / total

    @staticmethod
    def categorical_accuracy(predictions: List[List], targets: List[List]) -> float:
        correct = 0

        for prediction_row, target_row in zip(predictions, targets):
            prediction = prediction_row.index(max(prediction_row))
            target = target_row.index(max(target_row))

            correct += prediction == target

        return correct / len(predictions)

    @staticmethod
    def sse(predictions: List[List], targets: List[List]) -> float:
        sse = 0.0

        for prediction_list, target_list in zip(predictions, targets):
            sse += sum(
                [
                    (prediction - target) ** 2
                    for prediction, target in zip(prediction_list, target_list)
                ]
            )

        return sse

    @staticmethod
    def mae(predictions: List[List], targets: List[List]) -> float:
        mae = 0.0

        for prediction_list, target_list in zip(predictions, targets):
            mae += sum(
                [
                    abs(prediction - target)
                    for prediction, target in zip(prediction_list, target_list)
                ]
            ) / len(prediction_list)

        return mae / len(predictions)


class Perceptron:
    __slots__ = [
        "activations",
        "derivatives",
        "layers",
        "normalizer",
        "optimizer",
    ]

    def __init__(
        self,
        inputs: int,
        layer_sizes: List[int],
        activations: Union[str, List[str]],
        init_method: str = "gauss",
        normalization: str = None,
        optimizer: str = "SGD",
    ):
        # Initialize layers activations
        if isinstance(activations, str):
            activations = [activations] * len(layer_sizes)

        for activation in activations:
            assert activation in (
                "linear",
                "relu",
                "leaky_relu",
                "sigmoid",
                "heavyside",
            ), f"Invalid activation ({activation})"

        assert len(activations) == len(
            layer_sizes
        ), "Amount of activations must match layers"

        self.activations = [
            getattr(Activation, activation) for activation in activations
        ]
        if activation == "heavyside":
            assert len(layer_sizes) == 1, "Heavyside activation is invalid for MLP"
            self.derivatives = [lambda _: 1]
        else:
            self.derivatives = [
                getattr(Derivative, activation) for activation in activations
            ]

        # Initialize layers weights
        assert init_method in (
            "uniform",
            "gauss",
            "zeros",
            "he",
            "xavier",
        ), "Invalid weight initialization method"

        input_sizes = [inputs, *layer_sizes]

        init_method = getattr(WeightInit, init_method)
        self.layers = [
            [
                Neuron(weights=init_method(input_size), bias=0.0)
                for _ in range(layer_size)
            ]
            for (input_size, layer_size) in zip(input_sizes, layer_sizes)
        ]

        # Initialize input normalization method
        assert normalization in (
            "minmax",
            "zscore",
            None,
        ), "Unknown normalization method"
        if normalization is None:
            self.normalizer = None
        elif normalization == "minmax":
            self.normalizer = normalizers.MinMax()
        elif normalization == "zscore":
            self.normalizer = normalizers.ZScore()
        else:
            raise ValueError("Unknown normalization method")

        # Initialize model optimizer using default parameters
        optimizer = optimizer.lower()
        if optimizer == "sgd":
            self.optimizer = optimizers.SGD()
        elif optimizer == "momentum":
            self.optimizer = optimizers.Momentum()
        elif optimizer == "adagrad":
            self.optimizer = optimizers.Adagrad()
        elif optimizer == "rmsprop":
            self.optimizer = optimizers.RMSprop()
        elif optimizer == "adam":
            self.optimizer = optimizers.Adam()
        else:
            raise ValueError("Unknown optimization method")

        self.optimizer.init(self.layers)

    def predict(self, inputs: List[float]) -> List[float]:

        if self.normalizer is not None:
            inputs = self.normalizer(inputs)

        state = inputs
        for layer, activation in zip(self.layers, self.activations):
            state = [
                activation(
                    sum([weight * inp for weight, inp in zip(neuron.weights, state)])
                    + neuron.bias
                )
                for neuron in layer
            ]
        return state

    def update(
        self,
        inputs: List[float],
        targets: List[float],
        learning_rate: float,
    ) -> Tuple[List[float], float]:

        if self.normalizer is not None:
            inputs = self.normalizer(inputs)

        # Forward pass
        for layer, activation in zip(self.layers, self.activations):
            state = []
            for neuron in layer:
                neuron.inputs = inputs

                neuron.output = neuron.bias
                for w, i in zip(neuron.weights, neuron.inputs):
                    neuron.output += w * i
                neuron.output = activation(neuron.output)

                state.append(neuron.output)
            inputs = state
        output = state

        # Error backpropagation
        *hidden_layers, output_layer = self.layers

        for neuron, target in zip(output_layer, targets):
            neuron.error = target - neuron.output
            neuron.error *= self.derivatives[-1](neuron.output)

        for index in reversed(range(len(hidden_layers))):
            for neuron_index, neuron in enumerate(self.layers[index]):

                neuron.error = 0.0
                for front_neuron in self.layers[index + 1]:
                    neuron.error += (
                        front_neuron.weights[neuron_index] * front_neuron.error
                    )
                neuron.error *= self.derivatives[index](neuron.output)

        # Weight update
        self.optimizer(self.layers, learning_rate)

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

        if self.normalizer is not None:
            self.normalizer.adapt(training_inputs)

        progress = tqdm(
            range(epochs),
            unit="epochs",
            bar_format="Training: {percentage:3.0f}% |{bar:40}| {n_fmt}/{total_fmt}{postfix}",
        )

        validate = len(validation_inputs) > 0

        training_inputs = training_inputs.copy()
        training_targets = training_targets.copy()

        learning_rate = base_learning_rate
        for epoch in progress:

            training_inputs, training_targets = data_utils.shuffle(
                training_inputs, training_targets
            )

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


def cross_validation(
    inputs: List,
    targets: List,
    fold_count: int,
    epoch: int,
    base_learning_rate: float,
    learning_rate_decay: str,
    model_params: Dict,
    metrics: List[str] = ["sse"],
):
    order = list(range(len(inputs)))
    random.shuffle(order)

    fold_size = ceil(len(inputs) / fold_count)
    folds = [
        order[index : index + fold_size] for index in range(0, len(inputs), fold_size)
    ]

    for test_fold in folds:
        test_inputs = [inputs[idx] for idx in test_fold]
        test_targets = [targets[idx] for idx in test_fold]

        train_folds = [fold for fold in folds if fold is not test_fold]
        train_inputs = [inputs[idx] for fold in train_folds for idx in fold]
        train_targets = [targets[idx] for fold in train_folds for idx in fold]

        model = Perceptron(**model_params)
        model.train(
            training_inputs=train_inputs,
            training_targets=train_targets,
            epochs=epoch,
            base_learning_rate=base_learning_rate,
            learning_rate_decay=learning_rate_decay,
            metrics=metrics,
            validation_inputs=test_inputs,
            validation_targets=test_targets,
        )

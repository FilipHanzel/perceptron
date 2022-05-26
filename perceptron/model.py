import random
from math import ceil
from typing import List, Tuple, Union, Dict

from tqdm import tqdm

from perceptron.neuron import Neuron
from perceptron.neuron import WeightInitialization
from perceptron import data_utils
from perceptron import normalizers
from perceptron import optimizers
import perceptron.decay
import perceptron.activations
import perceptron.metrics


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
            getattr(perceptron.activations, activation) for activation in activations
        ]
        if activation == "heavyside":
            assert len(layer_sizes) == 1, "Heavyside activation is invalid for MLP"
            self.derivatives = [lambda _: 1]
        else:
            self.derivatives = [
                getattr(perceptron.activations, "d_" + activation)
                for activation in activations
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

        init_method = getattr(WeightInitialization, init_method)
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
        elif optimizer == "nesterov":
            self.optimizer = optimizers.Nesterov()
        elif optimizer == "adagrad":
            self.optimizer = optimizers.Adagrad()
        elif optimizer == "rmsprop":
            self.optimizer = optimizers.RMSprop()
        elif optimizer == "adam":
            self.optimizer = optimizers.Adam()
        else:
            raise ValueError("Unknown optimization method")

        self.optimizer.init(self)

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

        return self.optimizer(inputs, targets, learning_rate)

    def measure(
        self, inputs: List[List[float]], targets: List[List[float]], metrics: List[str]
    ) -> Dict[str, float]:
        for metric in metrics:
            assert hasattr(perceptron.metrics, metric), "Unsupported metric"

        predictions = [self.predict(inp) for inp in inputs]
        return {
            metric: getattr(perceptron.metrics, metric)(predictions, targets)
            for metric in metrics
        }

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
    ) -> Dict:
        assert learning_rate_decay in [
            None,
            "linear",
            "polynomial",
            "timebased",
            "exponential",
            "step",
        ], "Unsupported learning rate decay"

        if learning_rate_decay == "linear":
            decay = perceptron.decay.LinearDecay(
                base_learning_rate=base_learning_rate, epochs=epochs
            )
        elif learning_rate_decay == "polynomial":
            decay = perceptron.decay.PolynomialDecay(
                base_learning_rate=base_learning_rate, epochs=epochs, power=2
            )
        elif learning_rate_decay == "timebased":
            decay = perceptron.decay.TimeBasedDecay(
                base_learning_rate=base_learning_rate, epochs=epochs
            )
        elif learning_rate_decay == "exponential":
            decay = perceptron.decay.ExpDecay(
                base_learning_rate=base_learning_rate, decay_rate=0.1
            )
        elif learning_rate_decay == "step":
            decay = perceptron.decay.StepDecay(
                base_learning_rate=base_learning_rate, drop=0.5, interval=epochs // 10
            )
        else:
            decay = lambda x: x

        for metric in metrics:
            assert hasattr(perceptron.metrics, metric), "Unsupported metric"

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

        measured = self.measure(training_inputs, training_targets, metrics)
        history = {metric: [measured[metric]] for metric in measured}
        if validate:
            measured = self.measure(validation_inputs, validation_targets, metrics)
            history.update({"val_" + metric: [measured[metric]] for metric in measured})

        learning_rate = base_learning_rate
        for epoch in progress:

            training_inputs, training_targets = data_utils.shuffle(
                training_inputs, training_targets
            )

            learning_rate = decay(epoch)

            for inputs, target in zip(training_inputs, training_targets):
                prediction = self.update(inputs, target, learning_rate)

            measured = self.measure(training_inputs, training_targets, metrics)

            if validate:
                validated = self.measure(validation_inputs, validation_targets, metrics)
                measured.update(
                    {"val_" + metric: validated[metric] for metric in validated}
                )

            progress.set_postfix(**measured)

            for metric in measured:
                history[metric].append(measured[metric])

        return history


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

    history = []
    for test_fold in folds:
        test_inputs = [inputs[idx] for idx in test_fold]
        test_targets = [targets[idx] for idx in test_fold]

        train_folds = [fold for fold in folds if fold is not test_fold]
        train_inputs = [inputs[idx] for fold in train_folds for idx in fold]
        train_targets = [targets[idx] for fold in train_folds for idx in fold]

        model = Perceptron(**model_params)
        run = model.train(
            training_inputs=train_inputs,
            training_targets=train_targets,
            epochs=epoch,
            base_learning_rate=base_learning_rate,
            learning_rate_decay=learning_rate_decay,
            metrics=metrics,
            validation_inputs=test_inputs,
            validation_targets=test_targets,
        )
        history.append(run)
    return history

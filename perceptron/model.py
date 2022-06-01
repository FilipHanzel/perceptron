import random
from math import ceil
from typing import List, Tuple, Union, Dict

from tqdm import tqdm

from perceptron.neuron import Neuron
from perceptron.neuron import WeightInitialization
from perceptron import data_utils
from perceptron import normalizers
from perceptron import optimizers
from perceptron import decays
from perceptron.optimizers import Optimizer
from perceptron.decays import Decay
from perceptron.activations import Activation
from perceptron.metrics import Metric
from perceptron.loss import Loss
import perceptron.activations
import perceptron.metrics
import perceptron.loss


class Perceptron:
    __slots__ = [
        "activations",
        "derivatives",
        "layers",
        "normalizer",
        "optimizer",
        "loss_function",
    ]

    def __init__(
        self,
        inputs: int,
        layer_sizes: List[int],
        activations: Union[str, List[Union[str, Activation]]],
        init_method: str = "gauss",
        normalization: str = None,
        optimizer: Union[Optimizer, str] = "SGD",
        loss_function: Union[Loss, str] = "MSE",
    ):
        # Initialize layers activations
        if isinstance(activations, str):
            activations = [activations] * len(layer_sizes)

        assert len(activations) == len(
            layer_sizes
        ), "Amount of activations must match layers"

        loaded_activations = []
        for activation in activations:
            if isinstance(activation, Activation):
                loaded_activations.append(activation)
            else:
                activation = activation.lower()
                if activation == "heavyside":
                    if len(layer_sizes) > 1:
                        raise ValueError("Heavyside activation is invalid for MLP")
                    loaded_activations.append(perceptron.activations.Heavyside())
                elif activation == "linear":
                    loaded_activations.append(perceptron.activations.Linear())
                elif activation == "relu":
                    loaded_activations.append(perceptron.activations.Relu())
                elif activation == "leaky_relu":
                    loaded_activations.append(perceptron.activations.LeakyRelu())
                elif activation == "sigmoid":
                    loaded_activations.append(perceptron.activations.Sigmoid())
                elif activation == "softmax":
                    loaded_activations.append(perceptron.activations.Softmax())
                else:
                    raise ValueError(f"Invalid activation {activation}")
        self.activations = loaded_activations

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

        # Initialize model optimizer
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            optimizer = optimizer.lower()
            if optimizer == "gd":
                self.optimizer = optimizers.GD()
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

        if isinstance(loss_function, Loss):
            self.loss_function = loss_function
        else:
            loss_function = loss_function.lower()
            if loss_function == "mse":
                self.loss_function = perceptron.loss.MSE()
            else:
                raise ValueError("Unknown loss function")

    def predict(
        self,
        inputs: List[float],
        normalize_input: bool = True,
    ) -> List[float]:

        if self.normalizer is not None and normalize_input:
            inputs = self.normalizer(inputs)

        state = inputs
        for layer, activation in zip(self.layers, self.activations):
            state = [
                (
                    sum([weight * inp for weight, inp in zip(neuron.weights, state)])
                    + neuron.bias
                )
                for neuron in layer
            ]
            state = activation.activate(state)
        return state

    def measure(
        self,
        inputs: List[List[float]],
        targets: List[List[float]],
        metrics: List[str],
        normalize_input: bool = True,
    ) -> Dict[str, float]:

        for metric in metrics.values():
            assert isinstance(metric, Metric), f"Unsupported metric {metric}"

        predictions = [self.predict(inp, normalize_input) for inp in inputs]
        return {name: metric(predictions, targets) for name, metric in metrics.items()}

    def train(
        self,
        training_inputs: List[List[float]],
        training_targets: List[List[float]],
        epochs: int,
        batch_size: int = 1,
        base_learning_rate: float = 1e-4,
        learning_rate_decay: Union[Decay, str, None] = "linear",
        metrics: List[str] = ["mae"],
        validation_inputs: List[List[float]] = [],
        validation_targets: List[List[float]] = [],
    ) -> Dict:

        if batch_size > len(training_inputs):
            batch_size = len(training_inputs)

        if isinstance(learning_rate_decay, Decay):
            decay = learning_rate_decay
        elif learning_rate_decay is None:
            decay = lambda _: base_learning_rate
        elif learning_rate_decay == "linear":
            decay = decays.LinearDecay(
                base_learning_rate=base_learning_rate, epochs=epochs
            )
        elif learning_rate_decay == "polynomial":
            decay = decays.PolynomialDecay(
                base_learning_rate=base_learning_rate, epochs=epochs, power=2
            )
        elif learning_rate_decay == "timebased":
            decay = decays.TimeBasedDecay(
                base_learning_rate=base_learning_rate, epochs=epochs
            )
        elif learning_rate_decay == "exponential":
            decay = decays.ExpDecay(
                base_learning_rate=base_learning_rate, decay_rate=0.1
            )
        elif learning_rate_decay == "step":
            decay = decays.StepDecay(
                base_learning_rate=base_learning_rate, drop=0.5, interval=epochs // 10
            )
        else:
            raise ValueError("Unsupported learning rate decay")

        loaded_metrics = {}
        for metric in metrics:
            if isinstance(metric, Metric):
                loaded_metrics[metric.name, metric]
            else:
                metric = metric.lower()

                if metric == "mae":
                    metric = perceptron.metrics.MAE()
                elif metric == "mape":
                    metric = perceptron.metrics.MAPE()
                elif metric == "mse":
                    metric = perceptron.metrics.MSE()
                elif metric == "rmse":
                    metric = perceptron.metrics.RMSE()
                elif metric == ("cos_sim", "cos_similarity"):
                    metric = perceptron.metrics.CosSim()
                elif metric in ("bin_acc", "binary_accuracy"):
                    metric = perceptron.metrics.BinaryAccuracy()
                elif metric in ("cat_acc", "categorical_accuracy"):
                    metric = perceptron.metrics.CategoricalAccuracy()
                elif metric in ("top_k_cat_acc", "top_k_categorical_accuracy"):
                    metric = perceptron.metrics.TopKCategoricalAccuracy()
                else:
                    raise ValueError(f"Unsupported metric {metric}")

                loaded_metrics[metric.name] = metric

        metrics = loaded_metrics

        if self.normalizer is not None:
            self.normalizer.adapt(training_inputs, clean=False)
            training_inputs = [self.normalizer(inp) for inp in training_inputs]
            validation_inputs = [self.normalizer(inp) for inp in validation_inputs]
        else:
            training_inputs = training_inputs.copy()
        training_targets = training_targets.copy()

        progress = tqdm(
            range(epochs),
            unit="epochs",
            bar_format="Training: {percentage:3.0f}% |{bar:40}| {n_fmt}/{total_fmt}{postfix}",
        )

        validate = len(validation_inputs) > 0

        measured = self.measure(
            training_inputs,
            training_targets,
            metrics,
            normalize_input=False,
        )
        history = {metric: [measured[metric]] for metric in measured}
        if validate:
            measured = self.measure(
                validation_inputs,
                validation_targets,
                metrics,
                normalize_input=False,
            )
            history.update({"val_" + metric: [measured[metric]] for metric in measured})

        for epoch in progress:

            training_inputs, training_targets = data_utils.shuffle(
                training_inputs, training_targets
            )

            learning_rate = decay(epoch)

            for inputs, targets in zip(training_inputs, training_targets):

                self.optimizer.forward_pass(inputs)
                self.optimizer.backprop(targets)
                self.optimizer.accumulate_gradient()

                if (
                    self.optimizer.batch_size % batch_size == 0
                    or self.optimizer.batch_size == len(training_inputs)
                ):
                    self.optimizer.update(learning_rate)
                    self.optimizer.forget_gradient()

            measured = self.measure(
                training_inputs,
                training_targets,
                metrics,
                normalize_input=False,
            )
            if validate:
                validated = self.measure(
                    validation_inputs,
                    validation_targets,
                    metrics,
                    normalize_input=False,
                )
                measured.update(
                    {"val_" + metric: validated[metric] for metric in validated}
                )

            progress.set_postfix(**measured)

            for metric in measured:
                history[metric].append(measured[metric])

        return history


def cross_validation(
    inputs: List[List[float]],
    targets: List[List[float]],
    model_params: Dict,
    fold_count: int,
    epoch: int,
    batch_size: int,
    base_learning_rate: float = 1e-4,
    learning_rate_decay: Union[Decay, str, None] = "linear",
    metrics: List[str] = ["mae"],
) -> List[Dict]:

    folds = data_utils.kfold_split(
        inputs, targets, fold_count, stratified=True, random=True
    )

    history = []
    for test_fold in folds:
        test_inputs = test_fold["inputs"]
        test_targets = test_fold["targets"]

        train_inputs = []
        train_targets = []
        for fold in folds:
            if fold is not test_fold:
                train_inputs += fold["inputs"]
                train_targets += fold["targets"]

        model = Perceptron(**model_params)
        run = model.train(
            training_inputs=train_inputs,
            training_targets=train_targets,
            epochs=epoch,
            batch_size=batch_size,
            base_learning_rate=base_learning_rate,
            learning_rate_decay=learning_rate_decay,
            metrics=metrics,
            validation_inputs=test_inputs,
            validation_targets=test_targets,
        )
        history.append(run)
    return history

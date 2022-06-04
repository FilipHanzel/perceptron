import random
from math import ceil
from typing import List, Tuple, Union, Dict

from tqdm import tqdm

from perceptron.layer import Layer
from perceptron import data_utils
from perceptron import normalizers
from perceptron import optimizers
from perceptron import decays
from perceptron.optimizers import Optimizer
from perceptron.decays import Decay
from perceptron.activations import Activation
from perceptron.activations import Heavyside
from perceptron.metrics import Metric
from perceptron.loss import Loss
import perceptron.metrics
import perceptron.loss


class Model:
    __slots__ = [
        "layers",
        "normalizer",
        "optimizer",
    ]

    def __init__(
        self,
        inputs: int,
        layer_sizes: List[int],
        activations: Union[str, List[Union[str, Activation]]],
        init_method: str = "gauss",
        normalization: str = None,
        optimizer: Union[Optimizer, str] = "GD",
        loss_function: Union[Loss, str] = "MSE",
    ):
        # Initialize layers
        if isinstance(activations, str):
            activations = [activations] * len(layer_sizes)

        if len(activations) != len(layer_sizes):
            raise ValueError("Amount of activations must match layers")

        if len(layer_sizes) > 1:
            for activation in activations:
                if isinstance(activation, Heavyside) or activation == "heavyside":
                    raise ValueError("Heavyside activation is invalud for MLP")

        input_sizes = [inputs, *layer_sizes]
        self.layers = [
            Layer(input_size, layer_size, activation, init_method)
            for input_size, layer_size, activation in zip(
                input_sizes, layer_sizes, activations
            )
        ]

        # Initialize input normalization method
        if normalization not in ("minmax", "zscore", None):
            raise ValueError("Invalid input normalization method")

        if normalization is None:
            self.normalizer = None
        elif normalization == "minmax":
            self.normalizer = normalizers.MinMax()
        elif normalization == "zscore":
            self.normalizer = normalizers.ZScore()
        else:
            raise ValueError("Unknown normalization method")

        # Initialize loss function
        if isinstance(loss_function, Loss):
            pass
        else:
            if not isinstance(loss_function, str):
                raise ValueError(
                    f"loss_function must be a string or inherit from Loss class, not {type(loss_function)}"
                )
            loss_function = loss_function.lower()
            if loss_function == "mse":
                loss_function = perceptron.loss.MSE()
            else:
                raise ValueError(f"Invalid loss function {loss_function}")

        # Initialize optimizer
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            if not isinstance(optimizer, str):
                raise ValueError(
                    f"optimizer must be a string or inherit from Optimizer class, not {type(optimizer)}"
                )
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
                raise ValueError(f"Invalid optimization method {optimizer}")

        self.optimizer.init(self.layers, loss_function)

    def predict(
        self,
        inputs: List[float],
        normalize_input: bool = True,
    ) -> List[float]:
        """Normalize inputs if needed and do forward pass."""

        if normalize_input and self.normalizer is not None:
            inputs = self.normalizer(inputs)

        for layer in self.layers:
            outputs = layer.forward_pass(inputs)
            inputs = outputs

        return outputs

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

        if learning_rate_decay is None:
            decay = lambda _: base_learning_rate
        elif isinstance(learning_rate_decay, Decay):
            decay = learning_rate_decay
        else:
            if not isinstance(learning_rate_decay, str):
                raise ValueError(
                    f"learning_rate_decay must be a string or inherit from Decay class, not {type(learning_rate_decay)}"
                )
            learning_rate_decay = learning_rate_decay.lower()
            if learning_rate_decay == "linear":
                decay = decays.LinearDecay(base_learning_rate, epochs)
            elif learning_rate_decay == "polynomial":
                decay = decays.PolynomialDecay(base_learning_rate, epochs)
            elif learning_rate_decay == "timebased":
                decay = decays.TimeBasedDecay(base_learning_rate, epochs)
            elif learning_rate_decay == "exponential":
                decay = decays.ExpDecay(base_learning_rate)
            elif learning_rate_decay == "step":
                decay = decays.StepDecay(base_learning_rate, interval=epochs // 10)
            else:
                raise ValueError(f"Invalid learning rate decay {learning_rate_decay}")

        loaded_metrics = {}
        for metric in metrics:
            if isinstance(metric, Metric):
                loaded_metrics[metric.name, metric]
            else:
                if not isinstance(metric, str):
                    raise ValueError(
                        f"metric must be a string or inherit from Metric class, not {type(metric)}"
                    )
                metric = metric.lower()
                if metric == "mae":
                    metric = perceptron.metrics.MAE()
                elif metric == "mape":
                    metric = perceptron.metrics.MAPE()
                elif metric == "mse":
                    metric = perceptron.metrics.MSE()
                elif metric == "rmse":
                    metric = perceptron.metrics.RMSE()
                elif metric == "cos_similarity":
                    metric = perceptron.metrics.CosSim()
                elif metric in "binary_accuracy":
                    metric = perceptron.metrics.BinaryAccuracy()
                elif metric in "categorical_accuracy":
                    metric = perceptron.metrics.CategoricalAccuracy()
                elif metric in "top_k_categorical_accuracy":
                    metric = perceptron.metrics.TopKCategoricalAccuracy()
                else:
                    raise ValueError(f"Invalid metric {metric}")
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

        progress.set_postfix(**measured)

        for epoch in progress:

            training_inputs, training_targets = data_utils.shuffle(
                training_inputs, training_targets
            )

            learning_rate = decay(epoch)

            for inputs, targets in zip(training_inputs, training_targets):
                self.optimizer.forward_pass(inputs)
                self.optimizer.backprop(targets)

                if (
                    self.optimizer.batch_size % batch_size == 0
                    or self.optimizer.batch_size == len(training_inputs)
                ):
                    self.optimizer.update(learning_rate)
                    self.optimizer.reset_gradients()

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

        model = Model(**model_params)
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

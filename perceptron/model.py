import random
from math import ceil
from typing import Callable, Dict, List, Tuple, Union

from tqdm import tqdm

from perceptron.layer import Layer
from perceptron import data_util
from perceptron.activation import Activation
from perceptron.decay import Decay, decay_from_string
from perceptron.loss import Loss, loss_from_string
from perceptron.metric import Metric, metric_from_string
from perceptron.normalizer import Normalizer, normalizer_from_string
from perceptron.optimizer import Optimizer, optimizer_from_string


class Model:
    __slots__ = [
        "layers",
        "trainable_layers",
        "normalizer",
        "optimizer",
    ]

    def __init__(
        self,
        normalizer: Union[Normalizer, str] = None,
    ):
        # Initialize input normalizer
        if normalizer is not None and not isinstance(normalizer, Normalizer):
            if not isinstance(normalizer, str):
                raise ValueError(
                    f"normalizer must be a string or inherit from Normalizer class, not {type(normalizer)}"
                )
            normalizer = normalizer_from_string(normalizer)
        self.normalizer = normalizer

        self.layers = []
        self.trainable_layers = []

    def add(self, layer: Union[Layer, Activation]) -> None:
        self.layers.append(layer)

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

    def compile(self, optimizer: Union[Optimizer, str] = "GD"):
        """Prepare optimizer and all trainable layers for training with given optimizer."""

        if not isinstance(optimizer, Optimizer):
            if not isinstance(optimizer, str):
                raise ValueError(
                    f"optimizer must be a string or inherit from Optimizer class, not {type(optimizer)}"
                )
            optimizer = optimizer_from_string(optimizer)

        self.optimizer = optimizer

        for layer in self.layers:
            if hasattr(layer, "weights"):
                self.optimizer.init(layer)

    def measure(
        self,
        training_inputs,
        training_targets,
        loss_function: Loss = None,
        metrics: List[Metric] = None,
        validation_inputs: List[List[float]] = [],
        validation_targets: List[List[float]] = [],
        normalize_input: bool = True,
    ):
        measurements = {}

        toutputs = [self.predict(inp, normalize_input) for inp in training_inputs]
        voutputs = [self.predict(inp, normalize_input) for inp in validation_inputs]
        ttargets = training_targets
        vtargets = validation_targets

        # Measure loss
        if loss_function is not None:

            loss = 0.0
            for output, target in zip(toutputs, ttargets):
                loss += loss_function(output, target)
            measurements["loss"] = loss / len(toutputs)

            if len(validation_inputs) > 0:
                loss = 0.0
                for output, target in zip(voutputs, vtargets):
                    loss += loss_function(output, target)
                measurements["val_loss"] = loss / len(voutputs)

        # Measure metrics
        if metrics is not None:
            for metric in metrics:
                measurements[metric.name] = metric(toutputs, ttargets)

                if len(validation_inputs) > 0:
                    measurements["val_" + metric.name] = metric(voutputs, vtargets)

        return measurements

    def train(
        self,
        training_inputs: List[List[float]],
        training_targets: List[List[float]],
        epochs: int,
        loss_function: Union[Loss, str],
        batch_size: int = 1,
        base_learning_rate: float = 1e-4,
        learning_rate_decay: Union[Decay, str, None] = "linear",
        metrics: List[str] = ["mae"],
        validation_inputs: List[List[float]] = [],
        validation_targets: List[List[float]] = [],
        session_name: str = "",
    ) -> Dict:

        if len(self.layers) == 0:
            raise Exception("Model has no layers")

        if not hasattr(self, "optimizer"):
            raise Exception("Model needs optimizer for training. Compile model first")

        # Prepare learning rate decay
        if learning_rate_decay is None:
            decay = lambda _: base_learning_rate
        elif isinstance(learning_rate_decay, Decay):
            decay = learning_rate_decay
        else:
            if not isinstance(learning_rate_decay, str):
                raise ValueError(
                    f"learning_rate_decay must be a string or inherit from Decay class, not {type(learning_rate_decay)}"
                )
            decay = decay_from_string(learning_rate_decay, base_learning_rate, epochs)

        # Prepare metrics
        loaded_metrics = []
        for metric in metrics:
            if not isinstance(metric, Metric):
                if not isinstance(metric, str):
                    raise ValueError(
                        f"metric must be a string or inherit from Metric class, not {type(metric)}"
                    )
                metric = metric_from_string(metric)
            loaded_metrics.append(metric)
        metrics = loaded_metrics

        # Prepare loss function
        if not isinstance(loss_function, Loss):
            if not isinstance(loss_function, str):
                raise ValueError(
                    f"loss_function must be a string or inherit from Loss class, not {type(loss_function)}"
                )
            loss_function = loss_from_string(loss_function)

        # Adapt normalizer and normalize data if normalizer is defined
        if self.normalizer is not None:
            self.normalizer.adapt(training_inputs, clean=False)

            tinputs = [self.normalizer(inp) for inp in training_inputs]
            vinputs = [self.normalizer(inp) for inp in validation_inputs]
        else:
            tinputs = training_inputs.copy()
            vinputs = validation_inputs

        ttargets = training_targets.copy()
        vtargets = validation_targets

        # Prepare history dict
        class History(dict):
            def add(self, record: Dict):
                for key, val in record.items():
                    if key in self:
                        self[key].append(val)
                    else:
                        self[key] = [val]

            def get_last(self):
                return {key: val[-1] for key, val in self.items()}

        history = History()

        # Measure performance before training
        measurements = self.measure(
            tinputs,
            ttargets,
            loss_function,
            metrics,
            vinputs,
            vtargets,
            False,
        )
        history.add(measurements)

        # Setup progress bar
        if session_name:
            session_name = f" {session_name}"

        progress = tqdm(
            range(epochs),
            unit="epochs",
            bar_format=f"Training{session_name}: "
            "{percentage:3.0f}% |{bar:40}| {n_fmt}/{total_fmt}{postfix}",
        )
        progress.set_postfix(**measurements)

        for epoch in progress:

            tinputs, ttargets = data_util.shuffle(tinputs, ttargets)
            learning_rate = decay(epoch)

            sample_counter = 1
            for inputs, targets in zip(tinputs, ttargets):

                # Forward pass through the model
                state = inputs
                for layer in self.layers:
                    if isinstance(layer, Layer):
                        state = self.optimizer.forward_pass(layer, state)
                    else:
                        state = layer.forward_pass(state)
                outputs = state

                # Loss derivative for sample
                dloss = loss_function.derivative(outputs, targets)

                # Backward pass through the model
                dstate = dloss

                for layer in reversed(self.layers):
                    dstate = layer.backprop(dstate)

                # Perform the update
                is_batch = sample_counter % batch_size == 0
                is_last = sample_counter == len(tinputs)

                if is_batch or is_last:
                    for layer in self.layers:
                        if isinstance(layer, Layer):
                            self.optimizer.update(layer, learning_rate, batch_size)

                sample_counter += 1

            # Measure performance
            measurements = self.measure(
                tinputs,
                ttargets,
                loss_function,
                metrics,
                vinputs,
                vtargets,
                False,
            )
            history.add(measurements)

            progress.set_postfix(**measurements)

        return history


def cross_validation(
    model_factory: Callable[[], Model],
    inputs: List[List[float]],
    targets: List[List[float]],
    fold_count: int,
    epochs: int,
    loss_function: Union[Loss, str],
    optimizer: Union[Optimizer, str],
    batch_size: int,
    base_learning_rate: float = 1e-4,
    learning_rate_decay: Union[Decay, str, None] = "linear",
    metrics: List[str] = ["mae"],
) -> List[Dict]:

    folds = data_util.kfold_split(
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

        model = model_factory()
        model.compile(optimizer)

        run = model.train(
            training_inputs=train_inputs,
            training_targets=train_targets,
            epochs=epochs,
            loss_function=loss_function,
            batch_size=batch_size,
            base_learning_rate=base_learning_rate,
            learning_rate_decay=learning_rate_decay,
            metrics=metrics,
            validation_inputs=test_inputs,
            validation_targets=test_targets,
        )
        history.append(run)
    return history

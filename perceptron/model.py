import random
from math import ceil
from typing import Callable, Dict, List, Tuple, Union

from tqdm import tqdm

from perceptron import data_util
from perceptron.layer import Layer
from perceptron.activation import Activation
from perceptron.dropout import Dropout
from perceptron.decay import Decay, decay_from_string
from perceptron.loss import Loss, loss_from_string
from perceptron.metric import Metric, metric_from_string
from perceptron.normalizer import Normalizer, normalizer_from_string
from perceptron.optimizer import Optimizer, optimizer_from_string


class Model:
    """Perceptron model.

    Main class that joins everything together."""

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
        """Create an empty model. Initialize input normalizer if specified."""

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

    def add(self, layer: Union[Layer, Activation, Dropout]) -> None:
        self.layers.append(layer)

    # This step could be moved to training, but it is more flexible to initialize model optimizer explicitly
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

    def forward_pass(
        self, inputs: List[float], normalize_input: bool = False
    ) -> List[float]:
        """Forward pass through the whole model.

        This method should be used for training. To make predictions use predict."""

        if normalize_input and self.normalizer is not None:
            inputs = self.normalizer(inputs)

        state = inputs
        for layer in self.layers:
            if isinstance(layer, Layer):
                # In case of Nesterov optimizer, forward_pass needs to be overriden
                state = self.optimizer.forward_pass(layer, state)
            elif isinstance(layer, Dropout):
                state = layer.forward_pass(state, training=True)
            else:
                state = layer.forward_pass(state)
        outputs = state

        return outputs

    def backprop(self, outputs_gradients: List[float]) -> None:
        """Backpropagade gradients through the whole model."""

        for layer in reversed(self.layers):
            outputs_gradients = layer.backprop(outputs_gradients)
        return outputs_gradients

    def update(self, learning_rate: float, batch_size: int) -> None:
        """Update all trainable layers of the model using accumulated gradients."""

        for layer in self.layers:
            if isinstance(layer, Layer):
                self.optimizer.update(layer, learning_rate, batch_size)

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
    model: Model,
    training_inputs: List[List[float]],
    training_targets: List[List[float]],
    validation_inputs: List[List[float]] = [],
    validation_targets: List[List[float]] = [],
    loss_function: Loss = None,
    normalize_input: bool = True,
    metrics: List[Metric] = None,
    include_l1_loss: bool = False,
    include_l2_loss: bool = False,
) -> Dict[str, List[float]]:
    """Helper function to measure model performance during training.

    Note that l1 and l2 losses are not added to the loss of a loss function."""

    measurements = {}

    toutputs = [model.predict(inp, normalize_input) for inp in training_inputs]
    voutputs = [model.predict(inp, normalize_input) for inp in validation_inputs]
    ttargets = training_targets
    vtargets = validation_targets

    # Measure l1 loss
    if include_l1_loss:
        l1 = 0.0
        for layer in model.layers:
            if isinstance(layer, Layer):
                l1 += layer.l1_regularization_loss()
        measurements["l1_loss"] = l1

    # Measure l2 loss
    if include_l2_loss:
        l2 = 0.0
        for layer in model.layers:
            if isinstance(layer, Layer):
                l2 += layer.l2_regularization_loss()
        measurements["l2_loss"] = l2

    # Measure loss
    if loss_function is not None:
        loss = loss_function.calculate_avg(toutputs, ttargets)
        measurements["loss"] = loss

        if len(validation_inputs) > 0:
            loss = loss_function.calculate_avg(voutputs, vtargets)
            measurements["val_loss"] = loss

    # Measure metrics
    if metrics is not None:
        for m in metrics:
            measurements[m.name] = m.calculate_avg(toutputs, ttargets)

            if len(validation_inputs) > 0:
                measurements["val_" + m.name] = m.calculate_avg(voutputs, vtargets)

    return measurements


class History(dict):
    """Helper class to keep track of model training history."""

    def add(self, record: Dict):
        for key, val in record.items():
            if key in self:
                self[key].append(val)
            else:
                self[key] = [val]


def train(
    model: Model,
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
    include_l1_loss_in_history: bool = False,
    include_l2_loss_in_history: bool = False,
    session_name: str = "",
) -> Dict:
    """Implementation of a training loop."""

    # Check if model has any layers
    if len(model.layers) == 0:
        raise Exception("model has no layers")

    # Check if model has any trainable layers
    for layer in model.layers:
        if isinstance(layer, Layer):
            break
    else:
        raise Exception("model has no trainable layers")

    # Check if model was compiled (has optimizer)
    if not hasattr(model, "optimizer"):
        raise Exception("model needs optimizer for training. Compile model first")

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
    if model.normalizer is not None:
        model.normalizer.adapt(training_inputs, clean=False)

        tinputs = [model.normalizer(inp) for inp in training_inputs]
        vinputs = [model.normalizer(inp) for inp in validation_inputs]
    else:
        tinputs = training_inputs.copy()
        vinputs = validation_inputs

    ttargets = training_targets.copy()
    vtargets = validation_targets

    # Prepare history dict
    history = History()

    # Measure performance before training
    measurements = measure(
        model=model,
        training_inputs=tinputs,
        training_targets=ttargets,
        validation_inputs=vinputs,
        validation_targets=vtargets,
        loss_function=loss_function,
        metrics=metrics,
        normalize_input=False,
        include_l1_loss=include_l1_loss_in_history,
        include_l2_loss=include_l2_loss_in_history,
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
            outputs = model.forward_pass(inputs)

            # Loss derivative for sample
            dloss = loss_function.derivative(outputs, targets)

            # Backward pass through the model
            model.backprop(dloss)

            # Perform the update
            is_batch = sample_counter % batch_size == 0
            is_last = sample_counter == len(tinputs)

            if is_batch or is_last:
                model.update(learning_rate, batch_size)

            sample_counter += 1

        # Measure performance
        measurements = measure(
            model=model,
            training_inputs=tinputs,
            training_targets=ttargets,
            validation_inputs=vinputs,
            validation_targets=vtargets,
            loss_function=loss_function,
            metrics=metrics,
            normalize_input=False,
            include_l1_loss=include_l1_loss_in_history,
            include_l2_loss=include_l2_loss_in_history,
        )
        history.add(measurements)

        progress.set_postfix(**measurements)

    return history


def cross_validate(
    model_factory: Callable[[], Model],
    inputs: List[List[float]],
    targets: List[List[float]],
    fold_count: int,
    epochs: int,
    loss_function: Union[Loss, str],
    optimizer: Union[Optimizer, str],
    batch_size: int = 1,
    base_learning_rate: float = 1e-4,
    learning_rate_decay: Union[Decay, str, None] = "linear",
    metrics: List[str] = ["mae"],
    include_l1_loss_in_history: bool = False,
    include_l2_loss_in_history: bool = False,
) -> List[Dict]:
    """Implementation of k-fold cross-validation algorithm."""

    folds = data_util.kfold_split(
        inputs, targets, fold_count, stratified=True, random=True
    )

    history = {}
    for fold_idx, test_fold in enumerate(folds):
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

        run = train(
            model=model,
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
            include_l1_loss_in_history=include_l1_loss_in_history,
            include_l2_loss_in_history=include_l2_loss_in_history,
        )
        history[f"fold_{fold_idx}"] = run
    return history

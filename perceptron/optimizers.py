from typing import List
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, *args, **kwargs):
        """Initialize all necessary parameters."""

    def init(self, model: "Perceptron") -> None:
        """Store reference to the model. Initialize neurons parameters if needed.

        Invoking this method is necessary in order to use any optimizer."""

        self.model = model
        self.batch_size = 0

        # Initialize gradients with zeros
        self.forget_gradient()

    def forward_pass(self, inputs: List[float]) -> List[float]:
        """Pass inputs through the model.

        Each neuron has to store its inputs and output.
        It is needed for backprop and weight update."""

        layers = self.model.layers
        activations = self.model.activations

        for layer, activation in zip(layers, activations):
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

        return output

    def backprop(self, targets: List[float]) -> None:
        """Calculate error and propagate it backwards through the model.

        Each neuron has to store its error needed for weight update."""

        derivatives = self.model.derivatives
        layers = self.model.layers
        *hidden_layers, output_layer = layers

        predictions = [neuron.output for neuron in output_layer]
        loss_derivatives = self.model.loss_function.partial_derivatives(
            predictions, targets
        )

        for neuron, loss in zip(output_layer, loss_derivatives):
            neuron.error = loss * derivatives[-1](neuron.output)

        for layer_index in reversed(range(len(hidden_layers))):
            for neuron_index, neuron in enumerate(layers[layer_index]):

                neuron.error = 0.0
                for front_neuron in layers[layer_index + 1]:
                    neuron.error += (
                        front_neuron.weights[neuron_index] * front_neuron.error
                    )
                neuron.error *= derivatives[layer_index](neuron.output)

    def accumulate_gradient(self):
        for layer in self.model.layers:
            for neuron in layer:
                for index, inp in enumerate(neuron.inputs):
                    neuron.gradients[index] += neuron.error * inp
                neuron.bias_gradient += neuron.error
                neuron.error = 0.0

        self.batch_size += 1

    def forget_gradient(self):
        for layer in self.model.layers:
            for neuron in layer:
                neuron.gradients = [0.0] * len(neuron.weights)
                neuron.bias_gradient = 0.0

        self.batch_size = 0

    def update(self, learning_rate: float) -> List[float]:
        """Update weights based on calculated gradient. Synonym to __call__ method."""
        return self(learning_rate)

    @abstractmethod
    def __call__(self, learning_rate: float) -> List[float]:
        """Update weights based on calculated gradient. Synonym to update method."""


class GD(Optimizer):
    def __call__(self, learning_rate: float) -> None:
        for layer in self.model.layers:
            for neuron in layer:
                neuron.weights = [
                    weight - learning_rate * gradient / self.batch_size
                    for weight, gradient in zip(neuron.weights, neuron.gradients)
                ]
                neuron.bias -= learning_rate * neuron.bias_gradient / self.batch_size


class Momentum(Optimizer):
    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma

    def init(self, model: "Perceptron") -> None:
        super().init(model)

        for layer in self.model.layers:
            for neuron in layer:
                neuron.velocities = [0] * len(neuron.weights)
                neuron.bias_velocity = 0

    def __call__(self, learning_rate: float) -> None:
        for layer in self.model.layers:
            for neuron in layer:
                for index, grad in enumerate(neuron.gradients):
                    grad /= self.batch_size

                    neuron.velocities[index] *= self.gamma
                    neuron.velocities[index] += learning_rate * grad

                    neuron.weights[index] -= neuron.velocities[index]

                bias_grad = neuron.bias_gradient / self.batch_size
                neuron.bias_velocity = (
                    neuron.bias_velocity * self.gamma + learning_rate * bias_grad
                )
                neuron.bias -= neuron.bias_velocity


class Nesterov(Optimizer):
    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma

    def init(self, model: "Perceptron") -> None:
        super().init(model)

        for layer in self.model.layers:
            for neuron in layer:
                neuron.velocities = [0.0] * len(neuron.weights)
                neuron.bias_velocity = 0.0

    def forward_pass(self, inputs: List[float]) -> List[float]:
        for layer in self.model.layers:
            for neuron in layer:
                neuron.weights_cache = neuron.weights
                neuron.weights = [
                    weight - self.gamma * velocity
                    for weight, velocity in zip(neuron.weights, neuron.velocities)
                ]

        output = super().forward_pass(inputs)

        for layer in self.model.layers:
            for neuron in layer:
                neuron.weights = neuron.weights_cache

        return output

    def __call__(self, learning_rate: float) -> None:
        for layer in self.model.layers:
            for neuron in layer:

                for index, grad in enumerate(neuron.gradients):
                    grad /= self.batch_size
                    neuron.velocities[index] *= self.gamma
                    neuron.velocities[index] += learning_rate * grad

                    neuron.weights[index] -= neuron.velocities[index]

                bias_grad = neuron.bias_gradient / self.batch_size
                neuron.bias_velocity = (
                    neuron.bias_velocity * self.gamma + learning_rate * bias_grad
                )
                neuron.bias -= neuron.bias_velocity


class Adagrad(Optimizer):
    def __init__(self, epsilon: float = 1e-8, initial_accumulator_value: float = 0.1):
        self.epsilon = epsilon
        self.init_acc = initial_accumulator_value

    def init(self, model: "Perceptron") -> None:
        super().init(model)

        for layer in self.model.layers:
            for neuron in layer:
                neuron.accumulator = [self.init_acc] * len(neuron.weights)
                neuron.bias_accumulator = self.init_acc

    def __call__(self, learning_rate: float) -> None:
        for layer in self.model.layers:
            for neuron in layer:
                for index, grad in enumerate(neuron.gradients):
                    grad /= self.batch_size

                    neuron.accumulator[index] += grad**2
                    scale = self.epsilon + neuron.accumulator[index] ** 0.5
                    neuron.weights[index] -= learning_rate * grad / scale

                bias_grad = neuron.bias_gradient / self.batch_size
                neuron.bias_accumulator += bias_grad**2

                bias_scale = self.epsilon + neuron.bias_accumulator**0.5
                neuron.bias -= learning_rate * bias_grad / bias_scale


class RMSprop(Optimizer):
    def __init__(
        self,
        epsilon: float = 1e-8,
        initial_accumulator_value: float = 0.0,
        decay_rate: float = 0.9,
    ):
        self.epsilon = epsilon
        self.init_acc = initial_accumulator_value
        self.decay_rate = decay_rate

    def init(self, model: "Perceptron") -> None:
        super().init(model)

        for layer in self.model.layers:
            for neuron in layer:
                neuron.accumulator = [self.init_acc] * len(neuron.weights)
                neuron.bias_accumulator = self.init_acc

    def __call__(self, learning_rate: float) -> None:
        for layer in self.model.layers:
            for neuron in layer:
                for index, grad in enumerate(neuron.gradients):
                    grad /= self.batch_size

                    neuron.accumulator[index] *= self.decay_rate
                    neuron.accumulator[index] += (1 - self.decay_rate) * grad**2

                    scale = self.epsilon + neuron.accumulator[index] ** 0.5
                    neuron.weights[index] -= learning_rate * grad / scale

                bias_grad = neuron.bias_gradient / self.batch_size
                neuron.bias_accumulator *= self.decay_rate
                neuron.bias_accumulator += (1 - self.decay_rate) * bias_grad**2

                bias_scale = self.epsilon + neuron.bias_accumulator**0.5
                neuron.bias -= learning_rate * bias_grad / bias_scale


class Adam(Optimizer):
    def __init__(
        self, epsilon: float = 1e-8, beta_1: float = 0.9, beta_2: float = 0.999
    ):
        self.epsilon = epsilon
        self.b1 = beta_1
        self.b2 = beta_2

        self.step = 1

    def init(self, model: "Perceptron") -> None:
        super().init(model)

        for layer in self.model.layers:
            for neuron in layer:
                neuron.first_moment_accumulator = [0.0] * len(neuron.weights)
                neuron.second_moment_accumulator = [0.0] * len(neuron.weights)
                neuron.first_moment_bias_accumulator = 0.0
                neuron.second_moment_bias_accumulator = 0.0

    def __call__(self, learning_rate: float) -> None:
        for layer in self.model.layers:
            for neuron in layer:
                for index, grad in enumerate(neuron.gradients):
                    grad /= self.batch_size

                    neuron.first_moment_accumulator[index] *= self.b1
                    neuron.first_moment_accumulator[index] += (1 - self.b1) * grad

                    neuron.second_moment_accumulator[index] *= self.b2
                    neuron.second_moment_accumulator[index] += (1 - self.b2) * grad**2

                    f_corrected = neuron.first_moment_accumulator[index] / (
                        1 - self.b1**self.step
                    )
                    s_corrected = neuron.second_moment_accumulator[index] / (
                        1 - self.b2**self.step
                    )

                    neuron.weights[index] -= (
                        learning_rate
                        * f_corrected
                        / (self.epsilon + s_corrected**0.5)
                    )

                bias_grad = neuron.bias_gradient / self.batch_size
                neuron.first_moment_bias_accumulator *= self.b1
                neuron.first_moment_bias_accumulator += (1 - self.b1) * bias_grad

                neuron.second_moment_bias_accumulator *= self.b2
                neuron.second_moment_bias_accumulator += (1 - self.b2) * bias_grad**2

                f_corrected = neuron.first_moment_bias_accumulator / (
                    1 - self.b1**self.step
                )
                s_corrected = neuron.second_moment_bias_accumulator / (
                    1 - self.b2**self.step
                )

                neuron.bias -= (
                    learning_rate * f_corrected / (self.epsilon + s_corrected**0.5)
                )

        self.step += 1

from typing import List
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, *args, **kwargs):
        """Initialize all necessary parameters."""

    def init(self, model: "Perceptron") -> None:
        """Store reference to the model. Initialize neurons parameters if needed.

        Invoking this method is necessary in order to use any optimizer."""

        self.model = model

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

        for neuron, target in zip(output_layer, targets):
            neuron.error = (target - neuron.output) * derivatives[-1](neuron.output)

        for layer_index in reversed(range(len(hidden_layers))):
            for neuron_index, neuron in enumerate(layers[layer_index]):

                neuron.error = 0.0
                for front_neuron in layers[layer_index + 1]:
                    neuron.error += (
                        front_neuron.weights[neuron_index] * front_neuron.error
                    )
                neuron.error *= derivatives[layer_index](neuron.output)

    def update(
        self, inputs: List[float], targets: List[float], learning_rate: float
    ) -> List[float]:
        """Update weights. Synonym to __call__."""
        return self(inputs, targets, learning_rate)

    @abstractmethod
    def __call__(
        self, inputs: List[float], targets: List[float], learning_rate: float
    ) -> List[float]:
        """Update weights."""


class SGD(Optimizer):
    def __call__(
        self, inputs: List[float], targets: List[float], learning_rate: float
    ) -> List[float]:
        output = self.forward_pass(inputs)
        self.backprop(targets)

        for layer in self.model.layers:
            for neuron in layer:
                neuron.weights = [
                    weight + learning_rate * neuron.error * inp
                    for weight, inp in zip(neuron.weights, neuron.inputs)
                ]
                neuron.bias += learning_rate * neuron.error

        return output


class Momentum(Optimizer):
    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma

    def init(self, model: "Perceptron") -> None:
        super().init(model)

        for layer in self.model.layers:
            for neuron in layer:
                neuron.velocities = [0] * len(neuron.weights)
                neuron.bias_velocity = 0

    def __call__(
        self, inputs: List[float], targets: List[float], learning_rate: float
    ) -> List[float]:
        output = self.forward_pass(inputs)
        self.backprop(targets)

        for layer in self.model.layers:
            for neuron in layer:
                lre = learning_rate * neuron.error

                for weight_index, inp in enumerate(neuron.inputs):
                    neuron.velocities[weight_index] *= self.gamma
                    neuron.velocities[weight_index] += lre * inp

                    neuron.weights[weight_index] += neuron.velocities[weight_index]

                neuron.bias_velocity = neuron.bias_velocity * self.gamma + lre
                neuron.bias += neuron.bias_velocity

        return output


class Nesterov(Optimizer):
    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma

    def init(self, model: "Perceptron") -> None:
        super().init(model)

        for layer in self.model.layers:
            for neuron in layer:
                neuron.velocities = [0] * len(neuron.weights)
                neuron.bias_velocity = 0

    def __call__(
        self, inputs: List[float], targets: List[float], learning_rate: float
    ) -> List[float]:

        for layer in self.model.layers:
            for neuron in layer:
                neuron.weights_cache = neuron.weights
                neuron.weights = [
                    weight + self.gamma * velocity
                    for weight, velocity in zip(neuron.weights, neuron.velocities)
                ]

        output = self.forward_pass(inputs)
        self.backprop(targets)

        for layer in self.model.layers:
            for neuron in layer:
                neuron.weights = neuron.weights_cache

        for layer in self.model.layers:
            for neuron in layer:
                lre = learning_rate * neuron.error

                for weight_index, inp in enumerate(neuron.inputs):
                    neuron.velocities[weight_index] *= self.gamma
                    neuron.velocities[weight_index] += lre * inp

                    neuron.weights[weight_index] += neuron.velocities[weight_index]

                neuron.bias_velocity = neuron.bias_velocity * self.gamma + lre
                neuron.bias += neuron.bias_velocity

        return output


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

    def __call__(
        self, inputs: List[float], targets: List[float], learning_rate: float
    ) -> List[float]:
        output = self.forward_pass(inputs)
        self.backprop(targets)

        for layer in self.model.layers:
            for neuron in layer:
                for weight_index, inp in enumerate(neuron.inputs):
                    gradient = neuron.error * inp
                    neuron.accumulator[weight_index] += gradient**2

                    scale = self.epsilon + neuron.accumulator[weight_index] ** 0.5
                    neuron.weights[weight_index] += learning_rate * gradient / scale

                bias_gradient = neuron.error
                neuron.bias_accumulator += bias_gradient**2

                bias_scale = self.epsilon + neuron.bias_accumulator**0.5
                neuron.bias += learning_rate * bias_gradient / bias_scale

        return output


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

    def __call__(
        self, inputs: List[float], targets: List[float], learning_rate: float
    ) -> List[float]:
        output = self.forward_pass(inputs)
        self.backprop(targets)

        for layer in self.model.layers:
            for neuron in layer:
                for weight_index, inp in enumerate(neuron.inputs):
                    gradient = neuron.error * inp
                    neuron.accumulator[weight_index] *= self.decay_rate
                    neuron.accumulator[weight_index] += (
                        1 - self.decay_rate
                    ) * gradient**2

                    scale = self.epsilon + neuron.accumulator[weight_index] ** 0.5
                    neuron.weights[weight_index] += learning_rate * gradient / scale

                bias_gradient = neuron.error
                neuron.bias_accumulator *= self.decay_rate
                neuron.bias_accumulator += (1 - self.decay_rate) * bias_gradient**2

                bias_scale = self.epsilon + neuron.bias_accumulator**0.5
                neuron.bias += learning_rate * bias_gradient / bias_scale

        return output


class Adam(Optimizer):
    def __init__(
        self, epsilon: float = 1e-8, beta_1: float = 0.9, beta_2: float = 0.999
    ):
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.step = 1

    def init(self, model: "Perceptron") -> None:
        super().init(model)

        for layer in self.model.layers:
            for neuron in layer:
                neuron.first_moment_accumulator = [0.0] * len(neuron.weights)
                neuron.second_moment_accumulator = [0.0] * len(neuron.weights)
                neuron.first_moment_bias_accumulator = 0.0
                neuron.second_moment_bias_accumulator = 0.0

    def __call__(
        self, inputs: List[float], targets: List[float], learning_rate: float
    ) -> List[float]:
        output = self.forward_pass(inputs)
        self.backprop(targets)

        for layer in self.model.layers:
            for neuron in layer:
                for weight_index, inp in enumerate(neuron.inputs):
                    gradient = neuron.error * inp

                    neuron.first_moment_accumulator[weight_index] *= self.beta_1
                    neuron.first_moment_accumulator[weight_index] += (
                        1 - self.beta_1
                    ) * gradient

                    neuron.second_moment_accumulator[weight_index] *= self.beta_2
                    neuron.second_moment_accumulator[weight_index] += (
                        1 - self.beta_2
                    ) * gradient**2

                    f_corrected = neuron.first_moment_accumulator[weight_index] / (
                        1 - self.beta_1**self.step
                    )
                    s_corrected = neuron.second_moment_accumulator[weight_index] / (
                        1 - self.beta_2**self.step
                    )

                    neuron.weights[weight_index] += (
                        learning_rate
                        * f_corrected
                        / (self.epsilon + s_corrected**0.5)
                    )

                bias_gradient = neuron.error
                neuron.first_moment_bias_accumulator *= self.beta_1
                neuron.first_moment_bias_accumulator += (
                    1 - self.beta_1
                ) * bias_gradient

                neuron.second_moment_bias_accumulator *= self.beta_2
                neuron.second_moment_bias_accumulator += (
                    1 - self.beta_2
                ) * bias_gradient**2

                f_corrected = neuron.first_moment_bias_accumulator / (
                    1 - self.beta_1**self.step
                )
                s_corrected = neuron.second_moment_bias_accumulator / (
                    1 - self.beta_2**self.step
                )

                neuron.bias += (
                    learning_rate * f_corrected / (self.epsilon + s_corrected**0.5)
                )

        self.step += 1

        return output

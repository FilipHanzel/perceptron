from math import exp, tanh
from abc import ABC, abstractmethod
from typing import List


class Activation:
    def __init__(self):
        self.outputs: List[float] = None
        self.inputs_gradients: List[float] = None

    @abstractmethod
    def forward_pass(self, inputs: List[float]) -> List[float]:
        """Calculate the activation value."""

    @abstractmethod
    def backprop(self, outputs_gradients: List[float]) -> List[float]:
        """Propagate error backwards through the function."""


class Heavyside(Activation):
    """Heavyside, binary step or unit step function.

    Heavyside activation is non-differentiable in 0
    and anywhere else the derivative is equal to 0
    thus is incompatible with backpropagation algorithm.
    This activation should be used only with single layer models."""

    def forward_pass(self, inputs: List[float]) -> List[float]:
        self.outputs = [1.0 if value >= 0.0 else 0.0 for value in inputs]
        return self.outputs

    def backprop(self, outputs_gradients: List[float]) -> List[float]:
        """This method returns input value for compatibility with weight updates."""
        return outputs_gradients


class Linear(Activation):
    def forward_pass(self, inputs: List[float]) -> List[float]:
        self.outputs = inputs.copy()
        return self.outputs

    def backprop(self, outputs_gradients: List[float]) -> List[float]:
        # Calculate the derivative
        outputs_derivatives = [1.0] * len(self.outputs)

        # Backpropagate outputs gradients through the activation
        self.inputs_gradients = [
            doutput * gradient
            for doutput, gradient in zip(outputs_derivatives, outputs_gradients)
        ]

        return self.inputs_gradients


class Relu(Activation):
    """Rectified linear unit."""

    def forward_pass(self, inputs: List[float]) -> List[float]:
        self.outputs = [max(0.0, value) for value in inputs]
        return self.outputs

    def backprop(self, outputs_gradients: List[float]) -> List[float]:
        # Calculate the derivative
        outputs_derivatives = [1.0 if value >= 0.0 else 0.0 for value in self.outputs]

        # Backpropagate outputs gradients through the activation
        self.inputs_gradients = [
            doutput * gradient
            for doutput, gradient in zip(outputs_derivatives, outputs_gradients)
        ]

        return self.inputs_gradients


class LeakyRelu(Activation):
    """Leaky rectified linear unit."""

    def __init__(self, leak_coefficient: List[float] = 0.1):
        super().__init__()

        self.lc = leak_coefficient

    def forward_pass(self, inputs: List[float]) -> List[float]:
        self.outputs = [value if value >= 0.0 else self.lc * value for value in inputs]
        return self.outputs

    def backprop(self, outputs_gradients: List[float]) -> List[float]:
        # Calculate the derivative
        outputs_derivatives = [
            1.0 if value >= 0.0 else self.lc for value in self.outputs
        ]

        # Backpropagate outputs gradients through the activation
        self.inputs_gradients = [
            doutput * gradient
            for doutput, gradient in zip(outputs_derivatives, outputs_gradients)
        ]

        return self.inputs_gradients


class Sigmoid(Activation):
    """Logistic, sigmoid, or soft step function."""

    def forward_pass(self, inputs: List[float]) -> List[float]:
        self.outputs = [1.0 / (1.0 + exp(-value)) for value in inputs]
        return self.outputs

    def backprop(self, outputs_gradients: List[float]) -> List[float]:
        # Calculate the derivative
        outputs_derivatives = [output * (1.0 - output) for output in self.outputs]

        # Backpropagate outputs gradients through the activation
        self.inputs_gradients = [
            doutput * gradient
            for doutput, gradient in zip(outputs_derivatives, outputs_gradients)
        ]

        return self.inputs_gradients


class Tanh(Activation):
    """Hyperbolic tangent."""

    def forward_pass(self, inputs: List[float]) -> List[float]:
        self.outputs = [tanh(value) for value in inputs]
        return self.outputs

    def backprop(self, outputs_gradients: List[float]) -> List[float]:
        # Calculate the derivative
        outputs_derivatives = [1.0 - tanh(value) ** 2 for value in self.outputs]

        # Backpropagate outputs gradients through the activation
        self.inputs_gradients = [
            doutput * gradient
            for doutput, gradient in zip(outputs_derivatives, outputs_gradients)
        ]

        return self.inputs_gradients


class Softmax(Activation):
    """Softmax, softargmax or normalized exponential function."""

    def forward_pass(self, inputs: List[float]) -> List[float]:
        shifted_values = [value - max(inputs) for value in inputs]
        exps = [exp(value) for value in shifted_values]
        sum_ = sum(exps)
        self.outputs = [exp_ / sum_ for exp_ in exps]

        return self.outputs

    def backprop(self, outputs_gradients: List[float]) -> List[float]:
        # Calculate the derivative
        outputs_derivatives = [[0.0] * len(self.outputs) for _ in self.outputs]

        for i, ivalue in enumerate(self.outputs):
            for j, jvalue in enumerate(self.outputs):
                if i == j:
                    outputs_derivatives[i][j] += ivalue * (1 - jvalue)
                else:
                    outputs_derivatives[i][j] -= ivalue * jvalue

        # Backpropagate outputs gradients through the activation
        self.inputs_gradients = [
            sum([output * grad for output, grad in zip(outputs, outputs_gradients)])
            for outputs in outputs_derivatives
        ]

        return self.inputs_gradients

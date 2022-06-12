import unittest

from perceptron.activation import (
    Activation,
    Heavyside,
    Linear,
    Relu,
    LeakyRelu,
    Sigmoid,
    Tanh,
    Softmax,
)


class TestActivationsIntegrity(unittest.TestCase):
    def check_activation(self, activation: Activation):
        inputs = [0.0, 1.1, -1.0, 15.0, -15.4]

        forward_result = activation.forward_pass(inputs=inputs)
        self.assertIsInstance(forward_result, list)
        for element in forward_result:
            self.assertIsInstance(element, float)

        backprop_result = activation.backprop(outputs_gradients=inputs)
        self.assertIsInstance(backprop_result, list)
        for element in backprop_result:
            self.assertIsInstance(element, float)

        self.assertEqual(len(forward_result), len(backprop_result))

    def test_heavyside(self):
        activation = Heavyside()
        self.check_activation(activation)

    def test_linear(self):
        activation = Linear()
        self.check_activation(activation)

    def test_relu(self):
        activation = Relu()
        self.check_activation(activation)

    def test_leaky_relu(self):
        activation = LeakyRelu(leak_coefficient=0.4)
        self.check_activation(activation)

        activation = LeakyRelu()
        self.check_activation(activation)

    def test_sigmoid(self):
        activation = Sigmoid()
        self.check_activation(activation)

    def test_tanh(self):
        activation = Tanh()
        self.check_activation(activation)

    def test_softmax(self):
        activation = Softmax()
        self.check_activation(activation)

import unittest

from perceptron.layer import Layer


class TestLayerIntegrity(unittest.TestCase):
    def check_init_method(self, method):
        input_size = 5
        layer_size = 10
        layer = Layer(input_size, layer_size, init_method=method)

        self.assertEqual(len(layer.weights), layer_size)
        self.assertEqual(len(layer.biases), layer_size)

        for neuron in layer.weights:
            self.assertEqual(len(neuron), input_size)
            self.assertIsInstance(neuron, list)

            for weight in neuron:
                self.assertIsInstance(weight, float)

        for bias in layer.biases:
            self.assertIsInstance(bias, float)

    def test_uniform(self):
        self.check_init_method("uniform")

    def test_gauss(self):
        self.check_init_method("gauss")

    def test_zeros(self):
        self.check_init_method("zeros")

    def test_he(self):
        self.check_init_method("he")

    def test_xavier(self):
        self.check_init_method("xavier")

    def test_forward_pass_and_backprop(self):
        input_size = 5
        layer_size = 10

        layer = Layer(input_size, layer_size)
        layer.init_gradients()

        inputs = [0.0, 0.1, 100.0, -0.1, -0.005]
        result = layer.forward_pass(inputs=inputs)
        self.assertEqual(len(result), layer_size)
        for element in result:
            self.assertIsInstance(element, float)

        inputs = [0.0, 0.1, 100.0, -0.1, -0.005, 0.2, 0.3, 4.0, 0.1, 0.1]
        result = layer.backprop(outputs_gradients=inputs)
        self.assertEqual(len(result), input_size)
        for element in result:
            self.assertIsInstance(element, float)

    def test_inits(self):
        input_size = 5
        layer_size = 10
        inputs = [0.0, 0.1, 100.0, -0.1, -0.005]

        layer = Layer(input_size, layer_size)

        self.assertIsNone(layer.inputs)

        self.assertIsNone(layer.weights_gradients)
        self.assertIsNone(layer.biases_gradients)

        self.assertIsNone(layer.weights_velocities)
        self.assertIsNone(layer.biases_velocities)

        self.assertIsNone(layer.weights_cache)
        self.assertIsNone(layer.biases_cache)

        self.assertIsNone(layer.weights_accumulators)
        self.assertIsNone(layer.biases_accumulators)

        self.assertIsNone(layer.step)
        self.assertIsNone(layer.first_moment_weights_accumulators)
        self.assertIsNone(layer.second_moment_weights_accumulators)
        self.assertIsNone(layer.first_moment_biases_accumulators)
        self.assertIsNone(layer.second_moment_biases_accumulators)

        layer.forward_pass(inputs)
        self.assertIsNotNone(layer.inputs)

        layer.init_gradients()
        self.assertIsNotNone(layer.weights_gradients)
        self.assertIsNotNone(layer.biases_gradients)

        layer.init_velocities()
        self.assertIsNotNone(layer.weights_velocities)
        self.assertIsNotNone(layer.biases_velocities)

        layer.init_accumulators()
        self.assertIsNotNone(layer.weights_accumulators)
        self.assertIsNotNone(layer.biases_accumulators)

        layer.init_step()
        self.assertIsNotNone(layer.step)

        layer.init_first_and_second_moment_accumulators()
        self.assertIsNotNone(layer.first_moment_weights_accumulators)
        self.assertIsNotNone(layer.second_moment_weights_accumulators)
        self.assertIsNotNone(layer.first_moment_biases_accumulators)
        self.assertIsNotNone(layer.second_moment_biases_accumulators)

import unittest
from typing import List

from perceptron.loss import (
    Loss,
    MSE,
    MSLE,
    MAE,
    BinaryCrossentropy,
    CategoricalCrossentropy,
)


class TestLossIntegrity(unittest.TestCase):
    def check_loss(
        self, loss: Loss, test_outputs: List[float], test_targets: List[int]
    ):
        call_result = loss.calculate(outputs=test_outputs, targets=test_targets)
        self.assertIsInstance(call_result, float)

        derivatice_result = loss.derivative(test_outputs, test_targets)
        self.assertIsInstance(derivatice_result, list)
        self.assertEqual(len(derivatice_result), len(test_outputs))
        for element in derivatice_result:
            self.assertIsInstance(element, float)

    def test_mse(self):
        test_outputs = [0.0, 0.1, 0.2, -0.1, -0.2, -100.0, 102.0]
        test_targets = [0.1, -0.1, 0.0, -15.0, 0.0, -90.0, -2.0]

        loss = MSE()
        self.check_loss(loss, test_outputs, test_targets)

    def test_msle(self):
        test_outputs = [0.0, 0.1, 0.2, -0.1, -0.2, -0.999, 1.0]
        test_targets = [0.1, -0.1, 0.0, -0.999, 0.0, -0.5, -0.2]

        loss = MSLE()
        self.check_loss(loss, test_outputs, test_targets)

    def test_mae(self):
        test_outputs = [0.0, 0.1, 0.2, -0.1, -0.2, -100.0, 102.0]
        test_targets = [0.1, -0.1, 0.0, -15.0, 0.0, -90.0, -2.0]

        loss = MAE()
        self.check_loss(loss, test_outputs, test_targets)

    def test_binary_crossentropy(self):
        test_outputs = [0.0, 0.1, 0.2, -0.1, -0.2, -1.0, 1.0]
        test_targets = [0.1, -0.1, 0.0, -1.0, 0.0, -0.5, -0.2]

        loss = BinaryCrossentropy()
        self.check_loss(loss, test_outputs, test_targets)

    def test_categorical_crossentropy(self):
        test_outputs = [0.0, 0.1, 0.2, -0.1, -0.2, -1.0, 1.0]
        test_targets = [0.1, -0.1, 0.0, -1.0, 0.0, -0.5, -0.2]

        loss = CategoricalCrossentropy()
        self.check_loss(loss, test_outputs, test_targets)

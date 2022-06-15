import unittest

from perceptron.metric import (
    MAE,
    MAPE,
    MSE,
    RMSE,
    CosSim,
    BinaryAccuracy,
    CategoricalAccuracy,
    TopKCategoricalAccuracy,
)


class TestMAE(unittest.TestCase):
    def test(self):
        metric = MAE()

        result = metric.calculate_avg(
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.2, 1.8], [3.8, 4.0]],
        )
        self.assertAlmostEqual(result, 0.3, places=8)

        result = metric.calculate_avg(
            [[-2.2, 2.0], [3.0, 4.0]],
            [[1.2, 1.8], [3.8, 4.0]],
        )
        self.assertAlmostEqual(result, 1.1, places=8)


class TestMAPE(unittest.TestCase):
    def test(self):
        metric = MAPE()

        result = metric.calculate_avg(
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.2, 1.8], [3.8, 4.0]],
        )
        self.assertAlmostEqual(result, 0.12207602, places=8)

        result = metric.calculate_avg(
            [[-2.2, 2.0], [3.0, 4.0]],
            [[1.2, 1.8], [3.8, 4.0]],
        )
        self.assertAlmostEqual(result, 0.78874269, places=8)


class TestMSE(unittest.TestCase):
    def test(self):
        metric = MSE()

        result = metric.calculate_avg(
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.2, 1.8], [3.8, 4.0]],
        )
        self.assertAlmostEqual(result, 0.18, places=8)

        result = metric.calculate_avg(
            [[-2.2, 2.0], [3.0, 4.0]],
            [[1.2, 1.8], [3.8, 4.0]],
        )
        self.assertAlmostEqual(result, 3.06, places=8)


class TestRMSE(unittest.TestCase):
    def test(self):
        metric = RMSE()

        result = metric.calculate_avg(
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.2, 1.8], [3.8, 4.0]],
        )
        self.assertAlmostEqual(result, 0.382842712, places=8)

        result = metric.calculate_avg(
            [[-2.2, 2.0], [3.0, 4.0]],
            [[1.2, 1.8], [3.8, 4.0]],
        )
        self.assertAlmostEqual(result, 1.487002170, places=8)


class TestCosSim(unittest.TestCase):
    def test(self):
        metric = CosSim()

        result = metric.calculate_avg(
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.2, 1.8], [3.8, 4.0]],
        )
        self.assertAlmostEqual(result, 0.992763550, places=8)

        result = metric.calculate_avg(
            [[-2.2, 2.0], [3.0, 4.0]],
            [[1.2, 1.8], [3.8, 4.0]],
        )
        self.assertAlmostEqual(result, 0.571250958, places=8)


class TestBinaryAccuracy(unittest.TestCase):
    def test(self):
        metric = BinaryAccuracy()

        result = metric.calculate_avg(
            [[0.1], [0.3], [-0.1, 1.2, 0.51], [0.49]],
            [[0], [0], [1, 1, 0], [1]],
        )
        self.assertAlmostEqual(result, 7 / 12, places=8)

        result = metric.calculate_avg(
            [[0.1], [0.3], [-0.1, 1.2, 0.51], [0.49]],
            [[0.4], [0.1], [0.52, 0.6, 0.01], [0.78]],
        )
        self.assertAlmostEqual(result, 7 / 12, places=8)


class TestCategoricalAccuracy(unittest.TestCase):
    def test(self):
        metric = CategoricalAccuracy()

        result = metric.calculate_avg(
            [[0.1, 0.2], [0.3, 0.1], [-0.1, 1.2, 0.51], [0.3, 0.1]],
            [[0, 1], [1, 0], [0, 1, 0], [0, 1]],
        )
        self.assertAlmostEqual(result, 3 / 4, places=8)

        result = metric.calculate_avg(
            [[0.1, 0.2], [0.3, 0.1], [-0.1, 1.2, 0.51], [0.3, 0.1]],
            [[0.2, 0.9], [0.6, 0.3], [0.1, 0.3, 0.2], [0.6, 0.7]],
        )
        self.assertAlmostEqual(result, 3 / 4, places=8)


class TestTopKCategoricalAccuracy(unittest.TestCase):
    def test(self):
        metric = TopKCategoricalAccuracy()

        result = metric.calculate_avg(
            [[-0.1, 1.2, 0.51, 0.7]] * 4,
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
        self.assertAlmostEqual(result, 3 / 4, places=8)

import unittest

from perceptron.normalizers import MinMax, ZScore


class TestMinMax(unittest.TestCase):
    def setUp(self):
        self.normalizer = MinMax()

    def test_non_adapted(self):
        # return input if not adapted
        record = [-1, -2, 0, 3, 4]
        self.assertEqual(self.normalizer(record), record)
        record = []
        self.assertEqual(self.normalizer(record), record)
        record = [-1, -2, 0, 3, 4]
        self.assertEqual(self.normalizer(record, inverse=True), record)
        record = []
        self.assertEqual(self.normalizer(record, inverse=True), record)

    def test_adapted(self):
        # raise ValueError if column has only one unique value
        self.assertRaises(ValueError, self.normalizer.adapt, [[0], [0]])
        self.assertRaises(ValueError, self.normalizer.adapt, [[1], [1]])

        # correctly normalize and denormalize input after adapting
        self.normalizer.adapt([[-2, -2, 0, 3, 4], [-1, 1, 1, 1, 1]])
        self.assertEqual(self.normalizer([1, 1, 1, 1, 1]), [3, 1, 1, 0, 0])
        self.assertEqual(
            self.normalizer([3, 1, 1, 0, 0], inverse=True), [1, 1, 1, 1, 1]
        )

    def test_iterative_adaptation(self):
        # raise ValueError if readapted with records of different length
        self.normalizer.adapt([[0, 1], [2, 3], [-1, 0], [2, 2]])
        self.assertRaises(ValueError, self.normalizer.adapt, [[2]], clean=False)

        # correctly normalize and denormalize input after clean second adaptation
        expected_result = self.normalizer([0, 0])
        self.normalizer.adapt([[0, 1], [2, 3], [-1, 0], [2, 2]])
        self.assertEqual(self.normalizer([0, 0]), expected_result)
        self.assertEqual(self.normalizer(expected_result, inverse=True), [0, 0])

        # correctly normalize input after second adapt
        expected_result = self.normalizer([0, 0])
        self.normalizer.adapt([[0, 1], [2, 3], [-1, 0]], clean=True)
        self.normalizer.adapt([[2, 2]], clean=False)
        self.assertAlmostEqual(self.normalizer([0, 0]), expected_result, places=8)
        self.assertAlmostEqual(
            self.normalizer(expected_result, inverse=True), [0, 0], places=8
        )


class TestZScore(unittest.TestCase):
    def setUp(self):
        self.normalizer = ZScore()

    def test_non_adapted(self):
        # return input if not adapted
        record = [-1, -2, 0, 3, 4]
        self.assertEqual(self.normalizer(record), record)
        record = []
        self.assertEqual(self.normalizer(record), record)

    def test_adapted(self):
        # raise ValueError if column has only one unique value
        self.assertRaises(ValueError, self.normalizer.adapt, [[0], [0]])
        self.assertRaises(ValueError, self.normalizer.adapt, [[1], [1]])

        # correctly normalize input after adapting
        self.normalizer.adapt([[-2, -2, 0, 3, 4], [-1, 1, 1, 1, 1]])
        self.assertEqual(self.normalizer([1, 1, 1, 1, 1]), [5, 1, 1, -1, -1])
        self.assertEqual(
            self.normalizer([5, 1, 1, -1, -1], inverse=True), [1, 1, 1, 1, 1]
        )

    def test_iterative_adaptation(self):
        # raise ValueError if readapted with records of different length
        self.normalizer.adapt([[0, 1], [2, 3], [-1, 0], [2, 2]])
        self.assertRaises(ValueError, self.normalizer.adapt, [[2]], clean=False)

        # correctly normalize input after clean second adaptation
        expected_result = self.normalizer([0, 0])
        self.normalizer.adapt([[0, 1], [2, 3], [-1, 0], [2, 2]])
        self.assertEqual(self.normalizer([0, 0]), expected_result)
        self.assertEqual(self.normalizer(expected_result, inverse=True), [0, 0])

        # correctly normalize input after second adapt
        expected_result = self.normalizer([0, 0])
        self.normalizer.adapt([[0, 1], [2, 3], [-1, 0]], clean=True)
        self.normalizer.adapt([[2, 2]], clean=False)
        self.assertAlmostEqual(self.normalizer([0, 0]), expected_result, places=8)
        self.assertAlmostEqual(
            self.normalizer(expected_result, inverse=True), [0, 0], places=8
        )

import unittest

from perceptron.data_util import (
    transpose,
    clip,
    shuffle,
    kfold_split,
    to_binary,
    to_categorical,
)


class TestTranspose(unittest.TestCase):
    def test(self):
        data = [[1, 2, 3], [4, 5, 6]]

        result = transpose(data)
        self.assertEqual(result, [[1, 4], [2, 5], [3, 6]])

        result = transpose(result)
        self.assertEqual(result, data)

        self.assertNotEqual(id(result), id(data))


class TestClip(unittest.TestCase):
    def test(self):
        data = [1, 2, 3, 4]

        result = clip(data, 2, 3)
        self.assertEqual(result, [2, 2, 3, 3])

        result = clip(data, 1.5, 3.5)
        self.assertEqual(result, [1.5, 2, 3, 3.5])

        self.assertRaises(ValueError, clip, data, 2.5, 1.5)


class TestShuffle(unittest.TestCase):
    def test(self):
        inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        inputs_ids = [id(inp) for inp in inputs]

        targets = [["a"], ["b"], ["c"], ["d"], ["e"]]
        targets_ids = [id(target) for target in targets]

        shuffled_inputs, shuffled_targets = shuffle(inputs, targets)

        # shuffled targets should not be in the same order as original
        self.assertTrue(any(a != b for a, b in zip(targets, shuffled_targets)))

        # shuffled inputs list has to have all original inputs
        for inp in shuffled_inputs:
            self.assertTrue(inp in inputs)
            self.assertTrue(id(inp) in inputs_ids)

        # shuffled targets list has to have all original targets
        for target in shuffled_targets:
            self.assertTrue(target in targets)
            self.assertTrue(id(target) in targets_ids)

        # shuffle cannot mix up inputs and targets
        for inp, target in zip(inputs, targets):
            expected_target_index = shuffled_inputs.index(inp)
            self.assertEqual(target, shuffled_targets[expected_target_index])


class TestKFoldSplit(unittest.TestCase):
    def test(self):
        inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        inputs_ids = [id(inp) for inp in inputs]

        targets = [["a"], ["b"], ["a"], ["b"], ["c"]]
        targets_ids = [id(target) for target in targets]

        folds = kfold_split(inputs, targets, 2, stratified=True, random=True)

        # folds should be similar in size
        folds_lengths = [len(fold["inputs"]) for fold in folds]
        self.assertTrue(max(folds_lengths) - min(folds_lengths) <= 1)

        for fold in folds:
            # stratified shuffle should ensure close to equal split of targets
            self.assertEqual(fold["targets"].count(["a"]), 1)
            self.assertEqual(fold["targets"].count(["b"]), 1)

            # inputs list has to have all original inputs
            for inp in fold["inputs"]:
                self.assertTrue(inp in inputs)
                self.assertTrue(id(inp) in inputs_ids)

            # targets list has to have all original targets
            for target in fold["targets"]:
                self.assertTrue(target in targets)
                self.assertTrue(id(target) in targets_ids)

            # random picking cannot mix up inputs and targets
            for inp, target in zip(fold["inputs"], fold["targets"]):
                expected_target_index = inputs.index(inp)
                self.assertEqual(target, targets[expected_target_index])


class TestToBinary(unittest.TestCase):
    def test(self):
        mapping, encoded = to_binary(["a", "b", "a", "b", "b"])
        self.assertDictEqual(mapping, {"a": 0, "b": 1})
        self.assertListEqual(encoded, [[0], [1], [0], [1], [1]])

        mapping, encoded = to_binary(["b", "b", "a", "b", "b"])
        self.assertDictEqual(mapping, {"a": 0, "b": 1})
        self.assertListEqual(encoded, [[1], [1], [0], [1], [1]])

        mapping, encoded = to_binary([1, 2, 1, 2, 2])
        self.assertDictEqual(mapping, {1: 0, 2: 1})
        self.assertListEqual(encoded, [[0], [1], [0], [1], [1]])

        mapping, encoded = to_binary([2, 2, 1, 2, 2])
        self.assertDictEqual(mapping, {1: 0, 2: 1})
        self.assertListEqual(encoded, [[1], [1], [0], [1], [1]])


class TestToCategorical(unittest.TestCase):
    def test(self):
        mapping, encoded = to_categorical(["a", "b", "c", "c", "b"])
        self.assertDictEqual(mapping, {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]})
        self.assertListEqual(
            encoded, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0]]
        )

        mapping, encoded = to_categorical(["c", "b", "c", "a", "b"])
        self.assertDictEqual(mapping, {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]})
        self.assertListEqual(
            encoded, [[0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
        )

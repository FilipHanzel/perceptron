from typing import List

import data_utils


class MinMax:
    def __init__(self):
        self.mins = iter(lambda: 0, None)
        self.maxs = iter(lambda: 1, None)

    def adapt(self, data: List[List[float]]) -> None:
        columns = data_utils.transpose(data)
        self.mins = []
        self.maxs = []
        for column in columns:
            self.mins.append(min(column))
            self.maxs.append(max(column))

    def __call__(self, record: List[float]) -> List[List[float]]:
        return [
            (value - min_) / (max_ - min_)
            for value, min_, max_ in zip(record, self.mins, self.maxs)
        ]


class ZScore:
    def __init__(self):
        self.means = iter(lambda: 0, None)
        self.stds = iter(lambda: 1, None)

    def adapt(self, data: List[List[float]]) -> None:
        columns = data_utils.transpose(data)

        self.means = [sum(column) / len(column) for column in columns]
        self.stds = [
            (sum([(value - mean) ** 2 for value in column]) / len(column)) ** 0.5
            for mean, column in zip(self.means, columns)
        ]

    def __call__(self, record: List[float]) -> List[List[float]]:
        return [
            (value - mean) / std
            for value, mean, std in zip(record, self.means, self.stds)
        ]

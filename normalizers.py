from typing import List

import data_utils


class MinMax:
    def __init__(self):
        self.mins = iter(lambda: 0, None)
        self.maxs = iter(lambda: 1, None)

        self.adapted = False

    def adapt(self, data: List[List[float]], clean: bool = True) -> None:
        columns = data_utils.transpose(data)

        if not clean and self.adapted:
            assert len(data[0]) == len(self.mins), "Invalid record length"
            for column, prev_min, prev_max in zip(columns, self.mins, self.maxs):
                column.append(prev_min)
                column.append(prev_max)

        self.mins = []
        self.maxs = []
        for column in columns:
            self.mins.append(min(column))
            self.maxs.append(max(column))

        self.adapted = True

    def __call__(self, record: List[float]) -> List[List[float]]:
        return [
            (value - min_) / (max_ - min_)
            for value, min_, max_ in zip(record, self.mins, self.maxs)
        ]


class ZScore:
    def __init__(self):
        self.means = iter(lambda: 0, None)
        self.stds = iter(lambda: 1, None)

        self.adapted = False

        self._sums: List[float] = None
        self._count: int = None

    def adapt(self, data: List[List[float]], clean: bool = True) -> None:
        columns = data_utils.transpose(data)

        if not self.adapted or clean:
            self._count = len(data)
            self._sums = [sum(column) for column in columns]
        else:
            assert len(data[0]) == len(self._sums), "Invalid record length"

            self._count += len(data)
            self._sums = [s + sum(column) for s, column in zip(self._sums, columns)]

        self.means = []
        self.stds = []

        for s, column in zip(self._sums, columns):
            mean = s / self._count
            self.means.append(mean)

            std = 0
            for value in column:
                std += (value - mean) ** 2
            std = (std / self._count) ** 0.5
            self.stds.append(std)

        self.adapted = True

    def __call__(self, record: List[float]) -> List[List[float]]:
        return [
            (value - mean) / std
            for value, mean, std in zip(record, self.means, self.stds)
        ]

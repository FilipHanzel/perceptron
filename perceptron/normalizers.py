from typing import List
from abc import ABC, abstractmethod

from perceptron import data_utils


class Normalizer(ABC):
    def __init__(self, *args, **kwargs):
        """Initialize all necessary parameters."""

    @abstractmethod
    def adapt(self, data: List[List[float]], clean: bool = True) -> None:
        """Adjust normalization parameters to fit the data.

        If clean = True - reset parameters before adapting.
        Otherwise adjust already adapted parameters."""

    @abstractmethod
    def __call__(self, record: List[float], inverse: bool = False) -> List[float]:
        """Normalize record. If inverse is True - denormalize instead."""


class MinMax(Normalizer):
    def __init__(self):
        self.mins = iter(lambda: 0, None)
        self.maxs = iter(lambda: 1, None)

        self.adapted = False

    def adapt(self, data: List[List[float]], clean: bool = True) -> None:
        columns = data_utils.transpose(data)

        if not clean and self.adapted:
            if len(data[0]) != len(self.mins):
                raise ValueError("Invalid record length")

            for column, prev_min, prev_max in zip(columns, self.mins, self.maxs):
                column.append(prev_min)
                column.append(prev_max)

        self.mins = []
        self.maxs = []
        for column in columns:
            min_ = min(column)
            max_ = max(column)

            if min_ == max_:
                raise ValueError("Column has to contain more than one unique value")

            self.mins.append(min_)
            self.maxs.append(max_)

        self.adapted = True

    def __call__(self, record: List[float], inverse: bool = False) -> List[float]:
        if inverse:
            return [
                value * (max_ - min_) + min_
                for value, min_, max_ in zip(record, self.mins, self.maxs)
            ]
        else:
            return [
                (value - min_) / (max_ - min_)
                for value, min_, max_ in zip(record, self.mins, self.maxs)
            ]


class ZScore(Normalizer):
    def __init__(self):
        self.means = iter(lambda: 0, None)
        self.stds = iter(lambda: 1, None)

        self.adapted = False

        self._s2_buffer: List[float] = None
        self._count: int = None

    def adapt(self, data: List[List[float]], clean: bool = True) -> None:
        columns = data_utils.transpose(data)

        if not self.adapted or clean:
            self._count = len(data)
            self._sums = [sum(column) for column in columns]

            self.means = [s / self._count for s in self._sums]
            self.stds = []
            self._s2_buffer = []
            for column, mean in zip(columns, self.means):
                std = 0
                for value in column:
                    std += (value - mean) ** 2

                if std == 0:
                    raise ValueError("Column has to contain more than one unique value")

                self._s2_buffer.append(std)
                std = (std / self._count) ** 0.5

                self.stds.append(std)
        else:
            if len(data[0]) != len(self.means):
                raise ValueError("Invalid record length")

            # Update with Welford's algorithm
            old_means = self.means
            old_s2_buffer = self._s2_buffer

            self.means = []
            self._s2_buffer = []

            for column, old_mean, old_s2 in zip(columns, old_means, old_s2_buffer):
                count = self._count
                for value in column:
                    count += 1
                    new_mean = old_mean + (value - old_mean) / count
                    new_s2 = old_s2 + (value - old_mean) * (value - new_mean)

                    old_mean = new_mean
                    old_s2 = new_s2
                else:
                    self.means.append(new_mean)
                    self._s2_buffer.append(new_s2)

            self._count += len(data)
            self.stds = [(s2 / self._count) ** 0.5 for s2 in self._s2_buffer]

        self.adapted = True

    def __call__(self, record: List[float], inverse: bool = False) -> List[float]:
        if inverse:
            return [
                value * std + mean
                for value, mean, std in zip(record, self.means, self.stds)
            ]
        else:
            return [
                (value - mean) / std
                for value, mean, std in zip(record, self.means, self.stds)
            ]

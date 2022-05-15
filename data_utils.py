import random
from typing import List, Tuple, Union


def transpose(data: List[List]) -> List[List]:
    return [list(column) for column in zip(*data)]


def shuffle(features: List, targets: List) -> Tuple[List, List]:
    order = list(range(len(features)))
    random.shuffle(order)

    features = [features[index] for index in order]
    targets = [targets[index] for index in order]

    return features, targets


def normalize(dataset: List[List[float]]) -> List[List[float]]:
    row_length = len(dataset[0])

    total_min = [None] * row_length
    total_max = [None] * row_length
    for record in dataset:
        for idx, column in enumerate(record):
            if total_min[idx] is None or total_min[idx] > column:
                total_min[idx] = column
            if total_max[idx] is None or total_max[idx] < column:
                total_max[idx] = column

    normalized_dataset = [
        [
            (column - min_) / (max_ - min_)
            for column, min_, max_ in zip(record, total_min, total_max)
        ]
        for record in dataset
    ]

    return normalized_dataset


def drop_columns(dataset, column_index: Union[int, List[int]]):
    """Drop columns of the dataset (inplace). Supports negative indexes."""
    if isinstance(column_index, int):
        column_index = [column_index]

    record_length = len(dataset[0])

    # Support negative indexes
    for index in column_index:
        if not (-record_length <= index < record_length):
            raise IndexError("index out of range")
    column_index = [
        index if index >= 0 else record_length + index for index in column_index
    ]

    for index in sorted(column_index, reverse=True):
        for line in dataset:
            del line[index]

    return dataset


def encode_as_int(column: List[str]) -> List[int]:
    """Encode column values as 0 or 1."""
    values = set(column)
    assert len(values) == 2, "Too many values for binary encoding"

    mapping = {label: index for index, label in enumerate(values)}
    encoded = [mapping[value] for value in column]

    return mapping, encoded


def encode(column: List[str]) -> List[List[int]]:
    """Encode column values as binary vectors."""
    values = set(column)

    index_mapping = {label: index for index, label in enumerate(values)}

    encoded = [[0] * len(values) for _ in range(len(column))]
    for vector, value in zip(encoded, column):
        vector[index_mapping[value]] = 1

    mapping = {label: [0] * len(values) for label in index_mapping}
    for label, index in index_mapping.items():
        mapping[label][index] = 1

    return mapping, encoded
import os
import csv
import random
from typing import List, Dict, Union, Any


def load_mpg_data():
    with open(os.path.join("data", "auto-mpg-data.csv"), "rt") as f:
        dataset = [
            [float(value) for value in features] + [car_name]
            for *features, car_name in [
                line for line in csv.reader(f) if line and "?" not in line
            ]
        ]

    dataset = drop_columns(dataset, [-1])
    dataset = swap_target_placement(dataset, 0)
    random.shuffle(dataset)

    return normalize(dataset)


def load_sonar_data():
    with open(os.path.join("data", "sonar-data.csv"), "rt") as f:
        dataset = [
            [float(value) for value in features] + [label]
            for *features, label in [line for line in csv.reader(f) if line]
        ]

    mapping, encoded = encode_labels(dataset)
    random.shuffle(encoded)

    return mapping, encoded


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


def swap_target_placement(dataset: List[List[Any]], label_index: int):
    """Move target column to the end of each row (inplace)."""
    for line in dataset:
        line.append(line.pop(label_index))
    return dataset


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


def encode_labels(dataset: List[List[Any]]) -> (Dict[Any, int], List[List[Any]]):
    """Encode dataset labels. Return label: encoding mapping and mapped dataset."""

    mapping = {label: idx for idx, label in enumerate({label for *_, label in dataset})}
    mapped_dataset = [[*features, mapping[label]] for *features, label in dataset]

    return mapping, mapped_dataset

import os
import csv
from typing import List, Dict, Union, Any


def load_mpg_data():
    with open(os.path.join("data", "auto-mpg-data.csv"), "rt") as f:
        dataset = [
            [float(value) for value in features] + [car_name]
            for *features, car_name in [
                line for line in csv.reader(f) if line and "?" not in line
            ]
        ]

    dataset = drop_columns(dataset, [1, 2, -1])
    dataet = swap_target_placement(dataset, 0)

    return normalize(dataset)


def load_sonar_data():
    with open(os.path.join("data", "sonar-data.csv"), "rt") as f:
        dataset = [
            [float(value) for value in features] + [label]
            for *features, label in [line for line in csv.reader(f) if line]
        ]

    mapping, encoded = encode_labels(dataset)

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
    feature_count = len(dataset[0]) - 1

    total_min = [None] * feature_count
    total_max = [None] * feature_count
    for *features, _ in dataset:
        for idx, feature in enumerate(features):
            if total_min[idx] is None or total_min[idx] > feature:
                total_min[idx] = feature
            if total_max[idx] is None or total_max[idx] < feature:
                total_max[idx] = feature

    normalized_dataset = []
    for *features, label in dataset:
        normalized_dataset.append(
            [
                (feature - min_) / (max_ - min_)
                for feature, min_, max_ in zip(features, total_min, total_max)
            ]
            + [label]
        )
    return normalized_dataset


def encode_labels(dataset: List[List[Any]]) -> (Dict[Any, int], List[List[Any]]):
    """Encode dataset labels. Return label: encoding mapping and mapped dataset."""

    mapping = {label: idx for idx, label in enumerate({label for *_, label in dataset})}
    mapped_dataset = [[*features, mapping[label]] for *features, label in dataset]

    return mapping, mapped_dataset

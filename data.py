import os
import csv
from typing import List, Dict, Any


def load_sonar_data():
    with open(os.path.join("data", "sonar-data.csv"), "rt") as f:
        dataset = [
            [float(value) for value in features] + [label]
            for *features, label in [line for line in csv.reader(f) if line]
        ]

    mapping, encoded = encode_labels(dataset)

    return mapping, encoded


def encode_labels(dataset: List[List[Any]]) -> (Dict[Any, int], List[List[Any]]):
    """Encode dataset labels. Return label: encoding mapping and mapped dataset."""

    mapping = {label: idx for idx, label in enumerate({label for *_, label in dataset})}
    mapped_dataset = [[*features, mapping[label]] for *features, label in dataset]

    return mapping, mapped_dataset

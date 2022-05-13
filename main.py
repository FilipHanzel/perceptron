import os
import csv
import random
from pprint import pprint

from data_utils import transpose, normalize, encode_as_int, encode

from perceptron import Perceptron
from perceptron import MultilayerPerceptron


def load_xor(targets_as_lists: bool = False):
    features = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    targets = [0, 1, 1, 0]
    if targets_as_lists:
        targets = [[value] for value in targets]

    return features, targets


def load_mpg(targets_as_lists: bool = False):
    with open(os.path.join("data", "auto-mpg.csv"), "rt") as f:
        data = [
            [float(value) for value in features]
            for *features, _ in [
                line for line in csv.reader(f) if line and "?" not in line
            ]
        ]

    data = normalize(data)

    targets, *features = transpose(data)
    features = transpose(features)
    if targets_as_lists:
        targets = [[value] for value in targets]

    return features, targets


def load_sonar(targets_as_lists: bool = False):
    with open(os.path.join("data", "sonar.csv"), "rt") as f:
        data = [
            [float(value) for value in features] + [target]
            for *features, target in [line for line in csv.reader(f) if line]
        ]

    *features, targets = transpose(data)
    features = transpose(features)
    if targets_as_lists:
        mapping, targets = encode(targets)
    else:
        mapping, targets = encode_as_int(targets)

    return features, targets, mapping


if __name__ == "__main__":
    print("Loading xor data...")

    print("Integer output:")
    features, targets = load_xor()
    pprint(features)
    pprint(targets)

    print("Vector output:")
    features, targets = load_xor(targets_as_lists=True)
    pprint(features)
    pprint(targets)

    print("Loading sonar data...")

    print("Integer output:")
    features, targets, mapping = load_sonar()
    pprint(mapping)
    for feature in features[:2]:
        print(feature)
    for target in targets[:2]:
        print(target)

    print("Vector output:")
    features, targets, mapping = load_sonar(targets_as_lists=True)
    pprint(mapping)
    for feature in features[:2]:
        print(feature)
    for target in targets[:2]:
        print(target)

    print("Loading mpg data...")

    print("Integer output:")
    features, targets = load_mpg()
    for feature in features[:2]:
        print(feature)
    for target in targets[:2]:
        print(target)

    print("Vector output:")
    features, targets = load_mpg(targets_as_lists=True)
    for feature in features[:2]:
        print(feature)
    for target in targets[:2]:
        print(target)

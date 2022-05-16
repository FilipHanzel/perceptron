import os
import csv
import random
from pprint import pprint

from data_utils import transpose, normalize, to_categorical, to_binary

from perceptron import Perceptron


def load_xor(categorical: bool = False):
    features = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]

    if categorical:
        mapping, targets = to_categorical([0, 1, 1, 0])
        return features, targets, mapping
    else:
        targets = [[0], [1], [1], [0]]
        return features, targets


def load_sonar(categorical: bool = False):
    with open(os.path.join("data", "sonar.csv"), "rt") as f:
        data = [
            [float(value) for value in features] + [target]
            for *features, target in [line for line in csv.reader(f) if line]
        ]

    *features, targets = transpose(data)
    features = transpose(features)

    if categorical:
        mapping, targets = to_categorical(targets)
    else:
        mapping, targets = to_binary(targets)

    return features, targets, mapping


def load_mpg():
    with open(os.path.join("data", "auto-mpg.csv"), "rt") as f:
        data = [
            [float(value) for value in features]
            for *features, _ in [
                line for line in csv.reader(f) if line and "?" not in line
            ]
        ]

    data = normalize(data)

    targets, *features = transpose(data)
    targets = [[value] for value in targets]
    features = transpose(features)

    return features, targets


def load_iris():
    with open(os.path.join("data", "iris.csv"), "rt") as f:
        data = [
            [float(value) for value in features] + [target]
            for *features, target in [line for line in csv.reader(f) if line]
        ]

    *features, targets = transpose(data)
    features = normalize(transpose(features))
    mapping, targets = to_categorical(targets)

    return features, targets, mapping


if __name__ == "__main__":
    print("Running xor (classification)...")

    print("Solving with single perceptron as binary classification...")
    features, targets = load_xor()
    inputs = 2
    epochs = 5000
    base_learning_rate = 0.2

    random.seed(0)

    model = Perceptron(inputs, [1], "sigmoid", init_method="he")
    model.train(
        features,
        targets,
        epochs,
        base_learning_rate,
        metrics=["binary_accuracy", "sse"],
    )

    for feature, target in zip(features, targets):
        print(f"{feature=}, {target=}, prediction={model.predict(feature)}")

    print("Solving with single layer as categorical classification...")
    features, targets, mapping = load_xor(categorical=True)
    inputs = 2
    epochs = 5000
    base_learning_rate = 0.2

    random.seed(0)

    model = Perceptron(inputs, [2], "sigmoid", init_method="he")
    model.train(
        features,
        targets,
        epochs,
        base_learning_rate,
        metrics=["categorical_accuracy", "binary_accuracy", "sse"],
    )

    for feature, target in zip(features, targets):
        print(f"{feature=}, {target=}, prediction={model.predict(feature)}")

    print("Solving with MLP...")
    features, targets, mapping = load_xor(categorical=True)

    inputs = 2
    epochs = 5000
    base_learning_rate = 0.2

    random.seed(0)

    model = Perceptron(inputs, [2, 2], "leaky_relu", init_method="he")
    model.train(
        features,
        targets,
        epochs,
        base_learning_rate,
        metrics=["categorical_accuracy", "binary_accuracy", "sse"],
    )

    for feature, target in zip(features, targets):
        print(f"{feature=}, {target=}, prediction={model.predict(feature)}")

    print("Running sonar (classification)...")

    print("Solving with single perceptron as binary classification...")
    features, targets, mapping = load_sonar()
    inputs = 60
    epochs = 500
    base_learning_rate = 0.2

    random.seed(0)

    model = Perceptron(inputs, [1], "sigmoid", init_method="he")
    model.train(
        features,
        targets,
        epochs,
        base_learning_rate,
        metrics=["binary_accuracy", "sse"],
    )

    print("Solving with single layer as categorical classification...")
    features, targets, mapping = load_sonar(categorical=True)
    inputs = 60
    epochs = 500
    base_learning_rate = 0.2

    random.seed(0)

    model = Perceptron(inputs, [2], "sigmoid", init_method="he")
    model.train(
        features,
        targets,
        epochs,
        base_learning_rate,
        metrics=["categorical_accuracy", "binary_accuracy", "sse"],
    )

    print("Solving with MLP...")
    features, targets, mapping = load_sonar(categorical=True)
    inputs = 60
    epochs = 500
    base_learning_rate = 0.2

    random.seed(0)

    model = Perceptron(inputs, [2, 2], "leaky_relu", init_method="he")
    model.train(
        features,
        targets,
        epochs,
        base_learning_rate,
        metrics=["categorical_accuracy", "binary_accuracy", "sse"],
    )

    print("Running mpg (regression)...")

    print("Solving with single perceptron...")
    features, targets = load_mpg()
    inputs = 7
    epochs = 500
    base_learning_rate = 0.01

    random.seed(0)

    model = Perceptron(inputs, [1], "sigmoid", init_method="he")
    model.train(
        features,
        targets,
        epochs,
        base_learning_rate,
        metrics=["binary_accuracy", "sse"],
    )

    for feature, target in zip(features[:5], targets[:5]):
        print(f"{target=}, prediction={model.predict(feature)}")

    print("Solving with MLP...")
    features, targets = load_mpg()
    inputs = 7
    epochs = 500
    base_learning_rate = 0.1

    random.seed(0)

    model = Perceptron(inputs, [10, 5, 1], "leaky_relu", init_method="he")
    model.train(features, targets, epochs, base_learning_rate)

    for feature, target in zip(features[:5], targets[:5]):
        print(f"{target=}, prediction={model.predict(feature)}")

    print("Running iris (multiclass classification)...")
    features, targets, mapping = load_iris()
    inputs = 4
    epochs = 500
    base_learning_rate = 0.01

    random.seed(0)

    model = Perceptron(inputs, [4, 3, 3], "leaky_relu", init_method="he")
    model.train(
        features,
        targets,
        epochs,
        base_learning_rate,
        metrics=["categorical_accuracy", "sse"],
    )

    for feature, target in zip(features[:5], targets[:5]):
        print(f"{target=}, prediction={model.predict(feature)}")

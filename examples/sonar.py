import os
import sys
import random
import csv
from pprint import pprint

script_path = os.path.dirname(os.path.abspath(__file__))
data_scanner_path = os.path.join(script_path, "..")
sys.path.append(data_scanner_path)

from perceptron import Perceptron, cross_validation
from data_utils import transpose, to_categorical, to_binary

if __name__ == "__main__":
    with open(os.path.join(script_path, "data", "sonar.csv"), "rt") as f:
        data = [
            [float(value) for value in features] + [target]
            for *features, target in [line for line in csv.reader(f) if line]
        ]

    *features, targets = transpose(data)
    features = transpose(features)
    binary_mapping, binary_targets = to_binary(targets)
    categorical_mapping, categorical_targets = to_categorical(targets)

    print("Labels binary mapping:")
    pprint(binary_mapping)
    print("Labels categorical mapping:")
    pprint(categorical_mapping)

    print("Solving with single perceptron as binary classification...")
    random.seed(0)

    model = Perceptron(
        inputs=60, layer_sizes=[1], activations="sigmoid", init_method="he"
    )
    model.train(
        training_inputs=features,
        training_targets=binary_targets,
        epochs=100,
        base_learning_rate=0.2,
        metrics=["binary_accuracy", "sse"],
    )

    print("Solving with single layer as categorical classification...")
    random.seed(0)

    model = Perceptron(
        inputs=60, layer_sizes=[2], activations="sigmoid", init_method="he"
    )
    model.train(
        training_inputs=features,
        training_targets=categorical_targets,
        epochs=100,
        base_learning_rate=0.2,
        metrics=["categorical_accuracy", "sse"],
    )

    print("Solving with MLP as binary classification...")
    model = Perceptron(
        inputs=60, layer_sizes=[2, 1], activations="sigmoid", init_method="he"
    )
    model.train(
        training_inputs=features,
        training_targets=binary_targets,
        epochs=100,
        base_learning_rate=0.2,
        metrics=["binary_accuracy", "sse"],
    )

    print("Solving with MLP as categorical classification...")
    model = Perceptron(
        inputs=60, layer_sizes=[2, 2], activations="sigmoid", init_method="he"
    )
    model.train(
        training_inputs=features,
        training_targets=categorical_targets,
        epochs=100,
        base_learning_rate=0.2,
        metrics=["categorical_accuracy", "sse"],
    )

    print("Cross validating MLP for categorical classification...")
    random.seed(0)

    model_params = dict(
        inputs=60, layer_sizes=[2, 2], activations="sigmoid", init_method="he"
    )
    cross_validation(
        inputs=features,
        targets=categorical_targets,
        fold_count=5,
        epoch=100,
        base_learning_rate=0.2,
        learning_rate_decay="linear",
        model_params=model_params,
        metrics=["categorical_accuracy"],
    )

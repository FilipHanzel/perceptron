import os
import sys
import random

script_path = os.path.dirname(os.path.abspath(__file__))
data_scanner_path = os.path.join(script_path, "..")
sys.path.append(data_scanner_path)

from perceptron import Perceptron
from data_utils import normalize


if __name__ == "__main__":
    features = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    binary_targets = [[0], [1], [1], [0]]
    categorical_targets = [[0, 1], [1, 0], [1, 0], [0, 1]]

    print("Solving with single perceptron as binary classification...")
    random.seed(0)

    model = Perceptron(
        inputs=2, layer_sizes=[1], activations="sigmoid", init_method="he"
    )
    model.train(
        training_inputs=features,
        training_targets=binary_targets,
        epochs=5000,
        base_learning_rate=0.2,
        metrics=["binary_accuracy", "sse"],
    )

    for feature, target in zip(features, binary_targets):
        print(f"\t{feature=}, {target=}, prediction={model.predict(feature)}")

    print("Solving with single layer as categorical classification...")
    random.seed(0)

    model = Perceptron(
        inputs=2, layer_sizes=[2], activations="sigmoid", init_method="he"
    )
    model.train(
        training_inputs=features,
        training_targets=categorical_targets,
        epochs=5000,
        base_learning_rate=0.2,
        metrics=["categorical_accuracy", "sse"],
    )

    for feature, target in zip(features, categorical_targets):
        print(f"\t{feature=}, {target=}, prediction={model.predict(feature)}")

    print("Solving with MLP as binary classification...")
    random.seed(0)

    model = Perceptron(
        inputs=2, layer_sizes=[2, 1], activations="leaky_relu", init_method="he"
    )
    model.train(
        training_inputs=features,
        training_targets=binary_targets,
        epochs=5000,
        base_learning_rate=0.2,
        metrics=["binary_accuracy", "sse"],
    )

    for feature, target in zip(features, binary_targets):
        print(f"\t{feature=}, {target=}, prediction={model.predict(feature)}")

    print("Solving with MLP as categorical classification...")
    random.seed(0)

    model = Perceptron(
        inputs=2, layer_sizes=[2, 2], activations="leaky_relu", init_method="he"
    )
    model.train(
        training_inputs=features,
        training_targets=categorical_targets,
        epochs=5000,
        base_learning_rate=0.2,
        metrics=["binary_accuracy", "sse"],
    )

    for feature, target in zip(features, categorical_targets):
        print(f"\t{feature=}, {target=}, prediction={model.predict(feature)}")

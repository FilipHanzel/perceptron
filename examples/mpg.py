import os
import sys
import random
import csv

script_path = os.path.dirname(os.path.abspath(__file__))
data_scanner_path = os.path.join(script_path, "..")
sys.path.append(data_scanner_path)

from perceptron import Perceptron, cross_validation
from data_utils import normalize, transpose

if __name__ == "__main__":
    with open(os.path.join(script_path, "data", "auto-mpg.csv"), "rt") as f:
        data = [
            [float(value) for value in features]
            for *features, _ in [
                line for line in csv.reader(f) if line and "?" not in line
            ]
        ]

    targets, *features = transpose(data)
    targets = [[value] for value in targets]
    features = normalize(transpose(features))

    print("Solving with single perceptron...")
    random.seed(0)

    model = Perceptron(
        inputs=7, layer_sizes=[1], activations="linear", init_method="he"
    )
    model.train(
        training_inputs=features,
        training_targets=targets,
        epochs=100,
        base_learning_rate=0.01,
        metrics=["sse", "mae"],
    )

    for feature, target in zip(features[:5], targets[:5]):
        print(f"{target=}, prediction={model.predict(feature)}")

    print("Solving with MLP...")
    random.seed(0)

    model = Perceptron(
        inputs=7,
        layer_sizes=[10, 5, 5, 1],
        activations=["leaky_relu"] * 3 + ["linear"],
        init_method="he",
    )
    model.train(
        training_inputs=features,
        training_targets=targets,
        epochs=100,
        base_learning_rate=0.0001,
        metrics=["sse", "mae"],
    )

    for feature, target in zip(features[:5], targets[:5]):
        print(f"{target=}, prediction={model.predict(feature)}")

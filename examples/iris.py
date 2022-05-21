import os
import sys
import random
import csv
from pprint import pprint

script_path = os.path.dirname(os.path.abspath(__file__))
data_scanner_path = os.path.join(script_path, "..")
sys.path.append(data_scanner_path)

from perceptron import Perceptron, cross_validation
from data_utils import transpose, normalize, to_categorical

if __name__ == "__main__":
    with open(os.path.join(script_path, "data", "iris.csv"), "rt") as f:
        data = [
            [float(value) for value in features] + [target]
            for *features, target in [line for line in csv.reader(f) if line]
        ]

    *features, targets = transpose(data)
    features = transpose(features)
    normalized_features = normalize(features)
    mapping, targets = to_categorical(targets)

    print("Labels mapping:")
    pprint(mapping)

    print("Solving with MLP...")
    random.seed(0)

    model = Perceptron(
        inputs=4,
        layer_sizes=[4, 3, 3],
        activations="leaky_relu",
        init_method="he",
        optimizer="momentum",
    )
    model.train(
        training_inputs=normalized_features,
        training_targets=targets,
        epochs=100,
        base_learning_rate=0.01,
        metrics=["categorical_accuracy", "sse"],
    )

    for feature, target in zip(normalized_features[:5], targets[:5]):
        print(f"\t{target=}, prediction={model.predict(feature)}")

    print("Predicting with untrained model with builtin minmax normalization...")
    model = Perceptron(
        inputs=4,
        layer_sizes=[4, 3, 3],
        activations="leaky_relu",
        init_method="he",
        normalization="minmax",
    )
    for feature, target in zip(features[:5], targets[:5]):
        print(f"\t{target=}, prediction={model.predict(feature)}")

    print("Predicting with untrained model with builtin zscore normalization...")
    model = Perceptron(
        inputs=4,
        layer_sizes=[4, 3, 3],
        activations="leaky_relu",
        init_method="he",
        normalization="zscore",
    )
    for feature, target in zip(features[:5], targets[:5]):
        print(f"\t{target=}, prediction={model.predict(feature)}")

    print("Cross validating...")
    random.seed(0)

    model_params = dict(
        inputs=4, layer_sizes=[4, 3, 3], activations="leaky_relu", init_method="he"
    )
    cross_validation(
        inputs=normalized_features,
        targets=targets,
        fold_count=5,
        epoch=30,
        base_learning_rate=0.01,
        learning_rate_decay="linear",
        model_params=model_params,
        metrics=["categorical_accuracy", "sse"],
    )

    print("Cross validating with momentum...")
    random.seed(0)

    model_params = dict(
        inputs=4,
        layer_sizes=[4, 3, 3],
        activations="leaky_relu",
        init_method="he",
        optimizer="momentum",
    )
    cross_validation(
        inputs=normalized_features,
        targets=targets,
        fold_count=5,
        epoch=30,
        base_learning_rate=0.01,
        learning_rate_decay="linear",
        model_params=model_params,
        metrics=["categorical_accuracy", "sse"],
    )

    print("Cross validating with builtin zscore normalization...")
    random.seed(0)

    model_params = dict(
        inputs=4,
        layer_sizes=[4, 3, 3],
        activations="leaky_relu",
        init_method="he",
        normalization="zscore",
    )
    cross_validation(
        inputs=features,
        targets=targets,
        fold_count=5,
        epoch=30,
        base_learning_rate=0.01,
        learning_rate_decay="linear",
        model_params=model_params,
        metrics=["categorical_accuracy", "sse"],
    )

    print("Cross validating with builtin zscore normalization and momentum...")
    random.seed(0)

    model_params = dict(
        inputs=4,
        layer_sizes=[4, 3, 3],
        activations="leaky_relu",
        init_method="he",
        normalization="zscore",
        optimizer="momentum",
    )
    cross_validation(
        inputs=features,
        targets=targets,
        fold_count=5,
        epoch=30,
        base_learning_rate=0.01,
        learning_rate_decay="linear",
        model_params=model_params,
        metrics=["categorical_accuracy", "sse"],
    )

    print("Cross validating with builtin zscore normalization and adagrad...")
    random.seed(0)

    model_params = dict(
        inputs=4,
        layer_sizes=[4, 3, 3],
        activations="leaky_relu",
        init_method="he",
        normalization="zscore",
        optimizer="adagrad",
    )
    cross_validation(
        inputs=features,
        targets=targets,
        fold_count=5,
        epoch=30,
        base_learning_rate=1.0,
        learning_rate_decay="linear",
        model_params=model_params,
        metrics=["categorical_accuracy", "sse"],
    )

import random
from math import exp
from typing import List, Tuple, Callable

from tqdm import tqdm

from data import load_sonar_data, load_mpg_data


class Activation:
    @staticmethod
    def heavyside(x):
        return 1 if x >= 0 else 0

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def relu(x):
        return max(0, x)

    @staticmethod
    def leaky_relu(x):
        return 0.3 * x if x < 0 else x

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + exp(-x))


class Perceptron:
    __slots__ = ["weights", "bias", "activation"]

    def __init__(self, inputs: int, activation: Callable[[float], float]):
        self.weights = [random.uniform(0, 1) for _ in range(inputs)]
        self.bias = 0
        self.activation = activation

    def predict(self, input_features: List[float]) -> float:
        state = 0
        for (w, s) in zip(self.weights, input_features):
            state += w * s
        state += self.bias
        return self.activation(state)

    def update(
        self, train_vector: List[float], learning_rate: float
    ) -> Tuple[float, float]:
        *features, target = train_vector
        prediction = self.predict(features)
        error = target - prediction

        self.bias += learning_rate * error
        for idx, feature in enumerate(features):
            self.weights[idx] = self.weights[idx] + learning_rate * error * feature

        return (prediction, error**2)


def main():
    print("[ Classification problem ]")
    labels_mapping, dataset = load_sonar_data()

    print("Label mapping:")
    for label in labels_mapping:
        print(f"\t{label + ':':20} {labels_mapping[label]}")

    assert (
        len(labels_mapping) == 2
    ), "Invalid dataset, only binary classification is supported"

    random.seed(0)

    sample = random.choice(dataset)
    *features, label = sample

    perceptron = Perceptron(len(features), Activation.heavyside)

    sse = 0.0
    correct = 0
    for *features, label in dataset:
        prediction = perceptron.predict(features)

        sse += (prediction - label) ** 2

        prediction = 1 if prediction >= 0.5 else 0
        if prediction == label:
            correct += 1
    accuracy = correct / len(dataset)
    print(f"Pre training accuracy: {accuracy:6.2f}")
    print(f"Pre training SSE: {sse:6.3f}")

    epochs = 100
    learning_rate = 0.001

    progress = tqdm(
        range(epochs),
        unit="epochs",
        ncols=100,
        bar_format="Training: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}",
    )

    for epoch in progress:
        sse = 0.0
        accuracy = 0
        for vector in dataset:
            *features, label = vector
            prediction, square_error = perceptron.update(vector, learning_rate)

            prediction = 1 if prediction >= 0.5 else 0
            if prediction == label:
                accuracy += 1
            sse += square_error
        accuracy /= len(dataset)
        progress.set_postfix(sse=sse, acc=round(accuracy, 3))

    print("[ Regression problem ]")
    dataset = load_mpg_data()

    random.seed(0)

    sample = random.choice(dataset)
    *features, label = sample

    perceptron = Perceptron(len(features), Activation.relu)

    print('Pre training predictions:')
    for *features, target in dataset[:3]:
        print(target, "\t", perceptron.predict(features))

    sse = 0.0
    for *features, target in dataset:
        prediction = perceptron.predict(features)
        sse += (prediction - target) ** 2
    print(f"Pre training SSE: {sse:6.3f}")

    epochs = 1000
    learning_rate = 0.001

    progress = tqdm(
        range(epochs),
        unit="epochs",
        ncols=100,
        bar_format="Training: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}",
    )

    for epoch in progress:
        sse = 0.0
        for vector in dataset:
            target, *features = vector
            prediction, square_error = perceptron.update(vector, learning_rate)
            sse += square_error
        progress.set_postfix(sse=sse)


    print('Predictions after training:')
    for *features, target in dataset[:3]:
        print(target, "\t", perceptron.predict(features))


if __name__ == "__main__":
    main()

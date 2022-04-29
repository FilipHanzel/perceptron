import random
from typing import List, Tuple

from tqdm import tqdm

from data import load_sonar_data


class Perceptron:
    __slots__ = ["weights", "bias", "activation"]

    def __init__(self, inputs: int):
        self.weights = [random.uniform(0, 1) for _ in range(inputs)]
        self.bias = 0

    def predict(self, input_features: List[float]) -> float:
        state = 0
        for (w, s) in zip(self.weights, input_features):
            state += w * s
        state += self.bias
        return 1 if state >= 0 else 0

    def update(
        self, train_vector: List[float], learning_rate: float
    ) -> Tuple[float, float]:
        *features, label = train_vector
        prediction = self.predict(features)
        error = label - prediction

        self.bias += learning_rate * error
        for idx, feature in enumerate(features):
            self.weights[idx] = self.weights[idx] + learning_rate * error * feature

        return (prediction, error**2)


def main():
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

    perceptron = Perceptron(len(features))

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


if __name__ == "__main__":
    main()

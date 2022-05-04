import random

from tqdm import tqdm

from data import load_sonar_data, load_mpg_data
from perceptron import Perceptron


def sonar():
    print("[ Classification ]")
    labels_mapping, dataset = load_sonar_data()

    print("Label mapping:")
    for label in labels_mapping:
        print(f"\t{label + ':':20} {labels_mapping[label]}")
    assert (
        len(labels_mapping) == 2
    ), "Invalid dataset, only binary classification is supported"

    random.seed(0)

    perceptron = Perceptron(len(dataset[0]), 'heavyside')

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

    epochs = 2500
    learning_rate = 0.0001

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

    sse = 0.0
    correct = 0
    for *features, label in dataset:
        prediction = perceptron.predict(features)
        sse += (prediction - label) ** 2
        prediction = 1 if prediction >= 0.5 else 0
        if prediction == label:
            correct += 1
    accuracy = correct / len(dataset)
    print(f"After training accuracy: {accuracy:6.2f}")
    print(f"After training SSE: {sse:6.3f}")


def mpg():
    print("[ Regression ]")
    dataset = load_mpg_data()

    random.seed(0)

    perceptron = Perceptron(len(dataset[0]), 'relu')

    print("Pre training predictions:")
    for *features, target in dataset[:3]:
        print(target, "\t", perceptron.predict(features))

    sse = 0.0
    for *features, target in dataset:
        prediction = perceptron.predict(features)
        sse += (prediction - target) ** 2
    print(f"Pre training SSE: {sse:6.3f}")

    epochs = 2500
    learning_rate = 0.0001

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

    print("Predictions after training:")
    for *features, target in dataset[:3]:
        print(target, "\t", perceptron.predict(features))

    sse = 0.0
    for *features, target in dataset:
        prediction = perceptron.predict(features)
        sse += (prediction - target) ** 2
    print(f"After training SSE: {sse:6.3f}")


def main():
    sonar()
    mpg()


if __name__ == "__main__":
    main()

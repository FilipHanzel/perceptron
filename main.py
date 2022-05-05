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

    perceptron = Perceptron(len(dataset[0]), "heavyside")

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

    list_of_inputs = []
    list_of_targets = []
    for *inputs, target in dataset:
        list_of_inputs.append(inputs)
        list_of_targets.append(target)

    perceptron.train(list_of_inputs, list_of_targets, epochs, learning_rate)

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

    perceptron = Perceptron(len(dataset[0]), "relu")

    print("Pre training predictions:")
    for *features, target in dataset[:3]:
        print(target, "\t", perceptron.predict(features))

    sse = 0.0
    for *features, target in dataset:
        prediction = perceptron.predict(features)
        sse += (prediction - target) ** 2
    print(f"Pre training SSE: {sse:6.3f}")

    epochs = 500
    learning_rate = 0.001

    list_of_inputs = []
    list_of_targets = []
    for *inputs, target in dataset:
        list_of_inputs.append(inputs)
        list_of_targets.append(target)

    perceptron.train(list_of_inputs, list_of_targets, epochs, learning_rate)

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

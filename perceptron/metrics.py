from typing import List


def binary_accuracy(predictions: List[List], targets: List[List]) -> float:
    threshold = 0.5

    correct = 0
    total = 0

    for predictions_row, target_row in zip(predictions, targets):
        total += len(predictions_row)
        for prediction, target in zip(predictions_row, target_row):
            prediction = 1 if prediction > threshold else 0
            correct += prediction == target

    return correct / total


def categorical_accuracy(predictions: List[List], targets: List[List]) -> float:
    correct = 0

    for prediction_row, target_row in zip(predictions, targets):
        prediction = prediction_row.index(max(prediction_row))
        target = target_row.index(max(target_row))

        correct += prediction == target

    return correct / len(predictions)


def sse(predictions: List[List], targets: List[List]) -> float:
    sse = 0.0

    for prediction_list, target_list in zip(predictions, targets):
        sse += sum(
            [
                (prediction - target) ** 2
                for prediction, target in zip(prediction_list, target_list)
            ]
        )

    return sse


def mae(predictions: List[List], targets: List[List]) -> float:
    mae = 0.0

    for prediction_list, target_list in zip(predictions, targets):
        mae += sum(
            [
                abs(prediction - target)
                for prediction, target in zip(prediction_list, target_list)
            ]
        ) / len(prediction_list)

    return mae / len(predictions)

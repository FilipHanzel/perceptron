import re
from typing import List
from abc import ABC, abstractmethod


class Metric:
    def __init__(self, name: str = None):
        self.__name = name

    @property
    def name(self):
        """Formatted metric name. Should be used while displaying metrics during model training."""
        if self.__name is not None:
            return self.__name
        default_name = self.__class__.__name__
        default_name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", default_name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", default_name).lower()

    @name.setter
    def name(self, name):
        self.__name = name

    @abstractmethod
    def __call__(
        self, predictions: List[List[float]], targets: List[List[float]]
    ) -> float:
        """Calculate metric."""


# Regression metrics


class MAE(Metric):
    """Mean Absolute Error."""

    def __call__(
        self, predictions: List[List[float]], targets: List[List[float]]
    ) -> float:
        ae = 0.0
        count = 0

        for prediction, target in zip(predictions, targets):
            for predicted_value, target_value in zip(prediction, target):
                ae += abs(predicted_value - target_value)
            count += len(prediction)

        return ae / count


class MAPE(Metric):
    """Mean Absolute Percentage Error."""

    def __call__(
        self, predictions: List[List[float]], targets: List[List[float]]
    ) -> float:
        ape = 0.0
        count = 0

        for prediction, target in zip(predictions, targets):
            for predicted_value, target_value in zip(prediction, target):
                ape += abs((predicted_value - target_value) / target_value)
            count += len(prediction)

        return ape / count


class MSE(Metric):
    """Mean Square Error."""

    def __call__(
        self, predictions: List[List[float]], targets: List[List[float]]
    ) -> float:
        se = 0.0
        count = 0

        for prediction, target in zip(predictions, targets):
            for predicted_value, target_value in zip(prediction, target):
                se += (predicted_value - target_value) ** 2
            count += len(prediction)

        return se / count


class RMSE(MSE):
    """Root Mean Square Error."""

    def __call__(
        self, predictions: List[List[float]], targets: List[List[float]]
    ) -> float:
        return super().__call__(predictions, targets) ** 0.5


class CosSim(Metric):
    """Cosine Similarity."""

    def __call__(
        self, predictions: List[List[float]], targets: List[List[float]]
    ) -> float:
        m_sum = 0.0
        p_sum = 0.0
        t_sum = 0.0

        for prediction, target in zip(predictions, targets):
            for predicted_value, target_value in zip(prediction, target):
                m_sum += predicted_value * target_value
                p_sum += predicted_value**2
                t_sum += target_value**2

        return m_sum / (p_sum**0.5 * t_sum**0.5)


# Classification metrics


class BinaryAccuracy(Metric):
    """Binary accuracy. Average of single output accuracy."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(
        self, predictions: List[List[float]], targets: List[List[float]]
    ) -> float:
        correct = 0
        total = 0

        for predictions_row, target_row in zip(predictions, targets):
            total += len(predictions_row)

            for prediction, target in zip(predictions_row, target_row):
                prediction = 1 if prediction > self.threshold else 0
                target = 1 if target > self.threshold else 0

                correct += prediction == target

        return correct / total


class CategoricalAccuracy(Metric):
    """Categorical accuracy. Compares argmax of predictions and targets."""

    def __call__(
        self, predictions: List[List[float]], targets: List[List[float]]
    ) -> float:
        correct = 0

        for prediction_row, target_row in zip(predictions, targets):
            prediction = prediction_row.index(max(prediction_row))
            target = target_row.index(max(target_row))

            correct += prediction == target

        return correct / len(predictions)


class TopKCategoricalAccuracy(Metric):
    """Top K Categorical accuracy.

    Check if target is amongst top k predictions.
    If k = 1, it's the same as CategoricalAccuracy."""

    def __init__(self, k: int = 3):
        assert isinstance(k, int), "Parameter k has to be of type int"
        self.name = f"top_{k}_cat_acc"
        self.k = k

    def __call__(
        self, predictions: List[List[float]], targets: List[List[float]]
    ) -> float:
        correct = 0

        for prediction_row, target_row in zip(predictions, targets):
            top_k_predictions = [
                index
                for index, _ in sorted(
                    enumerate(prediction_row), key=lambda x: x[1], reverse=True
                )[: self.k]
            ]
            target = target_row.index(max(target_row))

            correct += target in predictions

        return correct / len(predictions)

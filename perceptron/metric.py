import re
from abc import ABC, abstractmethod
from typing import List


def metric_from_string(name: str) -> "Metric":
    """Get metric object with default values, based on string. Convenience function."""

    name = name.lower()
    if name == "mae":
        metric = MAE()
    elif name == "mape":
        metric = MAPE()
    elif name == "mse":
        metric = MSE()
    elif name == "rmse":
        metric = RMSE()
    elif name == "cos_similarity":
        metric = CosSim()
    elif name in "binary_accuracy":
        metric = BinaryAccuracy()
    elif name in "categorical_accuracy":
        metric = CategoricalAccuracy()
    elif name in "top_k_categorical_accuracy":
        metric = TopKCategoricalAccuracy()
    else:
        raise ValueError(f"Invalid metric {name}")

    return metric


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
    def calculate(self, prediction: List[float], target: List[float]) -> float:
        """Calculate metric for a single prediction."""

    def calculate_avg(
        self, predictions: List[List[float]], targets: List[List[float]]
    ) -> float:
        """Calculate average metric for multiple predictions."""

        s = 0.0
        for prediction, target in zip(predictions, targets):
            s += self.calculate(prediction, target)

        return s / len(predictions)


# Regression metrics


class MAE(Metric):
    """Mean Absolute Error."""

    def calculate(self, prediction: List[float], target: List[float]) -> float:
        ae = 0.0
        for p, t in zip(prediction, target):
            ae += abs(p - t)

        return ae / len(prediction)


class MAPE(Metric):
    """Mean Absolute Percentage Error."""

    def calculate(self, prediction: List[float], target: List[float]) -> float:
        ape = 0.0
        for p, t in zip(prediction, target):
            ape += abs((p - t) / t)

        return ape / len(prediction)


class MSE(Metric):
    """Mean Square Error."""

    def calculate(self, prediction: List[float], target: List[float]) -> float:
        se = 0.0
        for p, t in zip(prediction, target):
            se += (p - t) ** 2

        return se / len(prediction)


class RMSE(MSE):
    """Root Mean Square Error."""

    def calculate(self, prediction: List[float], target: List[float]) -> float:
        return super().calculate(prediction, target) ** 0.5


class CosSim(Metric):
    """Cosine Similarity."""

    def calculate(self, prediction: List[float], target: List[float]) -> float:
        m_sum = 0.0
        p_sum = 0.0
        t_sum = 0.0

        for p, t in zip(prediction, target):
            m_sum += p * t
            p_sum += p**2
            t_sum += t**2

        return m_sum / (p_sum**0.5 * t_sum**0.5)


# Classification metrics


class BinaryAccuracy(Metric):
    """Binary accuracy. Average of single output accuracy."""

    def __init__(self, name: str = None, threshold: float = 0.5):
        super().__init__(name)
        self.threshold = threshold

    def calculate(self, prediction: List[float], target: List[float]) -> float:
        correct = 0
        for p, t in zip(prediction, target):
            p = 1 if p > self.threshold else 0
            t = 1 if t > self.threshold else 0

            correct += p == t

        return correct / len(prediction)


class CategoricalAccuracy(Metric):
    """Categorical accuracy. Compares argmax of predictions and targets."""

    def calculate(self, prediction: List[float], target: List[float]) -> float:
        prediction = prediction.index(max(prediction))
        target = target.index(max(target))

        return prediction == target


class TopKCategoricalAccuracy(Metric):
    """Top K Categorical accuracy.

    Check if target is amongst top k predictions.
    If k = 1, it's the same as CategoricalAccuracy."""

    def __init__(self, name: str = None, k: int = 3):
        super().__init__(name)
        assert isinstance(k, int), "Parameter k has to be of type int"
        self.name = f"top_{k}_cat_acc"
        self.k = k

    def calculate(self, prediction: List[float], target: List[float]) -> float:
        top_k_predictions_indexes = [
            index
            for index, _ in sorted(
                enumerate(prediction), key=lambda x: x[1], reverse=True
            )[: self.k]
        ]
        target_index = target.index(max(target))

        return target_index in top_k_predictions_indexes

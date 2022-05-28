from math import exp


# Activations


def heavyside(x: float) -> float:
    return 1 if x >= 0 else 0


def linear(x: float) -> float:
    return x


def relu(x: float) -> float:
    return max(0, x)


def leaky_relu(x: float) -> float:
    return 0.3 * x if x < 0 else x


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


# Derivatives


def d_linear(x: float) -> float:
    return 1.0


def d_relu(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0


def d_leaky_relu(x: float) -> float:
    return 0.3 if x < 0.0 else 1.0


def d_sigmoid(x: float) -> float:
    return x * (1.0 - x)

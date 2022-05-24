from math import exp


# Activations


def heavyside(x):
    return 1 if x >= 0 else 0


def linear(x):
    return x


def relu(x):
    return max(0, x)


def leaky_relu(x):
    return 0.3 * x if x < 0 else x


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


# Derivatives


def d_linear(x):
    return 1.0


def d_relu(x):
    return 0.0 if x < 0.0 else 1.0


def d_leaky_relu(x):
    return 0.3 if x < 0.0 else 1.0


def d_sigmoid(x):
    return x * (1.0 - x)

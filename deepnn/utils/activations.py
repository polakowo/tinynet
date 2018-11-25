import numpy as np


def ReLU(x, delta=False):
    if not delta:
        return np.maximum(0, x)
    else:
        return np.int64(x > 0)


def tanh(x, delta=False):
    if not delta:
        return np.tanh(x)
    else:
        return 1 - np.tanh(x) ** 2


def sigmoid(x, delta=False):
    if not delta:
        return 1 / (1 + np.exp(-x))
    else:
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)


def softmax(x):
    # Compute the softmax of vector x in a numerically stable way
    shiftx = x - np.max(x)
    e = np.exp(shiftx)
    return e / np.sum(e, axis=1, keepdims=True)


def softmax_delta(x, Y):
    return softmax(x) - Y

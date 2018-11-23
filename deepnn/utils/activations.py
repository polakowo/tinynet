import numpy as np

###########
# FORWARD #
###########


def ReLU(x):
    """
    RELU function
    """
    return np.maximum(0, x)


def tanh(x):
    """
    Hyperbolic tangent function
    """
    return np.tanh(x)


def sigmoid(x):
    """
    Sigmoid function
    """
    return 1 / (1 + np.exp(-x))

############
# BACKWARD #
############


def dReLU(x):
    """
    RELU function derivative
    """
    return np.int64(x > 0)


def dtanh(x):
    """
    Hyperbolic tangent function derivative
    """
    return 1 - np.tanh(x) ** 2


def dsigmoid(x):
    """
    Sigmoid function derivative
    """
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)

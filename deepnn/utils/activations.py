import numpy as np

###########
# FORWARD #
###########


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

############
# BACKWARD #
############


def dReLU(x):
    return np.int64(x > 0)


def dtanh(x):
    return 1 - np.tanh(x) ** 2


def dsigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)

import numpy as np

###########
# STAGE 1 #
###########


def linear_forward(A_prev, W, b):
    """
    Apply linear function to the parameters
    """
    Z = W.dot(A_prev) + b
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))

    linear_cache = (A_prev, W, b)
    return Z, linear_cache

###########
# STAGE 2 #
###########


def relu(Z):
    """
    RELU function
    """
    A = np.maximum(0, Z)

    activation_cache = Z
    return A, activation_cache


def tanh(Z):
    """
    Hyperbolic tangent function
    """
    A = np.tanh(Z)

    activation_cache = Z
    return A, activation_cache


def sigmoid(Z):
    """
    Sigmoid function
    """
    A = 1 / (1 + np.exp(-Z))

    activation_cache = Z
    return A, activation_cache


def activation_forward(Z, activation):
    """
    Apply activation function to the linear output
    """
    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'tanh':
        A, activation_cache = tanh(Z)
    elif activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)

    assert(A.shape == Z.shape)
    return A, activation_cache

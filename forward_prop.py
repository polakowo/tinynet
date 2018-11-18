import numpy as np

###########
# STAGE 1 #
###########


def linear_forward(A_prev, W, b):
    Z = W.dot(A_prev) + b
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))

    linear_cache = (A_prev, W, b)
    return Z, linear_cache

###########
# STAGE 2 #
###########


def relu(Z):
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)

    activation_cache = Z
    return A, activation_cache


def tanh(Z):
    A = np.tanh(Z)
    assert(A.shape == Z.shape)

    activation_cache = Z
    return A, activation_cache


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    assert(A.shape == Z.shape)

    activation_cache = Z
    return A, activation_cache


def activation_forward(Z, activation):
    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'tanh':
        A, activation_cache = tanh(Z)
    elif activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)

    return A, activation_cache

###########
# STAGE 3 #
###########


def dropout_forward(A, keep_prob):
    KEEP_MASK = np.random.rand(A.shape[0], A.shape[1])
    KEEP_MASK = KEEP_MASK < keep_prob
    A = A * KEEP_MASK
    A = A / keep_prob

    dropout_cache = KEEP_MASK
    return A, dropout_cache

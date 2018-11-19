import numpy as np

###########
# STAGE 1 #
###########


def dropout_backward(dA, dropout_cache, keep_prob):
    """
    Partial derivative of J with respect to activation output
    """
    # dJ/dA = dJ/dA' * dA'/dA
    KEEP_MASK = dropout_cache
    # Apply the mask to shut down the same neurons as during the forward propagation
    dA = dA * KEEP_MASK
    # Scale the value of neurons that haven't been shut down
    dA = dA / keep_prob

    return dA

###########
# STAGE 2 #
###########


def relu_derivative(Z):
    """
    RELU function derivative
    """
    dZ = np.int64(Z > 0)

    return dZ


def tanh_derivative(Z):
    """
    Hyperbolic tangent function derivative
    """
    dZ = 1 - np.tanh(Z) ** 2

    return dZ


def sigmoid_derivative(Z):
    """
    Sigmoid function derivative
    """
    s = 1 / (1 + np.exp(-Z))
    dZ = s * (1 - s)

    return dZ


def activation_backward(dA, activation_cache, activation):
    """
    Partial derivative of J with respect to linear output
    """
    Z = activation_cache

    if activation == 'relu':
        dA_dZ = relu_derivative(Z)
    elif activation == 'tanh':
        dA_dZ = tanh_derivative(Z)
    elif activation == 'sigmoid':
        dA_dZ = sigmoid_derivative(Z)

    # dJ/dZ = dJ/dA * dA/dZ
    dZ = dA * dA_dZ
    assert(dZ.shape == Z.shape)

    return dZ

###########
# STAGE 3 #
###########


def linear_backward(dZ, linear_cache, l2_lambda=0):
    """
    Partial derivative of J with respect to parameters
    """
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    # dJ/dW = dJ/dZ * dZ/dW
    dW = 1. / m * np.dot(dZ, A_prev.T)
    if l2_lambda > 0:
        # Penalize weights (weaken connections in the computational graph)
        dW += l2_lambda / m * W
    assert(dW.shape == W.shape)

    # dJ/db = dJ/dZ * dZ/db
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    assert(db.shape == b.shape)

    # dJ/dA_prev = dJ/dZ * dZ/dA_prev
    dA_prev = np.dot(W.T, dZ)
    assert(dA_prev.shape == A_prev.shape)

    return dA_prev, dW, db

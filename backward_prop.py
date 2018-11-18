import numpy as np

###########
# STAGE 1 #
###########


def dropout_backward(dA, dropout_cache, keep_prob):
    KEEP_MASK = dropout_cache
    dA = dA * KEEP_MASK
    dA = dA / keep_prob

    return dA

###########
# STAGE 2 #
###########


def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.multiply(dA, np.int64(Z > 0))
    assert(dZ.shape == Z.shape)

    return dZ


def tanh_backward(dA, activation_cache):
    Z = activation_cache
    s = 1 - np.tanh(Z) ** 2
    dZ = dA * s
    assert(dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert(dZ.shape == Z.shape)

    return dZ


def approx_gradient(fun, x, eps=0.1e-7):
    # Approximate derivative (gradient), error is eps^2
    return (fun(x + eps) - fun(x - eps)) / (2 * eps)


def activation_backward(dA, activation_cache, activation):
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'tanh':
        dZ = tanh_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)

    return dZ

###########
# STAGE 3 #
###########


def linear_backward(dZ, linear_cache, l2_lambda=0):
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    if l2_lambda > 0:
        # Penalize weights (weaken connections in the computational graph)
        dW += l2_lambda / m * W
    assert(dW.shape == W.shape)

    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    assert(db.shape == b.shape)

    dA_prev = np.dot(W.T, dZ)
    assert(dA_prev.shape == A_prev.shape)

    return dA_prev, dW, db

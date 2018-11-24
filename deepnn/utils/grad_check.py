import numpy as np


def roll_params(layers, grads=False):
    # Roll the parameters from layers into a single (n, 1) vector
    theta = np.zeros((0, 1))

    for layer in layers:
        if grads:
            vdict = layer.grads
        else:
            vdict = layer.params
        for k in vdict:
            vector = vdict[k]
            # Flatten the vector
            vector = np.reshape(vector, (-1, 1))
            # Append the vector
            theta = np.concatenate((theta, vector), axis=0)

    return theta


def unroll_params(theta, layers, grads=False):
    # Unroll the parameters from a vector and save to layers
    i = 0
    for layer in layers:
        if grads:
            vdict = layer.grads
        else:
            vdict = layer.params
        for k in vdict:
            vector = vdict[k]
            # Extract and reshape the parameter to the original form
            j = i + vector.shape[0] * vector.shape[1]
            vdict[k] = theta[i:j].reshape(vector.shape)
            i = j


def calculate_diff(grad_theta, grad_approx):
    # np.linalg.norm apply for matric equal to Frobenius norm
    numerator = np.linalg.norm(grad_theta - grad_approx)
    denominator = np.linalg.norm(grad_theta) + np.linalg.norm(grad_approx)
    diff = numerator / denominator
    return diff

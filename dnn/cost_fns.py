import numpy as np


def binary_crossentropy(out, Y, delta=False):
    # Binary classification
    m = out.shape[0]

    if not delta:
        with np.errstate(divide='ignore', invalid='ignore'):
            logprobs = Y * np.log(out) + (1 - Y) * np.log(1 - out)

            logprobs[logprobs == np.inf] = 0
            logprobs = np.nan_to_num(logprobs)
            return -1. / m * np.sum(logprobs)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            dX = -Y / (out) + (1 - Y) / (1 - out)

            dX[dX == np.inf] = 0
            dX = np.nan_to_num(dX)
            return dX


def categorical_crossentropy(out, Y, delta=False):
    # Multiclass classification
    m = out.shape[0]

    if not delta:
        with np.errstate(divide='ignore', invalid='ignore'):
            logprobs = Y * np.log(out)

            logprobs[logprobs == np.inf] = 0
            logprobs = np.nan_to_num(logprobs)
            return -1. / m * np.sum(logprobs)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            dX = -Y / out

            dX[dX == np.inf] = 0
            dX = np.nan_to_num(dX)
            return dX

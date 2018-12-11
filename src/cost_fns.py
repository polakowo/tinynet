import numpy as np


def cross_entropy(out, Y, delta=False):
    m = out.shape[0]

    if not delta:

        if Y.shape[1] == 1:
            # binary classification
            with np.errstate(divide='ignore', invalid='ignore'):
                logprobs = Y * np.log(out) + (1 - Y) * np.log(1 - out)
        else:
            # multiclass classification
            with np.errstate(divide='ignore', invalid='ignore'):
                logprobs = Y * np.log(out)

        logprobs[logprobs == np.inf] = 0
        logprobs = np.nan_to_num(logprobs)
        return -1. / m * np.sum(logprobs)

    else:
        if Y.shape[1] == 1:
            with np.errstate(divide='ignore', invalid='ignore'):
                dX = -Y / (out) + (1 - Y) / (1 - out)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                dX = -Y / out

        dX[dX == np.inf] = 0
        dX = np.nan_to_num(dX)
        return dX

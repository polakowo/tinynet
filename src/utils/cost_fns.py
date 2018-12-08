import numpy as np


def cross_entropy(output, Y, delta=False):
    m = output.shape[0]

    if not delta:

        if Y.shape[1] == 1:
            # binary classification
            with np.errstate(divide='ignore', invalid='ignore'):
                logprobs = Y * np.log(output) + (1 - Y) * np.log(1 - output)
        else:
            # multiclass classification
            with np.errstate(divide='ignore', invalid='ignore'):
                logprobs = Y * np.log(output)

        logprobs[logprobs == np.inf] = 0
        logprobs = np.nan_to_num(logprobs)
        return -1. / m * np.sum(logprobs)

    else:
        if Y.shape[1] == 1:
            with np.errstate(divide='ignore', invalid='ignore'):
                doutput = -Y / (output) + (1 - Y) / (1 - output)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                doutput = -Y / output

        doutput[doutput == np.inf] = 0
        doutput = np.nan_to_num(doutput)
        return doutput

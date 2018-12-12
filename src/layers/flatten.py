import numpy as np


class Flatten:
    """
    Flatten input
    """

    def __init__(self):
        pass

    def init_params(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = (1, np.prod(in_shape[1:]))

        self.params = None
        self.grads = None

    def forward(self, X, predict=False):
        out = X.reshape(X.shape[0], -1)

        if not predict:
            self.cache = X
        return out

    def backward(self, dout):
        X = self.cache

        dX = dout.reshape(X.shape)
        assert(dX.shape == X.shape)

        self.cache = None
        return dX

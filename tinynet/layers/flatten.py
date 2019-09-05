import numpy as np

from tinynet.layers import Layer

class Flatten(Layer):
    """Flatten layer"""

    def __init__(self):
        pass

    def init_params(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = (None, np.prod(in_shape[1:]))

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

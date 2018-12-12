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

    def forward(self, input, predict=False):
        output = input.reshape(input.shape[0], -1)

        if not predict:
            self.cache = input
        return output

    def backward(self, dout):
        input = self.cache

        dX = input.reshape(input.shape)
        assert(dX.shape == input.shape)

        self.cache = None
        return dX

from tinynet import activation_funcs
from tinynet.layers import Layer


class Activation(Layer):
    """Activation layer"""

    def __init__(self, activation_fn):
        self.activation_fn = activation_fn

    def init_params(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = in_shape

        self.params = None
        self.grads = None

    def forward(self, X, predict=False):
        out = self.activation_fn(X)

        if not predict:
            self.cache = X
        return out

    def backward(self, dout, Y):
        X = self.cache

        if self.activation_fn == activation_funcs.softmax:
            dX = activation_funcs.softmax_delta(X, Y)
            assert(dX.shape == X.shape)
        else:
            dX = dout * self.activation_fn(X, grad=True)
            assert(dX.shape == X.shape)

        self.cache = None
        return dX

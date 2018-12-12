import numpy as np

from src import initializers


class Dense:
    """
    Fully-connected layer
    """

    def __init__(self,
                 out_units,
                 weight_initializer=None,
                 bias_initializer=None):

        # The number of out_units in the layer
        self.out_units = out_units

        # Initializer for weights
        self.weight_initializer = weight_initializer

        # Initializer for biases
        self.bias_initializer = bias_initializer

    def init_params(self, in_shape):
        in_units = in_shape[1]
        self.in_shape = in_shape
        self.out_shape = (1, self.out_units)

        self.params = {}
        self.grads = {}

        # Poor initialization can lead to vanishing/exploding gradients
        # Random initialization is preferred to break symmetry
        if self.weight_initializer is None:
            weight_initializer = initializers.Xavier()
            self.params['W'] = weight_initializer.init_param(in_units, self.out_units)
        else:
            self.params['W'] = self.weight_initializer.init_param(in_units, self.out_units)

        if self.bias_initializer is None:
            self.params['b'] = np.zeros((1, self.out_units))
        else:
            self.params['b'] = self.bias_initializer.init_param(1, self.out_units)

    def forward(self, X, predict=False):
        W = self.params['W']
        b = self.params['b']

        out = np.dot(X, W) + b
        assert(out.shape == (X.shape[0], W.shape[1]))

        if not predict:
            self.cache = X
        return out

    def backward(self, dout):
        W = self.params['W']
        b = self.params['b']

        m = dout.shape[0]
        X = self.cache

        dX = np.dot(dout, W.T)
        assert(dX.shape == X.shape)

        dW = 1. / m * np.dot(X.T, dout)
        assert(dW.shape == W.shape)

        db = 1. / m * np.sum(dout, axis=0, keepdims=True)
        assert(db.shape == b.shape)

        self.grads['dW'] = dW
        self.grads['db'] = db

        self.cache = None
        return dX

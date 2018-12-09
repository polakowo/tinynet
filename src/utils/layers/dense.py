import numpy as np

from src.utils import initializers


class Dense:

    def __init__(self,
                 units,
                 weight_initializer=None,
                 bias_initializer=None):

        # The number of units in the layer
        self.units = units

        # Initializer for weights
        self.weight_initializer = weight_initializer

        # Initializer for biases
        self.bias_initializer = bias_initializer

    def init_params(self, prev_units):
        self.params = {}
        self.grads = {}

        # Poor initialization can lead to vanishing/exploding gradients
        # Random initialization is preferred to break symmetry
        if self.weight_initializer is None:
            weight_initializer = initializers.Xavier()
            self.params['W'] = weight_initializer.init_param(prev_units, self.units)
        else:
            self.params['W'] = self.weight_initializer.init_param(prev_units, self.units)

        if self.bias_initializer is None:
            self.params['b'] = np.zeros((1, self.units))
        else:
            self.params['b'] = self.bias_initializer.init_param(1, self.units)

    def forward(self, input, predict=False):
        W = self.params['W']
        b = self.params['b']

        output = np.dot(input, W) + b
        assert(output.shape == (input.shape[0], W.shape[1]))

        if not predict:
            self.cache = input
        return output

    def backward(self, dinput):
        W = self.params['W']
        b = self.params['b']

        m = dinput.shape[0]
        input = self.cache

        dW = 1. / m * np.dot(input.T, dinput)
        assert(dW.shape == W.shape)

        db = 1. / m * np.sum(dinput, axis=0, keepdims=True)
        assert(db.shape == b.shape)

        doutput = np.dot(dinput, W.T)
        assert(doutput.shape == input.shape)

        self.grads['dW'] = dW
        self.grads['db'] = db

        self.cache = None
        return doutput

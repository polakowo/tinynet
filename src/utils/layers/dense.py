import numpy as np

from src.utils import activation_fns
from src.utils import regularizers
from src.utils import initializers


class Dense:

    def __init__(self,
                 units,
                 activation_fn=activation_fns.tanh,
                 weight_initializer=None,
                 bias_initializer=None,
                 regularizer=None,
                 batch_norm=None,
                 rng=None):

        # The number of units in the layer
        self.units = units
        # (non-linear) activation function
        self.activation_fn = activation_fn
        # Initializer for weights
        self.weight_initializer = weight_initializer
        # Initializer for biases
        self.bias_initializer = bias_initializer
        # Layer-level regularization algorithm
        self.regularizer = regularizer
        # Batch normalization
        self.batch_norm = batch_norm

        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng

    def init_params(self, prev_units):
        self.params = {}
        self.cache = {}
        self.grads = {}

        # Poor initialization can lead to vanishing/exploding gradients
        # Random initialization is preferred to break symmetry
        if self.weight_initializer is None:
            weight_initializer = initializers.Xavier(rng=self.rng)
            self.params['W'] = weight_initializer.init_param(prev_units, self.units)
        else:
            self.params['W'] = self.weight_initializer.init_param(prev_units, self.units)

        if self.bias_initializer is None:
            self.params['b'] = np.zeros((1, self.units))
        else:
            self.params['b'] = self.bias_initializer.init_param(1, self.units)

        if self.batch_norm is not None:
            # Learn two extra parameters for every dimension to get optimum scaling and
            # shifting of activation outputs over zero means and unit variances towards
            # elimination of internal covariate shift.

            # There is no symmetry breaking to consider here
            # GD adapts their values to fit the corresponding feature's distribution
            self.params['gamma'] = np.ones((1, self.units))
            self.params['beta'] = np.zeros((1, self.units))

    #########################
    # FORWARD: FUN-1 -> FUN #
    #########################

    def linear_forward(self, input, W, b):
        output = np.dot(input, W) + b
        assert(output.shape == (input.shape[0], W.shape[1]))

        cache = (input, W, b)
        return output, cache

    def activation_forward(self, input):
        output = self.activation_fn(input)
        assert(output.shape == input.shape)

        cache = (input)
        return output, cache

    def propagate_forward(self, input, predict=False):
        output = input

        W = self.params['W']
        b = self.params['b']
        output, cache = self.linear_forward(output, W, b)
        if not predict:
            self.cache['linear'] = cache

        if self.batch_norm is not None:
            # Normalize the linear output
            # Cache is stored in the BatchNorm class
            # But parameters are stored in the layer
            gamma = self.params['gamma']
            beta = self.params['beta']
            if predict:
                output = self.batch_norm.forward_predict(output, gamma, beta)
            else:
                output, cache = self.batch_norm.forward(output, gamma, beta)
                self.cache['batch_norm'] = cache

        output, cache = self.activation_forward(output)
        if not predict:
            self.cache['activation'] = cache

        if not predict:
            if isinstance(self.regularizer, regularizers.Dropout):
                # Randomly shut down some neurons for each sample in input
                # Cache is stored in the Dropout class
                output, cache = self.regularizer.forward(output)
                self.cache['dropout'] = cache

        return output

    #########################
    # BACKWARD FUN-1 <- FUN #
    #########################

    def activation_backward(self, dinput, Y, cache):
        input = cache

        if self.activation_fn == activation_fns.softmax:
            doutput = activation_fns.softmax_delta(input, Y)
        else:
            doutput = dinput * self.activation_fn(input, delta=True)
        assert(doutput.shape == input.shape)

        return doutput

    def linear_backward(self, dinput, cache):
        m = dinput.shape[0]
        input, W, b = cache

        dW = 1. / m * np.dot(input.T, dinput)
        assert(dW.shape == W.shape)

        db = 1. / m * np.sum(dinput, axis=0, keepdims=True)
        assert(db.shape == b.shape)

        doutput = np.dot(dinput, W.T)
        assert(doutput.shape == input.shape)

        return doutput, dW, db

    def propagate_backward(self, dinput, Y):
        doutput = dinput

        if isinstance(self.regularizer, regularizers.Dropout):
            # Apply the mask to shut down the same neurons as in the forward propagation
            cache = self.cache['dropout']
            doutput = self.regularizer.backward(doutput, cache)

        cache = self.cache['activation']
        doutput = self.activation_backward(doutput, Y, cache)

        if self.batch_norm is not None:
            cache = self.cache['batch_norm']
            doutput, dgamma, dbeta = self.batch_norm.backward(doutput, cache)
            self.grads['dgamma'] = dgamma
            self.grads['dbeta'] = dbeta

        cache = self.cache['linear']
        doutput, dW, db = self.linear_backward(doutput, cache)
        self.grads['dW'] = dW
        self.grads['db'] = db

        return doutput

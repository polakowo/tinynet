import numpy as np

from deepnn.utils import activations
from deepnn.utils import regularizers


class Layer:
    def __init__(self,
                 n=None,
                 activation=activations.tanh,
                 init='xavier',
                 regularizer=None,
                 batch_norm=None,
                 rng=None):

        # The number of units in the layer
        assert(n is not None)
        self.n = n

        # (non-linear) activation function
        self.activation = activation

        # Initialization method
        self.init = init

        # Layer-level regularization algorithm
        self.regularizer = regularizer

        # Batch normalizer
        self.batch_norm = batch_norm

        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng

    def init_params(self, prev_n, **params):
        self.params = {}
        self.cache = {}
        self.grads = {}

        if 'W' in params:
            self.params['W'] = params['W']
        else:
            # Poor initialization can lead to vanishing/exploding gradients
            if self.init == 'xavier':
                # Random initialization is used to break symmetry
                self.params['W'] = self.rng.randn(prev_n, self.n) * np.sqrt(1. / prev_n)
            elif self.init == 'he':
                # He initialization works well for networks with ReLU activations
                self.params['W'] = self.rng.randn(prev_n, self.n) * np.sqrt(2. / prev_n)

        if 'b' in params:
            self.params['b'] = params['b']
        else:
            # Use zeros initialization for the biases
            self.params['b'] = np.zeros((1, self.n))

        if self.batch_norm is not None:
            # Learn two extra parameters for every dimension to get optimum scaling and
            # shifting of activation outputs over zero means and unit variances towards
            # elimination of internal covariate shift.

            if 'gamma' in params:
                self.params['gamma'] = params['gamma']
            else:
                # There is no symmetry breaking to consider here
                # GD adapts their values to fit the corresponding feature's distribution
                self.params['gamma'] = np.ones((1, self.n))

            if 'beta' in params:
                self.params['beta'] = params['beta']
            else:
                self.params['beta'] = np.zeros((1, self.n))

    #########################
    # FORWARD: FUN-1 -> FUN #
    #########################

    def linear_forward(self, input, W, b):
        output = np.dot(input, W) + b
        assert(output.shape == (input.shape[0], W.shape[1]))

        cache = (input, W, b)
        return output, cache

    def activation_forward(self, input):
        output = self.activation(input)
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

        if self.activation == activations.softmax:
            doutput = activations.softmax_delta(input, Y)
        else:
            doutput = dinput * self.activation(input, delta=True)
        assert(doutput.shape == input.shape)

        return doutput

    def linear_backward(self, dinput, cache):
        m = dinput.shape[0]
        input, W, b = cache

        dW = 1. / m * np.dot(input.T, dinput)
        assert(dW.shape == W.shape)

        if isinstance(self.regularizer, regularizers.L2):
            # Penalize weights (weaken connections in the computation graph)
            dW += self.regularizer.compute_term_derivative(W, m)

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

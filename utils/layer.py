import numpy as np

from utils import activations
from utils import regularizers


class Layer:
    def __init__(self,
                 n=None,
                 activation=activations.tanh,
                 init='xavier',
                 regularizer=None,
                 batch_normalizer=None,
                 rng=None):

        # The number of units in the layer
        assert(n is not None)
        self.n = n

        # Activation function
        # Must be a non-linear function
        self.activation = activation
        if activation == activations.tanh:
            self.dactivation = activations.dtanh
        elif activation == activations.sigmoid:
            self.dactivation = activations.dsigmoid
        elif activation == activations.ReLU:
            self.dactivation = activations.dReLU
        else:
            raise ValueError()

        # Initialization method
        self.init = init

        # Layer-level regularization algorithm
        self.regularizer = regularizer

        # Batch normalizer
        # # https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
        # Makes deeper layers more robust to changes to the weights in the previous layers
        # Also, similar to dropout, adds some noise to each hidden layer's activations
        # The smaller the mini-batches are, the more noise they produce, the more regularization takes place
        self.batch_normalizer = batch_normalizer

        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng

    def init_params(self, prev_n):
        self.params = {}
        self.cache = {}
        self.grads = {}

        # Poor initialization can lead to vanishing/exploding gradients
        if self.init == 'xavier':
            # Random initialization is used to break symmetry
            self.params['W'] = self.rng.randn(self.n, prev_n) * np.sqrt(1. / prev_n)
        elif self.init == 'he':
            # He initialization works well for networks with ReLU activations
            self.params['W'] = self.rng.randn(self.n, prev_n) * np.sqrt(2. / prev_n)
        # Use zeros initialization for the biases
        self.params['b'] = np.zeros((self.n, 1))

    #########################
    # FORWARD: FUN-1 -> FUN #
    #########################

    def linear_forward(self, input):
        """
        Apply linear function to the parameters
        """
        W = self.params['W']
        b = self.params['b']

        output = W.dot(input) + b
        assert(output.shape == (W.shape[0], input.shape[1]))

        return output

    def activation_forward(self, input):
        """
        Apply activation function to the previous output
        """
        output = self.activation(input)
        assert(output.shape == input.shape)

        return output

    def propagate_forward(self, input, train=True):
        """
        Forward propagation
        """
        output = input

        self.cache['linear'] = output
        output = self.linear_forward(output)

        self.cache['activation'] = output
        output = self.activation_forward(output)

        if train and isinstance(self.regularizer, regularizers.Dropout):
            # Randomly shut down some neurons for each sample in input
            # Cache is stored in the Dropout class
            output = self.regularizer.dropout_forward(output)

        return output

    #########################
    # BACKWARD FUN-1 <- FUN #
    #########################

    def activation_backward(self, dinput, cache):
        """
        Partial derivative of J with respect to linear output
        """
        doutput = dinput * self.dactivation(cache)
        assert(doutput.shape == cache.shape)

        return doutput

    def linear_backward(self, dinput, cache):
        """
        Partial derivative of J with respect to parameters
        """
        m = cache.shape[1]
        W = self.params['W']
        b = self.params['b']

        dW = 1. / m * np.dot(dinput, cache.T)
        assert(dW.shape == W.shape)

        if isinstance(self.regularizer, regularizers.L2):
            # Penalize weights (weaken connections in the computational graph)
            dW += self.regularizer.compute_term_derivative(W, m)

        db = 1. / m * np.sum(dinput, axis=1, keepdims=True)
        assert(db.shape == b.shape)

        doutput = np.dot(W.T, dinput)
        assert(doutput.shape == cache.shape)

        return doutput, dW, db

    def propagate_backward(self, dinput):
        """
        Backward propagation
        """
        doutput = dinput

        if isinstance(self.regularizer, regularizers.Dropout):
            # Apply the mask to shut down the same neurons as in the forward propagation
            doutput = self.regularizer.dropout_backward(doutput)

        cache = self.cache['activation']
        doutput = self.activation_backward(doutput, cache)

        cache = self.cache['linear']
        doutput, dW, db = self.linear_backward(doutput, cache)
        self.grads['dW'] = dW
        self.grads['db'] = db

        return doutput

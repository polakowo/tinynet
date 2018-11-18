import numpy as np

from forward_prop import linear_forward, activation_forward, dropout_forward
from backward_prop import linear_backward, activation_backward, dropout_backward


class DeepNN:

    def __init__(self, **hyperparams):
        self.layer_dims = hyperparams['layer_dims']
        self.activations = hyperparams['activations']
        assert(len(self.activations) == len(self.layer_dims))

        self.learning_rate = hyperparams['learning_rate']
        self.num_iterations = hyperparams['num_iterations']
        if 'initialization' not in hyperparams:
            self.initialization = 'xavier'
        else:
            self.initialization = hyperparams['initialization']
        # L2 regularization
        if 'l2_lambda' not in hyperparams:
            self.l2_lambda = 0.
        else:
            self.l2_lambda = hyperparams['l2_lambda']
        # Dropout regularization
        if 'keep_probs' not in hyperparams:
            self.keep_probs = [1.] * len(self.layer_dims)
        else:
            self.keep_probs = hyperparams['keep_probs']
            assert(len(self.keep_probs) == len(self.layer_dims))

    ################
    # FORWARD PROP #
    ################

    def propagate_forward(self, X, predict=False):
        caches = []
        A = X

        for l in range(len(self.layer_dims)):
            A_prev = A
            W = self.params['W' + str(l)]
            b = self.params['b' + str(l)]
            activation = self.activations[l]
            keep_prob = self.keep_probs[l]

            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = activation_forward(Z, activation)

            dropout_cache = None
            if not predict and keep_prob < 1:
                # Randomly shut down some neurons for each iteration and dataset
                A, dropout_cache = dropout_forward(A, keep_prob)

            cache = (linear_cache, activation_cache, dropout_cache)
            caches.append(cache)

        return A, caches

    ########
    # COST #
    ########

    def compute_cost(self, AL, Y):
        m = Y.shape[1]

        # Cross-entropy
        with np.errstate(divide='ignore', invalid='ignore'):
            # Handle inf in np.log
            logprobs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
            logprobs[logprobs == np.inf] = 0
            logprobs = np.nan_to_num(logprobs)

        cost = 1. / m * np.nansum(logprobs)
        if self.l2_lambda > 0:
            # L2 regularization cost
            L2 = np.sum([np.sum(np.square(self.params['W' + str(l)])) for l in range(len(self.layer_dims))])
            cost += 1 / 2 * self.l2_lambda / m * L2
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        return cost

    #################
    # BACKWARD PROP #
    #################

    def propagate_backward(self, AL, Y, caches):
        np.random.seed(1)
        grads = {}
        Y = Y.reshape(AL.shape)
        with np.errstate(divide='ignore', invalid='ignore'):
            # Handle division by zero in np.divide
            dA = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
            dA[dA == np.inf] = 0
            dA = np.nan_to_num(dA)

        for l in reversed(range(len(self.layer_dims))):
            linear_cache, activation_cache, dropout_cache = caches[l]
            keep_prob = self.keep_probs[l]
            activation = self.activations[l]

            if dropout_cache is not None:
                # Apply mask to shut down the same neurons
                dA = dropout_backward(dA, dropout_cache, keep_prob)

            dZ = activation_backward(dA, activation_cache, activation)
            dA_prev, dW, db = linear_backward(dZ, linear_cache, self.l2_lambda)
            grads['dA' + str(l - 1)] = dA_prev
            grads['dW' + str(l)] = dW
            grads['db' + str(l)] = db

            dA = dA_prev

        return grads

    #################
    # UPDATE PARAMS #
    #################

    def update_params(self, grads):
        for l in range(len(self.layer_dims)):
            self.params['W' + str(l)] = self.params['W' + str(l)] - self.learning_rate * grads['dW' + str(l)]
            self.params['b' + str(l)] = self.params['b' + str(l)] - self.learning_rate * grads['db' + str(l)]

    #########
    # TRAIN #
    #########

    def initialize_params(self):
        np.random.seed(3)
        self.params = {}

        for l in range(len(self.layer_dims)):
            prev_layer_dim = self.layer_dims[l - 1] if l > 0 else self.X.shape[0]
            this_layer_dim = self.layer_dims[l]

            # Poor initialization can lead to vanishing/exploding gradients
            # Random initialization is used to break symmetry
            # He initialization works well for networks with ReLU activations
            if self.initialization == 'xavier':
                self.params['W' + str(l)] = np.random.randn(this_layer_dim, prev_layer_dim) \
                    * np.sqrt(1. / prev_layer_dim)
            elif self.initialization == 'he':
                self.params['W' + str(l)] = np.random.randn(this_layer_dim, prev_layer_dim) \
                    * np.sqrt(2. / prev_layer_dim)
            self.params['b' + str(l)] = np.zeros((this_layer_dim, 1))

    def train(self, X, Y, print_output=False):
        self.X = X
        self.Y = Y
        self.initialize_params()

        costs = []
        for i in range(0, self.num_iterations):
            AL, caches = self.propagate_forward(X)
            cost = self.compute_cost(AL, Y)
            grads = self.propagate_backward(AL, Y, caches)
            self.update_params(grads)

            if print_output and i % 10000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
                costs.append(cost)

    ###########
    # PREDICT #
    ###########

    def predict(self, X, Y):
        m = X.shape[1]

        probs, caches = self.propagate_forward(X, predict=True)
        probs = np.array(probs, copy=True)
        probs[probs <= 0.5] = 0
        probs[probs > 0.5] = 1
        accuracy = str(np.sum((probs == Y) / m))

        return probs, accuracy

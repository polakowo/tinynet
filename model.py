import numpy as np

from forward_prop import linear_forward, activation_forward, dropout_forward
from backward_prop import linear_backward, activation_backward, dropout_backward
from gradient_check import params_to_vector, vector_to_params, calculate_diff


class DeepNN:

    def __init__(self, **hyperparams):
        """
        Initialize the class
        """
        # The number of units in each layer
        self.layer_dims = hyperparams['layer_dims']

        # Activation function in each layer
        self.activations = hyperparams['activations']
        assert(len(self.activations) == len(self.layer_dims))

        # The key differentiator between convergence and divergence
        # Don't set too high
        self.learning_rate = hyperparams['learning_rate']

        # Number of iterations of gradient descent
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
        # Randomly shut down some neurons in each iteration
        # With dropout, neurons become less sensitive to the activation of one other specific neuron
        # Use dropout only during training, not during test time
        if 'keep_probs' not in hyperparams:
            self.keep_probs = np.ones(len(self.layer_dims))
        else:
            self.keep_probs = np.array(hyperparams['keep_probs'])
            assert(len(self.keep_probs) == len(self.layer_dims))

        # Specify seed to yield different initializations and dropouts
        if 'seed' not in hyperparams:
            np.random.seed(1)
        else:
            np.random.seed(hyperparams['seed'])

    #########
    # TRAIN #
    #########

    def initialize_params(self, X):
        """
        Initialize the weights and the biases
        """
        params = {}

        for l in range(len(self.layer_dims)):
            prev_layer_dim = self.layer_dims[l - 1] if l > 0 else X.shape[0]
            this_layer_dim = self.layer_dims[l]

            # Poor initialization can lead to vanishing/exploding gradients
            # Don't intialize to values that are too large
            # Random initialization is used to break symmetry
            if self.initialization == 'xavier':
                params['W' + str(l)] = np.random.randn(this_layer_dim, prev_layer_dim) \
                    * np.sqrt(1. / prev_layer_dim)
            # He initialization works well for networks with ReLU activations
            elif self.initialization == 'he':
                params['W' + str(l)] = np.random.randn(this_layer_dim, prev_layer_dim) \
                    * np.sqrt(2. / prev_layer_dim)
            # Use zeros initialization for the biases
            params['b' + str(l)] = np.zeros((this_layer_dim, 1))

        return params

    def train(self, X, Y, print_output=False):
        """
        Train an n-layer neural network
        """
        # Initialize parameters dictionary
        params = self.initialize_params(X)

        costs = []
        # Gradient descent
        for i in range(0, self.num_iterations):
            AL, caches = self.propagate_forward(X, params)
            cost = self.compute_cost(AL, Y, params)
            grads = self.propagate_backward(AL, Y, caches)
            params = self.update_params(params, grads)

            if print_output and i % 10000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
                costs.append(cost)

        # Store parameters as a class variable for later predictions
        self.params = params

        return costs

    ################
    # FORWARD PROP #
    ################

    def propagate_forward(self, X, params, predict=False):
        """
        Propagate forwards to calculate the caches and the output
        """
        caches = []
        A = X

        for l in range(len(self.layer_dims)):
            A_prev = A
            W = params['W' + str(l)]
            b = params['b' + str(l)]
            activation = self.activations[l]
            keep_prob = self.keep_probs[l]

            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = activation_forward(Z, activation)

            dropout_cache = None
            if not predict and keep_prob < 1:
                # Randomly shut down some neurons for each iteration and dataset
                A, dropout_cache = dropout_forward(A, keep_prob)

            # Used in calculating derivatives
            cache = (linear_cache, activation_cache, dropout_cache)
            caches.append(cache)

        return A, caches

    ########
    # COST #
    ########

    def compute_cost(self, AL, Y, params):
        """
        Calculate the cost
        """
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
            L2 = np.sum([np.sum(np.square(params['W' + str(l)])) for l in range(len(self.layer_dims))])
            cost += 1 / 2 * self.l2_lambda / m * L2
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        return cost

    #################
    # BACKWARD PROP #
    #################

    def propagate_backward(self, AL, Y, caches):
        """
        Propagate backwards to derive the gradients
        """
        grads = {}
        Y = Y.reshape(AL.shape)

        with np.errstate(divide='ignore', invalid='ignore'):
            # Handle division by zero in np.divide
            dA = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
            dA[dA == np.inf] = 0
            dA = np.nan_to_num(dA)

        # For each layer, calculate the gradients of the parameters
        # Move from the last layer to the first
        for l in reversed(range(len(self.layer_dims))):
            linear_cache, activation_cache, dropout_cache = caches[l]
            keep_prob = self.keep_probs[l]
            activation = self.activations[l]

            if keep_prob < 1:
                # Apply the mask to shut down the same neurons as in the forward propagation
                dA = dropout_backward(dA, dropout_cache, keep_prob)

            dZ = activation_backward(dA, activation_cache, activation)
            dA_prev, dW, db = linear_backward(dZ, linear_cache, self.l2_lambda)
            grads['dW' + str(l)] = dW
            grads['db' + str(l)] = db

            dA = dA_prev

        return grads

    #################
    # UPDATE PARAMS #
    #################

    def update_params(self, params, grads):
        """
        Update the parameters using gradient descent
        """
        for l in range(len(self.layer_dims)):
            # Update rule for each parameter
            params['W' + str(l)] = params['W' + str(l)] - self.learning_rate * grads['dW' + str(l)]
            params['b' + str(l)] = params['b' + str(l)] - self.learning_rate * grads['db' + str(l)]

        return params

    ##################
    # GRADIENT CHECK #
    ##################

    def gradient_check(self, X, Y, epsilon=1e-7):
        """
        Check whether backpropagation computes the gradients correctly
        """
        # http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
        # Don't use with dropout
        assert(np.all(self.keep_probs == 1.))

        # One iteration of gradient descent to get gradients
        params = self.initialize_params(X)
        AL, caches = self.propagate_forward(X, params)
        grads = self.propagate_backward(AL, Y, caches)

        # Roll parameters dictionary into a large (n, 1) vector
        param_keys = [key + str(l)
                      for l in range(len(self.layer_dims))
                      for key in ('W', 'b')]
        param_theta, param_cache = params_to_vector(params, param_keys)

        grad_keys = [key + str(l)
                     for l in range(len(self.layer_dims))
                     for key in ('dW', 'db')]
        grad_theta, _ = params_to_vector(grads, grad_keys)

        num_params = param_theta.shape[0]
        J_plus = np.zeros((num_params, 1))
        J_minus = np.zeros((num_params, 1))
        gradapprox = np.zeros((num_params, 1))

        # Repeat for each number (parameter) in the vector
        for i in range(num_params):
            # Use two-sided Taylor approximation which is 2x more precise than one-sided
            # Add epsilon to the parameter
            theta_plus = np.copy(param_theta)
            theta_plus[i][0] = theta_plus[i][0] + epsilon
            # Calculate new cost
            theta_plus_params = vector_to_params(theta_plus, param_cache)
            AL_plus, _ = self.propagate_forward(X, theta_plus_params)
            J_plus[i] = self.compute_cost(AL_plus, Y, theta_plus_params)

            # Subtract epsilon from the parameter
            theta_minus = np.copy(param_theta)
            theta_minus[i][0] = theta_minus[i][0] - epsilon
            # Calculate new cost
            thetha_minus_params = vector_to_params(theta_minus, param_cache)
            AL_minus, _ = self.propagate_forward(X, thetha_minus_params)
            J_minus[i] = self.compute_cost(AL_minus, Y, thetha_minus_params)

            # Approximate the partial derivative, error is eps^2
            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

        # Difference between the approximated gradient and the backward propagation gradient
        diff = calculate_diff(grad_theta, gradapprox)
        if diff > 2e-7:
            print("\033[93m" + "Failed" + "\033[0m")
        else:
            print("\033[92m" + "Passed" + "\033[0m")

        return diff

        ###########
        # PREDICT #
        ###########

    def predict(self, X, Y, threshold=0.5):
        """
        Predict using forward propagation and a classification threshold
        """
        m = X.shape[1]

        # Propagate forward with the parameters learned previously
        probs, caches = self.propagate_forward(X, self.params, predict=True)
        # Classify the probabilities
        probs = np.array(probs, copy=True)
        probs[probs <= threshold] = 0
        probs[probs > threshold] = 1
        # Compute the fraction of correct predictions
        accuracy = str(np.sum((probs == Y) / m))

        return probs, accuracy

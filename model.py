import numpy as np
import math

from tqdm import trange
import asciichartpy
from colorama import Fore
from tabulate import tabulate

from utils import forward_prop
from utils import backward_prop
from utils import regularizers
from utils import optimizers

# TODO: Data augmentation


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
        self.num_epochs = hyperparams['num_epochs']
        if 'initialization' not in hyperparams:
            self.initialization = 'xavier'
        else:
            self.initialization = hyperparams['initialization']

        # Mini-batch gradient descent
        # Faster if the size is a power of 2, usually from 64 to 512
        # Make sure that a single mini-batch fits into the CPU/GPU memory
        if 'mini_batch_size' not in hyperparams:
            self.mini_batch_size = None
        else:
            self.mini_batch_size = hyperparams['mini_batch_size']

        # Regularization techniques
        if 'regularizer' not in hyperparams:
            self.regularizer = None
        else:
            self.regularizer = hyperparams['regularizer']

        # Optimizations techniques
        if 'optimizer' not in hyperparams:
            self.optimizer = None
        else:
            self.optimizer = hyperparams['optimizer']

        # Specify seed to yield different initializations and dropouts
        if 'seed' not in hyperparams:
            self.seed = 1
        else:
            self.seed = hyperparams['seed']

    #####################
    # INITIALIZE PARAMS #
    #####################

    def initialize_params(self, X):
        """
        Initialize the weights and the biases
        """
        np.random.seed(self.seed)
        # Some functions outside this class may need it
        params = {
            'L': len(self.layer_dims)
        }

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

            Z, linear_cache = forward_prop.linear_forward(A_prev, W, b)
            A, activation_cache = forward_prop.activation_forward(Z, activation)

            dropout_cache = None
            if not predict and isinstance(self.regularizer, regularizers.Dropout):
                # Randomly shut down some neurons for each iteration and dataset
                A, dropout_cache = self.regularizer.dropout_forward(A, l)

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

        if isinstance(self.regularizer, regularizers.L2):
            # Add L2 regularization term to the cost
            term = self.regularizer.compute_term(params, m)
            cost += term

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
        # Always specify L to iterate over keys safely
        grads = {
            'L': len(self.layer_dims)
        }
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
            activation = self.activations[l]

            if isinstance(self.regularizer, regularizers.Dropout):
                # Apply the mask to shut down the same neurons as in the forward propagation
                dA = self.regularizer.dropout_backward(dA, dropout_cache, l)

            dZ = backward_prop.activation_backward(dA, activation_cache, activation)

            dA_prev, dW, db = backward_prop.linear_backward(dZ, linear_cache, regularizer=self.regularizer)

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

    #########
    # TRAIN #
    #########

    def generate_mini_batches(self, X, Y, seed=None):
        if seed is not None:
            np.random.seed(seed)
        m = X.shape[1]
        mini_batches = []

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1, m))

        # Step 2: Partition (shuffled_X, shuffled_Y)
        num_mini_batches = math.floor(m / self.mini_batch_size)
        for k in range(num_mini_batches + 1):
            mini_batch_X = shuffled_X[:, k * self.mini_batch_size: (k + 1) * self.mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * self.mini_batch_size: (k + 1) * self.mini_batch_size]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def train(self, X, Y, print_overview=True, print_progress=True, print_cost_chart=True):
        """
        Train an L-layer neural network

        X must be of shape (n, m)
        Y must be of shape (1, m)
        where n is the number of features and m is the number of datasets
        """
        # Overview over the dataset
        if print_overview:
            print("Overview:")

            if self.mini_batch_size is None:
                mini_batch_size = X.shape[1]
            else:
                mini_batch_size = self.mini_batch_size
            print(tabulate([[
                X.shape[0],
                X.shape[1],
                mini_batch_size,
                math.floor(X.shape[1] / mini_batch_size),
                self.num_epochs
            ]],
                headers=[
                    'n',
                    'm',
                    'batch-size',
                    'batches',
                    'epochs'
            ]))

        # Initialize parameters dictionary
        params = self.initialize_params(X)

        # Initialize the optimizer
        if isinstance(self.optimizer, optimizers.Momentum):
            self.optimizer.initialize_params(params)
        elif isinstance(self.optimizer, optimizers.Adam):
            self.optimizer.initialize_params(params)

        costs = []
        # Progress information is displayed and updated dynamically in the console
        if print_progress:
            print("Progress:")
        with trange(self.num_epochs,
                    disable=not print_progress,
                    bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.YELLOW, Fore.RESET),
                    ncols=100) as t:
            for i in t:
                if self.mini_batch_size is not None:
                    # Divide the dataset into mini-batched based on their size
                    # We increment the seed to reshuffle differently the dataset after each epoch
                    mini_batches = self.generate_mini_batches(X, Y, seed=self.seed + i)
                else:
                    # Batch gradient descent
                    mini_batches = [(X, Y)]

                for mini_batch in mini_batches:
                    # Unpack the mini-batch
                    mini_X, mini_Y = mini_batch

                    # Forward propagation
                    AL, caches = self.propagate_forward(mini_X, params)

                    # Compute cost
                    cost = self.compute_cost(AL, mini_Y, params)
                    costs.append(cost)

                    # Backward propagation
                    grads = self.propagate_backward(AL, mini_Y, caches)

                    # Update parameters
                    if isinstance(self.optimizer, optimizers.Momentum):
                        params = self.optimizer.update_params(params, grads, self.learning_rate)
                    elif isinstance(self.optimizer, optimizers.Adam):
                        params = self.optimizer.update_params(params, grads, self.learning_rate)
                    else:
                        params = self.update_params(params, grads)

        # Store parameters as a class variable
        self.params = params
        # Success: The model has been trained

        # Plot the cost as function of time in the console
        if print_cost_chart:
            print("Cost chart:")
        cfg = {
            'height': 5
        }
        print("%s%s%s" % (Fore.YELLOW, asciichartpy.plot(costs[::(len(costs) // 50)], cfg), Fore.RESET))

        return costs

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

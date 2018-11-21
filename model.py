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


class DeepNN:

    def __init__(self, **cfg):

        # The number of units in each layer
        self.layer_dims = cfg['layer_dims']

        # Activation function in each layer
        self.activations = cfg['activations']
        assert(len(self.activations) == len(self.layer_dims))

        # The key differentiator between convergence and divergence
        # Can be a function of epoch
        self.learning_rate = cfg['learning_rate']

        # Number of iterations of gradient descent
        self.num_epochs = cfg['num_epochs']

        # Initialization of weights and biases
        self.initialization = cfg['initialization'] if 'initialization' in cfg else 'xavier'

        # Mini-batch gradient descent
        # Powers of two are often chosen to be the mini-batch size, e.g., 64, 128
        # Make sure that a single mini-batch fits into the CPU/GPU memory
        self.mini_batch_size = cfg['mini_batch_size'] if 'mini_batch_size' in cfg else None

        # Regularization algorithm
        self.regularizer = cfg['regularizer'] if 'regularizer' in cfg else None

        # Optimizations algorithm
        self.optimizer = cfg['optimizer'] if 'optimizer' in cfg else None

        # Batch normalizer
        # As a normalizer:
        # Makes deeper layers more robust to changes to the weights in the previous layers
        # Having the same mean and variance across nodes makes the job of later layers easier
        # As a regularizer:
        # Similar to dropout, it adds some noise to each hidden layer's activations
        # The smaller the mini-batches are, the more noise they produce, the more regularization takes place
        # At test time:
        # The dataset size during training usually differs from that during testing
        # Moving averages of mean and variance to the rescue!
        self.batch_norm = cfg['batch_norm'] if 'batch_norm' in cfg else False

        # Specify seed to yield different initializations and dropouts
        self.seed = cfg['seed'] if 'seed' in cfg else 1

    #####################
    # INITIALIZE PARAMS #
    #####################

    def initialize_params(self, X):
        """
        Initialize the weights and the biases
        """
        np.random.seed(self.seed)
        # Some functions outside this class may need it
        layer_params = []

        for l in range(len(self.layer_dims)):
            prev_layer_dim = self.layer_dims[l - 1] if l > 0 else X.shape[0]
            this_layer_dim = self.layer_dims[l]

            # Poor initialization can lead to vanishing/exploding gradients
            # Don't intialize to values that are too large
            # Random initialization is used to break symmetry
            if self.initialization == 'xavier':
                W = np.random.randn(this_layer_dim, prev_layer_dim) * np.sqrt(1. / prev_layer_dim)
            # He initialization works well for networks with ReLU activations
            elif self.initialization == 'he':
                W = np.random.randn(this_layer_dim, prev_layer_dim) * np.sqrt(2. / prev_layer_dim)
            # Use zeros initialization for the biases
            b = np.zeros((this_layer_dim, 1))

            layer_params.append({
                'W': W,
                'b': b
            })

        return layer_params

    ################
    # FORWARD PROP #
    ################

    def propagate_forward(self, X, layer_params, predict=False):
        """
        Propagate forwards to calculate the caches and the output
        """
        caches = []
        A = X

        for l in range(len(self.layer_dims)):
            A_prev = A
            W = layer_params[l]['W']
            b = layer_params[l]['b']
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

    def compute_cost(self, AL, Y, layer_params):
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
            term = self.regularizer.compute_term(layer_params, m)
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
        Y = Y.reshape(AL.shape)
        layer_grads = []

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

            layer_grads.append({
                'dW': dW,
                'db': db
            })
            dA = dA_prev

        # We iterated in reversed order
        layer_grads = layer_grads[::-1]
        return layer_grads

    #################
    # UPDATE PARAMS #
    #################

    def update_params(self, layer_params, layer_grads, learning_rate):
        """
        Update the parameters using gradient descent
        """
        for l in range(len(layer_params)):
            for k in layer_params[l]:
                # Update rule for each parameter
                layer_params[l][k] -= learning_rate * layer_grads[l]['d' + k]

        return layer_params

    #########
    # TRAIN #
    #########

    def generate_mini_batches(self, X, Y, seed=None):
        """
        Shuffe and partition the dataset to build mini-batches
        """
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

    def train(self, X, Y, print_dataset=False, print_progress=False, print_cost=False):
        """
        Train an L-layer neural network

        X must be of shape (n, m)
        Y must be of shape (1, m)
        where n is the number of features and m is the number of datasets
        """

        # Overview of the complexity
        if print_dataset:
            print(Fore.BLUE + '-' * 100 + Fore.RESET)
            print("Dataset:")

            columns = []
            columns.append(('features', X.shape[0]))
            columns.append(('examples', X.shape[1]))
            if self.mini_batch_size is not None:
                columns.append(('mini-batch-size', self.mini_batch_size))
                columns.append(('mini-batches', math.floor(X.shape[1] / self.mini_batch_size)))

            headers, row = zip(*columns)
            print(tabulate([row], headers=headers, tablefmt="presto"))

        # Initialize parameters dictionary
        layer_params = self.initialize_params(X)

        # Initialize the optimizer
        if isinstance(self.optimizer, optimizers.Momentum):
            self.optimizer.initialize_params(layer_params)
        elif isinstance(self.optimizer, optimizers.Adam):
            self.optimizer.initialize_params(layer_params)

        costs = []
        # Progress information is displayed and updated dynamically in the console
        if print_progress:
            print(Fore.BLUE + '-' * 100 + Fore.RESET)
            print("Progress:")
        with trange(self.num_epochs,
                    disable=not print_progress,
                    bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.YELLOW, Fore.RESET),
                    ascii=True,
                    ncols=100) as t:
            for epoch in t:
                if self.mini_batch_size is not None:
                    # Divide the dataset into mini-batched based on their size
                    # We increment the seed to reshuffle differently the dataset after each epoch
                    mini_batches = self.generate_mini_batches(X, Y, seed=self.seed + epoch)
                else:
                    # Batch gradient descent
                    mini_batches = [(X, Y)]

                for mini_batch in mini_batches:
                    # Unpack the mini-batch
                    mini_X, mini_Y = mini_batch

                    # Forward propagation
                    AL, caches = self.propagate_forward(mini_X, layer_params)

                    # Compute cost
                    cost = self.compute_cost(AL, mini_Y, layer_params)
                    costs.append(cost)
                    t.set_description("Cost %.2f" % cost)

                    # Backward propagation
                    layer_grads = self.propagate_backward(AL, mini_Y, caches)

                    # Update parameters
                    if callable(self.learning_rate):
                        learning_rate = self.learning_rate(epoch)
                    else:
                        learning_rate = self.learning_rate

                    if isinstance(self.optimizer, optimizers.Momentum):
                        layer_params = self.optimizer.update_params(layer_params, layer_grads, learning_rate)
                    elif isinstance(self.optimizer, optimizers.Adam):
                        layer_params = self.optimizer.update_params(layer_params, layer_grads, learning_rate)
                    else:
                        layer_params = self.update_params(layer_params, layer_grads, learning_rate)

        # Store parameters as a class variable
        self.layer_params = layer_params
        # Success: The model has been trained

        costs = np.array(costs)
        # Print cost as a function of time
        if print_cost:
            print(Fore.BLUE + '-' * 100 + Fore.RESET)
            print("Cost development:")
            points = 89
            step_costs = costs[[max(0, math.floor(i / points * len(costs)) - 1) for i in range(0, points + 1)]]
            print(Fore.YELLOW + asciichartpy.plot(step_costs, {'height': 4}) + Fore.RESET)

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
        probs, caches = self.propagate_forward(X, self.layer_params, predict=True)
        # Classify the probabilities
        probs = np.array(probs, copy=True)
        probs[probs <= threshold] = 0
        probs[probs > threshold] = 1
        # Compute the fraction of correct predictions
        accuracy = np.sum((probs == Y) / m)

        return probs, accuracy

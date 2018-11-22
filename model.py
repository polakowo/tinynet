import numpy as np
import math

from tqdm import trange
import asciichartpy
from colorama import Fore
from tabulate import tabulate

from utils import regularizers
from utils import optimizers


class DeepNN:

    def __init__(self,
                 layers=None,
                 lr=None,
                 num_epochs=None,
                 mini_batch_size=None,
                 optimizer=None,
                 regularizer=None):

        # A list of layer instances
        assert(layers is not None)
        self.layers = layers

        # Learning rate
        # A key differentiator between convergence and divergence
        # Can be a function of epoch
        assert(lr is not None)
        self.lr = lr

        # Number of iterations of gradient descent
        assert(num_epochs is not None)
        self.num_epochs = num_epochs

        # Mini-batch gradient descent
        # Powers of two are often chosen to be the mini-batch size, e.g., 64, 128
        # Make sure that a single mini-batch fits into the CPU/GPU memory
        self.mini_batch_size = mini_batch_size

        # Optimizations algorithm
        self.optimizer = optimizer

        # Network-level regularization algorithm
        self.regularizer = regularizer
        if isinstance(regularizer, regularizers.Dropout) or isinstance(regularizer, regularizers.L2):
            for layer in layers:
                layer.regularizer = regularizer

    ###############
    # INIT PARAMS #
    ###############

    def init_params(self, X):
        """
        Initialize params in each layer
        """
        for index, layer in enumerate(self.layers):
            prev_n = self.layers[index - 1].n if index > 0 else X.shape[0]

            layer.init_params(prev_n)

    #######################
    # FORWARD PROPAGATION #
    #######################

    def propagate_forward(self, X, train=True):
        """
        Propagate forwards to calculate the output
        """
        output = X
        for index, layer in enumerate(self.layers):
            output = layer.propagate_forward(output, train=train)

        return output

    ########
    # COST #
    ########

    def compute_cost(self, output, Y):
        """
        Calculate the cost
        """
        m = Y.shape[1]

        # Cross-entropy
        with np.errstate(divide='ignore', invalid='ignore'):
            # Handle inf in np.log
            logprobs = np.multiply(-np.log(output), Y) + np.multiply(-np.log(1 - output), 1 - Y)
            logprobs[logprobs == np.inf] = 0
            logprobs = np.nan_to_num(logprobs)
        cost = 1. / m * np.nansum(logprobs)

        if isinstance(self.regularizer, regularizers.L2):
            # Add L2 regularization term to the cost
            cost += self.regularizer.compute_term(self.layers, m)

        cost = np.squeeze(cost)
        assert(cost.shape == ())

        return cost

    ########################
    # BACKWARD PROPAGATION #
    ########################

    def propagate_backward(self, output, Y):
        """
        Propagate backwards to derive the gradients in each layer
        """
        Y = Y.reshape(output.shape)

        with np.errstate(divide='ignore', invalid='ignore'):
            # Handle division by zero in np.divide
            doutput = -(np.divide(Y, output) - np.divide(1 - Y, 1 - output))
            doutput[doutput == np.inf] = 0
            doutput = np.nan_to_num(doutput)

        # For each layer, calculate the gradients of the parameters
        # Move from the last layer to the first
        for layer in reversed(self.layers):
            doutput = layer.propagate_backward(doutput)
            # Gradients are now stored in the layer instance

    #################
    # UPDATE PARAMS #
    #################

    def update_params(self):
        """
        Update the parameters using gradient descent
        """
        for layer in self.layers:
            for key in layer.params:
                # Update the rule for each parameter in each layer
                layer.params[key] = layer.params[key] - self.lr * layer.grads['d' + key]

    #########
    # TRAIN #
    #########

    def generate_mini_batches(self, X, Y, rng=None):
        """
        Shuffe and partition the dataset to build mini-batches
        """
        if rng is None:
            pass
        m = X.shape[1]
        mini_batches = []

        # Step 1: Shuffle (X, Y)
        permutation = list(rng.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1, m))

        # Step 2: Partition (shuffled_X, shuffled_Y)
        num_mini_batches = math.floor(m / self.mini_batch_size)
        for i in range(num_mini_batches + 1):
            mini_batch_X = shuffled_X[:, i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            mini_batch_Y = shuffled_Y[:, i * self.mini_batch_size: (i + 1) * self.mini_batch_size]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def train(self, X, Y,
              print_datainfo=False,
              print_progress=True,
              print_coststats=False,
              print_costdev=False):
        """
        Train an L-layer neural network

        X must be of shape (n, m)
        Y must be of shape (1, m)
        where n is the number of features and m is the number of datasets
        """

        # Overview of the complexity
        if print_datainfo:
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
        self.init_params(X)

        # Initialize the optimizer
        if isinstance(self.optimizer, optimizers.Momentum):
            self.optimizer.init_params(self.layers)
        elif isinstance(self.optimizer, optimizers.Adam):
            self.optimizer.init_params(self.layers)

        costs = []
        # Progress information is displayed and updated dynamically in the console
        if print_progress:
            print(Fore.BLUE + '-' * 100 + Fore.RESET)
            print("Progress:")
        with trange(self.num_epochs,
                    disable=not print_progress,
                    ncols=100) as pbar:

            for epoch in pbar:
                if self.mini_batch_size is not None:
                    # Divide the dataset into mini-batched based on their size
                    # We increment the seed to reshuffle differently the dataset after each epoch
                    rng = np.random.RandomState(epoch)
                    mini_batches = self.generate_mini_batches(X, Y, rng=rng)
                else:
                    # Batch gradient descent
                    mini_batches = [(X, Y)]

                for mini_batch in mini_batches:
                    # Unpack the mini-batch
                    mini_X, mini_Y = mini_batch

                    # Forward propagation
                    output = self.propagate_forward(mini_X, train=True)

                    # Compute cost
                    cost = self.compute_cost(output, mini_Y)
                    costs.append(cost)

                    # Backward propagation
                    self.propagate_backward(output, mini_Y)

                    # Update parameters
                    if callable(self.lr):
                        lr = self.lr(epoch)
                    else:
                        lr = self.lr

                    if isinstance(self.optimizer, optimizers.Momentum):
                        self.optimizer.update_params(self.layers, lr)
                    elif isinstance(self.optimizer, optimizers.Adam):
                        self.optimizer.update_params(self.layers, lr)
                    else:
                        self.update_params()

        # Success: The model has been trained

        costs = np.array(costs)
        # Print cost statistics
        if print_coststats:
            print(Fore.BLUE + '-' * 100 + Fore.RESET)
            print("Cost stats:")
            print(tabulate([[
                '%.4f (%i)' % (np.max(costs), np.argmax(costs)),
                '%.4f (%i)' % (np.min(costs), np.argmin(costs)),
                '%.4f' % costs[-1],
                '%.4f' % np.mean(costs),
                '%.4f' % np.std(costs)
            ]],
                headers=['max', 'min', 'last', 'mean', 'std'],
                tablefmt="presto"))
        # Print cost as a function of time
        if print_costdev:
            print(Fore.BLUE + '-' * 100 + Fore.RESET)
            print("Cost development:")
            points = 89
            step_costs = costs[[max(0, math.floor(i / points * len(costs)) - 1) for i in range(0, points + 1)]]
            print(asciichartpy.plot(step_costs, {'height': 4}))

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
        output = self.propagate_forward(X, train=False)
        # Classify the probabilities
        output = np.array(output, copy=True)
        output[output <= threshold] = 0
        output[output > threshold] = 1
        # Compute the fraction of correct predictions
        accuracy = np.sum((output == Y) / m)

        return output, accuracy

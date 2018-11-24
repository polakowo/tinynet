import numpy as np
import math

from tqdm.auto import trange

from deepnn.utils import regularizers
from deepnn.utils import optimizers
from deepnn.utils import grad_check


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
        for index, layer in enumerate(layers):
            layer.index = index

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

    def propagate_forward(self, X, predict=False):
        output = X
        for l, layer in enumerate(self.layers):
            output = layer.propagate_forward(output, predict=predict)

        return output

    ########
    # COST #
    ########

    def compute_cost(self, output, Y):
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
        for layer in self.layers:
            for key in layer.params:
                # Update the rule for each parameter in each layer
                layer.params[key] = layer.params[key] - self.lr * layer.grads['d' + key]

    #########
    # TRAIN #
    #########

    def generate_mini_batches(self, X, Y, rng=None):
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

    def fit(self, X, Y, gradient_checking=False):
        """
        Train an L-layer neural network

        X must be of shape (n, m)
        Y must be of shape (1, m)
        where n is the number of features and m is the number of datasets
        """

        # Initialize parameters dictionary
        self.init_params(X)

        # Initialize the optimizer
        if isinstance(self.optimizer, optimizers.Momentum):
            self.optimizer.init_params(self.layers)
        elif isinstance(self.optimizer, optimizers.Adam):
            self.optimizer.init_params(self.layers)

        # Initialize output array of gradient checking
        if gradient_checking:
            relative_errors = []

        costs = []
        # Progress information is displayed and updated dynamically in the console
        with trange(self.num_epochs) as pbar:

            for epoch in pbar:
                # Diversify outputs by epoch but make them predictable
                rng = np.random.RandomState(epoch)

                if self.mini_batch_size is not None:
                    # Divide the dataset into mini-batched based on their size
                    # We increment the seed to reshuffle differently the dataset after each epoch
                    mini_batches = self.generate_mini_batches(X, Y, rng=rng)
                else:
                    # Batch gradient descent
                    mini_batches = [(X, Y)]

                for i, mini_batch in enumerate(mini_batches):
                    # Unpack the mini-batch
                    mini_X, mini_Y = mini_batch

                    # Forward propagation
                    output = self.propagate_forward(mini_X)

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

                    # Check the backpropagation algorithm after learning some parameters
                    # Only one mini batch every epoch
                    if gradient_checking and i == 0:
                        # Only one test every 10% epochs
                        if (epoch + 1) % (math.floor(self.num_epochs / 10)) == 0:
                            relative_error = self.gradient_checking(mini_X, mini_Y)

                            if relative_error <= 1e-6:
                                status = 'OK'
                            elif relative_error <= 1e-2:
                                status = 'WARNING'
                            else:
                                status = 'ERROR'
                            relative_errors.append((epoch, status, relative_error))

                    # Delegate the task to the optimizer if set
                    if isinstance(self.optimizer, optimizers.Momentum):
                        t = epoch + 1
                        self.optimizer.update_params(self.layers, lr, t)
                    elif isinstance(self.optimizer, optimizers.Adam):
                        t = epoch + 1
                        self.optimizer.update_params(self.layers, lr, t)
                    else:
                        self.update_params()

        if gradient_checking:
            return costs, relative_errors

        return costs

    ###########
    # PREDICT #
    ###########

    def predict(self, X):
        # Propagate forward with the parameters learned previously
        output = self.propagate_forward(X, predict=True)

        return output

    #####################
    # GRADIENT CHECKING #
    #####################

    def gradient_checking(self, X, Y, epsilon=1e-5, train=False):
        """
        Gradient Checking algorithm

        Gradient checking verifies closeness between the gradients from backpropagation and
        the numerical approximation of the gradient (computed using forward propagation)
        """
        # http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
        # Epsilon higher than 1e-5 likely to produce numeric instability

        if train:
            # Train the model first
            output = self.propagate_forward(X)
            self.propagate_backward(output, Y)

        # Roll parameters dictionary into a large (n, 1) vector
        param_theta = grad_check.roll_params(self.layers)
        grad_theta = grad_check.roll_params(self.layers, grads=True)

        # Initialize vectors of the same shape
        num_params = param_theta.shape[0]
        grad_approx = np.zeros((num_params, 1))

        # Repeat for each number (parameter) in the vector
        for i in range(num_params):
            # Use two-sided Taylor approximation which is 2x more precise than one-sided
            # Add epsilon to the parameter
            theta_plus = np.copy(param_theta)
            theta_plus[i] = theta_plus[i] + epsilon
            # Calculate new cost
            grad_check.unroll_params(theta_plus, self.layers)
            output_plus = self.propagate_forward(X, predict=True)
            cost_plus = self.compute_cost(output_plus, Y)

            # Subtract epsilon from the parameter
            theta_minus = np.copy(param_theta)
            theta_minus[i] = theta_minus[i] - epsilon
            # Calculate new cost
            grad_check.unroll_params(theta_minus, self.layers)
            output_minus = self.propagate_forward(X, predict=True)
            cost_minus = self.compute_cost(output_minus, Y)

            # Approximate the partial derivative, error is eps^2
            grad_approx[i] = (cost_plus - cost_minus) / (2 * epsilon)

        # Reset model params
        grad_check.unroll_params(param_theta, self.layers)

        # Difference between the approximated gradient and the backward propagation gradient
        relative_error = grad_check.calculate_diff(grad_theta, grad_approx)

        return relative_error

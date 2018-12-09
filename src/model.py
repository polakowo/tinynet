import numpy as np
import math

from tqdm.auto import trange

from src.utils import layers
from src.utils import regularizers
from src.utils import optimizers
from src.utils import cost_fns


class DNN:

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    ###########
    # COMPILE #
    ###########

    def configure(self,
                  input_shape,
                  optimizer,
                  cost_fn=cost_fns.cross_entropy,
                  regularizer=None):
        """
        Initialize layer and optimization params
        """
        self.input_shape = input_shape
        # Optimization algorithm
        self.optimizer = optimizer
        # Cost function
        self.cost_fn = cost_fn
        # Network-level regularization algorithm
        self.regularizer = regularizer

        for index, layer in enumerate(self.layers):
            if index == 0:
                prev_n = input_shape[1]
            else:
                prev_n = self.layers[index - 1].units

            # Layers know their shapes only at runtime
            layer.init_params(prev_n)

        # Initialize the optimizer
        if isinstance(self.optimizer, optimizers.Momentum):
            self.optimizer.init_params(self.layers)
        elif isinstance(self.optimizer, optimizers.Adam):
            self.optimizer.init_params(self.layers)

    #######################
    # FORWARD PROPAGATION #
    #######################

    def propagate_forward(self, X, predict=False):
        output = X
        for l, layer in enumerate(self.layers):
            output = layer.forward(output, predict=predict)

        return output

    ########
    # COST #
    ########

    def compute_cost(self, output, Y, epsilon=1e-12):
        m = output.shape[0]

        cost = self.cost_fn(output, Y, delta=False)

        if isinstance(self.regularizer, regularizers.L2):
            # Add L2 regularization term to the cost
            cost += self.regularizer.compute_term(self.layers, m)

        return cost

    ########################
    # BACKWARD PROPAGATION #
    ########################

    def propagate_backward(self, output, Y):
        doutput = self.cost_fn(output, Y, delta=True)

        # Calculate and store gradients in each layer with parameters
        # Move from the last layer to the first
        for layer in reversed(self.layers):
            if isinstance(layer, layers.activation.Activation):
                doutput = layer.backward(doutput, Y)
            else:
                doutput = layer.backward(doutput)

    #################
    # UPDATE PARAMS #
    #################

    def update_params(self):
        if isinstance(self.optimizer, optimizers.GradientDescent):
            self.optimizer.update_params(self.layers, regularizer=self.regularizer)
        if isinstance(self.optimizer, optimizers.Momentum):
            self.optimizer.update_params(self.layers, regularizer=self.regularizer)
        elif isinstance(self.optimizer, optimizers.Adam):
            self.optimizer.update_params(self.layers, regularizer=self.regularizer)

    #########
    # TRAIN #
    #########

    def generate_batches(self, X, Y, batch_size, rng=None):
        if rng is None:
            pass
        m = X.shape[0]
        batches = []

        # Step 1: Shuffle (X, Y)
        permutation = list(rng.permutation(m))
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :].reshape(Y.shape)

        # Step 2: Partition (shuffled_X, shuffled_Y)
        num_batches = math.floor(m / batch_size)
        for i in range(num_batches + 1):
            from_num = i * batch_size
            to_num = (i + 1) * batch_size

            if from_num < m:
                batch_X = shuffled_X[from_num:to_num, :]
                batch_Y = shuffled_Y[from_num:to_num, :]

                batch = (batch_X, batch_Y)
                batches.append(batch)

        return batches

    def fit(self, X, Y, nb_epoch, batch_size=None):
        """
        Train an L-layer neural network
        """

        costs = []
        # Progress information is displayed and updated dynamically in the console
        with trange(nb_epoch) as pbar:

            for epoch in pbar:
                # Diversify outputs by epoch but make them predictable
                rng = np.random.RandomState(epoch)

                if batch_size is not None:
                    # Divide the dataset into mini-batches based on their size
                    # Powers of two are often chosen to be the mini-batch size, e.g., 64, 128
                    # Make sure that a single mini-batch fits into the CPU/GPU memory
                    # We increment the seed to reshuffle differently the dataset after each epoch
                    batches = self.generate_batches(X, Y, batch_size, rng=rng)
                else:
                    # Batch gradient descent
                    batches = [(X, Y)]

                for i, batch in enumerate(batches):
                    # Unpack the mini-batch
                    X_batch, Y_batch = batch

                    # Forward propagation
                    output = self.propagate_forward(X_batch)

                    # Compute cost
                    cost = self.compute_cost(output, Y_batch)
                    costs.append(cost)

                    # Backward propagation
                    self.propagate_backward(output, Y_batch)

                    # Update params with an optimizer
                    self.update_params()

        return costs

    ###########
    # PREDICT #
    ###########

    def predict(self, X):
        # Propagate forward with the parameters learned previously
        output = self.propagate_forward(X, predict=True)

        return output

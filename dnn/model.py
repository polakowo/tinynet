import numpy as np

from tqdm.auto import tqdm
from tabulate import tabulate

from dnn import layers
from dnn import regularizers
from dnn import optimizers
from dnn import cost_fns


class Sequential:

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    ###########
    # COMPILE #
    ###########

    def configure(self, in_shape, optimizer, cost_fn, regularizer=None):
        """
        Initialize layer and optimization params
        """
        in_shape = (None, *in_shape[1:])
        self.optimizer = optimizer
        self.cost_fn = cost_fn
        self.regularizer = regularizer

        for index, layer in enumerate(self.layers):
            if index > 0:
                in_shape = self.layers[index - 1].out_shape

            # Layers know their shapes only at runtime
            layer.init_params(in_shape)

        # Initialize the optimizer
        if isinstance(self.optimizer, optimizers.Momentum):
            self.optimizer.init_params(self.layers)
        elif isinstance(self.optimizer, optimizers.Adam):
            self.optimizer.init_params(self.layers)

    def summary(self):
        """
        Get summary on layer shapes and parameters
        """
        rows = []
        for layer in self.layers:
            name = layer.__class__.__name__
            out_shape = layer.out_shape
            if layer.params is not None:
                num_params = sum([np.prod(p.shape) for k, p in layer.params.items()])
            else:
                num_params = 0
            rows.append((name, out_shape, num_params))
        print(tabulate(rows, headers=['Layer class', 'Output shape', 'Params']))

    #######################
    # FORWARD PROPAGATION #
    #######################

    def propagate_forward(self, X, predict=False):
        out = X
        for l, layer in enumerate(self.layers):
            out = layer.forward(out, predict=predict)

        return out

    ########
    # COST #
    ########

    def compute_cost(self, out, Y, epsilon=1e-12):
        m = out.shape[0]

        cost = self.cost_fn(out, Y, delta=False)

        if isinstance(self.regularizer, regularizers.L2):
            # Add L2 regularization term to the cost
            cost += self.regularizer.compute_term(self.layers, m)

        return cost

    ########################
    # BACKWARD PROPAGATION #
    ########################

    def propagate_backward(self, out, Y):
        dX = self.cost_fn(out, Y, delta=True)

        # Calculate and store gradients in each layer with parameters
        # Move from the last layer to the first
        for layer in reversed(self.layers):
            if isinstance(layer, layers.activation.Activation):
                dX = layer.backward(dX, Y)
            else:
                dX = layer.backward(dX)

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
            rng = np.random
        m = X.shape[0]
        batches = []

        # Step 1: Shuffle (X, Y)
        permutation = list(rng.permutation(m))
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :].reshape(Y.shape)

        # Step 2: Partition (shuffled_X, shuffled_Y)
        for i in range(0, m, batch_size):
            batch_X = shuffled_X[i:i + batch_size, :]
            batch_Y = shuffled_Y[i:i + batch_size, :]

            batch = (batch_X, batch_Y)
            batches.append(batch)

        return batches

    def fit(self, X, Y, nb_epoch, batch_size=None):
        """
        Train an L-layer neural network
        """

        costs = []
        # Progress information is displayed and updated dynamically in the console
        batches = self.generate_batches(X, Y, batch_size)
        with tqdm(total=nb_epoch * len(batches)) as pbar:

            for epoch in range(nb_epoch):
                # Diversify outputs by epoch but make them predictable
                rng = np.random.RandomState(epoch)

                if batch_size is not None:
                    # Divide the dataset into mini-batches based on their size
                    # We increment the seed to reshuffle differently the dataset after each epoch
                    batches = self.generate_batches(X, Y, batch_size, rng=rng)
                else:
                    # Batch gradient descent
                    batches = [(X, Y)]

                for i, batch in enumerate(batches):
                    # Unpack the mini-batch
                    X_batch, Y_batch = batch

                    # Forward propagation
                    out = self.propagate_forward(X_batch)

                    # Compute cost
                    cost = self.compute_cost(out, Y_batch)
                    costs.append(cost)

                    # Backward propagation
                    self.propagate_backward(out, Y_batch)

                    # Update params with an optimizer
                    self.update_params()

                    pbar.update(1)

        return costs

    ###########
    # PREDICT #
    ###########

    def predict(self, X):
        # Propagate forward with the parameters learned previously
        return self.propagate_forward(X, predict=True)

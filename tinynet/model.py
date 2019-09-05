import numpy as np

from tqdm.auto import tqdm
from tabulate import tabulate

from tinynet import layers
from tinynet import regularizers
from tinynet import optimizers
from tinynet import cost_funcs


class TinyNet:
    """The basic sequential model of TinyNet"""

    def __init__(self, layers):
        self.layers = layers

    def configure(self, in_shape, optimizer, cost_fn, regularizer=None):
        """Initialize the layers and the optimization params."""
        in_shape = (None, *in_shape[1:])
        self.optimizer = optimizer
        self.cost_fn = cost_fn
        self.regularizer = regularizer

        for index, layer in enumerate(self.layers):
            if index > 0:
                in_shape = self.layers[index - 1].out_shape

            # Layers know their shapes only after defining all layers
            # That's why we need an explicit configure method.
            layer.init_params(in_shape)

        # Initialize the optimizer
        self.optimizer.init_params(self.layers)

    def print_summary(self):
        """Print the summary of the layers' shapes and parameters."""
        rows = []
        for layer in self.layers:
            name = layer.__class__.__name__
            # Get the shape of the layer
            out_shape = layer.out_shape
            if layer.params is not None:
                # Get the number of parameters in the layer
                num_params = sum([np.prod(p.shape) for k, p in layer.params.items()])
            else:
                num_params = 0
            rows.append((name, out_shape, num_params))
        print(tabulate(rows, headers=['Layer class', 'Output shape', 'Params']))

    def propagate_forward(self, X, predict=False):
        """Propagate through the layers forwards."""
        out = X
        # The output tensor of the previous layer becomes the input tensor of the next layer
        for l, layer in enumerate(self.layers):
            out = layer.forward(out, predict=predict)

        return out

    def compute_cost(self, out, Y, epsilon=1e-12):
        """Compute the cost."""
        m = out.shape[0]

        cost = self.cost_fn(out, Y, grad=False)

        # Some cost calculations depend upon a regularizer
        if isinstance(self.regularizer, regularizers.L2):
            # Add L2 regularization term to the cost
            cost += self.regularizer.compute_term(self.layers, m)

        return cost

    def propagate_backward(self, out, Y):
        """Propagate through the layers backwards."""
        dX = self.cost_fn(out, Y, grad=True)

        # Calculate and store gradients in each layer with parameters
        # Move from the last layer backwards to the first layer
        for layer in reversed(self.layers):
            if isinstance(layer, layers.activation.Activation):
                # Activation functions such as softmax require Y to be provided as well
                dX = layer.backward(dX, Y)
            else:
                dX = layer.backward(dX)

    def update_params(self):
        """Update the model parameters based on the optimization method."""
        self.optimizer.update_params(self.layers, regularizer=self.regularizer)

    def generate_batches(self, X, Y, batch_size, rng=None):
        """Divide the dataset into batches."""
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

    def fit(self, X, Y, num_epochs, batch_size=None):
        """Train a multi-layer neural network.
        
        X and Y should be both tensors of rank at least 2.
        """
        costs = []
        
        # Get the number of butches for progress bar
        if batch_size is None:
            num_iters = num_epochs
        else:
            num_batches = len(self.generate_batches(X, Y, batch_size))
            num_iters = num_epochs * num_batches
            
        # Progress information is displayed and updated dynamically in the console
        with tqdm(total=num_iters) as pbar:
            for epoch in range(num_epochs):
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

    def predict(self, X, batch_size=None):
        """Propagate forward with the parameters learned previously."""
        if batch_size is None:
            # Predict on the whole bulk (must fit into RAM)
            return self.propagate_forward(X, predict=True)
        else:
            # Split into mini-batches and predict
            preds = []
            m = X.shape[0]
            for i in tqdm(range(0, m, batch_size)):
                batch = X[i:i + batch_size, :]
                preds.append(self.propagate_forward(batch, predict=True))
            return np.vstack(preds)
                

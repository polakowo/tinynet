import numpy as np

# Regularization is used for penalizing complex models
# It forces the downstream hidden units not to rely too much on the previous units by introducing noise
# http://ruder.io/optimizing-gradient-descent/index.html


class L2:
    """
    L2 regularization
    """

    def __init__(self, _lambda):
        # Encourages the mean of the weights toward 0, with a normal (bell-shaped or Gaussian) distribution
        self._lambda = _lambda

    def compute_term(self, layers, m):
        L2 = np.sum([np.sum(np.square(layer.params['W'])) for layer in layers])

        self.cache = m
        return self._lambda / (2 * m) * L2

    def compute_term_delta(self, W, m):
        m = self.cache

        return self._lambda / m * W

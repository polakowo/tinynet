import numpy as np


class Xavier:
    """
    Xavier initialization

    Works well for layers with tanh activation
    """

    def __init__(self, uniform=True, rng=None):
        self.uniform = uniform
        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng

    def init_param(self, n_in, n_out):
        if self.uniform:
            # uniform distribution
            return self.rng.randn(n_in, n_out) * np.sqrt(6. / (n_in + n_out))
        else:
            # normal distribution
            return self.rng.randn(n_in, n_out) * np.sqrt(2. / (n_in + n_out))


class He:
    """
    He et al. initialization

    Works well for layers with ReLU activation
    """

    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng

    def init_param(self, n_in, n_out):
        return self.rng.randn(n_in, n_out) * np.sqrt(2. / n_in)

import numpy as np


class Xavier:
    """
    Xavier initialization
    """

    def __init__(self, uniform=True, rng=None):
        self.uniform = uniform
        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng

    def init_param(self, in_units, out_units):
        if self.uniform:
            # uniform distribution
            return self.rng.randn(in_units, out_units) * np.sqrt(6. / (in_units + out_units))
        else:
            # normal distribution
            return self.rng.randn(in_units, out_units) * np.sqrt(2. / (in_units + out_units))


class He:
    """
    He et al. initialization
    """

    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng

    def init_param(self, in_units, out_units):
        return self.rng.randn(in_units, out_units) * np.sqrt(2. / in_units)

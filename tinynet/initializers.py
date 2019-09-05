from abc import ABC, abstractmethod
import numpy as np


class Initializer(ABC):
    """Base initializer class"""

    def __init__(self, *args, **kwargs):
        """Define the initialization hyperparameters."""
        pass

    @abstractmethod
    def init_params(self, shape, in_units, out_units, *args, **kwargs):
        """Initialize the parameters of a layer."""
        pass


class Xavier(Initializer):
    """Xavier initialization"""

    def __init__(self, uniform=True, rng=None):
        self.uniform = uniform
        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng

    def init_params(self, shape, in_units, out_units):
        if self.uniform:
            # uniform distribution
            return self.rng.randn(*shape) * np.sqrt(6. / (in_units + out_units))
        else:
            # normal distribution
            return self.rng.randn(*shape) * np.sqrt(2. / (in_units + out_units))


class He(Initializer):
    """He et al. initialization"""

    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng

    def init_params(self, shape, in_units, out_units):
        return self.rng.randn(*shape) * np.sqrt(2. / in_units)

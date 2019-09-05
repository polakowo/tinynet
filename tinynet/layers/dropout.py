import numpy as np

from tinynet.layers import Layer

class Dropout(Layer):
    """Dropout regularization layer"""

    def __init__(self, keep_prob, rng=None):
        # Probability of keeping a neuron
        self.keep_prob = keep_prob
        # Random state
        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng

    def init_params(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = in_shape

        self.params = None
        self.grads = None

    def forward(self, input, predict=False):
        if not predict:
            KEEP_MASK = self.rng.rand(*input.shape)
            # Shut down each neuron of the layer with a probability of 1−keep_prob
            KEEP_MASK = KEEP_MASK < self.keep_prob
            out = input * KEEP_MASK
            # Divide each dropout layer by keep_prob to keep the same expected value for the activation
            out = out / self.keep_prob

            self.cache = (input, KEEP_MASK)
            return out

        else:
            # Use dropout only during training, not during test time
            return input

    def backward(self, dout):
        input, KEEP_MASK = self.cache
        # Apply the mask to shut down the same neurons as during the forward propagation
        dX = dout * KEEP_MASK
        # Scale the value of neurons that haven't been shut down
        dX = dX / self.keep_prob
        assert(dX.shape == input.shape)

        self.cache = None
        return dX

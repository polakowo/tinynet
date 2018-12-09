import numpy as np


class Dropout:
    """
    Dropout regularization

    # Randomly shut down some neurons in each iteration
    # With dropout, neurons become less sensitive to the activation of one other specific neuron
    """

    def __init__(self, keep_prob, rng=None):
        # Probability of keeping a neuron
        self.keep_prob = keep_prob
        # Random state
        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng

    def init_params(self, prev_units):
        self.units = prev_units
        self.params = None
        self.grads = None

    def forward(self, input, predict=False):
        if not predict:
            KEEP_MASK = self.rng.rand(*input.shape)
            # Shut down each neuron of the layer with a probability of 1−keep_prob
            KEEP_MASK = KEEP_MASK < self.keep_prob
            output = input * KEEP_MASK
            # Divide each dropout layer by keep_prob to keep the same expected value for the activation
            output = output / self.keep_prob

            self.cache = KEEP_MASK
            return output

        else:
            # Use dropout only during training, not during test time
            return input

    def backward(self, dinput):
        KEEP_MASK = self.cache
        # Apply the mask to shut down the same neurons as during the forward propagation
        doutput = dinput * KEEP_MASK
        # Scale the value of neurons that haven't been shut down
        doutput = doutput / self.keep_prob

        self.cache = None
        return doutput

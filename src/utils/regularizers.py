import numpy as np

# Regularization is used for penalizing complex models
# It forces the downstream hidden units not to rely too much on the previous units by introducing noise
# http://ruder.io/optimizing-gradient-descent/index.html


class L2:
    """
    L2 regularization

    # If model complexity is a function of weights, a feature weight with a high absolute value is more complex
    # Step 1: A regularization term is added to the cost
    # Step 2: In the backpropagation, weights are penalized ("weight decay")
    """

    def __init__(self, _lambda):
        # Encourages the mean of the weights toward 0, with a normal (bell-shaped or Gaussian) distribution
        self._lambda = _lambda

    def compute_term(self, layers, n_samples):
        L2 = np.sum([np.sum(np.square(layer.params['W'])) for layer in layers])
        return 1 / 2 * self._lambda / n_samples * L2

    def compute_term_derivative(self, W, n_samples):
        return self._lambda / n_samples * W


class Dropout:
    """
    Dropout regularization

    # Randomly shut down some neurons in each iteration
    # With dropout, neurons become less sensitive to the activation of one other specific neuron
    # Apply dropout both during forward and backward propagation
    # Use dropout only during training, not during test time
    """

    def __init__(self, keep_prob, rng=None):
        # Probability of keeping a neuron
        self.keep_prob = keep_prob
        # Randomize
        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng

    def forward(self, input):
        KEEP_MASK = self.rng.rand(*input.shape)
        # Shut down each neuron of the layer with a probability of 1−keep_prob
        KEEP_MASK = KEEP_MASK < self.keep_prob
        output = input * KEEP_MASK
        # Divide each dropout layer by keep_prob to keep the same expected value for the activation
        output = output / self.keep_prob

        cache = KEEP_MASK
        return output, cache

    def backward(self, dinput, cache):
        KEEP_MASK = cache
        # Apply the mask to shut down the same neurons as during the forward propagation
        doutput = dinput * KEEP_MASK
        # Scale the value of neurons that haven't been shut down
        doutput = doutput / self.keep_prob

        return doutput

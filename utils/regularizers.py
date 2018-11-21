import numpy as np

# Regularization is used for penalizing complex models
# http://ruder.io/optimizing-gradient-descent/index.html#minibatchgradientdescent


class L2:
    """
    L2 regularization

    # If model complexity is a function of weights, a feature weight with a high absolute value is more complex
    # Step 1: A regularization term is added to the cost
    # Step 2: In the backpropagation, weights are penalized ("weight decay")
    """

    def __init__(self, _lambda):
        # Encourages weight values toward 0
        # Encourages the mean of the weights toward 0, with a normal (bell-shaped or Gaussian) distribution
        self._lambda = _lambda

    def compute_term(self, params, m):
        """
        Compute the L2 regularization term
        """
        L2 = np.sum([np.sum(np.square(params['W' + str(l)])) for l in range(params['L'])])
        return 1 / 2 * self._lambda / m * L2

    def compute_term_derivative(self, W, m):
        """
        Compute the derivative of the term with respect to the weights
        """
        return self._lambda / m * W


class Dropout:
    """
    Dropout regularization

    # Randomly shut down some neurons in each iteration
    # With dropout, neurons become less sensitive to the activation of one other specific neuron
    # Apply dropout both during forward and backward propagation
    # Use dropout only during training, not during test time
    """

    def __init__(self, keep_probs):
        # Probability of keeping a neuron in each layer
        self.keep_probs = keep_probs

    def dropout_forward(self, A, l):
        """
        Apply the dropout regularization to the activation output
        """
        keep_prob = self.keep_probs[l]

        KEEP_MASK = np.random.rand(A.shape[0], A.shape[1])
        # Shut down each neuron of the layer with a probability of 1−keep_prob
        KEEP_MASK = KEEP_MASK < keep_prob
        A = A * KEEP_MASK
        # Divide each dropout layer by keep_prob to keep the same expected value for the activation
        A = A / keep_prob

        dropout_cache = KEEP_MASK
        return A, dropout_cache

    def dropout_backward(self, dA, dropout_cache, l):
        """
        Partial derivative of J with respect to activation output
        """
        keep_prob = self.keep_probs[l]

        # dJ/dA = dJ/dA' * dA'/dA
        KEEP_MASK = dropout_cache
        # Apply the mask to shut down the same neurons as during the forward propagation
        dA = dA * KEEP_MASK
        # Scale the value of neurons that haven't been shut down
        dA = dA / keep_prob

        return dA

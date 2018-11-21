import numpy as np

# Optimization algorithms ‘denoise’ the data and bring it closer to the original function
# They help in navigating plateaus where learning is slow


class Momentum:
    """
    Momentum optimization algorithm

    Momentum takes past gradients into account to smooth out the steps of gradient descent
    It can be applied with batch, mini-batch and stochastic gradient descent
    """

    def __init__(self, beta=0.9):

        # Increasing beta will smooth out the gradients
        self.beta = beta

    def initialize_params(self, params):
        """
        Initialize parameters
        """
        self.v = {}

        for l in range(params['L']):
            self.v['dW' + str(l)] = np.zeros(params['W' + str(l)].shape)
            self.v['db' + str(l)] = np.zeros(params['b' + str(l)].shape)

    def update_params(self, params, grads, learning_rate):
        """
        Update parameters
        """
        # Momentum update for each parameter
        for l in range(params['L']):
            # Compute velocities
            self.v['dW' + str(l)] = self.beta * self.v['dW' + str(l)] + (1 - self.beta) * grads['dW' + str(l)]
            self.v['db' + str(l)] = self.beta * self.v['db' + str(l)] + (1 - self.beta) * grads['db' + str(l)]

            # Update parameters
            params['W' + str(l)] -= learning_rate * self.v['dW' + str(l)]
            params['b' + str(l)] -= learning_rate * self.v['db' + str(l)]

        return params


class Adam:
    """
    Adam optimization algorithm

    Adam combines ideas from RMSProp and Momentum
    The algorithm calculates an exponential moving average of the gradient and the squared gradient
    It's one of the most effective optimization algorithms
    """

    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # pPrameters beta1 and beta2 control the decay rates of these moving averages
        self.beta1 = beta1
        self.beta2 = beta2
        # Epsilon is required to prevent division by zero
        self.epsilon = epsilon

    def initialize_params(self, params):
        """
        Initialize parameters
        """
        self.v = {}
        self.s = {}
        self.t = 1

        for l in range(params['L']):
            self.v['dW' + str(l)] = np.zeros(params['W' + str(l)].shape)
            self.v['db' + str(l)] = np.zeros(params['b' + str(l)].shape)
            self.s['dW' + str(l)] = np.zeros(params['W' + str(l)].shape)
            self.s['db' + str(l)] = np.zeros(params['b' + str(l)].shape)

    def update_params(self, params, grads, learning_rate):
        """
        Update parameters
        """
        # Initialize estimate dictionaries
        v_corrected = {}
        s_corrected = {}

        # Perform Adam update on all parameters
        for l in range(params['L']):
            # Moving average of the gradients
            self.v["dW" + str(l)] = self.beta1 * self.v["dW" + str(l)] + (1 - self.beta1) * grads['dW' + str(l)]
            self.v["db" + str(l)] = self.beta1 * self.v["db" + str(l)] + (1 - self.beta1) * grads['db' + str(l)]

            # Compute the first bias-corrected estimate
            v_corrected["dW" + str(l)] = self.v["dW" + str(l)] / (1 - self.beta1 ** self.t)
            v_corrected["db" + str(l)] = self.v["db" + str(l)] / (1 - self.beta1 ** self.t)

            # Moving average of the squared gradients
            self.s["dW" + str(l)] = self.beta2 * self.s["dW" + str(l)] + \
                (1 - self.beta2) * np.square(grads['dW' + str(l)])
            self.s["db" + str(l)] = self.beta2 * self.s["db" + str(l)] + \
                (1 - self.beta2) * np.square(grads['db' + str(l)])

            # Compute the second bias-corrected estimate
            s_corrected["dW" + str(l)] = self.s["dW" + str(l)] / (1 - self.beta2 ** self.t)
            s_corrected["db" + str(l)] = self.s["db" + str(l)] / (1 - self.beta2 ** self.t)

            # Update parameters
            params["W" + str(l)] -= learning_rate * v_corrected["dW" + str(l)] / \
                (np.sqrt(s_corrected["dW" + str(l)]) + self.epsilon)
            params["b" + str(l)] -= learning_rate * v_corrected["db" + str(l)] / \
                (np.sqrt(s_corrected["db" + str(l)]) + self.epsilon)

        # Update epoch
        self.t += 1
        return params

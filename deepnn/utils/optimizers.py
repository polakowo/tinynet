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

    def init_params(self, layers):
        # Initialize moment vector
        self.layer_v = []

        for l, layer in enumerate(layers):
            v = {}

            for k in layer.params:
                v['d' + k] = np.zeros(layer.params[k].shape)

            self.layer_v.append(v)

    def update_params(self, layers, lr, t):
        # Momentum update for each parameter in a layer
        for l, layer in enumerate(layers):
            v = self.layer_v[l]

            for k in layer.params:
                grad = layer.grads['d' + k]

                # Compute velocities
                v['d' + k] = self.beta * v['d' + k] + (1 - self.beta) * grad

                # Compute bias-corrected first moment estimate
                v_corrected = v['d' + k] / (1 - self.beta ** t)

                # Update parameters
                layer.params[k] = layer.params[k] - lr * v_corrected


class Adam:
    """
    Adam optimization algorithm
    https://arxiv.org/pdf/1412.6980.pdf

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

    def init_params(self, layers):
        # Initialize 1st moment vector
        self.layer_v = []
        # Initialize 2nd moment vector
        self.layer_s = []

        for l, layer in enumerate(layers):
            v = {}
            s = {}

            for k in layer.params:
                v['d' + k] = np.zeros(layer.params[k].shape)
                s['d' + k] = np.zeros(layer.params[k].shape)

            self.layer_v.append(v)
            self.layer_s.append(s)

    def update_params(self, layers, lr, t):
        # Perform Adam update on all parameters in a layer
        for l, layer in enumerate(layers):
            v = self.layer_v[l]
            s = self.layer_s[l]

            for k in layer.params:
                grad = layer.grads['d' + k]

                # Update biased first moment estimate
                v['d' + k] = self.beta1 * v['d' + k] + (1 - self.beta1) * grad
                # Update biased second raw moment estimate
                s['d' + k] = self.beta2 * s['d' + k] + (1 - self.beta2) * np.square(grad)

                # Compute bias-corrected first moment estimate
                v_corrected = v['d' + k] / (1 - self.beta1 ** t)
                # Compute bias-corrected second raw moment estimate
                s_corrected = s['d' + k] / (1 - self.beta2 ** t)

                # Update parameters
                layer.params[k] = layer.params[k] - lr * v_corrected / (np.sqrt(s_corrected) + self.epsilon)

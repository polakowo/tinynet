import numpy as np

from dnn import regularizers

# Optimization algorithms ‘denoise’ the data and bring it closer to the original function
# They help in navigating plateaus where learning is slow


class GradientDescent:
    """
    Vanilla gradient descent
    """

    def __init__(self, lr):
        # Learning rate
        self.lr = lr

    def update_params(self, layers, regularizer=None):
        for layer in layers:
            # Does the layer have learnable parameters?
            if layer.params is not None:
                for k in layer.params:
                    param = layer.params[k]
                    grad = layer.grads['d' + k]

                    # Update the rule for each parameter in each layer
                    layer.params[k] -= self.lr * grad

                    if k is 'W':
                        if isinstance(regularizer, regularizers.L2):
                            # Weight decay
                            layer.params[k] -= self.lr * regularizer.compute_term_delta(param)


class Momentum:
    """
    Gradient descent with momentum
    """

    def __init__(self, lr, beta=0.9):
        # Learning rate
        self.lr = lr
        # Increasing beta will smooth out the gradients
        self.beta = beta

    def init_params(self, layers):
        # Timestamp of the update
        self.t = 0
        # Initialize moment vector
        self.layer_v = []

        for l, layer in enumerate(layers):
            if layer.params is not None:
                v = {}

                for k in layer.params:
                    v['d' + k] = np.zeros(layer.params[k].shape)

                self.layer_v.append(v)
            else:
                self.layer_v.append(None)

    def update_params(self, layers, regularizer=None):
        self.t += 1

        # Momentum update for each parameter in a layer
        for l, layer in enumerate(layers):
            if layer.params is not None:
                v = self.layer_v[l]

                for k in layer.params:
                    param = layer.params[k]
                    grad = layer.grads['d' + k]

                    # Compute velocities
                    v['d' + k] = self.beta * v['d' + k] + (1 - self.beta) * grad

                    # Compute bias-corrected first moment estimate
                    v_corrected = v['d' + k] / (1 - self.beta ** self.t)

                    # Update parameters
                    layer.params[k] -= self.lr * v_corrected

                    if k is 'W':
                        if isinstance(regularizer, regularizers.L2):
                            # Weight decay
                            layer.params[k] -= self.lr * regularizer.compute_term_delta(param)


class Adam:
    """
    Adaptive Moment Estimation (Adam)
    """

    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        # Learning rate
        self.lr = lr
        # Parameters beta1 and beta2 control the decay rates of these moving averages
        self.beta1 = beta1
        self.beta2 = beta2
        # Epsilon is required to prevent division by zero
        self.eps = eps

    def init_params(self, layers):
        # Timestamp of the update
        self.t = 0
        # Initialize 1st moment vector
        self.layer_v = []
        # Initialize 2nd moment vector
        self.layer_s = []

        for l, layer in enumerate(layers):
            if layer.params is not None:
                v = {}
                s = {}

                for k in layer.params:
                    v['d' + k] = np.zeros(layer.params[k].shape)
                    s['d' + k] = np.zeros(layer.params[k].shape)

                self.layer_v.append(v)
                self.layer_s.append(s)
            else:
                self.layer_v.append(None)
                self.layer_s.append(None)

    def update_params(self, layers, regularizer=None):
        self.t += 1

        # Perform Adam update on all parameters in a layer
        for l, layer in enumerate(layers):
            if layer.params is not None:
                v = self.layer_v[l]
                s = self.layer_s[l]

                for k in layer.params:
                    param = layer.params[k]
                    grad = layer.grads['d' + k]

                    # Update biased first moment estimate
                    v['d' + k] = self.beta1 * v['d' + k] + (1 - self.beta1) * grad
                    # Update biased second raw moment estimate
                    s['d' + k] = self.beta2 * s['d' + k] + (1 - self.beta2) * np.square(grad)

                    # Compute bias-corrected first moment estimate
                    v_corrected = v['d' + k] / (1 - self.beta1 ** self.t)
                    # Compute bias-corrected second raw moment estimate
                    s_corrected = s['d' + k] / (1 - self.beta2 ** self.t)

                    # Update parameters
                    layer.params[k] -= self.lr * v_corrected / (np.sqrt(s_corrected) + self.eps)

                    if k is 'W':
                        if isinstance(regularizer, regularizers.L2):
                            # Weight decay
                            # https://www.fast.ai/2018/07/02/adam-weight-decay/
                            layer.params[k] -= self.lr * regularizer.compute_term_delta(param)

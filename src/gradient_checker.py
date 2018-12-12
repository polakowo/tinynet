import numpy as np

from src.layers.dropout import Dropout
from src.layers.batchnorm import BatchNorm

# http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization


def roll_params(layers, param_type):
    # Roll the parameters from layers into a single vector
    theta = np.empty(0)

    for layer in layers:
        params = getattr(layer, param_type)
        if params is not None:
            for k in params:
                vector = params[k]
                # Flatten and append the vector
                vector = vector.flatten()
                theta = np.concatenate((theta, vector))

    return theta


def unroll_params(theta, layers, param_type):
    # Unroll the parameters from a vector and save them to layers
    i = 0
    for layer in layers:
        params = getattr(layer, param_type)
        if params is not None:
            for k in params:
                vector = params[k]
                # Extract and reshape the parameter to the original form
                j = i + np.prod(vector.shape)
                params[k] = theta[i:j].reshape(vector.shape)
                i = j


def calculate_diff(grad_theta, grad_approx):
    # np.linalg.norm apply for matric equal to Frobenius norm
    numerator = np.linalg.norm(grad_theta - grad_approx)
    denominator = np.linalg.norm(grad_theta) + np.linalg.norm(grad_approx)
    diff = numerator / denominator
    return diff


class GradientChecker:
    """
    Gradient Checking
    """

    def __init__(self, model, eps=1e-5):
        self.model = model
        # Important: Epsilon higher than 1e-5 likely to produce numeric instability
        self.eps = eps

    def compute_error(self, X, Y):
        # Import methods from the model
        layers = self.model.layers
        regularizer = self.model.regularizer
        propagate_forward = self.model.propagate_forward
        compute_cost = self.model.compute_cost
        propagate_backward = self.model.propagate_backward

        # Dirty regularizers such as dropout may yield errors
        assert(regularizer is None)
        for layer in layers:
            assert(not isinstance(layer, Dropout))
            assert(not isinstance(layer, BatchNorm))

        # Get params currently stored in the layers (for reset)
        params = roll_params(layers, 'params')
        grads = roll_params(layers, 'grads')

        # Perform one iteration on X and Y to compute and store new gradients
        out = propagate_forward(X)
        propagate_backward(out, Y)

        # Extract new gradients and roll them into a vector
        param_theta = roll_params(layers, 'params')
        grad_theta = roll_params(layers, 'grads')

        # Initialize vector of the same shape for approximated gradients
        num_params = len(param_theta)
        grad_approx = np.zeros(num_params)

        # Repeat for each number in the vector
        for i in range(num_params):
            # Use two-sided Taylor approximation which is 2x more precise than one-sided
            # Add epsilon to the number
            theta_plus = np.copy(param_theta)
            theta_plus[i] = theta_plus[i] + self.eps
            # Calculate new cost
            unroll_params(theta_plus, layers, 'params')
            out_plus = propagate_forward(X, predict=True)
            cost_plus = compute_cost(out_plus, Y)

            # Subtract epsilon from the number
            theta_minus = np.copy(param_theta)
            theta_minus[i] = theta_minus[i] - self.eps
            # Calculate new cost
            unroll_params(theta_minus, layers, 'params')
            out_minus = propagate_forward(X, predict=True)
            cost_minus = compute_cost(out_minus, Y)

            # Approximate the gradient, error is eps^2
            grad_approx[i] = (cost_plus - cost_minus) / (2 * self.eps)

        # Reset model params
        unroll_params(params, layers, 'params')
        unroll_params(grads, layers, 'grads')

        # Compute relative error
        relative_error = calculate_diff(grad_theta, grad_approx)

        return relative_error

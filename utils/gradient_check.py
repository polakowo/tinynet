import numpy as np

from colorama import Fore

from utils import regularizers


def params_to_vector(layer_params):
    """
    Roll parameters dictionary into a single (n, 1) vector
    """
    theta = np.zeros((0, 1))

    for param in layer_params:
        for k in param.keys():
            vector = param[k]

            # Flatten the vector
            vector = np.reshape(vector, (-1, 1))
            # Append the vector
            theta = np.concatenate((theta, vector), axis=0)

    cache = layer_params
    return theta, cache


def vector_to_params(theta, cache):
    """
    Unroll parameters dictionary from a single vector
    """
    _layer_params = cache
    layer_params = _layer_params.copy()

    i = 0
    for param in layer_params:
        for k in param.keys():
            vector = param[k]

            # Extract and reshape the parameter to the original form
            j = i + vector.shape[0] * vector.shape[1]
            param[k] = theta[i:j].reshape(vector.shape)
            i = j

    return layer_params


def calculate_diff(A, B):
    """
    Calculate the difference between two vectors using their Euclidean norm
    """
    numerator = np.linalg.norm(np.linalg.norm(A - B))
    denominator = np.linalg.norm(np.linalg.norm(A)) + np.linalg.norm(np.linalg.norm(B))
    difference = numerator / denominator
    return difference


class GradientCheck:
    """
    Gradient Checking algorithm

    Gradient checking verifies closeness between the gradients from backpropagation and
    the numerical approximation of the gradient (computed using forward propagation)
    You would usually run it only once to make sure the code is correct
    """

    def __init__(self, model, epsilon=1e-7):
        # http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
        self.model = model
        self.epsilon = epsilon

    def run(self, X, Y):
        """
        Check whether the model's backpropagation works properly
        """
        # Doesn't work well with dropout regularization
        assert(not isinstance(self.model.regularizer, regularizers.Dropout))

        # One iteration of gradient descent to get gradients
        layer_params = self.model.initialize_params(X)
        AL, caches = self.model.propagate_forward(X, layer_params)
        layer_grads = self.model.propagate_backward(AL, Y, caches)

        # Roll parameters dictionary into a large (n, 1) vector
        param_theta, param_cache = params_to_vector(layer_params)
        grad_theta, _ = params_to_vector(layer_grads)

        # Initialize vectors of the same shape
        num_params = param_theta.shape[0]
        J_plus = np.zeros((num_params, 1))
        J_minus = np.zeros((num_params, 1))
        gradapprox = np.zeros((num_params, 1))

        # Repeat for each number (parameter) in the vector
        for i in range(num_params):
            # Use two-sided Taylor approximation which is 2x more precise than one-sided
            # Add epsilon to the parameter
            theta_plus = np.copy(param_theta)
            theta_plus[i][0] = theta_plus[i][0] + self.epsilon
            # Calculate new cost
            theta_plus_params = vector_to_params(theta_plus, param_cache)
            AL_plus, _ = self.model.propagate_forward(X, theta_plus_params)
            J_plus[i] = self.model.compute_cost(AL_plus, Y, theta_plus_params)

            # Subtract epsilon from the parameter
            theta_minus = np.copy(param_theta)
            theta_minus[i][0] = theta_minus[i][0] - self.epsilon
            # Calculate new cost
            thetha_minus_params = vector_to_params(theta_minus, param_cache)
            AL_minus, _ = self.model.propagate_forward(X, thetha_minus_params)
            J_minus[i] = self.model.compute_cost(AL_minus, Y, thetha_minus_params)

            # Approximate the partial derivative, error is eps^2
            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * self.epsilon)

        # Difference between the approximated gradient and the backward propagation gradient
        diff = calculate_diff(grad_theta, gradapprox)
        if diff > 2e-7:
            print("%s%s%s" % (Fore.RED, "Failed gradient checking", Fore.RESET))
        else:
            print("%s%s%s" % (Fore.GREEN, "Passed gradient checking", Fore.RESET))

        return diff

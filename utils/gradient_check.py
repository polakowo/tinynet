import numpy as np


def roll_params(layers, grads=False):
    """
    Roll parameters into a single (n, 1) vector
    """
    theta = np.zeros((0, 1))

    for layer in layers:
        if grads:
            vdict = layer.grads
        else:
            vdict = layer.params
        for k in vdict:
            vector = vdict[k]
            # Flatten the vector
            vector = np.reshape(vector, (-1, 1))
            # Append the vector
            theta = np.concatenate((theta, vector), axis=0)

    return theta


def unroll_params(theta, layers, grads=False):
    """
    Unroll parameters from a single vector and save to the layers
    """
    i = 0

    for layer in layers:
        if grads:
            vdict = layer.grads
        else:
            vdict = layer.params
        for k in vdict:
            vector = vdict[k]
            # Extract and reshape the parameter to the original form
            j = i + vector.shape[0] * vector.shape[1]
            vdict[k] = theta[i:j].reshape(vector.shape)
            i = j


def calculate_diff(grad_theta, grad_approx):
    """
    Calculate the difference between two vectors using their Euclidean norm
    """
    numerator = np.linalg.norm(grad_theta - grad_approx)
    denominator = np.linalg.norm(grad_theta) + np.linalg.norm(grad_approx)
    diff = numerator / denominator
    return diff


class GradientCheck:
    """
    Gradient Checking algorithm

    Gradient checking verifies closeness between the gradients from backpropagation and
    the numerical approximation of the gradient (computed using forward propagation)
    You would usually run it only once to make sure the code is correct
    Doesn't work well with dropout regularization.
    """

    def __init__(self, model, epsilon=1e-5):
        # http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
        self.model = model
        # Higher than 1e-5 likely to produce numeric instability!
        self.epsilon = epsilon

    def test(self, X, Y):
        """
        Check whether the model's backpropagation works properly
        """
        output = self.model.propagate_forward(X, train=False)
        self.model.propagate_backward(output, Y)

        # Roll parameters dictionary into a large (n, 1) vector
        param_theta = roll_params(self.model.layers)
        grad_theta = roll_params(self.model.layers, grads=True)

        # Initialize vectors of the same shape
        num_params = param_theta.shape[0]
        J_plus = np.zeros((num_params, 1))
        J_minus = np.zeros((num_params, 1))
        grad_approx = np.zeros((num_params, 1))

        # Repeat for each number (parameter) in the vector
        for i in range(num_params):
            # Use two-sided Taylor approximation which is 2x more precise than one-sided
            # Add epsilon to the parameter
            theta_plus = np.copy(param_theta)
            theta_plus[i] = theta_plus[i] + self.epsilon
            # Calculate new cost
            unroll_params(theta_plus, self.model.layers)
            output_plus = self.model.propagate_forward(X, train=False)
            J_plus[i] = self.model.compute_cost(output_plus, Y)

            # Subtract epsilon from the parameter
            theta_minus = np.copy(param_theta)
            theta_minus[i] = theta_minus[i] - self.epsilon
            # Calculate new cost
            unroll_params(theta_minus, self.model.layers)
            output_minus = self.model.propagate_forward(X, train=False)
            J_minus[i] = self.model.compute_cost(output_minus, Y)

            # Approximate the partial derivative, error is eps^2
            grad_approx[i] = (J_plus[i] - J_minus[i]) / (2 * self.epsilon)

        # Reset model params
        unroll_params(param_theta, self.model.layers)

        # Difference between the approximated gradient and the backward propagation gradient
        diff = calculate_diff(grad_theta, grad_approx)

        if diff < 1e-7:
            print('Passed gradient checking -', diff)
        else:
            print('Failed gradient checking -', diff)
        return diff

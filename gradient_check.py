import numpy as np


def params_to_vector(params, keys):
    """
    Roll parameters dictionary into a single (n, 1) vector
    """
    theta = np.zeros((0, 1))
    # Start and end indices of each parameter in the vector
    positions = {}
    # Shape of each parameter
    shapes = {}

    for key in keys:
        matrix = params[key]
        # Flatten the parameter into a vector
        vector = np.reshape(matrix, (-1, 1))

        from_i = len(theta)
        # Append the vector
        theta = np.concatenate((theta, vector), axis=0)
        to_i = len(theta)
        positions[key] = (from_i, to_i)
        shapes[key] = matrix.shape

    cache = (positions, shapes)
    return theta, cache


def vector_to_params(theta, cache):
    """
    Unroll parameters dictionary from a single vector
    """
    positions, shapes = cache
    params = {}

    for key, (from_i, to_i) in positions.items():
        # Extract and reshape the parameter to the original form
        params[key] = theta[from_i:to_i].reshape(shapes[key])

    return params


def calculate_diff(A, B):
    """
    Calculate the difference between two vectors using their Euclidean norm
    """
    numerator = np.linalg.norm(np.linalg.norm(A - B))
    denominator = np.linalg.norm(np.linalg.norm(A)) + np.linalg.norm(np.linalg.norm(B))
    difference = numerator / denominator
    return difference

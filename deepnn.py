import numpy as np


class DeepNN:
    def __init__(self, **hyperparams):
        self.layer_dims = hyperparams['layer_dims']
        self.activations = hyperparams['activations']
        self.learning_rate = hyperparams['learning_rate']
        self.num_iterations = hyperparams['num_iterations']

    # INIT PARAMS
    def initialize_params(self):
        np.random.seed(1)
        self.params = {}

        for l in range(len(self.layer_dims)):
            prev_layer_dim = self.layer_dims[l - 1] if l > 0 else self.X.shape[0]
            this_layer_dim = self.layer_dims[l]

            self.params['W' + str(l)] = np.random.randn(this_layer_dim, prev_layer_dim) / np.sqrt(prev_layer_dim)
            self.params['b' + str(l)] = np.zeros((this_layer_dim, 1))

    # FORWARD PROPAGATION
    def linear_forward(self, A_prev, W, b):
        Z = W.dot(A_prev) + b
        assert(Z.shape == (W.shape[0], A_prev.shape[1]))

        linear_cache = (A_prev, W, b)
        return Z, linear_cache

    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        assert(A.shape == Z.shape)

        activation_cache = Z
        return A, activation_cache

    def relu(self, Z):
        A = np.maximum(0, Z)
        assert(A.shape == Z.shape)

        activation_cache = Z
        return A, activation_cache

    def activation_forward(self, A_prev, W, b, activation):
        if activation == 'sigmoid':
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)

        elif activation == 'relu':
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
        assert(A.shape == (W.shape[0], A_prev.shape[1]))

        cache = (linear_cache, activation_cache)
        return A, cache

    def propagate_forward(self, X):
        caches = []
        A = X

        for l in range(len(self.layer_dims)):
            A_prev = A
            A, cache = self.activation_forward(
                A_prev, self.params['W' + str(l)], self.params['b' + str(l)], self.activations[l])
            caches.append(cache)

        return A, caches

    # COST CALCULATION
    def compute_cost(self, AL, Y):
        m = Y.shape[1]

        cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        return cost

    # BACKWARD PROPAGATION
    def relu_backward(self, dA, activation_cache):
        Z = activation_cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        assert(dZ.shape == Z.shape)

        return dZ

    def sigmoid_backward(self, dA, activation_cache):
        Z = activation_cache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        assert(dZ.shape == Z.shape)

        return dZ

    def linear_backward(self, dZ, linear_cache):
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]

        dW = 1. / m * np.dot(dZ, A_prev.T)
        assert(dW.shape == W.shape)

        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        assert(db.shape == b.shape)

        dA_prev = np.dot(W.T, dZ)
        assert(dA_prev.shape == A_prev.shape)

        return dA_prev, dW, db

    def activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache

        if activation == 'relu':
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == 'sigmoid':
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def propagate_backward(self, AL, Y, caches):
        grads = {}
        Y = Y.reshape(AL.shape)
        dA_prev = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        for l in reversed(range(len(self.layer_dims))):
            dA_prev, dW, db = self.activation_backward(dA_prev, caches[l], self.activations[l])
            grads['dA' + str(l - 1)] = dA_prev
            grads['dW' + str(l)] = dW
            grads['db' + str(l)] = db

        return grads

    # UPDATE PARAMS
    def update_params(self, grads):
        for l in range(len(self.layer_dims)):
            self.params['W' + str(l)] -= self.learning_rate * grads['dW' + str(l)]
            self.params['b' + str(l)] -= self.learning_rate * grads['db' + str(l)]

    # TRAIN MODEL
    def train(self, X, Y, print_output=False):
        self.X = X
        self.Y = Y
        self.initialize_params()

        np.random.seed(1)
        costs = []
        for i in range(0, self.num_iterations):
            AL, caches = self.propagate_forward(X)
            cost = self.compute_cost(AL, Y)
            grads = self.propagate_backward(AL, Y, caches)
            self.update_params(grads)

            if print_output and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
                costs.append(cost)

    # PREDICT CLASS
    def predict(self, X, Y):
        m = X.shape[1]

        probs, caches = self.propagate_forward(X)
        probs = np.array(probs, copy=True)
        probs[probs <= 0.5] = 0
        probs[probs > 0.5] = 1
        accuracy = str(np.sum((probs == Y) / m))

        return probs, accuracy

import numpy as np

# https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html


class BatchNorm:
    """
    Batch normalization layer
    """

    def __init__(self, eps=1e-8, decay=0.9):
        self.eps = eps
        # Parameters for running averages
        self.decay = decay
        self.ma_mu = None
        self.ma_var = None

    def init_params(self, shape_in):
        self.shape_in = shape_in
        self.shape_out = shape_in

        self.params = {}
        self.grads = {}

        # Learn two extra parameters for every dimension to get optimum scaling and
        # shifting of activation outputs over zero means and unit variances towards
        # elimination of internal covariate shift.

        # There is no symmetry breaking to consider here
        # GD adapts their values to fit the corresponding feature's distribution
        self.params['gamma'] = np.ones(shape_in)
        self.params['beta'] = np.zeros(shape_in)

    def forward(self, input, predict=False):
        gamma = self.params['gamma']
        beta = self.params['beta']

        if not predict:
            mu = np.mean(input, axis=0, keepdims=True)
            var = np.var(input, axis=0, keepdims=True)

            # Normalize activation output within a mini-batch
            input_norm = (input - mu) / np.sqrt(var + self.eps)
            # Scale and shift these normalized activations
            out = gamma * input_norm + beta

            # Keep track of the moving averages of normalization parameters
            if self.ma_mu is None:
                self.ma_mu = mu
            if self.ma_var is None:
                self.ma_var = var

            self.ma_mu = self.decay * self.ma_mu + (1 - self.decay) * mu
            self.ma_var = self.decay * self.ma_var + (1 - self.decay) * var

            self.cache = (input, input_norm, mu, var)
        else:
            # Fix the normalization at test time
            input_norm = (input - self.ma_mu) / np.sqrt(self.ma_var + self.eps)
            out = gamma * input_norm + beta

        return out

    def backward(self, dout):
        gamma = self.params['gamma']
        beta = self.params['beta']

        input, input_norm, mu, var = self.cache
        m = input.shape[0]

        input_mu = input - mu
        std_inv = 1 / np.sqrt(var + self.eps)

        dinput_norm = dout * gamma
        dvar = -0.5 * np.sum(dinput_norm * input_mu, axis=0, keepdims=True) * std_inv ** 3
        dmu = np.sum(-dinput_norm * std_inv, axis=0, keepdims=True) + \
            dvar * np.mean(-2. * input_mu, axis=0, keepdims=True)

        # Gradients
        dX = (dinput_norm * std_inv) + (2 * dvar * input_mu / m) + (dmu / m)
        assert(dX.shape == input.shape)

        dgamma = np.sum(dout * input_norm, axis=0, keepdims=True)
        assert(dgamma.shape == gamma.shape)

        dbeta = np.sum(dout, axis=0, keepdims=True)
        assert(dbeta.shape == beta.shape)

        self.grads['dgamma'] = dgamma
        self.grads['dbeta'] = dbeta

        self.cache = None
        return dX

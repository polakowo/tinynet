import numpy as np

# https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html


class BatchNorm:
    """
    Batch normalizer

    Batch Normalization is a technique to normalize the internal representation of data for faster training
    BN performs whitening to the intermediate layers of the networks (mean=0 and variance=1)
    Makes deeper layers more robust to changes in previous layers
    In addition, BN works as a regularizer for the model which allows to use less dropout
    """

    def __init__(self, eps=1e-8, decay=0.9):
        self.eps = eps
        # Parameters for running averages
        self.decay = decay
        self.ma_mu = None
        self.ma_var = None

    def forward(self, input, gamma, beta):
        mu = np.mean(input, axis=0, keepdims=True)
        var = np.var(input, axis=0, keepdims=True)

        # Normalize activation output within a mini-batch
        input_norm = (input - mu) / np.sqrt(var + self.eps)
        # Scale and shift these normalized activations
        output = gamma * input_norm + beta
        assert(output.shape == input.shape)

        # Keep track of the moving averages of normalization parameters
        if self.ma_mu is None:
            self.ma_mu = mu
        if self.ma_var is None:
            self.ma_var = var

        self.ma_mu = self.decay * self.ma_mu + (1 - self.decay) * mu
        self.ma_var = self.decay * self.ma_var + (1 - self.decay) * var

        cache = (input, input_norm, mu, var, gamma, beta)
        return output, cache

    def forward_predict(self, input, gamma, beta):
        # Fix the normalization at test time
        input_norm = (input - self.ma_mu) / np.sqrt(self.ma_var + self.eps)
        output = gamma * input_norm + beta
        assert(output.shape == input.shape)

        return output

    def backward(self, dinput, cache):
        input, input_norm, mu, var, gamma, beta = cache
        n_samples = input.shape[0]

        input_mu = input - mu
        std_inv = 1 / np.sqrt(var + self.eps)

        dinput_norm = dinput * gamma
        dvar = -0.5 * np.sum(dinput_norm * input_mu, axis=0, keepdims=True) * std_inv ** 3
        dmu = np.sum(-dinput_norm * std_inv, axis=0, keepdims=True) + \
            dvar * np.mean(-2. * input_mu, axis=0, keepdims=True)

        # Gradients
        doutput = (dinput_norm * std_inv) + (2 * dvar * input_mu / n_samples) + (dmu / n_samples)
        assert(doutput.shape == dinput.shape)

        dgamma = np.sum(dinput * input_norm, axis=0, keepdims=True)
        assert(dgamma.shape == gamma.shape)

        dbeta = np.sum(dinput, axis=0, keepdims=True)
        assert(dbeta.shape == beta.shape)

        return doutput, dgamma, dbeta

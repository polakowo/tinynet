import numpy as np

from src.im2col import im2col_indices
from src.im2col import col2im_indices

# Adopted from
# https://github.com/wiseodd/hipsternet/blob/master/hipsternet/layer.py


class Conv2D:
    """
    Convolutional layer (2D)
    """

    def __init__(self, filters, kernel_size, stride=1, pad=0):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def init_params(self, shape_in):
        # Input volume in format NC
        channels = shape_in[1]
        h_in = shape_in[2]
        w_in = shape_in[3]
        self.shape_in = shape_in

        # Output volume
        h_out = int((h_in - self.kernel_size[0] + 2 * self.pad) / self.stride) + 1
        w_out = int((w_in - self.kernel_size[1] + 2 * self.pad) / self.stride) + 1
        self.shape_out = (1, self.filters, h_out, w_out)

        # Learnable parameters
        self.params = {}
        self.grads = {}
        np.random.seed(1)
        self.params['W'] = np.random.randn(self.filters, channels, self.kernel_size[0], self.kernel_size[1])
        self.params['b'] = np.random.randn(self.filters, 1)

    def forward(self, X, predict=False):
        W = self.params['W']
        b = self.params['b']

        m = X.shape[0]

        X_col = im2col_indices(X,
                               self.kernel_size[0],
                               self.kernel_size[1],
                               padding=self.pad,
                               stride=self.stride)
        W_col = W.reshape(self.filters, -1)

        out = W_col @ X_col + b
        _, _, h_out, w_out = self.shape_out
        out = out.reshape(self.filters, h_out, w_out, m)
        out = out.transpose(3, 0, 1, 2)

        if not predict:
            self.cache = (X, X_col)
        return out

    def backward(self, dout):
        X, X_col = self.cache
        W = self.params['W']
        b = self.params['b']

        db = np.sum(dout, axis=(0, 2, 3))
        db = db.reshape(self.filters, -1)
        assert(db.shape == b.shape)

        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.filters, -1)
        dW = dout_reshaped @ X_col.T
        dW = dW.reshape(W.shape)
        assert(dW.shape == W.shape)

        W_reshape = W.reshape(self.filters, -1)
        dX_col = W_reshape.T @ dout_reshaped
        dX = col2im_indices(dX_col,
                            X.shape,
                            self.kernel_size[0],
                            self.kernel_size[1],
                            padding=self.pad,
                            stride=self.stride)
        assert(dX.shape == X.shape)

        self.grads['dW'] = dW
        self.grads['db'] = db

        self.cache = None
        return dX

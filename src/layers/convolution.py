import numpy as np

from src.im2col import im2col_indices
from src.im2col import col2im_indices

# Adopted from
# https://github.com/wiseodd/hipsternet/blob/master/hipsternet/layer.py


class Conv2D:
    """
    Convolutional layer (2D)
    """

    def __init__(self, out_channels, field, stride=1, pad=0):
        self.out_channels = out_channels
        self.field = field
        self.stride = stride
        self.pad = pad

    def init_params(self, in_shape):
        # Input volume
        in_channels = in_shape[1]
        in_height = in_shape[2]
        in_width = in_shape[3]
        self.in_shape = in_shape

        # Output volume
        out_height = int((in_height - self.field[0] + 2 * self.pad) / self.stride) + 1
        out_width = int((in_width - self.field[1] + 2 * self.pad) / self.stride) + 1
        self.out_shape = (1, self.out_channels, out_height, out_width)

        # Learnable parameters
        self.params = {}
        self.grads = {}
        np.random.seed(1)
        self.params['W'] = np.random.randn(self.out_channels, in_channels, self.field[0], self.field[1])
        self.params['b'] = np.random.randn(self.out_channels, 1)

    def forward(self, X, predict=False):
        W = self.params['W']
        b = self.params['b']

        m = X.shape[0]

        X_col = im2col_indices(X,
                               self.field[0],
                               self.field[1],
                               padding=self.pad,
                               stride=self.stride)
        W_col = W.reshape(self.out_channels, -1)

        out = W_col @ X_col + b
        _, _, out_height, out_width = self.out_shape
        out = out.reshape(self.out_channels, out_height, out_width, m)
        out = out.transpose(3, 0, 1, 2)

        if not predict:
            self.cache = (X, X_col)
        return out

    def backward(self, dout):
        X, X_col = self.cache
        W = self.params['W']
        b = self.params['b']

        db = np.sum(dout, axis=(0, 2, 3))
        db = db.reshape(self.out_channels, -1)
        assert(db.shape == b.shape)

        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        dW = dout_reshaped @ X_col.T
        dW = dW.reshape(W.shape)
        assert(dW.shape == W.shape)

        W_reshape = W.reshape(self.out_channels, -1)
        dX_col = W_reshape.T @ dout_reshaped
        dX = col2im_indices(dX_col,
                            X.shape,
                            self.field[0],
                            self.field[1],
                            padding=self.pad,
                            stride=self.stride)
        assert(dX.shape == X.shape)

        self.grads['dW'] = dW
        self.grads['db'] = db

        self.cache = None
        return dX

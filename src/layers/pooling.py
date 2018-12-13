import numpy as np

from src.im2col import im2col_indices
from src.im2col import col2im_indices

# Adopted from
# https://github.com/wiseodd/hipsternet/blob/master/hipsternet/layer.py


def maxpool(X_col):
    max_idx = np.argmax(X_col, axis=0)
    out = X_col[max_idx, range(max_idx.size)]
    return out, max_idx


def dmaxpool(dX_col, dout_col, pool_cache):
    dX_col[pool_cache, range(dout_col.size)] = dout_col
    return dX_col


def avgpool(X_col):
    out = np.mean(X_col, axis=0)
    cache = None
    return out, cache


def davgpool(dX_col, dout_col, pool_cache):
    dX_col[:, range(dout_col.size)] = 1. / dX_col.shape[0] * dout_col
    return dX_col


class Pool2D:
    """
    Pooling layer (2D)
    """

    def __init__(self, field, stride=1, pad=0, mode='max'):
        self.field = field
        self.stride = stride
        self.pad = pad

        if mode == 'max':
            self.pool_fn = maxpool
            self.dpool_fn = dmaxpool
        elif mode == 'avg':
            self.pool_fn = avgpool
            self.dpool_fn = davgpool

    def init_params(self, in_shape):
        # Input volume (channels first)
        in_channels = in_shape[1]
        in_height = in_shape[2]
        in_width = in_shape[3]
        self.in_shape = in_shape

        # Output volume
        out_height = int((in_height - self.field[0] + 2 * self.pad) / self.stride) + 1
        out_width = int((in_width - self.field[1] + 2 * self.pad) / self.stride) + 1
        self.out_shape = (None, in_channels, out_height, out_width)

        self.params = None
        self.grads = None

    def forward(self, X, predict=False):
        m, in_channels, in_height, in_width = X.shape
        X_reshaped = X.reshape(m * in_channels, 1, in_height, in_width)
        X_col = im2col_indices(X_reshaped,
                               self.field[0],
                               self.field[1],
                               pad=self.pad,
                               stride=self.stride)

        out, pool_cache = self.pool_fn(X_col)

        _, _, out_height, out_width = self.out_shape
        out = out.reshape(out_height, out_width, m, in_channels)
        out = out.transpose(2, 3, 0, 1)

        if not predict:
            self.cache = (X, X_col, pool_cache)
        return out

    def backward(self, dout):
        X, X_col, pool_cache = self.cache

        dX_col = np.zeros_like(X_col)
        dout_col = dout.transpose(2, 3, 0, 1).ravel()

        dX = self.dpool_fn(dX_col, dout_col, pool_cache)

        m, in_channels, in_height, in_width = X.shape
        dX = col2im_indices(dX_col,
                            (m * in_channels, 1, in_height, in_width),
                            self.field[0],
                            self.field[1],
                            pad=self.pad,
                            stride=self.stride)
        dX = dX.reshape(X.shape)
        assert(dX.shape == X.shape)

        self.cache = None
        return dX

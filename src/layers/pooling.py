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

    def __init__(self, pool_size, stride=1, pad=0, mode='max'):
        self.pool_size = pool_size
        self.stride = stride
        self.pad = pad
        self.mode = mode

    def init_params(self, shape_in):
        # Input volume in format NC
        channels = shape_in[1]
        h_in = shape_in[2]
        w_in = shape_in[3]
        self.shape_in = shape_in

        # Output volume
        h_out = int((h_in - self.pool_size[0] + 2 * self.pad) / self.stride) + 1
        w_out = int((w_in - self.pool_size[1] + 2 * self.pad) / self.stride) + 1
        self.shape_out = (1, channels, h_out, w_out)

        self.params = None
        self.grads = None

    def forward(self, X, predict=False):
        m, channels, h_in, w_in = X.shape
        X_reshaped = X.reshape(m * channels, 1, h_in, w_in)
        X_col = im2col_indices(X_reshaped,
                               self.pool_size[0],
                               self.pool_size[1],
                               padding=self.pad,
                               stride=self.stride)

        if self.mode == 'max':
            out, pool_cache = maxpool(X_col)
        elif self.mode == 'avg':
            out, pool_cache = avgpool(X_col)

        _, _, h_out, w_out = self.shape_out
        out = out.reshape(h_out, w_out, m, channels)
        out = out.transpose(2, 3, 0, 1)

        if not predict:
            self.cache = (X, X_col, pool_cache)
        return out

    def backward(self, dout):
        X, X_col, pool_cache = self.cache

        dX_col = np.zeros_like(X_col)
        dout_col = dout.transpose(2, 3, 0, 1).ravel()

        if self.mode == 'max':
            dX = dmaxpool(dX_col, dout_col, pool_cache)
        elif self.mode == 'avg':
            dX = davgpool(dX_col, dout_col, pool_cache)

        m, channels, h_in, w_in = X.shape
        dX = col2im_indices(dX_col,
                            (m * channels, 1, h_in, w_in),
                            self.pool_size[0],
                            self.pool_size[1],
                            padding=self.pad,
                            stride=self.stride)
        dX = dX.reshape(X.shape)
        assert(dX.shape == X.shape)

        self.cache = None
        return dX

import numpy as np

# https://github.com/wiseodd/hipsternet/blob/master/hipsternet/im2col.py
# https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/

# Im2col is a helper for doing the image-to-column transformation.
# This is used in original convolution to do matrix multiplication
# by laying out all patches into a matrix.


def get_im2col_indices(x_shape, field_height, field_width, pad=1, stride=1):
    # First figure out what the size of the output should be
    m, c, h, w = x_shape
    out_height = int((h + 2 * pad - field_height) / stride + 1)
    out_width = int((w + 2 * pad - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, c)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * c)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(c), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, pad=1, stride=1):
    """An implementation of im2col based on some fancy indexing"""
    # Zero-pad the input
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, pad, stride)
    cols = x_padded[:, k, i, j]
    c = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * c, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, pad=1, stride=1):
    """An implementation of col2im based on fancy indexing and np.add.at"""
    m, c, h, w = x_shape
    h_padded, w_padded = h + 2 * pad, w + 2 * pad
    x_padded = np.zeros((m, c, h_padded, w_padded), dtype=cols.dtype)

    k, i, j = get_im2col_indices(x_shape, field_height, field_width, pad, stride)
    cols_reshaped = cols.reshape(c * field_height * field_width, -1, m)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if pad == 0:
        return x_padded
    return x_padded[:, :, pad:-pad, pad:-pad]

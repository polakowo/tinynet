import numpy as np
import h5py
import scipy.io


def load_images():
    with h5py.File('datasets/train_catvnoncat.h5', 'r') as train_dataset:
        with h5py.File('datasets/test_catvnoncat.h5', 'r') as test_dataset:

            train_X = np.array(train_dataset['train_set_x'][:])
            train_Y = np.array(train_dataset['train_set_y'][:])

            test_X = np.array(test_dataset['test_set_x'][:])
            test_Y = np.array(test_dataset['test_set_y'][:])

            train_X = train_X.reshape(train_X.shape[0], -1).T
            test_X = test_X.reshape(test_X.shape[0], -1).T

            train_X = train_X / 255.
            test_X = test_X / 255.

            train_Y = train_Y.reshape((1, train_Y.shape[0]))
            test_Y = test_Y.reshape((1, test_Y.shape[0]))

            return train_X, train_Y, test_X, test_Y


def load_2D_dataset():
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    return train_X, train_Y, test_X, test_Y

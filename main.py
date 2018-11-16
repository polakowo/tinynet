import numpy as np
import h5py

from deepnn import DeepNN

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

        hyperparams = {
            'learning_rate': 0.0075,
            'num_iterations': 1000,
            'layer_dims': [20, 7, 5, 1],
            'activations': ['relu', 'relu', 'relu', 'sigmoid']
        }

        dnn = DeepNN(**hyperparams)
        dnn.train(train_X, train_Y, print_output=True)

        _, accuracy = dnn.predict(train_X, train_Y)
        print("Training accuracy:", accuracy)
        _, accuracy = dnn.predict(test_X, test_Y)
        print("Test accuracy:", accuracy)

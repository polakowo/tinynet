import sklearn
import sklearn.datasets


def load_data(train_m=300, test_m=100):
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=train_m, noise=.2)
    test_X, test_Y = sklearn.datasets.make_moons(n_samples=test_m, noise=.2)
    # Visualize the data
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))

    return train_X, train_Y, test_X, test_Y

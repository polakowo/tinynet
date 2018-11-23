from data import load_2D_dataset
from deepnn.model import DeepNN

from deepnn.utils.layer import Layer
from deepnn.utils import activations


train_X, train_Y, test_X, test_Y = load_2D_dataset()

dnn = DeepNN([
    Layer(n=30, activation=activations.ReLU),
    Layer(n=2, activation=activations.ReLU),
    Layer(n=1, activation=activations.sigmoid)
], lr=0.3, num_epochs=1000)

costs = dnn.train(train_X, train_Y, print_progress=True, gradient_checking=True)

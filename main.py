import numpy as np
from tabulate import tabulate
from colorama import Fore

from data import load_2D_dataset
from model import DeepNN
from utils.regularizers import Dropout

from utils.layer import Layer
from utils import activations


# Load data
train_X, train_Y, test_X, test_Y = load_2D_dataset()

# Set up the model
dnn = DeepNN([
    Layer(n=20, activation=activations.ReLU),
    Layer(n=3, activation=activations.ReLU),
    Layer(n=1, activation=activations.sigmoid)
], lr=0.3, num_epochs=10000)

# Train the model
print()
costs = dnn.train(train_X, train_Y, print_progress=True, print_coststats=True)

# Check the performance
print(Fore.BLUE + '-' * 100 + Fore.RESET)
print('Performance:')
_, train_accuracy = dnn.predict(train_X, train_Y)
_, test_accuracy = dnn.predict(test_X, test_Y)
print(tabulate([['train', '%.4f' % train_accuracy], ['test', '%.4f' % test_accuracy]],
               headers=['', 'accuracy'],
               tablefmt="presto"))
print()

from tabulate import tabulate
from colorama import Fore

from data import load_2D_dataset
from model import DeepNN

from utils.layer import Layer
from utils import activations
from utils.regularizers import L2

from utils.gradient_check import GradientCheck


# Load data
train_X, train_Y, test_X, test_Y = load_2D_dataset()

# Set up the model
dnn = DeepNN([
    Layer(n=5, activation=activations.ReLU),
    Layer(n=2, activation=activations.ReLU),
    Layer(n=1, activation=activations.sigmoid)
], lr=0.1, num_epochs=10000)

gradient_check = GradientCheck(dnn)
gradient_check.test(train_X, train_Y)

"""
# Train the model
print()
costs = dnn.train(train_X, train_Y, print_progress=True, print_coststats=True, print_costdev=True)

# Check the performance
print(Fore.BLUE + '-' * 100 + Fore.RESET)
print('Performance:')
_, train_accuracy = dnn.predict(train_X, train_Y)
_, test_accuracy = dnn.predict(test_X, test_Y)
print(tabulate([['train', '%.4f' % train_accuracy], ['test', '%.4f' % test_accuracy]],
               headers=['', 'accuracy'],
               tablefmt="presto"))
print()
"""

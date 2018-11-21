from tabulate import tabulate
from colorama import Fore

from data import load_2D_dataset
from model import DeepNN
from utils.regularizers import L2
from utils.optimizers import Adam, Momentum
from utils.gradient_check import GradientCheck


# Load data
train_X, train_Y, test_X, test_Y = load_2D_dataset()

# Define hypyerparameters
hyperparams = {
    'learning_rate': 0.3,
    'num_epochs': 100,
    'layer_dims': [20, 3, 1],
    'activations': ['relu', 'relu', 'sigmoid']
}

# Set up the model
dnn = DeepNN(**hyperparams)

# Check the backpropagation algorithm
#gradient_check = GradientCheck(dnn)
#gradient_check.run(train_X, train_Y)

# Train the model
print()
costs = dnn.train(train_X, train_Y, print_dataset=True, print_progress=True, print_cost=True)

# Check the performance
print(Fore.BLUE + '-' * 100 + Fore.RESET)
print('Performance:')
_, train_accuracy = dnn.predict(train_X, train_Y)
_, test_accuracy = dnn.predict(test_X, test_Y)
print(tabulate([['train', '%.4f' % train_accuracy], ['test', '%.4f' % test_accuracy]],
               headers=['', 'accuracy'],
               tablefmt="presto"))
print()

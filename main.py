from tabulate import tabulate
from colorama import Fore

from load_data import load_data
from deepnn.model import DeepNN

from deepnn.utils.layer import Layer
from deepnn.utils import activations


train_X, train_Y, test_X, test_Y = load_data()

dnn = DeepNN([
    Layer(n=30, activation=activations.ReLU),
    Layer(n=2, activation=activations.ReLU),
    Layer(n=1, activation=activations.sigmoid)
], lr=0.3, num_epochs=1000)

costs = dnn.train(train_X, train_Y, print_progress=True)

print(Fore.BLUE + '-' * 100 + Fore.RESET)
print('Performance:')
_, train_accuracy = dnn.predict(train_X, train_Y)
_, test_accuracy = dnn.predict(test_X, test_Y)
print(tabulate([['train', '%.4f' % train_accuracy],
                ['test', '%.4f' % test_accuracy]],
               headers=['', 'accuracy'],
               tablefmt="presto"))
print()

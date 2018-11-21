from model import DeepNN

from utils.regularizers import L2
from utils.optimizers import Adam
from utils.gradient_check import GradientCheck

from data import load_2D_dataset

# Load data
train_X, train_Y, test_X, test_Y = load_2D_dataset()

# Define hypyerparameters
hyperparams = {
    'learning_rate': lambda epoch: 1 / (1 + 0.8 * epoch) * 0.3,
    'num_epochs': 100,
    'mini_batch_size': 32,
    'layer_dims': [20, 3, 1],
    'activations': ['relu', 'relu', 'sigmoid'],
    'regularizer': L2(0.5)
}

# Set up the model
dnn = DeepNN(**hyperparams)

# Check the backpropagation algorithm
#gradient_check = GradientCheck(dnn)
#gradient_check.run(train_X, train_Y)

# Train the model
costs = dnn.train(train_X, train_Y)

# Check the performance
_, accuracy = dnn.predict(train_X, train_Y)
print("Training accuracy:", accuracy)
_, accuracy = dnn.predict(test_X, test_Y)
print("Test accuracy:", accuracy)

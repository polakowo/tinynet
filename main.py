from model import DeepNN
from data import load_2D_dataset


train_X, train_Y, test_X, test_Y = load_2D_dataset()

hyperparams = {
    'learning_rate': 0.3,
    'num_iterations': 30000,
    'layer_dims': [20, 3, 1],
    'activations': ['relu', 'relu', 'sigmoid'],
    # 'l2_lambda': 0,
    # 'keep_probs': [0.86, 0.86, 1]
}

dnn = DeepNN(**hyperparams)

dnn.gradient_check(train_X, train_Y)

dnn.train(train_X, train_Y, print_output=True)

_, accuracy = dnn.predict(train_X, train_Y)
print("Training accuracy:", accuracy)
_, accuracy = dnn.predict(test_X, test_Y)
print("Test accuracy:", accuracy)

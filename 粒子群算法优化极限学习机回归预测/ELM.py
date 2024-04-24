import numpy as np


class ELM:
    def __init__(self, n_hidden, activation_func='sigmoid'):
        self.n_hidden = n_hidden
        self.activation_func = activation_func
        self.input_weights = None
        self.biases = None
        self.output_weights = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def activation(self, x):
        if self.activation_func == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_func == 'tanh':
            return self.tanh(x)
        elif self.activation_func == 'relu':
            return self.relu(x)
        else:
            raise ValueError("Unknown activation function.")

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.input_weights = np.random.normal(0, 1, (n_features, self.n_hidden))
        self.biases = np.random.normal(0, 1, (1, self.n_hidden))
        H = self.activation(np.dot(X, self.input_weights) + self.biases)
        pseudo_inverse_H = np.linalg.pinv(np.dot(H.T, H))
        self.output_weights = np.dot(pseudo_inverse_H, np.dot(H.T, y.reshape(-1, 1)))

    def predict(self, X):
        H = self.activation(np.dot(X, self.input_weights) + self.biases)
        return np.dot(H, self.output_weights).flatten()

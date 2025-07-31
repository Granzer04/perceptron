# See README.md for a detailed explanation of this class and all concepts.
import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, x):
        x = np.insert(x, 0, 1)
        activation = np.dot(self.weights, x)
        return 1 if activation >= 0 else 0

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                xi = np.insert(xi, 0, 1)
                prediction = self.predict(xi[1:])
                self.weights += self.learning_rate * (target - prediction) * xi

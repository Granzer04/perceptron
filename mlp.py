# See README.md for a detailed explanation of this class and all concepts.
import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=10000):
        self.W1 = np.random.randn(hidden_size, input_size + 1) * 0.1
        self.W2 = np.random.randn(output_size, hidden_size + 1) * 0.1
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def forward(self, x):
        x = np.insert(x, 0, 1)
        self.z1 = np.dot(self.W1, x)
        self.a1 = self.sigmoid(self.z1)
        self.a1 = np.insert(self.a1, 0, 1)
        self.z2 = np.dot(self.W2, self.a1)
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                output = self.forward(xi)
                error = target - output
                d2 = error * self.sigmoid_deriv(output)
                d1 = self.sigmoid_deriv(self.a1[1:]) * np.dot(self.W2[:,1:].T, d2)
                self.W2 += self.learning_rate * np.outer(d2, self.a1)
                self.W1 += self.learning_rate * np.outer(d1, np.insert(xi, 0, 1))

    def predict(self, x):
        output = self.forward(x)
        return (output >= 0.5).astype(int)

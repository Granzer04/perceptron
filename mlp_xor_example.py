# See README.md for a detailed explanation of this script and all concepts.
import numpy as np
from mlp import MLP

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])

mlp = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1, epochs=10000)
mlp.train(X, y)

print("Testing MLP on XOR logic gate:")
for xi, target in zip(X, y):
    pred = mlp.predict(xi)
    print(f"Input: {xi}, Predicted: {pred[0]}, Actual: {target[0]}")

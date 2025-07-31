import numpy as np
from perceptron import Perceptron
import matplotlib.pyplot as plt

# See README.md for a detailed explanation of this script and all concepts.

data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
labels = np.array([0, 0, 0, 1])

perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
perceptron.train(data, labels)


# Classic output: print each input, prediction, and actual, and plot actual vs predicted
print("Testing Perceptron on AND logic gate:")
preds = []
for x, y in zip(data, labels):
    pred = perceptron.predict(x)
    preds.append(pred)
    print(f"Input: {x}, Predicted: {pred}, Actual: {y}")

# Scatter plot: x-axis = actual, y-axis = predicted
plt.figure(figsize=(5,5))
plt.scatter(labels, preds, color='blue', s=100)
plt.plot([0,1],[0,1],'r--',label='Perfect Prediction')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Perceptron AND Gate: Predicted vs Actual')
plt.xticks([0,1])
plt.yticks([0,1])
plt.legend()
plt.grid(True)
plt.show()

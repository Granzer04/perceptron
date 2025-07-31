# See README.md for a detailed explanation of this script and all concepts.
import numpy as np
from mlp import MLP
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.rand(1000, 2)
Y = np.sum(X, axis=1, keepdims=True)
Y_scaled = Y / 2.0

mlp = MLP(input_size=2, hidden_size=5, output_size=1, learning_rate=0.1, epochs=5000)
mlp.train(X, Y_scaled)

print("Testing MLP on addition (a + b):")
test_samples = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8], [0.3, 0.7]])
for sample in test_samples:
    pred_scaled = mlp.forward(sample)[0]
    pred = pred_scaled * 2
    actual = np.sum(sample)
    print(f"Input: {sample}, Predicted sum: {pred:.2f}, Actual sum: {actual:.2f}")

# Scatter plot for a larger test set
X_test = np.random.rand(200, 2)
Y_test = np.sum(X_test, axis=1)
Y_pred = np.array([mlp.forward(x)[0] * 2 for x in X_test])

plt.figure(figsize=(7,5))
plt.scatter(Y_test, Y_pred, alpha=0.6, label='Predicted vs Actual')
plt.plot([0,2],[0,2],'r--',label='Perfect Prediction')
plt.xlabel('Actual sum (a + b)')
plt.ylabel('Predicted sum (a + b)')
plt.title('MLP Addition Prediction')
plt.legend()
plt.grid(True)
plt.show()

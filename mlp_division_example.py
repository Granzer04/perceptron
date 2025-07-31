
# See README.md for a detailed explanation of this script and all concepts.
# mlp_division_example.py

import numpy as np
from mlp import MLP
import matplotlib.pyplot as plt

# Create a dataset for division: f(a, b) = a / b
# We'll use numbers between 0.1 and 1 for b to avoid division by zero in training
np.random.seed(42)
X = np.random.rand(1000, 2)
X[:,1] = X[:,1] * 0.9 + 0.1  # Ensure b is in [0.1, 1]
Y = (X[:,0] / X[:,1]).reshape(-1, 1)  # Their quotients

# Scale the target to [0, 1] for sigmoid output (max possible is 1/0.1 = 10)
Y_scaled = Y / 10.0

# Create and train the MLP
mlp = MLP(input_size=2, hidden_size=8, output_size=1, learning_rate=0.1, epochs=8000)
mlp.train(X, Y_scaled)


# Test on new data, including a case with b=0 (division by zero)
print("Testing MLP on division (a / b):")
test_samples = np.array([
    [0.5, 0.5],
    [0.9, 0.3],
    [0.2, 0.8],
    [0.7, 0.0],  # Division by zero case
])
preds = []
actuals = []
for sample in test_samples:
    a, b = sample
    if b == 0:
        print(f"Input: {sample}, Predicted: (see below), Actual: undefined (division by zero)")
        try:
            pred_scaled = mlp.forward(sample)[0]
            pred = pred_scaled * 10
            print(f"  MLP output: {pred:.2f} (should be undefined or very large)")
            preds.append(pred)
            actuals.append(np.nan)
        except Exception as e:
            print(f"  Error: {e}")
            preds.append(np.nan)
            actuals.append(np.nan)
    else:
        pred_scaled = mlp.forward(sample)[0]
        pred = pred_scaled * 10
        actual = a / b
        print(f"Input: {sample}, Predicted: {pred:.2f}, Actual: {actual:.2f}")
        preds.append(pred)
        actuals.append(actual)

# Plot predicted vs actual for a larger test set
X_test = np.random.rand(200, 2)
X_test[:,1] = X_test[:,1] * 0.9 + 0.1
Y_test = (X_test[:,0] / X_test[:,1])
Y_pred = np.array([mlp.forward(x)[0] * 10 for x in X_test])

plt.figure(figsize=(8,5))
plt.scatter(Y_test, Y_pred, alpha=0.6, label='Predicted vs Actual')
plt.plot([0,10],[0,10],'r--',label='Perfect Prediction')
avg_error = np.mean(np.abs(Y_pred - Y_test))
plt.axhline(np.mean(Y_pred), color='g', linestyle=':', label=f'Avg Predicted ({np.mean(Y_pred):.2f})')
plt.xlabel('Actual (a / b)')
plt.ylabel('Predicted (a / b)')
plt.title('MLP Division Prediction')
plt.legend()
plt.grid(True)
plt.show()

# ---
# Note: The MLP has never seen b=0 during training, so its prediction for division by zero is not meaningful.
# In real applications, always check for division by zero before using the result!

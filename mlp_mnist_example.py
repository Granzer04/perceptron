# mlp_mnist_example.py

# See README.md for a detailed explanation of this script and all concepts.
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from mlp import MLP

print("Downloading/loading MNIST dataset (this may take a minute)...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_small, _, y_small, _ = train_test_split(X_scaled, y, train_size=1000, stratify=y, random_state=42)

y_small = y_small.astype(int)
y_onehot = np.zeros((y_small.size, 10))
y_onehot[np.arange(y_small.size), y_small] = 1

mlp = MLP(input_size=784, hidden_size=64, output_size=10, learning_rate=0.1, epochs=20)
mlp.train(X_small, y_onehot)

print("\nTesting MLP on MNIST (first 10 samples):")
for i in range(10):
    sample = X_small[i]
    pred = mlp.forward(sample)
    predicted_digit = np.argmax(pred)
    actual_digit = np.argmax(y_onehot[i])
    print(f"Sample {i+1}: Predicted: {predicted_digit}, Actual: {actual_digit}")

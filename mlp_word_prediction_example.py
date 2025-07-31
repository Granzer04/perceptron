# mlp_word_prediction_example.py
"""
A toy example: MLP tries to predict the last letter of a 4-letter English word given the first 3 letters.
This is for demonstration only—MLPs are not good at real language tasks!
"""
import numpy as np
from mlp import MLP
import matplotlib.pyplot as plt



# Define a much larger vocabulary of 6-letter words (to make the task harder)
base_words = [
    "planet", "rocket", "python", "object", "module", "random", "vector", "matrix", "string", "number",
    "window", "button", "screen", "circle", "square", "border", "canvas", "editor", "layout", "widget",
    "action", "method", "import", "export", "result", "output", "system", "memory", "buffer", "thread"
]
# Add more words by permuting and combining
import random
random.seed(42)
letters = list(set(''.join(base_words)))
words = base_words[:]
while len(words) < 200:
    w = ''.join(random.choices(letters, k=6))
    if w not in words:
        words.append(w)


# Build character-to-index and index-to-character mappings
alphabet = sorted(list(set(''.join(words))))
char2idx = {c: i for i, c in enumerate(alphabet)}
idx2char = {i: c for c, i in char2idx.items()}
num_chars = len(alphabet)


# Predict the second half of each word given the first half
word_len = len(words[0])
half = word_len // 2
X = []
Y = []
for w in words:
    x = np.zeros(num_chars * half)
    for i in range(half):
        x[i * num_chars + char2idx[w[i]]] = 1
    y = np.zeros(num_chars * (word_len - half))
    for j in range(half, word_len):
        y[(j - half) * num_chars + char2idx[w[j]]] = 1
    X.append(x)
    Y.append(y)
X = np.array(X)
Y = np.array(Y)




# Make the MLP smaller and train for fewer epochs to make it struggle
mlp = MLP(input_size=num_chars * half, hidden_size=32, output_size=num_chars * (word_len - half), learning_rate=0.05, epochs=500)
mlp.train(X, Y)



# Test on all words and plot results for each predicted letter
actual = []
predicted = []
for i, w in enumerate(words):
    x = X[i]
    y_true = [char2idx[c] for c in w[half:]]
    y_pred_vec = mlp.forward(x)
    y_pred = [np.argmax(y_pred_vec[j*num_chars:(j+1)*num_chars]) for j in range(word_len - half)]
    actual.extend(y_true)
    predicted.extend(y_pred)
    print(f"Input: {w[:half]}_, Actual: {''.join([idx2char[j] for j in y_true])}, Predicted: {''.join([idx2char[j] for j in y_pred])}")

# Scatter plot: x-axis = actual letter index, y-axis = predicted letter index (for all predicted letters)
plt.figure(figsize=(10,7))
plt.scatter(actual, predicted, s=80, c='purple', alpha=0.6, label='Predicted vs Actual (all letters)')
plt.plot([0,num_chars-1],[0,num_chars-1],'r--',label='Perfect Prediction')
plt.xlabel('Actual letter index')
plt.ylabel('Predicted letter index')
plt.title(f'MLP {word_len}-letter Word Second Half Prediction')
plt.xticks(range(num_chars), [idx2char[i] for i in range(num_chars)])
plt.yticks(range(num_chars), [idx2char[i] for i in range(num_chars)])
plt.legend()
plt.grid(True)
plt.show()

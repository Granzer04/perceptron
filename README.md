

# What is a Perceptron?

A perceptron is the simplest kind of artificial neural network. You can think of it as a tiny decision-maker or a very basic robot brain:

- It looks at several input numbers (like 0 or 1),
- Multiplies each by a weight (which tells it how important that input is),
- Adds them up (plus a bias, which is like a nudge),
- Passes the result through a simple rule (the "activation function") to decide if the answer is 0 or 1.

**Simple analogy:** Imagine a judge with a checklist. For each item, the judge gives it a score (weight), adds up the scores, and if the total is high enough, says "yes" (1); otherwise, "no" (0).

This is how computers can learn to make simple decisions, like figuring out the AND logic gate.

---

# Perceptron & LSTM Stock Prediction Project

Welcome! This project is designed to help you learn about neural networks, starting from the basics (a single-layer perceptron) and moving up to more advanced models (LSTM for stock prediction). No prior experience is required—everything is explained in simple terms.

---

## Project Structure (What’s in Each File?)
- `perceptron.py`: The Perceptron class. Handles training and making predictions. Code is commented for beginners.
- `main.py`: Shows how to train the perceptron on the AND logic gate and test it.
- `lstm_stock_prediction_example.py`: Uses a more advanced LSTM model to predict stock prices. Includes features for saving progress and tracking accuracy.
- `model_weights/`: Folder where the LSTM model saves what it has learned (see below).

---

## How LSTM Model Weights & Incremental Learning Work

When you run the LSTM stock prediction script, it saves two things for each stock ticker:
- The model’s weights (what it has learned) as an `.h5` file.
- The scaler (how your data was normalized) as a `.pkl` file.
These are stored in the `model_weights/` folder. For example, for the ticker `SPY`, you’ll see:
  - `model_weights/SPY_weights.h5`
  - `model_weights/SPY_scaler.pkl`

**Why does this matter?**
- Next time you run the script, it loads these files automatically. This means your model keeps learning from where it left off—no need to start over!
- This is called **incremental learning**. Each run makes your model a little smarter.

**Want to start over?**
- Just delete the files in `model_weights/` for the ticker you want to reset. The script will create new ones next time.

**Moving to another computer?**
- Copy the files in `model_weights/` to the same folder on your new machine.

**If the script crashes:**
- As long as the files in `model_weights/` are still there, you won’t lose your progress.

---

## LSTM Accuracy Statistics & Overfitting Guidance

After each run, the script prints out how well your model is doing for each stock ticker:
- **MSE** (Mean Squared Error): Lower is better.
- **MAE** (Mean Absolute Error): Lower is better.
- **R²** (Coefficient of Determination): Closer to 1 is better.

**How to read these stats:**
```
If MSE/MAE get lower and R² gets closer to 1, your model is improving.
If MSE/MAE start increasing or R² drops as you train more, you may be overfitting.
```
Use these numbers to decide if you should keep training, stop, or adjust your settings.

---

## How the Perceptron Works (Step by Step)
1. **Initialization:** The perceptron starts with all weights set to zero.
2. **Training:** For each input, it predicts an output. If it’s wrong, it tweaks its weights to do better next time. This repeats for several rounds (called epochs).
3. **Prediction:** After training, it can predict the output for new inputs.

---

## How to Run and Test Everything
1. Make sure you have Python 3 and all required packages installed (see below).
2. Run each script to try different features:
   - `python main.py` — Test the basic perceptron on the AND logic gate.
   - `python mlp_xor_example.py` — Test a multi-layer perceptron (MLP) on the XOR logic gate.
   - `python mlp_addition_example.py` — Test the MLP on simple addition.
   - `python mlp_division_example.py` — Test the MLP on division (with a graph).
   - `python mlp_mnist_example.py` — Test the MLP on handwritten digit recognition (MNIST).
   - `python lstm_stock_prediction_example.py` — Test the LSTM on stock price prediction.
3. Check the outputs in the terminal and look at any graphs that pop up.

---

## How to Lower the Cost (Loss/Error)
"Cost" (also called loss or error) tells you how far off your model’s predictions are. Lower cost means better performance. Here’s how to lower it:

- **Train for more epochs:** More training can help, but too much can cause overfitting.
- **Adjust the learning rate:** Try values like 0.01, 0.05, 0.1, or 0.2. Too high can make learning unstable, too low can make it slow.
- **Add more hidden neurons or layers:** For MLPs, more neurons/layers can help learn more complex patterns.
- **Use more or better data:** More examples help the model learn.
- **Scale your data:** Make sure your inputs and outputs are scaled to similar ranges (as in the examples).

### How to Track the Cost
You can add code to your training loop to print or plot the average error after each epoch. For example, in your MLP’s `train` method:

```python
for epoch in range(self.epochs):
    epoch_loss = 0
    for xi, target in zip(X, y):
        output = self.forward(xi)
        error = target - output
        epoch_loss += np.mean(error ** 2)
        # ... weight updates ...
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(X):.4f}")
```

If the loss stops decreasing, try changing the learning rate or model size.

---

## Example Output (AND logic gate)
```
Testing Perceptron on AND logic gate:
Input: [0 0], Predicted: 0, Actual: 0
Input: [0 1], Predicted: 0, Actual: 0
Input: [1 0], Predicted: 0, Actual: 0
Input: [1 1], Predicted: 1, Actual: 1
```

---

## Next Steps & Tips for Beginners

- Try changing the dataset in `main.py` to OR or XOR logic gates.
- Experiment with the learning rate and number of epochs.
- Read the explanations in this README to understand each part of the code.

**Don’t feel dumb!** Learning about neural networks is challenging for everyone at first. Here are some extra explanations to help you:

- **What is a perceptron, really?**
  - Imagine a perceptron as a very simple decision maker. It looks at some numbers (inputs), multiplies each by a weight (how important that input is), adds them up, and then decides "yes" (1) or "no" (0) based on whether the total is above or below zero.
  - The bias is like a starting point or a "nudge" to help the perceptron make better decisions.

- **What does training mean?**
  - Training is just the process of showing the perceptron lots of examples and telling it when it gets the answer wrong. Each time it makes a mistake, it adjusts its weights a little bit to try to do better next time. This is called "learning."

- **What is an epoch?**
  - An epoch is one complete pass through all the training data. Usually, we train for several epochs so the perceptron can keep improving.

- **What is the learning rate?**
  - The learning rate controls how big the weight updates are when the perceptron makes a mistake. If it's too high, learning can be unstable. If it's too low, learning can be very slow.

- **Why can't a perceptron learn XOR?**
  - A single perceptron can only learn problems that can be separated with a straight line (like AND and OR). XOR is more complicated and needs more layers (a multi-layer neural network).

**Remember:**
- Everyone starts somewhere. If you feel lost, that's normal!
- Try changing things and see what happens. Break the code and fix it again. That's how you learn.
- Google is your friend. If you don't understand something, search for it or ask for help.

---

## Want to Go Further?

### Can I add more perceptrons?
Yes! If you connect several perceptrons in layers (called a multi-layer perceptron or MLP), you can solve more complex problems. For example, an MLP can learn the XOR logic gate, which a single perceptron cannot do. This is the foundation of deep learning.

### Can a perceptron predict my text or solve math equations?
Not by itself. A single perceptron (or even a basic MLP) is not powerful enough for tasks like text prediction, language understanding, or solving general math equations. Those tasks require much more advanced neural networks:
- **Text prediction:** Uses models like RNNs (Recurrent Neural Networks), LSTMs, or Transformers (like ChatGPT!) trained on huge amounts of text.
- **Math equation solving:** Needs either symbolic math solvers or very large neural networks trained specifically for math tasks.

### What can you try next?
- Try building a multi-layer perceptron (MLP) for the XOR problem. There are many tutorials online!
- Explore libraries like scikit-learn, TensorFlow, or PyTorch for more advanced neural networks.
- If you want to work with text or math, look into natural language processing (NLP) or symbolic computation libraries.

**Keep experimenting!** Every step you take builds your understanding. If you want, I can help you write a simple MLP or point you to resources for text/math prediction.

You're doing great! Keep experimenting and learning.

## LSTM Accuracy Statistics & Overfitting Guidance

- After each run, the script prints accuracy statistics for each ticker:
  - **MSE** (Mean Squared Error)
  - **MAE** (Mean Absolute Error)
  - **R²** (Coefficient of Determination)
- The output includes guidance:
  ```
  If MSE/MAE get lower and R² gets closer to 1, your model is improving.
  If MSE/MAE start increasing or R² drops as you train more, you may be overfitting.
  ```
- Use these stats to monitor your model’s performance and decide when to stop training or adjust parameters.

## How it Works
1. **Initialization**: The perceptron starts with all weights set to zero.
2. **Training**: For each input in the dataset, the perceptron predicts an output. If the prediction is wrong, it updates its weights to reduce future errors. This process repeats for several epochs (full passes over the data).
3. **Prediction**: After training, the perceptron can predict the output for new inputs.


## Running and Testing Everything
1. Make sure you have Python 3 and all required packages installed (see below).
2. Run each script to test different features:
   - `python main.py` — Test the basic perceptron on the AND logic gate.
   - `python mlp_xor_example.py` — Test the MLP on the XOR logic gate.
   - `python mlp_addition_example.py` — Test the MLP on simple addition.
   - `python mlp_division_example.py` — Test the MLP on division (with a graph).
   - `python mlp_mnist_example.py` — Test the MLP on handwritten digit recognition (MNIST).
   - `python lstm_stock_prediction_example.py` — Test the LSTM on stock price prediction.
3. Check the outputs in the terminal and look at any graphs that pop up.

## How to Lower the Cost (Loss/Error)
"Cost" (also called loss or error) is a measure of how far off your model's predictions are from the correct answers. Lower cost means better performance. Here are ways to lower it:

- **Train for more epochs:** More training can help, but too much can cause overfitting.
- **Adjust the learning rate:** Try values like 0.01, 0.05, 0.1, or 0.2. Too high can make learning unstable, too low can make it slow.
- **Add more hidden neurons or layers:** For MLPs, more neurons/layers can help learn more complex patterns.
- **Use more or better data:** More examples help the model learn.
- **Scale your data:** Make sure your inputs and outputs are scaled to similar ranges (as in the examples).

### How to Track the Cost
You can add code to your training loop to print or plot the average error after each epoch. For example, in your MLP's `train` method, you can calculate the mean squared error (MSE) and print it every 1000 epochs:

```python
for epoch in range(self.epochs):
    epoch_loss = 0
    for xi, target in zip(X, y):
        output = self.forward(xi)
        error = target - output
        epoch_loss += np.mean(error ** 2)
        # ... weight updates ...
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(X):.4f}")
```

This helps you see if the model is improving. If the loss stops decreasing, try changing the learning rate or model size.

---
You should see output showing the perceptron's predictions for the AND logic gate, and similar results for the other scripts.

## Example Output (AND logic gate)
```
Testing Perceptron on AND logic gate:
Input: [0 0], Predicted: 0, Actual: 0
Input: [0 1], Predicted: 0, Actual: 0
Input: [1 0], Predicted: 0, Actual: 0
Input: [1 1], Predicted: 1, Actual: 1
```

## Next Steps
- Try changing the dataset in `main.py` to OR or XOR logic gates.
- Experiment with the learning rate and number of epochs.
- Read the explanations in this README to understand each part of the code.

---
If you have questions, feel free to ask!

---

## Notes for Beginners

**Don't feel dumb!** Learning about neural networks and perceptrons is challenging for everyone at first. Here are some extra explanations to help you:

- **What is a perceptron, really?**
  - Imagine a perceptron as a very simple decision maker. It looks at some numbers (inputs), multiplies each by a weight (how important that input is), adds them up, and then decides "yes" (1) or "no" (0) based on whether the total is above or below zero.
  - The bias is like a starting point or a "nudge" to help the perceptron make better decisions.

- **What does training mean?**
  - Training is just the process of showing the perceptron lots of examples and telling it when it gets the answer wrong. Each time it makes a mistake, it adjusts its weights a little bit to try to do better next time. This is called "learning."

- **What is an epoch?**
  - An epoch is one complete pass through all the training data. Usually, we train for several epochs so the perceptron can keep improving.

- **What is the learning rate?**
  - The learning rate controls how big the weight updates are when the perceptron makes a mistake. If it's too high, learning can be unstable. If it's too low, learning can be very slow.

- **Why can't a perceptron learn XOR?**
  - A single perceptron can only learn problems that can be separated with a straight line (like AND and OR). XOR is more complicated and needs more layers (a multi-layer neural network).

**Remember:**
- Everyone starts somewhere. If you feel lost, that's normal!
- Try changing things and see what happens. Break the code and fix it again. That's how you learn.
- Google is your friend. If you don't understand something, search for it or ask for help.


---

## What if I want to do more?

### Can I add more perceptrons?
Yes! If you connect several perceptrons in layers (called a multi-layer perceptron or MLP), you can solve more complex problems. For example, an MLP can learn the XOR logic gate, which a single perceptron cannot do. This is the foundation of deep learning.

### Can a perceptron predict my text or solve math equations?
Not by itself. A single perceptron (or even a basic MLP) is not powerful enough for tasks like text prediction, language understanding, or solving general math equations. Those tasks require much more advanced neural networks:
- **Text prediction**: Uses models like RNNs (Recurrent Neural Networks), LSTMs, or Transformers (like ChatGPT!) trained on huge amounts of text.
- **Math equation solving**: Needs either symbolic math solvers or very large neural networks trained specifically for math tasks.

### What can you try next?
- Try building a multi-layer perceptron (MLP) for the XOR problem. There are many tutorials online!
- Explore libraries like scikit-learn, TensorFlow, or PyTorch for more advanced neural networks.
- If you want to work with text or math, look into natural language processing (NLP) or symbolic computation libraries.

**Keep experimenting!** Every step you take builds your understanding. If you want, I can help you write a simple MLP or point you to resources for text/math prediction.

You're doing great! Keep experimenting and learning.

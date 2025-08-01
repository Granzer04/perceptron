"""
Weather Predictor for Jersey City, NJ
-------------------------------------
This script uses a simple neural network (MLP) to predict the next day's temperature in Jersey City, NJ, using recent weather data from the Open-Meteo API.

- The model's weights and scaler are saved in the 'model_weights/' folder for incremental learning.
- Each run loads previous weights (if available) and continues training, so the model can improve over time.
- Designed for beginners: clear comments and simple structure.
"""

import os
import numpy as np
import requests
import pickle
from datetime import datetime, timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# How does the neural network learn?
# -------------------------------------------------
# The MLPRegressor is a type of neural network called a Multi-Layer Perceptron (MLP).
# It learns by adjusting its internal weights (numbers that control how much each input matters)
# to reduce the difference between its predictions and the actual temperatures.
#
# - Each input (recent temperatures) is multiplied by a weight.
# - The network adds up these weighted inputs, passes them through hidden layers (with their own weights),
#   and finally produces a prediction (the output).
# - After making a prediction, it checks how far off it was (the error).
# - It then tweaks the weights a little bit in the direction that would have made the prediction better.
# - This process is called backpropagation and is repeated for each training example.
# - Over time, the network "learns" the patterns in the data and gets better at predicting.
#
# In this script, the model is trained incrementally: each time you run it, it loads previous weights (if any),
# trains a bit more on the latest data, and saves the updated weights for next time.

MODEL_WEIGHTS_PATH = 'model_weights/jersey_city_weather_weights.pkl'
SCALER_PATH = 'model_weights/jersey_city_weather_scaler.pkl'

# Jersey City, NJ coordinates
LAT, LON = 40.7282, -74.0776

# Fetch past 14 days of daily temperature data from Open-Meteo
# Fetches the past 14 days of daily mean temperatures for Jersey City, NJ from the Open-Meteo API.
# Returns a numpy array of temperatures.
def fetch_weather_data():
    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=14)
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean&timezone=America/New_York"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()['daily']
    temps = np.array(data['temperature_2m_mean'], dtype=np.float32)
    # Remove or fill NaN values (missing data)
    if np.isnan(temps).any():
        print("Warning: Some temperature data is missing. Filling missing values with the previous day's temperature.")
        # Fill NaNs with previous value, or zero if at the start
        for i in range(len(temps)):
            if np.isnan(temps[i]):
                temps[i] = temps[i-1] if i > 0 else 0.0
    return temps

# Prepares the data for training.
# Uses a sliding window to create input (X) and target (y) pairs.
# For each set of 'window' days, predicts the next day's temperature.
def prepare_data(temps, window=3):
    X, y = [], []
    for i in range(len(temps) - window):
        X.append(temps[i:i+window])
        y.append(temps[i+window])
    return np.array(X), np.array(y)

# Loads a previously saved scaler (for normalizing input data), or creates and saves a new one if it doesn't exist.
def load_or_create_scaler(X):
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = StandardScaler().fit(X)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
    return scaler

# Loads a previously saved neural network model (with learned weights),
# or creates a new one if it doesn't exist yet.
def load_or_create_model():
    if os.path.exists(MODEL_WEIGHTS_PATH):
        with open(MODEL_WEIGHTS_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        model = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1, warm_start=True, random_state=42)
    return model

# Saves the trained model and scaler to disk so they can be loaded next time.
def save_model_and_scaler(model, scaler):
    with open(MODEL_WEIGHTS_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

# Main function: orchestrates data loading, training, prediction, and output.
def main():

    print("Fetching recent weather data for Jersey City, NJ...")
    temps = fetch_weather_data()
    X, y = prepare_data(temps)
    # Check for any remaining NaNs in X or y (should not happen, but just in case)
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask]
    if len(X) == 0:
        print("Error: Not enough valid data to train the model. Try again later.")
        return
    scaler = load_or_create_scaler(X)
    X_scaled = scaler.transform(X)
    model = load_or_create_model()

    print("Training model (incremental learning)...")
    model.fit(X_scaled, y)
    save_model_and_scaler(model, scaler)

    # Predict the next 7 days' mean temperatures
    print("\nPredicted mean temperatures for the next 7 days in Jersey City, NJ (°F):")
    window = list(temps[-3:])
    preds = []
    for i in range(7):
        if np.isnan(window).any():
            print(f"Warning: Not enough recent data to make prediction for day {i+1}.")
            preds.append(float('nan'))
            window.append(float('nan'))
            window = window[1:]
            continue
        window_scaled = scaler.transform([window])
        pred_c = model.predict(window_scaled)[0]
        pred_f = pred_c * 9/5 + 32
        preds.append(pred_f)
        print(f"Day +{i+1}: {pred_f:.2f}°F")
        # Use the prediction as the next day's input (in Celsius)
        window.append(pred_c)
        window = window[1:]
    print("(Model weights and scaler saved for next run.)")

    # Print actual mean temperatures for the past week for comparison
    print("\nActual mean temperatures for the past 7 days in Jersey City, NJ (°F):")
    for i, t in enumerate(temps[-7:]):
        t_f = t * 9/5 + 32
        print(f"Day -{7-i}: {t_f:.2f}°F")
    print("\nCompare the predictions above with the actual values to see how accurate the model is over time.")

if __name__ == "__main__":
    main()

# --- GPU Check ---
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {[gpu.name for gpu in gpus]}")
    try:
        # Remove memory limit: allow TensorFlow to use all available GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow is set to use all available GPU memory (memory growth enabled).")
    except Exception as e:
        print(f"Could not set GPU memory growth: {e}")
else:
    print("No GPU detected. Running on CPU.")

# --- Explicit device check ---
print("TensorFlow logical devices:", tf.config.list_logical_devices())
print("TensorFlow GPU logical devices:", tf.config.list_logical_devices('GPU'))
print("Is built with CUDA:", tf.test.is_built_with_cuda())
print("Is GPU available:", tf.config.list_physical_devices('GPU') != [])

from tensorflow.keras.models import Sequential
# lstm_stock_prediction_example.py
"""
LSTM stock price prediction demo for SPY, AMZN, NVDA, MSFT.
Predicts the first hour of tomorrow's open prices from 1-min data.
Requires: yfinance, pandas, numpy, matplotlib, tensorflow, scikit-learn
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Input
import datetime
import os
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# --- Feature Engineering ---
class FeatureEngineer:
    @staticmethod
    def compute_rsi(series, period=14):
        delta = np.diff(series, prepend=series[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean()
        avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = 50  # neutral for initial period
        return rsi.values

    @staticmethod
    def make_features(data):
        opens = data['Open'].values.flatten()
        highs = data['High'].values.flatten()
        lows = data['Low'].values.flatten()
        closes = data['Close'].values.flatten()
        volumes = data['Volume'].values.flatten()
        rsi = FeatureEngineer.compute_rsi(opens, period=14)
        # 10-period moving average of open price
        ma10 = pd.Series(opens).rolling(window=10, min_periods=1).mean().values
        features = np.column_stack([
            opens, highs, lows, closes, rsi, ma10, volumes
        ])
        return features

# --- LSTM Model ---
class StockLSTMModel:
    def __init__(self, n_past, n_features, epochs=100, batch_size=256):
        self.n_past = n_past
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.n_past, self.n_features)),
            LSTM(256, return_sequences=True),
            LSTM(128, return_sequences=True),
            LSTM(64),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, X):
        return self.model.predict(X).flatten()

# --- Stock Predictor ---
class StockPredictor:
    def __init__(self, ticker, n_past=30, future_minutes=60, epochs=10, batch_size=32):
        self.ticker = ticker
        self.n_past = n_past
        self.future_minutes = future_minutes
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.data = None
        self.features = None
        self.times = None
        self.X = None
        self.Y = None
        self.tY = None
        self.model = None

        # Add weight/scaler persistence
        self.weights_dir = "model_weights"
        os.makedirs(self.weights_dir, exist_ok=True)
        self.weights_path = os.path.join(self.weights_dir, f"{ticker}_weights.h5")
        self.scaler_path = os.path.join(self.weights_dir, f"{ticker}_scaler.pkl")

    def load_data(self):
        # Yahoo only allows up to 7 days of 1m data as of 2025
        data = yf.download(self.ticker, period="7d", interval="1m")
        if data.empty or len(data) < 500:
            raise ValueError(f"Not enough data for {self.ticker}.")
        data = data.between_time("09:30", "16:00")
        self.data = data
        return data

    def save_scaler(self):
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_scaler(self):
        try:
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Loaded existing scaler for {self.ticker}")
            return True
        except FileNotFoundError:
            print(f"No existing scaler found for {self.ticker}")
            return False

    def prepare_features(self):
        scaler_loaded = self.load_scaler()
        self.features = FeatureEngineer.make_features(self.data)
        self.times = self.data.index.to_pydatetime()
        if scaler_loaded:
            self.features_scaled = self.scaler.transform(self.features)
        else:
            self.features_scaled = self.scaler.fit_transform(self.features)
        X, Y, tY = [], [], []
        for i in range(self.n_past, len(self.features_scaled)-1):
            X.append(self.features_scaled[i-self.n_past:i, :])
            Y.append(self.features_scaled[i, 0])
            tY.append(self.times[i])
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.tY = tY
        return self.X, self.Y, self.tY

    def train_test_split(self):
        test_day = self.tY[-1].date()
        train_idx = [i for i, t in enumerate(self.tY) if t.date() < test_day]
        test_idx = [i for i, t in enumerate(self.tY) if t.date() == test_day]
        X_train, Y_train = self.X[train_idx], self.Y[train_idx]
        X_test, Y_test = self.X[test_idx], self.Y[test_idx]
        t_test = [self.tY[i] for i in test_idx]
        return X_train, Y_train, X_test, Y_test, t_test

    def fit_model(self):
        self.model = StockLSTMModel(self.n_past, self.X.shape[2], self.epochs, self.batch_size)
        X_train, Y_train, X_test, Y_test, t_test = self.train_test_split()
        # Try to load existing weights
        if os.path.exists(self.weights_path):
            try:
                self.model.model.load_weights(self.weights_path)
                print(f"Loaded existing weights for {self.ticker} - continuing training...")
            except Exception as e:
                print(f"Could not load weights for {self.ticker}: {e}")
                print("Training from scratch...")
        else:
            print(f"No existing weights found for {self.ticker} - training new model...")
        print(f"Training {self.ticker} for {self.epochs} epochs...")
        self.model.fit(X_train, Y_train)
        # Save weights after training
        try:
            self.model.model.save_weights(self.weights_path)
            self.save_scaler()
            print(f"Saved updated weights and scaler for {self.ticker}")
        except Exception as e:
            print(f"Could not save weights for {self.ticker}: {e}")
        return X_train, Y_train, X_test, Y_test, t_test

    def predict_today(self, X_test, Y_test):
        Y_pred_test = self.model.predict(X_test)
        Y_pred_test_inv = self.scaler.inverse_transform(
            np.concatenate([
                Y_pred_test.reshape(-1,1),
                np.zeros((len(Y_pred_test), self.X.shape[2]-1))
            ], axis=1)
        )[:,0]
        Y_test_inv = self.scaler.inverse_transform(
            np.concatenate([
                Y_test.reshape(-1,1),
                np.zeros((len(Y_test), self.X.shape[2]-1))
            ], axis=1)
        )[:,0]
        return Y_pred_test_inv, Y_test_inv

    def predict_tomorrow(self):
        last_seq = self.features_scaled[-self.n_past:].copy().reshape(1, self.n_past, self.X.shape[2])
        future_preds = []
        n_features = self.X.shape[2]
        for _ in range(self.future_minutes):
            pred_scaled = self.model.predict(last_seq)[0]
            # Build next feature vector: predicted open, rest from last known
            next_feat = last_seq[0, -1, :].copy()
            next_feat[0] = pred_scaled  # predicted open
            # Optionally, update other features if you want to simulate e.g. RSI/MA/volume drift
            pred = self.scaler.inverse_transform(next_feat.reshape(1, -1))[0,0]
            future_preds.append(pred)
            last_seq = np.roll(last_seq, -1, axis=1)
            last_seq[0, -1, :] = next_feat
        return future_preds

    def get_future_times(self, t_test):
        next_day = t_test[-1].date() + datetime.timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += datetime.timedelta(days=1)
        market_open = datetime.datetime.combine(next_day, datetime.time(9, 30))
        future_times = [market_open + datetime.timedelta(minutes=i) for i in range(self.future_minutes)]
        return future_times


# --- Main Orchestration ---
def main():
    TICKERS = ["SPY", "AMZN", "NVDA", "FI"]
    N_PAST = 60  # Use more past data for better learning
    FUTURE_MINUTES = 150  # Predict until mid-day (2.5 hours from 9:30am)
    EPOCHS = 100  # Maximize training
    BATCH_SIZE = 256  # Maximize batch size for GPU

    fig, axs = plt.subplots(len(TICKERS), 2, figsize=(14, 3*len(TICKERS)))
    if len(TICKERS) == 1:
        axs = axs.reshape(1, 2)

    for idx, ticker in enumerate(TICKERS):
        print(f"\n=== {ticker} ===")
        try:
            predictor = StockPredictor(ticker, N_PAST, FUTURE_MINUTES, EPOCHS, BATCH_SIZE)
            predictor.load_data()
            predictor.prepare_features()
            X_train, Y_train, X_test, Y_test, t_test = predictor.fit_model()
            Y_pred_test_inv, Y_test_inv = predictor.predict_today(X_test, Y_test)
            future_preds = predictor.predict_tomorrow()
            future_times = predictor.get_future_times(t_test)
        except Exception as e:
            print(f"Error for {ticker}: {e}")
            continue

        # --- Add accuracy stats here ---
        mse = mean_squared_error(Y_test_inv, Y_pred_test_inv)
        mae = mean_absolute_error(Y_test_inv, Y_pred_test_inv)
        r2 = r2_score(Y_test_inv, Y_pred_test_inv)
        print(f"Stats for {ticker}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        print("If MSE/MAE get lower and R² gets closer to 1, your model is improving.")
        print("If MSE/MAE start increasing or R² drops as you train more, you may be overfitting.")

        # Plot today (actual + predicted) - clean style
        ax_today = axs[idx, 0]
        ax_today.plot(t_test, Y_test_inv, label='Actual Today', color='black')
        ax_today.plot(t_test, Y_pred_test_inv, label='Predicted Today', color='blue')
        ax_today.set_title(f"{ticker} Today (Market Hours)")
        ax_today.set_xlabel("Time")
        ax_today.set_ylabel("Open Price")
        ax_today.legend()
        ax_today.grid(True)

        # Plot tomorrow (predicted) - clean style
        ax_tomorrow = axs[idx, 1]
        ax_tomorrow.plot(future_times, future_preds, label='Tomorrow Predicted', color='red', linestyle='--')
        ax_tomorrow.set_title(f"{ticker} Tomorrow (First Hour, Predicted)")
        ax_tomorrow.set_xlabel("Time")
        ax_tomorrow.set_ylabel("Open Price")
        ax_tomorrow.legend()
        ax_tomorrow.grid(True)
        # Annotate open and close in red
        y_min, y_max = min(future_preds), max(future_preds)
        y_offset = (y_max - y_min) * 0.10 if y_max > y_min else 1.0
        ax_tomorrow.annotate(f"Open: {future_preds[0]:.2f}",
                            xy=(future_times[0], future_preds[0]),
                            xytext=(future_times[0], future_preds[0] + y_offset),
                            va='center', color='red', fontsize=11,
                            arrowprops=dict(arrowstyle='->', color='red'))
        ax_tomorrow.annotate(f"Close: {future_preds[-1]:.2f}",
                            xy=(future_times[-1], future_preds[-1]),
                            xytext=(future_times[-1], future_preds[-1] + y_offset),
                            va='center', color='red', fontsize=11,
                            arrowprops=dict(arrowstyle='->', color='red'))
        print(f"Tomorrow's predicted open: {future_preds[0]:.2f}, close: {future_preds[-1]:.2f}")

    plt.tight_layout()
    plt.show()
    print("\nDone. If you get an ImportError, run: pip install yfinance pandas matplotlib tensorflow scikit-learn")

if __name__ == "__main__":
    main()

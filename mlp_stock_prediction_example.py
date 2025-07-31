# mlp_stock_prediction_example.py
"""
MLP stock price prediction demo for SPY, AMZN, NVDA, MSFT.
Predicts the next day's closing price from the previous N days.
Requires: yfinance, pandas, numpy, matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from mlp import MLP


# Parameters
TICKERS = ["SPY", "AMZN", "NVDA", "MSFT"]
N_PAST = 5  # Number of previous days to use as input
HIDDEN_SIZE = 128  # Increased for better accuracy
EPOCHS = 1000      # Increased for better accuracy

import datetime
import pytz

# Parameters
TICKERS = ["SPY", "AMZN", "NVDA", "MSFT"]
N_PAST = 10  # Number of previous minutes to use as input
HIDDEN_SIZE = 128
EPOCHS = 500
FUTURE_MINUTES = 60  # Only predict the first hour of tomorrow's market
fig, axs = plt.subplots(4, 2, figsize=(14, 12))
all_results = []
for idx, ticker in enumerate(TICKERS):
    print(f"\n=== {ticker} ===")
    # Download 1-minute data for the last 5 days (max allowed by yfinance)
    data = yf.download(ticker, period="5d", interval="1m")
    if data.empty:
        print(f"No intraday data for {ticker}.")
        continue
    # Filter to most recent trading day
    data = data[data.index.date == data.index[-1].date()]
    # Filter to regular market hours (9:30am to 4:00pm US/Eastern)
    data = data.between_time("09:30", "16:00")
    # Use the Open price
    opens = data['Open'].values
    times = data.index.to_pydatetime()
    # Prepare dataset: predict next minute's open from previous N_PAST opens
    X, Y, tY = [], [], []
    for i in range(len(opens) - N_PAST):
        X.append(opens[i:i+N_PAST])
        Y.append(opens[i+N_PAST])
        tY.append(times[i+N_PAST])
    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)
    # Normalize
    x_mean, x_std = X.mean(), X.std()
    y_mean, y_std = Y.mean(), Y.std()
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std
    # Train MLP
    mlp = MLP(input_size=N_PAST, hidden_size=HIDDEN_SIZE, output_size=1, learning_rate=0.01, epochs=EPOCHS)
    mlp.train(Xn, Yn)
    # Predict
    preds = []
    for x in Xn:
        pred = mlp.forward(x)
        preds.append(pred[0])
    preds = np.array(preds) * y_std + y_mean

    # Store results for printing
    all_results.append((Y, preds, ticker, tY))

    # --- Predict the entire next market day's 1-min opens ---
    next_day = tY[-1].date() + datetime.timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += datetime.timedelta(days=1)
    market_open = datetime.datetime.combine(next_day, datetime.time(9, 30))
    market_close = datetime.datetime.combine(next_day, datetime.time(16, 0))
    # Only predict the first FUTURE_MINUTES timestamps for next market day
    future_times = []
    t = market_open
    for _ in range(FUTURE_MINUTES):
        future_times.append(t)
        t += datetime.timedelta(minutes=1)
    future_preds = []
    input_window = [float(x) for x in opens[-N_PAST:]]
    for _ in range(len(future_times)):
        norm_input = (np.array(input_window) - x_mean) / x_std
        pred_norm = mlp.forward(norm_input)
        pred = float(pred_norm[0] * y_std + y_mean)
        future_preds.append(pred)
        input_window = input_window[1:] + [pred]

    # Plot today in left subplot, tomorrow in right subplot
    axs[idx, 0].plot(tY, Y.flatten(), label='Actual', color='black')
    axs[idx, 0].plot(tY, preds, label='Predicted', color='blue')
    axs[idx, 0].set_title(f"{ticker} Today {tY[0].date()} Market Hours")
    axs[idx, 0].set_xlabel("Time")
    axs[idx, 0].set_ylabel("Open Price")
    axs[idx, 0].legend()
    axs[idx, 0].grid(True)
    last_x = tY[-1]
    last_pred = preds[-1]
    axs[idx, 0].annotate(f"Pred: {last_pred:.2f}",
                        xy=(last_x, last_pred),
                        xytext=(last_x, last_pred + (Y.max() - Y.min()) * 0.05),
                        va='center', color='blue', fontsize=10,
                        arrowprops=dict(arrowstyle='->', color='blue'))

    axs[idx, 1].plot(future_times, future_preds, label='Tomorrow Predicted', color='red', linestyle='--')
    axs[idx, 1].set_title(f"{ticker} Tomorrow {future_times[0].date()} Market Hours")
    axs[idx, 1].set_xlabel("Time")
    axs[idx, 1].set_ylabel("Open Price")
    axs[idx, 1].legend()
    axs[idx, 1].grid(True)
    axs[idx, 1].annotate(f"Tomorrow Open: {future_preds[0]:.2f}",
                        xy=(future_times[0], future_preds[0]),
                        xytext=(future_times[0], future_preds[0] + (Y.max() - Y.min()) * 0.10),
                        va='center', color='red', fontsize=11,
                        arrowprops=dict(arrowstyle='->', color='red'))
    axs[idx, 1].annotate(f"Tomorrow Close: {future_preds[-1]:.2f}",
                        xy=(future_times[-1], future_preds[-1]),
                        xytext=(future_times[-1], future_preds[-1] + (Y.max() - Y.min()) * 0.10),
                        va='center', color='red', fontsize=11,
                        arrowprops=dict(arrowstyle='->', color='red'))
    print(f"Tomorrow's predicted open: {future_preds[0]:.2f}, close: {future_preds[-1]:.2f}")

plt.tight_layout()
plt.show()

# Print last 5 predictions for each ticker
for Y, preds, ticker, tY in all_results:
    print(f"\nLast 5 predictions for {ticker}:")
    for i in range(-5, 0):
        print(f"{tY[i].strftime('%Y-%m-%d %H:%M')}: Actual={Y[i][0]:.2f}, Predicted={preds[i]:.2f}")

print("\nDone. If you get an ImportError, run: pip install yfinance pandas matplotlib pytz")

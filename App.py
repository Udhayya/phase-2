# Install packages if needed
# !pip install yfinance tensorflow pandas scikit-learn matplotlib

import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Suppress warnings and TensorFlow logs
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

def predict_stock_price(ticker, start_date='2010-01-01', end_date='2023-12-31', epochs=50, batch_size=64):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            print("Invalid ticker or no data available.")
            return

        # Target variable: 'Close' price
        data = stock_data['Close'].values.reshape(-1, 1)

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Create dataset sequences
        def create_dataset(data, time_step=100):
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:i + time_step, 0])
                Y.append(data[i + time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 100
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build the LSTM model
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Predict
        train_predict = model.predict(X_train, verbose=0)
        test_predict = model.predict(X_test, verbose=0)

        # Inverse transform predictions and targets
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

        # RMSE
        train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predict))
        test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict))

        print(f"Actual price : {train_rmse:.2f}")
        print(f"Close price : {test_rmse:.2f}")

        # Plot
        plt.figure(figsize=(15, 6))
        plt.plot(scaler.inverse_transform(scaled_data), label='Actual Close Price')

        train_plot = np.empty_like(scaled_data)
        train_plot[:, :] = np.nan
        train_plot[time_step:len(train_predict) + time_step, :] = train_predict
        plt.plot(train_plot, label='Train Prediction')

        test_plot = np.empty_like(scaled_data)
        test_plot[:, :] = np.nan
        test_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1, :] = test_predict
        plt.plot(test_plot, label='Test Prediction')

        plt.title(f"{ticker.upper()} Close Price Prediction")
        plt.xlabel('Time')
        plt.ylabel('Close Price')
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

# ✅ Ask user for input
ticker_input = input("Enter stock ticker symbol (e.g., AAPL, TCS.NS, INFY.NS): ").upper()
predict_stock_price(ticker_input)

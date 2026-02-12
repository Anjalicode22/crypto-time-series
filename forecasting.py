import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ================= ARIMA =================
def arima_forecast(df, steps=30):
    """
    Perform ARIMA forecasting.
    Parameters:
        df (pd.DataFrame): Must contain 'Price' column
        steps (int): Number of days to forecast
    Returns:
        np.array: Forecasted values
    """
    model = ARIMA(df['Price'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast.values

# ================= SARIMA =================
def sarima_forecast(df, steps=30):
    """
    Perform SARIMA forecasting (with seasonality).
    Parameters:
        df (pd.DataFrame): Must contain 'Price' column
        steps (int): Number of days to forecast
    Returns:
        np.array: Forecasted values
    """
    model = SARIMAX(
        df['Price'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=steps)
    return forecast.values

# ================= PROPHET =================
def prophet_forecast(df, steps=30):
    """
    Perform Prophet forecasting.
    Parameters:
        df (pd.DataFrame): Must contain 'Date' and 'Price' columns
        steps (int): Number of days to forecast
    Returns:
        np.array: Forecasted values
    """
    prophet_df = df.rename(columns={'Date': 'ds', 'Price': 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    return forecast['yhat'].tail(steps).values

# ================= LSTM =================
def lstm_forecast(df, steps=30, epochs=5, batch_size=32):
    """
    Perform LSTM forecasting.
    Parameters:
        df (pd.DataFrame): Must contain 'Price' column
        steps (int): Number of days to forecast
        epochs (int): Training epochs
        batch_size (int): Training batch size
    Returns:
        np.array: Forecasted values
    """
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Price'].values.reshape(-1, 1))

    # Prepare sequences
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    # Forecast future steps
    temp_input = list(scaled_data[-60:].flatten())
    preds = []

    for _ in range(steps):
        x_input = np.array(temp_input[-60:]).reshape(1, 60, 1)
        yhat = model.predict(x_input, verbose=0)[0][0]
        preds.append(yhat)
        temp_input.append(yhat)

    # Inverse scale
    forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return forecast

# (imports unchanged)
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.statespace.sarimax import SARIMAX
import mplfinance as mpf
import os

st.set_page_config(
    page_title="Time Series Forecasting Dashboard",
    page_icon="🪙",
    layout="wide"
)

DATA_PATH = "data.csv"

@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()
data = df.copy()

def evaluate_model(actual, predicted, model_name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

page = st.session_state.active_page

# ================= FORECASTING MODELS =================
if page == "Forecasting Models":

    st.subheader("Forecasting Models")

    # Prophet
    df_prophet = data[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
    train_size = int(len(df_prophet) * 0.8)
    train, test = df_prophet[:train_size], df_prophet[train_size:]

    train_prophet = train.copy()
    train_prophet["ds"] = pd.to_datetime(train_prophet["ds"])
    train_prophet["y"] = pd.to_numeric(train_prophet["y"], errors="coerce")
    train_prophet = train_prophet.dropna()
    train_prophet = train_prophet[["ds", "y"]]

    model_p = Prophet()
    model_p.fit(train_prophet)

    future = model_p.make_future_dataframe(periods=len(test), freq='D')
    forecast = model_p.predict(future)
    prophet_pred = forecast['yhat'][-len(test):].values

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(train['ds'], train['y'], label='Train')
    ax1.plot(test['ds'], test['y'], label='Test')
    ax1.plot(test['ds'], prophet_pred, label='Prophet')
    ax1.legend()
    ax1.set_title("Prophet Forecast")

    prophet_metrics = evaluate_model(test['y'], prophet_pred, "Prophet")

    # ARIMA
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)

    train_size = int(len(data) * 0.8)
    train_arima = data['close'][:train_size]
    test_arima = data['close'][train_size:]

    arima_fit = ARIMA(train_arima, order=(5,1,0)).fit()
    arima_pred = arima_fit.forecast(steps=len(test_arima))

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(train_arima.index, train_arima, label='Train')
    ax2.plot(test_arima.index, test_arima, label='Test')
    ax2.plot(test_arima.index, arima_pred, label='ARIMA')
    ax2.legend()
    ax2.set_title("ARIMA Forecast")

    arima_metrics = evaluate_model(test_arima, arima_pred, "ARIMA")

    # SARIMA
    sarima_model = SARIMAX(train_arima, order=(2,1,2),
                           seasonal_order=(1,1,1,12)).fit(disp=False)

    sarima_pred = sarima_model.predict(
        start=len(train_arima),
        end=len(data)-1,
        dynamic=False
    )

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(train_arima.index, train_arima, label='Train')
    ax3.plot(test_arima.index, test_arima, label='Test')
    ax3.plot(test_arima.index, sarima_pred, label='SARIMA')
    ax3.legend()
    ax3.set_title("SARIMA Forecast")

    sarima_metrics = evaluate_model(test_arima, sarima_pred, "SARIMA")

    # LSTM
    prices = data['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    split = int(len(scaled) * 0.8)
    train_data = scaled[:split]
    test_data = scaled[split:]

    def create_dataset(ds, step=60):
        X, y = [], []
        for i in range(step, len(ds)):
            X.append(ds[i-step:i, 0])
            y.append(ds[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    lstm_pred = model.predict(X_test)
    lstm_pred = scaler.inverse_transform(lstm_pred)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.plot(actual_prices, label='Actual')
    ax4.plot(lstm_pred, label='Predicted')
    ax4.legend()
    ax4.set_title("LSTM Prediction")

    lstm_metrics = evaluate_model(actual_prices, lstm_pred, "LSTM")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(fig3)
    with col4:
        st.pyplot(fig4)

    st.session_state["model_metrics"] = [
        prophet_metrics,
        arima_metrics,
        sarima_metrics,
        lstm_metrics
    ]

# ================= MODEL EVALUATION =================
elif page == "Model Evaluation":

    st.subheader("📊 Model Evaluation Metrics")

    if "model_metrics" in st.session_state:
        metrics_df = pd.DataFrame(st.session_state["model_metrics"])
        metrics_df = metrics_df.sort_values(by="RMSE")
        st.dataframe(metrics_df)

        best_model = metrics_df.iloc[0]["Model"]
        st.success(f"🏆 Best Performing Model (Based on RMSE): {best_model}")

        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(metrics_df["Model"], metrics_df["RMSE"])
        ax.set_ylabel("RMSE")
        ax.set_title("Model RMSE Comparison")
        st.pyplot(fig)
    else:
        st.info("Please run models in 'Forecasting Models' first.")

# ================= POWER BI =================
elif page == "Power BI Dashboard":

    st.markdown("## 📊 Interactive Power BI Dashboard")

    powerbi_url = "https://app.powerbi.com/view?r=eyJrIjoiYjA3YWQyN2MtMDM4ZC00YWUxLTlkNGQtNWIxYTc2MTZiZTI1IiwidCI6IjM0YTYzMzMwLWU2MWUtNGMwZC04ODIyLTQ4MjViZTk0YTNkYiJ9"

    components.iframe(powerbi_url, width=1200, height=650)

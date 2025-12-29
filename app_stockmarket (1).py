import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------- Load assets ----------
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_model.keras")

@st.cache_resource
def load_scaler_and_config():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("config.json", "r") as f:
        config = json.load(f)
    return scaler, config

model = load_lstm_model()
scaler, config = load_scaler_and_config()

LOOK_BACK = config["look_back"]
FEATURE_COLS = config["feature_columns"]
FREQ = config.get("freq", "B")

# ---------- Helper: build future forecast ----------
def forecast_30_days_lstm(df):
    data = df[FEATURE_COLS].copy()
    scaled = scaler.transform(data.values)

    if len(scaled) < LOOK_BACK:
        raise ValueError(f"Need at least {LOOK_BACK} rows, found {len(scaled)}.")

    window = scaled[-LOOK_BACK:, :]
    future_steps = 30
    future_ts_log_diff = []

    for _ in range(future_steps):
        inp = window[np.newaxis, :, :]
        next_scaled_close = model.predict(inp, verbose=0)[0, 0]

        dummy = np.zeros((1, scaled.shape[1]))
        dummy[0, -1] = next_scaled_close
        inv = scaler.inverse_transform(dummy)
        next_ts_log_diff = inv[0, -1]
        future_ts_log_diff.append(next_ts_log_diff)

        new_row_scaled = np.zeros((1, scaled.shape[1]))
        new_row_scaled[0, -1] = next_scaled_close
        window = np.vstack([window[1:], new_row_scaled])

    # Use user-provided last close date
    last_date = df.attrs.get("LastCloseDate", df.index.max())

    future_dates = pd.date_range(
        start=last_date,
        periods=future_steps + 1,
        freq=FREQ
    )[1:]

    # Convert log-diff → price path
    base_price = df["BasePriceForPlot"].iloc[-1]
    last_log_price = np.log(base_price)

    log_prices_future = [last_log_price]
    for delta in future_ts_log_diff:
        log_prices_future.append(log_prices_future[-1] + delta)

    price_future = np.exp(log_prices_future[1:])

    return pd.DataFrame(
        {
            "ts_log_diff_forecast": future_ts_log_diff,
            "price_forecast": price_future,
        },
        index=future_dates,
    )

# ---------- Streamlit UI ----------
st.set_page_config(
    page_title="AAPL 30-Day LSTM Forecast",
    layout="centered"
)

st.title("AAPL 30-Day Forecast (LSTM)")
st.write(
    "Upload recent Apple stock data to generate a **30-business-day** price forecast."
)

uploaded = st.file_uploader(
    "Upload CSV with Date index and columns: Open, High, Low, Adj Close, Volume, Close(ts_log_diff)",
    type=["csv"]
)

if uploaded is not None:
    df = pd.read_csv(uploaded)

    # ✅ FIXED DATE PARSING (DD-MM-YYYY)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            format="%d-%m-%Y",
            errors="coerce"
        )
        df = df.dropna(subset=["Date"]).set_index("Date")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        st.success(
            f"Data loaded: {len(df)} rows "
            f"({df.index.min().date()} → {df.index.max().date()})"
        )

        # User inputs
        base_price_input = st.number_input(
            "Last actual close price (for plotting future path, in USD):",
            min_value=1.0,
            value=150.0
        )

        last_close_date_input = st.date_input(
            "Last actual close date:",
            value=df.index.max().date()
        )

        df["BasePriceForPlot"] = base_price_input
        df.attrs["LastCloseDate"] = pd.to_datetime(last_close_date_input)

        try:
            forecast_df = forecast_30_days_lstm(df)
        except ValueError as e:
            st.error(str(e))
        else:
            st.subheader("Forecasted Prices (Next 30 Business Days)")
            st.dataframe(forecast_df[["price_forecast"]].round(2))

else:
    st.info("Upload a CSV file to start forecasting.")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

nifty_50 = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "LT", "KOTAKBANK", "SBIN",
    "HCLTECH", "BHARTIARTL", "ITC", "ASIANPAINT", "BAJFINANCE", "WIPRO", "HINDUNILVR",
    "ADANIENT", "ADANIPORTS", "MARUTI", "M&M", "POWERGRID", "TITAN", "NESTLEIND",
    "AXISBANK", "ULTRACEMCO", "NTPC", "BAJAJFINSV", "SUNPHARMA", "TECHM", "COALINDIA",
    "HDFCLIFE", "BRITANNIA", "INDUSINDBK", "CIPLA", "EICHERMOT", "JSWSTEEL", "DIVISLAB",
    "DRREDDY", "HINDALCO", "SBILIFE", "GRASIM", "TATASTEEL", "BPCL", "BAJAJ-AUTO",
    "HEROMOTOCO", "ONGC", "UPL", "SHREECEM", "APOLLOHOSP", "TATACONSUM"
]

indian_bank_holidays_2025 = [
    "2025-01-26", "2025-03-08", "2025-03-29", "2025-04-14", "2025-05-01",
    "2025-08-15", "2025-10-02", "2025-10-24", "2025-12-25"
]

def is_trading_day(date):
    return date.weekday() < 5 and date.strftime("%Y-%m-%d") not in indian_bank_holidays_2025

def create_enhanced_dataset(series, window_size=5):
    X, y = [], []
    moving_avg = pd.Series(series).rolling(3).mean().fillna(0).values
    returns = np.diff(series, prepend=series[0])
    for i in range(len(series) - window_size):
        window = series[i:i+window_size]
        window_ma = moving_avg[i:i+window_size]
        window_ret = returns[i:i+window_size]
        features = np.concatenate([window, window_ma, window_ret])
        X.append(features)
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

def forecast_future_prices(model, norm_series, window_size, days=7):
    future_preds, future_dates = [], []
    series = np.ravel(norm_series.tolist())
    current_date = datetime.today()
    while len(future_preds) < days:
        current_date += timedelta(days=1)
        if not is_trading_day(current_date):
            continue
        series = pd.to_numeric(series, errors='coerce')
        ma = pd.Series(series).rolling(3).mean().fillna(0).values
        returns = np.diff(series, prepend=series[0])
        x_win = series[-window_size:]
        ma_win = ma[-window_size:]
        ret_win = returns[-window_size:]
        features = np.concatenate([x_win, ma_win, ret_win]).reshape(1, -1)
        pred = model.predict(features)[0]
        future_preds.append(pred)
        future_dates.append(current_date)
        series = np.append(series, pred)
    return future_dates, future_preds

def inverse_normalize(norm_vals, mean, std):
    return np.array(norm_vals) * std + mean

# Streamlit app
st.title("📈 FINANCIAL FORECASTING")
stock = st.selectbox("Select Stock", options=nifty_50)
from_date = st.date_input("From Date", value=datetime(2022, 1, 1))
to_date = st.date_input("To Date", value=datetime.today())
forecast_days = st.slider("Select number of future trading days to forecast", min_value=3, max_value=15, value=7)

if stock and from_date < to_date:
    ticker = stock.strip().upper() + ".NS"
    data = yf.download(ticker, start=str(from_date), end=str(to_date + timedelta(days=1)))

    if 'Close' not in data.columns or len(data) < 50:
        st.warning("Data not available or insufficient for selected range.")
    else:
        close_prices = data['Close'].dropna().values.flatten()
        mean = np.mean(close_prices)
        std = np.std(close_prices)
        norm_prices = (close_prices - mean) / std

        best_val_rmse = float("inf")
        for window_size in range(5, 30, 5):
            X, y = create_enhanced_dataset(norm_prices, window_size)
            if len(X) < 40:
                continue
            split_idx = int(len(X) * 0.8)
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_val, y_val = X[split_idx:], y[split_idx:]
            for alpha in np.round(np.arange(0.1, 2.1, 0.3), 2):
                model = Ridge(alpha=alpha)
                model.fit(X_train, y_train)
                val_rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
                if val_rmse < best_val_rmse:
                    best_model = model
                    best_window = window_size
                    best_alpha = alpha
                    best_train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
                    best_val_rmse = val_rmse

        if to_date >= datetime.today().date():
            future_dates, future_preds_norm = forecast_future_prices(best_model, norm_prices, best_window, days=forecast_days)
            future_preds_real = inverse_normalize(future_preds_norm, mean, std)
            past_df = pd.DataFrame({"Date": data.index, "Actual": close_prices})
            future_df = pd.DataFrame({"Date": future_dates, "Predicted": future_preds_real})
            combined = pd.concat([past_df, future_df], ignore_index=True)
            st.subheader("📊 Forecast for Next Trading Days")
            st.line_chart(combined.set_index("Date"))
            st.dataframe(future_df)

            csv = future_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Forecast CSV", data=csv, file_name=f"{stock}_forecast.csv", mime="text/csv")
        else:
            X_all, y_all = create_enhanced_dataset(norm_prices, best_window)
            preds_norm = best_model.predict(X_all)
            preds = inverse_normalize(preds_norm, mean, std)
            y_true = close_prices[best_window:]
            dates = data.index[best_window:]
            compare_df = pd.DataFrame({"Date": dates, "Actual": y_true, "Predicted": preds})
            st.subheader("📊Predicted vs Actual")
            st.line_chart(compare_df.set_index("Date"))
            st.dataframe(compare_df)

            csv_backtest = compare_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", data=csv_backtest, file_name=f"{stock}.csv", mime="text/csv")

if "last_run" not in st.session_state:
    st.session_state.last_run = datetime.now()
elapsed = datetime.now() - st.session_state.last_run
if elapsed.total_seconds() > 86400:
    st.session_state.last_run = datetime.now()
    st.experimental_rerun()
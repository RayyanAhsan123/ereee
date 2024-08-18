import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta

# Streamlit title and description
st.title("Tin Price Prediction")
st.write("This application predicts the price of Tin using Prophet and ARIMA models.")

# API and base URL for fetching data
api_key = "l333ljg4122qws9kxkb4hly7a8dje27vk46c7zkceih11wmnrj7lqreku176"
base_url = "https://metals-api.com/api"

# Function to fetch data for a given timeframe
def fetch_data(start_date, end_date):
    params = {
        "access_key": api_key,
        "base": "USD",
        "symbols": "TIN",
        "start_date": start_date,
        "end_date": end_date
    }
    response = requests.get(f"{base_url}/timeseries", params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get('success', False):
            return data.get("rates", {})
        else:
            st.error(f"API request failed: {data.get('error', {}).get('info')}")
            return None
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return None

# Fetch data in smaller timeframes and combine results
start_dates = ["2024-07-01", "2024-07-16"]
end_dates = ["2024-07-15", "2024-07-31"]
all_data = {}
for start_date, end_date in zip(start_dates, end_dates):
    data = fetch_data(start_date, end_date)
    if data:
        all_data.update(data)

if all_data:
    df = pd.DataFrame.from_dict(all_data, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.reset_index().rename(columns={"index": "ds", "TIN": "y"})
    df = df[["ds", "y"]]
    st.write("Data Preview:")
    st.dataframe(df.head(30))  # Show more rows to verify data integrity
else:
    st.error("No data fetched.")
    st.stop()

# Check for anomalies and outliers
st.subheader("Price of Tin Over Time")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['ds'], df['y'], marker='o', linestyle='-')
ax.set_title('Price of Tin Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
st.pyplot(fig)

# Check for missing values and fill them if any
st.write("Missing values before filling:")
st.write(df.isnull().sum())
df.fillna(method='ffill', inplace=True)
st.write("Missing values after filling:")
st.write(df.isnull().sum())

# Decompose the time series if there is enough data
if len(df) >= 60:  # Requires at least 2 cycles for period=30
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(df['y'], model='additive', period=30)
    st.subheader("Seasonal Decomposition")
    fig = result.plot()
    st.pyplot(fig)
else:
    st.warning(f"Insufficient data for seasonal decomposition. Only {len(df)} observations available.")

# Initialize the Prophet model with adjusted parameters
model = Prophet(
    changepoint_prior_scale=0.1,  # Adjust based on performance
    yearly_seasonality=True,
    weekly_seasonality=True
)

# Fit the model
model.fit(df)

# Create a dataframe for future dates
future = model.make_future_dataframe(periods=15)
forecast = model.predict(future)

# Visualize the forecast
st.subheader("Prophet Forecast")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Evaluate the model using cross-validation
df_cv = cross_validation(model, initial='14 days', period='7 days', horizon='7 days')
df_performance = performance_metrics(df_cv)
st.write("Cross-validation Performance Metrics:")
st.dataframe(df_performance)

# Train-test split for evaluation
split_index = int(0.8 * len(df))
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# Create a new Prophet model for the training set
train_model = Prophet(
    changepoint_prior_scale=0.1,
    yearly_seasonality=True,
    weekly_seasonality=True
)

# Fit the new model on the training set
train_model.fit(train_df)

# Make predictions on the test set
future_test = train_model.make_future_dataframe(periods=len(test_df), include_history=False)
forecast_test = train_model.predict(future_test)

# Evaluate the model
y_test = test_df['y'].values
y_pred = forecast_test['yhat'].values

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"Mean Squared Error: {mse}")
st.write(f"Mean Absolute Error: {mae}")
st.write(f"R^2 Score: {r2}")

# Calculate prediction accuracy as a percentage
accuracy = 100 - (mae / y_test.mean() * 100)
st.write(f"Prediction Accuracy: {accuracy:.2f}%")

# Visualize the forecast vs actual values
st.subheader("Forecast vs Actual")
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(test_df['ds'], y_test, label='Actual')
ax2.plot(test_df['ds'], y_pred, label='Predicted')
ax2.legend()
ax2.set_title('Forecast vs Actual')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
st.pyplot(fig2)

# Consider alternative models: ARIMA
# For ARIMA, ensure the series is stationary
# Check for stationarity
result = adfuller(df['y'])
st.write(f'ADF Statistic: {result[0]}')
st.write(f'p-value: {result[1]}')

# Fit ARIMA model (adjust p, d, q as necessary)
arima_model = ARIMA(df['y'], order=(5,1,0))  # Example order
arima_result = arima_model.fit()

# Forecast using ARIMA
arima_forecast = arima_result.get_forecast(steps=15)
arima_conf_int = arima_forecast.conf_int()
arima_pred = arima_forecast.predicted_mean

# Plot ARIMA forecast
st.subheader("ARIMA Forecast")
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(df['ds'], df['y'], label='Historical')
ax3.plot(pd.date_range(start=df['ds'].iloc[-1], periods=16, freq='D')[1:], arima_pred, label='ARIMA Forecast')
ax3.fill_between(pd.date_range(start=df['ds'].iloc[-1], periods=16, freq='D')[1:],
                 arima_conf_int.iloc[:, 0], arima_conf_int.iloc[:, 1], color='pink', alpha=0.3)
ax3.legend()
ax3.set_title('ARIMA Forecast')
ax3.set_xlabel('Date')
ax3.set_ylabel('Price')
st.pyplot(fig3)

# Function to predict the price for a specific future date using Prophet
def predict_price_for_date(date_str):
    future_date = pd.to_datetime(date_str)
    future = pd.DataFrame({'ds': [future_date]})
    forecast = model.predict(future)
    return forecast['yhat'].values[0]

# User input for date prediction
st.subheader("Predict Tin Price for a Specific Date")
user_input = st.text_input("Enter the date for which you want to predict the price (YYYY-MM-DD): ")
if user_input:
    try:
        predicted_price = predict_price_for_date(user_input)
        st.write(f"The predicted price of tin on {user_input} is: {predicted_price}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

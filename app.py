import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model # type: ignore
import streamlit as st

# Define the date range and stock ticker
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

st.title('Stock Trend Prediction')

# User input for stock ticker
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Fetch data from Yahoo Finance
try:
    df = yf.download(user_input, start=start_date, end=end_date)
    if df.empty:
        st.error("No data found for the given stock ticker.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Reset index and drop unnecessary columns
df = df.reset_index()

# Descriptive statistics
st.subheader('Data from 2010 - 2023')
st.write(df.describe())

# Visualization 1: Closing Price vs Time Chart
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

# Visualization 2: Closing Price vs Time Chart with 100 Moving Average (100MA)
st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()  # Add () to calculate the rolling mean
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close, label='Closing Price')
plt.plot(ma100, label='100MA', color='orange')
plt.legend()
st.pyplot(fig)

# Visualization 3: Closing Price vs Time Chart with 100MA & 200MA
st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()  # Add () to calculate the rolling mean
ma200 = df.Close.rolling(200).mean()  # Add () to calculate the rolling mean
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close, label='Closing Price')
plt.plot(ma100, label='100MA', color='orange')
plt.plot(ma200, label='200MA', color='red')
plt.legend()
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)


import os
from keras.models import load_model

# Resolve the model path
model_path = os.path.join(os.path.dirname(__file__), 'keras_model.keras')
print("Resolved Model Path:", model_path)

# Check if the file exists
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")
else:
    print("Model file found. Loading the model...")

# Load the model
try:
    model = load_model(model_path, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise


#testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scaler_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor

#final graph

st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, '#0ea7b5', label='Actual Price')
plt.plot(y_predicted, '#e8702a', label='Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

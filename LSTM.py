# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Section 1: Fetching Stock Data
start = '2010-01-01'
end = '2023-12-31'

print("Fetching stock data...")
df = yf.download('AAPL', start=start, end=end)

# Display first and last rows of the dataset
print("First few rows of data:")
print(df.head())

print("Last few rows of data:")
print(df.tail())

# Reset index and drop unnecessary columns
df = df.reset_index()
df = df.drop(['Date', 'Adj Close'], axis=1)
print("Processed data (after resetting index and dropping columns):")
print(df.head())

# Plot the closing price
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price')
plt.title('AAPL Closing Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Section 2: Prepare Data for Training and Testing
print("Splitting data into training and testing sets...")
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

print("Training data shape:", data_training.shape)
print("Testing data shape:", data_testing.shape)

# Scale the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Display the scaled training data
print("Scaled training data array:")
print(data_training_array[:5])

# Section 3: Prepare Data for LSTM Model
x_train = []
y_train = []

print("Preparing data for LSTM model...")
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
print("Shape of x_train:", x_train.shape)
print("Shape of y_train:", y_train.shape)

# Section 4: Build the LSTM Model
print("Building the LSTM model...")
model = Sequential()

model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.summary()

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
print("Training the LSTM model...")
model.fit(x_train, y_train, epochs=50, batch_size=32)

# Save the model
model.save('keras_model.keras')
print("Model saved as 'keras_model.keras'.")

# Section 5: Testing the Model
print("Preparing testing data...")
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
print("Shape of x_test:", x_test.shape)
print("Shape of y_test:", y_test.shape)

# Make Predictions
print("Making predictions...")
y_predicted = model.predict(x_test)
scaler_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor

# Section 6: Plot the Results
plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Actual Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
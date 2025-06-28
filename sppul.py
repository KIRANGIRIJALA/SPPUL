# !pip install yfinance pandas matplotlib scikit-learn tensorflow

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Step 1: Download data
stock_symbol = 'AAPL'  # Try 'TSLA', 'RELIANCE.NS', 'INFY.NS' for India
start_date = '2010-01-01'
end_date = '2023-12-31'

print(f"📦 Fetching data for {stock_symbol} from {start_date} to {end_date}")
df = yf.download(stock_symbol, start=start_date, end=end_date)

# Step 2: Validate download
if df.empty:
    raise ValueError("❌ No data fetched. Please check the stock symbol and internet connection.")

if 'Close' not in df.columns:
    raise KeyError("❌ 'Close' column not found in the dataset!")


# Grab the Close column (as Series)
data = df['Close']

# Check if it's completely missing
if data.empty:
    raise ValueError("❌ 'Close' column is empty.")

# Drop NaNs
data = data.dropna()

# Check if anything remains
if data.empty:
    raise ValueError("❌ All values in 'Close' are NaN.")



# Step 3: Prepare and scale data
dataset = data.values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Step 4: Training data
training_data_len = int(len(dataset) * 0.8)
train_data = scaled_data[:training_data_len]

X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Step 5: Build model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
model.add(LSTM(50))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=10)

# Step 6: Test data
test_data = scaled_data[training_data_len - 60:]
X_test = []

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 7: Predict and inverse scale
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Step 8: Plotting
train = data[:training_data_len]
valid = data[training_data_len:].copy()
valid['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.plot(train, label='Training Data') # Changed 'train['Close']' to 'train'
plt.plot(valid['AAPL'], label='Actual Price') # Changed 'valid['Close']' to 'valid['AAPL']'
plt.plot(valid['Predictions'], label='Predicted Price')
plt.legend()
plt.grid(True)
plt.show()

# Step 9: Evaluate
rmse = np.sqrt(mean_squared_error(valid['AAPL'], valid['Predictions'])) # Changed 'valid['Close']' to 'valid['AAPL']'
print(f"✅ RMSE: {rmse:.2f}")

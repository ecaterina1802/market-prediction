# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset (Assuming you have a CSV file with stock market data)
# Make sure the CSV file is in the same directory as main.py
df = pd.read_csv('your_dataset.csv')

# Check the first few rows of the dataset and the data size
print(df.head())
print(f"Data size: {df.shape}")

# Convert the 'Date' column to datetime format (if it's not already)
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the index for proper plotting
df.set_index('Date', inplace=True)

# Preview the first few rows of the dataset
print(df.head())

# Assuming the dataset has columns 'Date' and 'Close' for stock prices
# We will predict the 'Close' price based on the previous day's 'Close' price
df['Prev_Close'] = df['Close'].shift(1)

# Drop rows with missing 'Prev_Close' values
df = df.dropna()

# Features (Previous day's close price)
X = df[['Prev_Close']]

# Target (Today's close price)
y = df['Close']

# Split the data into training and test sets (Ensuring we don't shuffle time-series data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Verify the sizes of the training and test sets
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Initialize and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print the first few predicted and actual values
print("Predicted values:", y_pred[:10])  # Print first 10 predicted values
print("Actual values:", y_test.head())  # Print first 5 actual values

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the results
plt.figure(figsize=(10, 6))

# Plotting the actual prices vs predicted prices
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Prices', color='red')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction (Previous Close vs. Current Close)')

# Rotate the date labels for better readability
plt.xticks(rotation=45)

# Show legend
plt.legend()

# Display the plot
plt.show()

# Example of making a prediction for the next day based on the most recent 'Prev_Close'
latest_prev_close = df['Close'].iloc[-1]  # Most recent 'Close' value
predicted_next_close = model.predict([[latest_prev_close]])
print(f"Predicted next day's close price: {predicted_next_close[0]}")

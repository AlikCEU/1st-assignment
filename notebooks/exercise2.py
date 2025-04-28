# Exercise 2: Model Fitting and Forecast Evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import os

# === Setup ===
# Set visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("muted")
plt.rcParams.update({
    'figure.figsize': (14, 6), 
    'figure.dpi': 120,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black'
})

# Create output directories
figures_dir = os.path.join(os.getcwd(), 'output', 'figures', '2 exercise')
os.makedirs(figures_dir, exist_ok=True)

# === Data Loading ===
# Load the gold price data - building on Exercise 1 results
data = pd.read_csv('data/Gold_hourly.csv')
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%Y.%m.%d %H:%M')
data.set_index('Datetime', inplace=True)

# Process data as in Exercise 1
data['High_filled'] = data['High'].fillna(method='ffill')
weekly_max_prices = data['High_filled'].resample('W').max()

# === Task (a): Split Weekly Max Prices into Training/Test Sets ===
train_size = int(len(weekly_max_prices) * 0.8)
train_weekly = weekly_max_prices[:train_size]
test_weekly = weekly_max_prices[train_size:]

print(f"Weekly price data split: Training set size = {len(train_weekly)}, Test set size = {len(test_weekly)}")

# === Task (b): Apply Drift Method for Forecasting ===
# Calculate the average change in the training set
first_value = train_weekly.iloc[0]
last_value = train_weekly.iloc[-1]
average_change = (last_value - first_value) / (len(train_weekly) - 1)

# Generate drift forecasts
drift_forecasts = []
for h in range(1, len(test_weekly) + 1):
    forecast = last_value + h * average_change
    drift_forecasts.append(forecast)

# Create DataFrame for drift forecasts
drift_forecasts_df = pd.DataFrame(drift_forecasts, index=test_weekly.index, columns=['forecast'])

# === Task (c): Calculate MSFE for Drift Method ===
mse_drift = mean_squared_error(test_weekly, drift_forecasts_df['forecast'])
print(f"\nMean Squared Forecast Error - Drift Method: {mse_drift:.2f}")

# Visualize the drift forecasts against actual values
plt.figure(figsize=(12, 6))
plt.plot(train_weekly.index, train_weekly, label='Training Data', color='#4C72B0')
plt.plot(test_weekly.index, test_weekly, label='Test Data', color='black')
plt.plot(drift_forecasts_df.index, drift_forecasts_df['forecast'], 
         label='Drift Forecast', color='#C44E52', linestyle='--')
plt.title('Exercise 2(b-c): Weekly Maximum Gold Price - Actual vs. Drift Forecast')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)

# Add exercise label
plt.text(0.02, 0.98, 'Exercise 2(b-c)', fontsize=14, weight='bold',
         transform=plt.gca().transAxes, ha='left', va='top',
         bbox=dict(facecolor='white', alpha=0.7))

plt.savefig(os.path.join(figures_dir, 'drift_forecast_gold.png'))
plt.close()

# === Task (d): One-Step-Ahead Forecasts for Daily Volume ===
# Get the daily trading volume without zeros
daily_volume = data['Volume'].resample('D').sum()
daily_volume = daily_volume[daily_volume > 0]

# Split into training (80%) and test (20%) sets
train_size_volume = int(len(daily_volume) * 0.8)
train_volume = daily_volume[:train_size_volume]
test_volume = daily_volume[train_size_volume:]

print(f"\nDaily volume data split: Training set size = {len(train_volume)}, Test set size = {len(test_volume)}")

# Average method - use mean of all training observations
average_forecast = np.mean(train_volume)
average_forecasts = np.repeat(average_forecast, len(test_volume))

# Naïve method - use the last observed value
naive_forecasts = []
for i in range(len(test_volume)):
    # For one-step-ahead forecasts, we use the actual previous value
    if i == 0:
        naive_forecasts.append(train_volume.iloc[-1])
    else:
        naive_forecasts.append(test_volume.iloc[i-1])

# === Task (e): Calculate and Compare MSFE for Both Methods ===
mse_average = mean_squared_error(test_volume, average_forecasts)
mse_naive = mean_squared_error(test_volume, naive_forecasts)

print(f"\nMean Squared Error - Average Method: {mse_average:.2f}")
print(f"Mean Squared Error - Naïve Method: {mse_naive:.2f}")

# Improved visualization of forecasts
plt.figure(figsize=(14, 10))  # Increased figure size

# Main plot with actual data
plt.subplot(3, 1, (1, 2))  # Use 2/3 of the space for main plot
plt.plot(test_volume.index, test_volume, label='Actual', color='black', linewidth=1.5)
plt.plot(test_volume.index, average_forecasts, 
         label='Average Forecast', color='#4C72B0', linestyle='--', linewidth=1, alpha=0.9)
plt.plot(test_volume.index, naive_forecasts, 
         label='Naïve Forecast', color='#C44E52', linestyle='-.', linewidth=1, alpha=0.9)
plt.title('Exercise 2(d-e): Daily Gold Trading Volume - Actual vs. Forecasts', fontsize=14)
plt.ylabel('Volume', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='upper right')

# Add subplot for forecast errors
plt.subplot(3, 1, 3)  # Bottom third for error plot
avg_error = test_volume.values - average_forecasts
naive_error = test_volume.values - naive_forecasts
plt.plot(test_volume.index, avg_error, label='Average Error', color='#4C72B0', alpha=0.8)
plt.plot(test_volume.index, naive_error, label='Naïve Error', color='#C44E52', alpha=0.8)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.fill_between(test_volume.index, 0, avg_error, color='#4C72B0', alpha=0.3)
plt.fill_between(test_volume.index, 0, naive_error, color='#C44E52', alpha=0.3)
plt.title('Forecast Errors', fontsize=12)
plt.ylabel('Error (Actual - Forecast)', fontsize=10)
plt.xlabel('Date', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='upper right')

# Add exercise label
plt.gcf().text(0.02, 0.98, 'Exercise 2(d-e)', fontsize=14, weight='bold',
              ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'volume_forecasts_comparison.png'))
plt.close()

# Print comparison results
better_method = 'Average' if mse_average < mse_naive else 'Naïve'
improvement = abs(mse_average - mse_naive)
improvement_percent = abs(mse_average - mse_naive)/max(mse_average, mse_naive)*100

print(f"\nComparison: The {better_method} method performed better for daily trading volume forecasting.")
print(f"Improvement: {improvement:.2f} ({improvement_percent:.2f}%)")

print(f"\nExercise 2 complete. Outputs saved to {figures_dir}")
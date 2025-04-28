# Exercise 4: Model Fitting and Forecast Evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import os

# === Setup ===
# Set visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("muted")
plt.rcParams.update({
    'figure.figsize': (12, 6), 
    'figure.dpi': 120,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black'
})

# Create output directories
figures_dir = os.path.join(os.getcwd(), 'output', 'figures', '4 exercise')
os.makedirs(figures_dir, exist_ok=True)

# === Data Loading and Preprocessing ===
# Load the energy consumption data - Using the results from Exercise 3
data = pd.read_csv('data/energyconsumption_hourly.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace=True)

# Fill missing values using previous day's same hour
data_filled = data.copy()
for idx in data_filled[data_filled['energyconsumption_hourly'].isna()].index:
    prev_day_same_hour = idx - pd.Timedelta(days=1)
    if prev_day_same_hour in data_filled.index:
        data_filled.loc[idx, 'energyconsumption_hourly'] = data_filled.loc[prev_day_same_hour, 'energyconsumption_hourly']
    else:
        data_filled.loc[idx, 'energyconsumption_hourly'] = data_filled['energyconsumption_hourly'].fillna(method='ffill').loc[idx]

# Resample to daily and monthly
daily_consumption = data_filled.resample('D').sum()
# Using 'ME' instead of 'M' to avoid the deprecation warning
monthly_consumption = data_filled.resample('ME').sum()

# === Task (a): Split into Training and Test Sets ===
# For daily data
train_size_daily = int(len(daily_consumption) * 0.8)
train_daily = daily_consumption.iloc[:train_size_daily]
test_daily = daily_consumption.iloc[train_size_daily:]

print(f"Daily data split: Training set size = {len(train_daily)}, Test set size = {len(test_daily)}")

# For monthly data
train_size_monthly = int(len(monthly_consumption) * 0.8)
train_monthly = monthly_consumption.iloc[:train_size_monthly]
test_monthly = monthly_consumption.iloc[train_size_monthly:]

print(f"Monthly data split: Training set size = {len(train_monthly)}, Test set size = {len(test_monthly)}")

# === Task (b): Fit Decomposition Models to Daily Data ===
# For additive decomposition - handle NaN values
# Ensure the data has no NaN values
decomposition_additive = seasonal_decompose(train_daily.dropna(), model='additive', period=7)

# Plot additive decomposition
plt.figure(figsize=(14, 10))
plt.subplot(411)
plt.plot(decomposition_additive.observed)
plt.title('Exercise 4(b): Additive Decomposition - Observed')
plt.subplot(412)
plt.plot(decomposition_additive.trend)
plt.title('Exercise 4(b): Additive Decomposition - Trend')
plt.subplot(413)
plt.plot(decomposition_additive.seasonal)
plt.title('Exercise 4(b): Additive Decomposition - Seasonality')
plt.subplot(414)
plt.plot(decomposition_additive.resid)
plt.title('Exercise 4(b): Additive Decomposition - Residuals')

# Add exercise label
plt.gcf().text(0.02, 0.98, 'Exercise 4(b) - Additive', fontsize=14, weight='bold',
              ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'additive_decomposition.png'))
plt.close()

# For multiplicative decomposition - handle NaN values and non-positive values
# Ensure data has no NaN or non-positive values for multiplicative model
train_daily_mult = train_daily.copy()
train_daily_mult[train_daily_mult <= 0] = 0.1  # Replace zeros or negative values
decomposition_multiplicative = seasonal_decompose(train_daily_mult.dropna(), model='multiplicative', period=7)

# Plot multiplicative decomposition
plt.figure(figsize=(14, 10))
plt.subplot(411)
plt.plot(decomposition_multiplicative.observed)
plt.title('Exercise 4(b): Multiplicative Decomposition - Observed')
plt.subplot(412)
plt.plot(decomposition_multiplicative.trend)
plt.title('Exercise 4(b): Multiplicative Decomposition - Trend')
plt.subplot(413)
plt.plot(decomposition_multiplicative.seasonal)
plt.title('Exercise 4(b): Multiplicative Decomposition - Seasonality')
plt.subplot(414)
plt.plot(decomposition_multiplicative.resid)
plt.title('Exercise 4(b): Multiplicative Decomposition - Residuals')

# Add exercise label
plt.gcf().text(0.02, 0.98, 'Exercise 4(b) - Multiplicative', fontsize=14, weight='bold',
              ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'multiplicative_decomposition.png'))
plt.close()

# Generate forecasts for the test period using the decomposition results
# For additive model - handle NaN values in trend
additive_trend = decomposition_additive.trend.dropna().iloc[-1]  # Get last non-NaN trend value
additive_seasonal_cycle = decomposition_additive.seasonal.iloc[-7:].values  # Last week

additive_forecasts = []
for i in range(len(test_daily)):
    season_idx = i % 7
    forecast = additive_trend + additive_seasonal_cycle[season_idx]
    additive_forecasts.append(forecast)

# For multiplicative model - handle NaN values in trend
multiplicative_trend = decomposition_multiplicative.trend.dropna().iloc[-1]  # Get last non-NaN trend value
multiplicative_seasonal_cycle = decomposition_multiplicative.seasonal.iloc[-7:].values  # Last week

multiplicative_forecasts = []
for i in range(len(test_daily)):
    season_idx = i % 7
    forecast = multiplicative_trend * multiplicative_seasonal_cycle[season_idx]
    multiplicative_forecasts.append(forecast)

# Create forecast DataFrames with the same index as test_daily
additive_forecasts_df = pd.DataFrame(additive_forecasts, index=test_daily.index, columns=['forecast'])
multiplicative_forecasts_df = pd.DataFrame(multiplicative_forecasts, index=test_daily.index, columns=['forecast'])

# === Task (c): Calculate MSFE and Compare Performance ===
# For additive model - ensure no NaN values
mse_additive = mean_squared_error(test_daily['energyconsumption_hourly'].values, 
                                 additive_forecasts_df['forecast'].values)
print(f"\nMean Squared Error - Additive Decomposition: {mse_additive:.2f}")

# For multiplicative model - ensure no NaN values
mse_multiplicative = mean_squared_error(test_daily['energyconsumption_hourly'].values, 
                                       multiplicative_forecasts_df['forecast'].values)
print(f"Mean Squared Error - Multiplicative Decomposition: {mse_multiplicative:.2f}")

# Improved visualization of forecasts
plt.figure(figsize=(14, 10))  # Increased figure size

# Main plot with actual data
plt.subplot(3, 1, (1, 2))  # Use 2/3 of the space for main plot
plt.plot(test_daily.index, test_daily['energyconsumption_hourly'], 
         label='Actual', color='black', linewidth=1.5)
plt.plot(additive_forecasts_df.index, additive_forecasts_df['forecast'], 
         label='Additive Forecast', color='#4C72B0', linestyle='--', linewidth=1.2, alpha=0.8)
plt.plot(multiplicative_forecasts_df.index, multiplicative_forecasts_df['forecast'], 
         label='Multiplicative Forecast', color='#C44E52', linestyle='-.', linewidth=1.2, alpha=0.8)
plt.title('Exercise 4(c): Daily Energy Consumption - Actual vs. Forecasts', fontsize=14)
plt.ylabel('Energy Consumption (kWh)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='upper right')

# Add subplot for forecast errors
plt.subplot(3, 1, 3)  # Bottom third for error plot
add_error = test_daily['energyconsumption_hourly'].values - additive_forecasts_df['forecast'].values
mult_error = test_daily['energyconsumption_hourly'].values - multiplicative_forecasts_df['forecast'].values
plt.plot(test_daily.index, add_error, label='Additive Error', color='#4C72B0', alpha=0.8)
plt.plot(test_daily.index, mult_error, label='Multiplicative Error', color='#C44E52', alpha=0.8)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.fill_between(test_daily.index, 0, add_error, color='#4C72B0', alpha=0.3)
plt.fill_between(test_daily.index, 0, mult_error, color='#C44E52', alpha=0.3)
plt.title('Forecast Errors', fontsize=12)
plt.ylabel('Error (Actual - Forecast)', fontsize=10)
plt.xlabel('Date', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='upper right')

# Add exercise label
plt.gcf().text(0.02, 0.98, 'Exercise 4(c)', fontsize=14, weight='bold',
              ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'daily_forecasts_comparison.png'))
plt.close()

# === Task (d): Apply Holt-Winters Method to Monthly Data ===
# Determine seasonality period for monthly data (likely 12 for yearly seasonality)
seasonality_period = 12

# Ensure data has enough seasonal periods
if len(train_monthly) >= seasonality_period * 2:  # Need at least 2 full seasons
    # Fit Holt-Winters with multiplicative seasonality
    hw_multiplicative = ExponentialSmoothing(
        train_monthly['energyconsumption_hourly'],
        seasonal_periods=seasonality_period,
        trend='add',
        seasonal='mul',
        initialization_method="estimated"
    ).fit()

    # Forecast for the test period
    hw_multiplicative_forecast = hw_multiplicative.forecast(steps=len(test_monthly))

    # Fit Holt-Winters with additive seasonality
    hw_additive = ExponentialSmoothing(
        train_monthly['energyconsumption_hourly'],
        seasonal_periods=seasonality_period,
        trend='add',
        seasonal='add',
        initialization_method="estimated"
    ).fit()

    # Forecast for the test period
    hw_additive_forecast = hw_additive.forecast(steps=len(test_monthly))

    # Calculate mean squared errors for both methods
    mse_hw_multiplicative = mean_squared_error(test_monthly['energyconsumption_hourly'], hw_multiplicative_forecast)
    mse_hw_additive = mean_squared_error(test_monthly['energyconsumption_hourly'], hw_additive_forecast)

    print(f"\nMean Squared Error - Holt-Winters Multiplicative: {mse_hw_multiplicative:.2f}")
    print(f"Mean Squared Error - Holt-Winters Additive: {mse_hw_additive:.2f}")

    # Create a DataFrame for easier plotting
    forecast_comparison = pd.DataFrame({
        'Actual': test_monthly['energyconsumption_hourly'],
        'HW Multiplicative': hw_multiplicative_forecast.values,
        'HW Additive': hw_additive_forecast.values
    })

    # Plot the forecasts against actual values
    plt.figure(figsize=(14, 7))
    plt.plot(test_monthly.index, forecast_comparison['Actual'], label='Actual', color='black')
    plt.plot(test_monthly.index, forecast_comparison['HW Multiplicative'], 
             label='HW Multiplicative', color='#4C72B0', linestyle='--')
    plt.plot(test_monthly.index, forecast_comparison['HW Additive'], 
             label='HW Additive', color='#C44E52', linestyle='-.')
    plt.title('Exercise 4(d): Monthly Energy Consumption - Actual vs. Holt-Winters Forecasts')
    plt.xlabel('Date')
    plt.ylabel('Energy Consumption (kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add exercise label
    plt.text(0.02, 0.98, 'Exercise 4(d)', fontsize=14, weight='bold',
             transform=plt.gca().transAxes, ha='left', va='top',
             bbox=dict(facecolor='white', alpha=0.7))

    plt.savefig(os.path.join(figures_dir, 'monthly_hw_forecasts.png'))
    plt.close()

    # === Task (e): Discussion of Holt-Winters performance ===
    # Print which model performed better
    better_hw_model = 'Multiplicative' if mse_hw_multiplicative < mse_hw_additive else 'Additive'
    print(f"\nHolt-Winters Comparison: The {better_hw_model} approach performs better for monthly energy consumption forecasting.")
else:
    print("Not enough monthly data for Holt-Winters with seasonal period of 12. Need at least 24 months.")
    # Fall back to simple Holt (trend-only) model without seasonality
    holt_model = ExponentialSmoothing(
        train_monthly['energyconsumption_hourly'],
        trend='add',
        seasonal=None
    ).fit()
    
    holt_forecast = holt_model.forecast(steps=len(test_monthly))
    mse_holt = mean_squared_error(test_monthly['energyconsumption_hourly'], holt_forecast)
    print(f"Mean Squared Error - Holt (no seasonality): {mse_holt:.2f}")
    
    # Plot the forecasts against actual values
    plt.figure(figsize=(14, 7))
    plt.plot(test_monthly.index, test_monthly['energyconsumption_hourly'], label='Actual', color='black')
    plt.plot(test_monthly.index, holt_forecast, 
             label='Holt Forecast (no seasonality)', color='#4C72B0', linestyle='--')
    plt.title('Exercise 4(d): Monthly Energy Consumption - Actual vs. Holt Forecast')
    plt.xlabel('Date')
    plt.ylabel('Energy Consumption (kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figures_dir, 'monthly_holt_forecast.png'))
    plt.close()

print(f"\nExercise 4 complete. Outputs saved to {figures_dir}")
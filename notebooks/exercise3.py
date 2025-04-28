# Exercise 3: Data Preparation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
figures_dir = os.path.join(os.getcwd(), 'output', 'figures', '3 exercise')
os.makedirs(figures_dir, exist_ok=True)

# === Data Loading ===
# Load the energy consumption data
data = pd.read_csv('data/energyconsumption_hourly.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace=True)

# Display basic information about the data
print("First few rows of the original data:")
print(data.head())
print("\nMissing values before filling:")
print(data.isnull().sum())

# === Task (a): Fill Missing Values ===
# Create a copy to avoid changing the original data
data_filled = data.copy()

# For each missing value, find the value from the same hour the day before
for idx in data_filled[data_filled['energyconsumption_hourly'].isna()].index:
    # Calculate the datetime 24 hours before
    prev_day_same_hour = idx - pd.Timedelta(days=1)
    
    # If the previous day's value exists, use it
    if prev_day_same_hour in data_filled.index:
        data_filled.loc[idx, 'energyconsumption_hourly'] = data_filled.loc[prev_day_same_hour, 'energyconsumption_hourly']
    else:
        # If not, use the next available value (forward fill)
        data_filled.loc[idx, 'energyconsumption_hourly'] = data_filled['energyconsumption_hourly'].fillna(method='ffill').loc[idx]

# Check if all missing values were filled
print("\nMissing values after filling:")
print(data_filled.isnull().sum())

# === Task (b): Resample to Daily and Monthly Frequency ===
# Daily consumption - sum of hourly values for each day
daily_consumption = data_filled.resample('D').sum()
print("\nFirst few rows of daily consumption:")
print(daily_consumption.head())

# Monthly consumption - sum of hourly values for each month
monthly_consumption = data_filled.resample('M').sum()
print("\nFirst few rows of monthly consumption:")
print(monthly_consumption.head())

# === Task (c): Plot Time Series and Analyze Characteristics ===
plt.figure(figsize=(14, 10))

# Daily consumption plot
plt.subplot(2, 1, 1)
plt.plot(daily_consumption, color='#4C72B0', linewidth=1)
plt.title('Exercise 3(c): Daily Energy Consumption')
plt.ylabel('Energy (kWh)')
plt.grid(True, alpha=0.3)

# Monthly consumption plot
plt.subplot(2, 1, 2)
plt.plot(monthly_consumption, color='#55A868', linewidth=2, marker='o')
plt.title('Exercise 3(c): Monthly Energy Consumption')
plt.ylabel('Energy (kWh)')
plt.grid(True, alpha=0.3)

# Add exercise label
plt.gcf().text(0.02, 0.98, 'Exercise 3(c)', fontsize=14, weight='bold',
              ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'energy_consumption_plots.png'))
plt.close()

# === Additional Analysis: Hourly and Monthly Patterns ===
# Calculate and plot the average consumption by hour of day
hourly_pattern = data_filled.groupby(data_filled.index.hour).mean()
plt.figure(figsize=(12, 6))
plt.bar(hourly_pattern.index, hourly_pattern['energyconsumption_hourly'], color='#4C72B0')
plt.title('Exercise 3: Average Energy Consumption by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Energy (kWh)')
plt.xticks(range(0, 24))
plt.grid(axis='y', alpha=0.3)

# Add exercise label
plt.text(0.02, 0.98, 'Exercise 3 - Hourly Pattern', fontsize=14, weight='bold',
         transform=plt.gca().transAxes, ha='left', va='top',
         bbox=dict(facecolor='white', alpha=0.7))

plt.savefig(os.path.join(figures_dir, 'hourly_pattern.png'))
plt.close()

# Calculate and plot the average consumption by month
monthly_pattern = data_filled.groupby(data_filled.index.month).mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.figure(figsize=(12, 6))
plt.bar(months, monthly_pattern['energyconsumption_hourly'], color='#55A868')
plt.title('Exercise 3: Average Energy Consumption by Month')
plt.xlabel('Month')
plt.ylabel('Average Energy (kWh)')
plt.grid(axis='y', alpha=0.3)

# Add exercise label
plt.text(0.02, 0.98, 'Exercise 3 - Monthly Pattern', fontsize=14, weight='bold',
         transform=plt.gca().transAxes, ha='left', va='top',
         bbox=dict(facecolor='white', alpha=0.7))

plt.savefig(os.path.join(figures_dir, 'monthly_pattern.png'))
plt.close()

print(f"\nExercise 3 complete. Outputs saved to {figures_dir}")
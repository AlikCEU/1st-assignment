# Exercise 1: Data Preparation

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
figures_dir = os.path.join(os.getcwd(), 'output', 'figures', '1 exercise')
os.makedirs(figures_dir, exist_ok=True)

# === Data Loading ===
data = pd.read_csv('data/Gold_hourly.csv')
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%Y.%m.%d %H:%M')
data.set_index('Datetime', inplace=True)

# === Task (a): Daily Trading Volume ===
daily_volume = data['Volume'].resample('D').sum()
daily_volume = daily_volume[daily_volume > 0]

# === Task (b): Plot Daily Volume with ACF/PACF ===
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot time series
axes[0].plot(daily_volume, color='#4C72B0')
axes[0].set_title('Exercise 1(b): Daily Gold Trading Volume')
axes[0].grid(True, alpha=0.3)

# Plot ACF
plot_acf(daily_volume, lags=30, ax=axes[1])
axes[1].set_title('Exercise 1(b): ACF of Daily Volume')
axes[1].grid(True, alpha=0.3)

# Plot PACF
plot_pacf(daily_volume, lags=30, ax=axes[2])
axes[2].set_title('Exercise 1(b): PACF of Daily Volume')
axes[2].grid(True, alpha=0.3)

fig.text(0.02, 0.98, 'Exercise 1(b)', fontsize=14, weight='bold', 
         ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'daily_volume_analysis.png'))
plt.close()

# === Task (c): Fill Missing Values in High Prices ===
data['High_filled'] = data['High'].ffill()

# === Task (d): Weekly Max Prices ===
weekly_max_prices = data['High_filled'].resample('W').max()

# === Task (e): Plot Weekly Max Prices with ACF/PACF ===
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot time series
axes[0].plot(weekly_max_prices, color='#C44E52')
axes[0].set_title('Exercise 1(e): Weekly Maximum Gold Prices')
axes[0].grid(True, alpha=0.3)

# Plot ACF
plot_acf(weekly_max_prices, lags=30, ax=axes[1])
axes[1].set_title('Exercise 1(e): ACF of Weekly Max Prices')
axes[1].grid(True, alpha=0.3)

# Plot PACF
plot_pacf(weekly_max_prices, lags=30, ax=axes[2])
axes[2].set_title('Exercise 1(e): PACF of Weekly Max Prices')
axes[2].grid(True, alpha=0.3)

fig.text(0.02, 0.98, 'Exercise 1(e)', fontsize=14, weight='bold', 
         ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'weekly_max_price_analysis.png'))
plt.close()

print(f"Exercise 1 complete. Outputs saved to {figures_dir}")
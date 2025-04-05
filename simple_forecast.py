import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Paths
processed_data_path = os.path.join('data', 'processed')
processed_file = os.path.join(processed_data_path, 'processed_sales.csv')  # Use processed data instead of engineered
forecast_file = os.path.join(processed_data_path, 'sales_forecasts.csv')

# Load processed data
print(f"Loading processed data from {processed_file}")
df = pd.read_csv(processed_file, nrows=100000)  # Load first 100,000 rows

# Create a simple forecast
print("Creating simple forecast...")

# Calculate average daily sales
avg_daily_sales = df['Purchase'].mean()  # Use 'Purchase' column instead of 'sales'
print(f"Average daily sales: {avg_daily_sales:.2f}")

# Calculate sales statistics for better forecasting
daily_std = df['Purchase'].std()
min_sales = df['Purchase'].min()
max_sales = df['Purchase'].max()
print(f"Sales statistics:")
print(f"Standard deviation: {daily_std:.2f}")
print(f"Minimum sales: {min_sales:.2f}")
print(f"Maximum sales: {max_sales:.2f}")

# Create a date range for the forecast (next 30 days)
last_date = datetime(2020, 1, 1) + timedelta(days=len(df)-1)
forecast_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(30)]

# Create a simple forecast with some randomness
np.random.seed(42)  # For reproducibility
# Generate random values between -2 and 2 standard deviations
random_factors = np.random.normal(0, 0.5, 30)  # Using 0.5 std for less variation
forecast_values = avg_daily_sales * (1 + random_factors)

# Ensure forecast values are within reasonable bounds
forecast_values = np.clip(forecast_values, min_sales, max_sales)

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'date': forecast_dates,
    'forecast': forecast_values
})

# Save forecast
print(f"Saving forecast to {forecast_file}")
forecast_df.to_csv(forecast_file, index=False)

print("Forecast generated successfully!")
print("\nSample of forecast:")
print(forecast_df.head()) 
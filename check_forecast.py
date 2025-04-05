import pandas as pd
import os
import time

# Path to the forecast CSV file
forecast_path = os.path.join('data', 'processed', 'sales_forecasts.csv')

# Check if file exists
if not os.path.exists(forecast_path):
    print(f"Forecast file not found at {forecast_path}")
    print("Waiting for the forecasting model to complete...")
    
    # Wait for the file to be created (check every 5 seconds)
    max_wait_time = 300  # 5 minutes
    wait_time = 0
    while not os.path.exists(forecast_path) and wait_time < max_wait_time:
        time.sleep(5)
        wait_time += 5
        print(f"Waiting... ({wait_time} seconds)")
    
    if not os.path.exists(forecast_path):
        print("Forecast file not found after waiting. The forecasting model might still be running.")
        exit(1)

# Try to read the forecast file
try:
    # Read the forecast data
    forecast_df = pd.read_csv(forecast_path)
    
    # Print forecast data
    print("Forecast data:")
    print(forecast_df)
    
    # Print summary statistics
    print("\nSummary statistics:")
    print(forecast_df.describe())
    
except Exception as e:
    print(f"Error reading forecast file: {e}") 
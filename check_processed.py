import pandas as pd
import os

# Path to the processed CSV file
csv_path = os.path.join('data', 'processed', 'processed_sales.csv')

# Check if file exists
if not os.path.exists(csv_path):
    print(f"Error: File not found at {csv_path}")
    exit(1)

# Try to read the CSV file
try:
    # Read just the first few rows to examine structure
    df = pd.read_csv(csv_path, nrows=5)
    
    # Print column names
    print("Columns in the processed CSV file:")
    for col in df.columns:
        print(f"- {col}")
    
    # Print first few rows
    print("\nFirst 5 rows of processed data:")
    print(df)
    
except Exception as e:
    print(f"Error reading CSV file: {e}") 
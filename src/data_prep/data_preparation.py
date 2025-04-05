import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataPreparation:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        
    def load_data(self, file_name, nrows=None):
        """Load data from CSV file"""
        file_path = os.path.join(self.raw_data_path, file_name)
        print(f"Attempting to load file from: {file_path}")
        return pd.read_csv(file_path, nrows=nrows)
    
    def clean_data(self, df):
        """Clean the dataset"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna(method='ffill')  # Forward fill
        df = df.fillna(method='bfill')  # Backward fill for any remaining NaNs
        
        return df
    
    def create_time_features(self, df):
        """Create time-based features for the dataset"""
        # Instead of creating a full date range, we'll use the index as a time proxy
        # and create cyclical features based on position
        
        # Create a time index (0 to len-1)
        time_index = np.arange(len(df))
        
        # Create cyclical time features
        # Assuming data is daily, create day of week (0-6)
        df['day_of_week'] = time_index % 7
        
        # Create month (1-12) - assuming roughly 30 days per month
        df['month'] = (time_index // 30) % 12 + 1
        
        # Create year (starting from 2020)
        df['year'] = 2020 + (time_index // 365)
        
        # Create quarter (1-4)
        df['quarter'] = ((df['month'] - 1) // 3) + 1
        
        # Create lag features for Purchase (which represents sales)
        df['sales'] = df['Purchase']  # Rename Purchase to sales for consistency
        df['sales_lag1'] = df['sales'].shift(1)
        df['sales_lag7'] = df['sales'].shift(7)
        df['sales_lag30'] = df['sales'].shift(30)
        
        # Create rolling means
        df['sales_rolling_mean_7'] = df['sales'].rolling(window=7).mean()
        df['sales_rolling_mean_30'] = df['sales'].rolling(window=30).mean()
        
        return df
    
    def process_data(self, input_file, output_file, max_rows=None):
        """Main data processing pipeline"""
        # Load data
        df = self.load_data(input_file, nrows=max_rows)
        print(f"Loaded data with {len(df)} rows")
        
        # Clean data
        df = self.clean_data(df)
        print(f"After cleaning: {len(df)} rows")
        
        # Create time features
        df = self.create_time_features(df)
        
        # Save processed data
        output_path = os.path.join(self.processed_data_path, output_file)
        df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
        
        return df

if __name__ == "__main__":
    # Get the absolute path to the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    
    # Create absolute paths for data directories
    raw_data_path = os.path.join(project_root, "data", "raw")
    processed_data_path = os.path.join(project_root, "data", "processed")
    
    print(f"Raw data path: {raw_data_path}")
    print(f"Processed data path: {processed_data_path}")
    
    # Process a subset of the data to avoid memory issues
    # You can adjust this number based on your system's capabilities
    max_rows = 100000  # Process first 100,000 rows
    
    data_prep = DataPreparation(raw_data_path, processed_data_path)
    processed_df = data_prep.process_data("walmart.csv", "processed_sales.csv", max_rows=max_rows)
    
    print("Data processing completed successfully!")
    print(f"Processed data shape: {processed_df.shape}")
    print("\nSample of processed data:")
    print(processed_df.head()) 
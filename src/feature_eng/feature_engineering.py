import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class FeatureEngineering:
    def __init__(self, processed_data_path):
        self.processed_data_path = processed_data_path
        self.scaler = StandardScaler()
        
    def load_processed_data(self, file_name, nrows=None):
        """Load processed data"""
        file_path = os.path.join(self.processed_data_path, file_name)
        print(f"Loading processed data from: {file_path}")
        return pd.read_csv(file_path, nrows=nrows)
    
    def create_advanced_features(self, df):
        """Create advanced features for forecasting"""
        # Sales velocity (rate of change)
        df['sales_velocity'] = df['sales'].diff()
        
        # Acceleration (change in velocity)
        df['sales_acceleration'] = df['sales_velocity'].diff()
        
        # Exponential moving averages
        df['sales_ema_7'] = df['sales'].ewm(span=7).mean()
        df['sales_ema_30'] = df['sales'].ewm(span=30).mean()
        
        # Volatility (rolling standard deviation)
        df['sales_volatility_7'] = df['sales'].rolling(window=7).std()
        df['sales_volatility_30'] = df['sales'].rolling(window=30).std()
        
        # Seasonal features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['day_of_week'].isin([0, 1, 2]).astype(int)  # Assuming first 3 days of week are month start
        df['is_month_end'] = df['day_of_week'].isin([4, 5, 6]).astype(int)    # Assuming last 3 days of week are month end
        
        # Interaction features
        df['sales_per_day_of_week'] = df['sales'] / df.groupby('day_of_week')['sales'].transform('mean')
        
        # Product category features
        df['avg_sales_by_category'] = df.groupby('Product_Category')['sales'].transform('mean')
        df['sales_to_category_avg'] = df['sales'] / df['avg_sales_by_category']
        
        # User features
        df['user_purchase_frequency'] = df.groupby('User_ID')['User_ID'].transform('count')
        df['user_avg_purchase'] = df.groupby('User_ID')['sales'].transform('mean')
        
        return df
    
    def create_holiday_features(self, df):
        """Create holiday-related features"""
        # Example holiday periods (customize based on your data)
        # Since we don't have actual dates, we'll use position-based holidays
        
        # Simulate holidays at regular intervals (e.g., every 30 days)
        df['is_holiday'] = ((df.index % 30) == 0).astype(int)
        
        return df
    
    def scale_features(self, df, features_to_scale):
        """Scale numerical features"""
        df_scaled = df.copy()
        df_scaled[features_to_scale] = self.scaler.fit_transform(df[features_to_scale])
        return df_scaled
    
    def engineer_features(self, input_file, output_file, max_rows=None):
        """Main feature engineering pipeline"""
        # Load processed data
        df = self.load_processed_data(input_file, nrows=max_rows)
        print(f"Loaded data with {len(df)} rows")
        
        # Create advanced features
        df = self.create_advanced_features(df)
        print("Created advanced features")
        
        # Create holiday features
        df = self.create_holiday_features(df)
        print("Created holiday features")
        
        # Scale numerical features
        features_to_scale = ['sales', 'sales_velocity', 'sales_acceleration', 
                           'sales_volatility_7', 'sales_volatility_30',
                           'sales_per_day_of_week', 'sales_to_category_avg',
                           'user_purchase_frequency', 'user_avg_purchase']
        df_scaled = self.scale_features(df, features_to_scale)
        print("Scaled numerical features")
        
        # Save engineered features
        output_path = os.path.join(self.processed_data_path, output_file)
        df_scaled.to_csv(output_path, index=False)
        print(f"Saved engineered features to {output_path}")
        
        return df_scaled

if __name__ == "__main__":
    # Get the absolute path to the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    
    # Create absolute paths for data directories
    processed_data_path = os.path.join(project_root, "data", "processed")
    
    print(f"Processed data path: {processed_data_path}")
    
    # Process a subset of the data to avoid memory issues
    max_rows = 100000  # Process first 100,000 rows
    
    feature_eng = FeatureEngineering(processed_data_path)
    engineered_df = feature_eng.engineer_features(
        "processed_sales.csv",
        "engineered_sales.csv",
        max_rows=max_rows
    )
    
    print("Feature engineering completed successfully!")
    print(f"Engineered data shape: {engineered_df.shape}")
    print("\nSample of engineered features:")
    print(engineered_df.head()) 
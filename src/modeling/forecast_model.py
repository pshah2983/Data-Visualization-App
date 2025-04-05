import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
import os

class SalesForecastModel:
    def __init__(self, processed_data_path, models_path):
        self.processed_data_path = processed_data_path
        self.models_path = models_path
        
    def load_engineered_data(self, file_name, nrows=None):
        """Load engineered features data"""
        file_path = os.path.join(self.processed_data_path, file_name)
        print(f"Loading engineered data from: {file_path}")
        return pd.read_csv(file_path, nrows=nrows)
    
    def prepare_data(self, df, target_col='sales', forecast_horizon=30):
        """Prepare data for modeling"""
        # Create target variable (future sales)
        df['target'] = df[target_col].shift(-forecast_horizon)
        
        # Remove rows with NaN target
        df = df.dropna(subset=['target'])
        
        # Select features for modeling
        feature_cols = [col for col in df.columns if col not in 
                       ['User_ID', 'Product_ID', target_col, 'target']]
        
        X = df[feature_cols]
        y = df['target']
        
        return X, y, feature_cols
    
    def train_xgboost_model(self, X, y):
        """Train XGBoost model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"XGBoost Model Performance:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        
        return model, X_test, y_test, y_pred
    
    def train_holt_winters(self, df, target_col='sales'):
        """Train Holt-Winters model"""
        # For Holt-Winters, we need a time series
        # We'll use the index as time
        ts = pd.Series(df[target_col].values, index=pd.date_range(start='2020-01-01', periods=len(df), freq='D'))
        
        model = ExponentialSmoothing(
            ts,
            seasonal_periods=7,
            trend='add',
            seasonal='add'
        ).fit()
        
        # Make forecast
        forecast = model.forecast(30)
        
        return model, forecast
    
    def save_model(self, model, model_name):
        """Save trained model"""
        # Create models directory if it doesn't exist
        os.makedirs(self.models_path, exist_ok=True)
        
        model_path = os.path.join(self.models_path, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    def generate_forecast(self, input_file, output_file, max_rows=None):
        """Main forecasting pipeline"""
        # Load data
        df = self.load_engineered_data(input_file, nrows=max_rows)
        print(f"Loaded data with {len(df)} rows")
        
        # Prepare data
        X, y, feature_cols = self.prepare_data(df)
        print(f"Prepared data with {len(X)} rows and {len(feature_cols)} features")
        
        # Train XGBoost model
        xgb_model, X_test, y_test, y_pred = self.train_xgboost_model(X, y)
        
        # Train Holt-Winters model
        hw_model, hw_forecast = self.train_holt_winters(df)
        
        # Save models
        self.save_model(xgb_model, "xgboost_model")
        self.save_model(hw_model, "holt_winters_model")
        
        # Create forecast dataframe
        # For XGBoost, we'll use the last 30 rows to predict the next 30 days
        last_30_rows = X.iloc[-30:]
        xgb_forecast = xgb_model.predict(last_30_rows)
        
        # Create a date range for the forecast
        last_date = pd.Timestamp('2020-01-01') + pd.Timedelta(days=len(df)-1)
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'xgboost_forecast': xgb_forecast,
            'holt_winters_forecast': hw_forecast.values
        })
        
        # Save forecasts
        output_path = os.path.join(self.processed_data_path, output_file)
        forecast_df.to_csv(output_path, index=False)
        print(f"Saved forecasts to {output_path}")
        
        return forecast_df

if __name__ == "__main__":
    # Get the absolute path to the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    
    # Create absolute paths for data directories
    processed_data_path = os.path.join(project_root, "data", "processed")
    models_path = os.path.join(project_root, "models")
    
    print(f"Processed data path: {processed_data_path}")
    print(f"Models path: {models_path}")
    
    # Process a subset of the data to avoid memory issues
    max_rows = 100000  # Process first 100,000 rows
    
    forecast_model = SalesForecastModel(processed_data_path, models_path)
    forecasts = forecast_model.generate_forecast(
        "engineered_sales.csv",
        "sales_forecasts.csv",
        max_rows=max_rows
    )
    
    print("\nForecast generated successfully!")
    print("\nSample of forecasts:")
    print(forecasts.head()) 
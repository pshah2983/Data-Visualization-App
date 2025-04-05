import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn')
sns.set_palette("husl")

# Paths
processed_data_path = os.path.join('data', 'processed')
processed_file = os.path.join(processed_data_path, 'processed_sales.csv')
forecast_file = os.path.join(processed_data_path, 'sales_forecasts.csv')
output_path = os.path.join(processed_data_path, 'forecast_visualization')

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv(processed_file, nrows=100000)
forecast_df = pd.read_csv(forecast_file)

# Calculate daily actual sales
daily_sales = df.groupby('day_of_week')['Purchase'].agg(['mean', 'std']).reset_index()
daily_sales['day_name'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create visualizations
print("Creating visualizations...")

# 1. Daily Sales Pattern
plt.figure(figsize=(12, 6))
plt.bar(daily_sales['day_name'], daily_sales['mean'])
plt.title('Average Daily Sales Pattern')
plt.xlabel('Day of Week')
plt.ylabel('Average Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'daily_sales_pattern.png'))
plt.close()

# 2. Sales Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Purchase', bins=50)
plt.title('Sales Distribution')
plt.xlabel('Sales Amount')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'sales_distribution.png'))
plt.close()

# 3. Forecast Plot
plt.figure(figsize=(12, 6))
plt.plot(forecast_df['date'], forecast_df['forecast'], marker='o')
plt.title('Sales Forecast for Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Forecasted Sales')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'sales_forecast.png'))
plt.close()

# Create a summary report
print("Creating summary report...")
with open(os.path.join(output_path, 'forecast_summary.txt'), 'w') as f:
    f.write("Sales Forecast Summary\n")
    f.write("====================\n\n")
    
    f.write("Historical Sales Statistics:\n")
    f.write(f"Average Daily Sales: ${df['Purchase'].mean():.2f}\n")
    f.write(f"Standard Deviation: ${df['Purchase'].std():.2f}\n")
    f.write(f"Minimum Sales: ${df['Purchase'].min():.2f}\n")
    f.write(f"Maximum Sales: ${df['Purchase'].max():.2f}\n\n")
    
    f.write("Forecast Statistics:\n")
    f.write(f"Average Forecasted Sales: ${forecast_df['forecast'].mean():.2f}\n")
    f.write(f"Standard Deviation: ${forecast_df['forecast'].std():.2f}\n")
    f.write(f"Minimum Forecast: ${forecast_df['forecast'].min():.2f}\n")
    f.write(f"Maximum Forecast: ${forecast_df['forecast'].max():.2f}\n")

print("Visualization completed!")
print(f"Output files saved to: {output_path}")
print("\nYou can now use these visualizations in Power BI:") 
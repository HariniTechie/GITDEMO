# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = r'C:\Users\DELL\OneDrive\Desktop\CSM\GitDemo\traffic_accident_analysis.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Analyze accidents by time of day (if a 'time' column exists)
if 'time' in df.columns:
    df['hour'] = pd.to_datetime(df['time'], format='%H:%M').dt.hour
    plt.figure(figsize=(10, 6))
    sns.countplot(x='hour', data=df, palette='viridis')
    plt.title('Accidents by Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Accidents')
    plt.show()

# Analyze accidents by weather condition (if a 'weather' column exists)
if 'weather' in df.columns:
    plt.figure(figsize=(10, 6))
    weather_counts = df['weather'].value_counts().head(10)  # Top 10 weather conditions
    sns.barplot(x=weather_counts.values, y=weather_counts.index, palette='magma')
    plt.title('Top 10 Weather Conditions During Accidents')
    plt.xlabel('Number of Accidents')
    plt.ylabel('Weather Condition')
    plt.show()

# Analyze accidents by road condition (if a 'road_condition' column exists)
if 'road_condition' in df.columns:
    plt.figure(figsize=(10, 6))
    road_counts = df['road_condition'].value_counts().head(10)  # Top 10 road conditions
    sns.barplot(x=road_counts.values, y=road_counts.index, palette='plasma')
    plt.title('Top 10 Road Conditions During Accidents')
    plt.xlabel('Number of Accidents')
    plt.ylabel('Road Condition')
    plt.show()

# Save the cleaned dataset (optional)
df.to_csv('cleaned_traffic_accidents.csv', index=False)
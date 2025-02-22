# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

# Load the dataset
df = pd.read_csv('Motor_Vehicle_Collisions_Crashes.csv')

# Display the first few rows of the dataset
print(df.head())

# Check the columns in the dataset
print(df.columns)

# Check for missing values
print(df.isnull().sum())

# Handle missing values (drop or fill as needed)
df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'CRASH TIME', 'WEATHER'])  # Drop rows with missing critical data

# Convert 'CRASH TIME' to datetime format (if applicable)
df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME'], format='%H:%M', errors='coerce')

# Extract hour from 'CRASH TIME'
df['HOUR'] = df['CRASH TIME'].dt.hour

# Exploratory Data Analysis (EDA)

# 1. Accidents by Time of Day
plt.figure(figsize=(10, 6))
sns.countplot(x='HOUR', data=df, palette='viridis')
plt.title('Accidents by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Accidents')
plt.show()

# 2. Accidents by Weather Condition
plt.figure(figsize=(10, 6))
weather_counts = df['WEATHER'].value_counts().head(10)  # Top 10 weather conditions
sns.barplot(x=weather_counts.values, y=weather_counts.index, palette='magma')
plt.title('Top 10 Weather Conditions During Accidents')
plt.xlabel('Number of Accidents')
plt.ylabel('Weather Condition')
plt.show()

# 3. Accidents by Road Surface Condition (if available)
if 'ROAD SURFACE' in df.columns:
    plt.figure(figsize=(10, 6))
    road_counts = df['ROAD SURFACE'].value_counts().head(10)  # Top 10 road conditions
    sns.barplot(x=road_counts.values, y=road_counts.index, palette='plasma')
    plt.title('Top 10 Road Surface Conditions During Accidents')
    plt.xlabel('Number of Accidents')
    plt.ylabel('Road Surface Condition')
    plt.show()

# 4. Accident Hotspots (Geographical Visualization)

# Create a map centered at the mean latitude and longitude
map_center = [df['LATITUDE'].mean(), df['LONGITUDE'].mean()]
accident_map = folium.Map(location=map_center, zoom_start=12)

# Add a heatmap to visualize accident hotspots
heat_data = [[row['LATITUDE'], row['LONGITUDE']] for index, row in df.iterrows()]
HeatMap(heat_data).add_to(accident_map)

# Save the map to an HTML file
accident_map.save('accident_hotspots.html')

# Display the map (in Jupyter Notebook or open the saved HTML file)
accident_map

# 5. Contributing Factors (if available)
if 'CONTRIBUTING FACTOR' in df.columns:
    plt.figure(figsize=(10, 6))
    factor_counts = df['CONTRIBUTING FACTOR'].value_counts().head(10)  # Top 10 contributing factors
    sns.barplot(x=factor_counts.values, y=factor_counts.index, palette='coolwarm')
    plt.title('Top 10 Contributing Factors to Accidents')
    plt.xlabel('Number of Accidents')
    plt.ylabel('Contributing Factor')
    plt.show()

# Summary of Findings
print("Summary of Findings:")
print(f"1. Most accidents occur between {df['HOUR'].mode()[0]}:00 and {df['HOUR'].mode()[0] + 1}:00.")
print(f"2. Most common weather condition during accidents: {df['WEATHER'].mode()[0]}.")
if 'ROAD SURFACE' in df.columns:
    print(f"3. Most common road surface condition during accidents: {df['ROAD SURFACE'].mode()[0]}.")
if 'CONTRIBUTING FACTOR' in df.columns:
    print(f"4. Most common contributing factor to accidents: {df['CONTRIBUTING FACTOR'].mode()[0]}.")
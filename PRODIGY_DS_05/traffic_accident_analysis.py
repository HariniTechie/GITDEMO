import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Motor_Vehicle_Collisions_-_Crashes.csv")

# Display basic info about dataset
print("Dataset Overview:")
print(df.info())

# Display first few rows
print(df.head())

# Check missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Drop columns with too many missing values (adjust as needed)
df = df.drop(columns=['ON STREET NAME', 'CROSS STREET NAME', 'OFF STREET NAME'], errors='ignore')

# Drop rows with missing critical data
df = df.dropna(subset=['BOROUGH', 'CRASH DATE', 'CRASH TIME', 'WEATHER CONDITION'])

# Convert Date and Time to Datetime format
df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME'], format='%H:%M').dt.hour  # Extract hour only

# 🚦 1. Accidents by Time of Day
plt.figure(figsize=(10,5))
sns.countplot(x=df['CRASH TIME'], palette='coolwarm')
plt.title("Number of Accidents by Hour of Day")
plt.xlabel("Hour of the Day")
plt.ylabel("Accident Count")
plt.xticks(range(0, 24))
plt.grid()
plt.show()

# 🌧️ 2. Accidents by Weather Condition
plt.figure(figsize=(12,5))
sns.countplot(y=df['WEATHER CONDITION'], order=df['WEATHER CONDITION'].value_counts().index, palette='viridis')
plt.title("Accidents by Weather Condition")
plt.xlabel("Accident Count")
plt.ylabel("Weather Condition")
plt.show()

# 🏙️ 3. Accidents by Borough
plt.figure(figsize=(10,5))
sns.countplot(x=df['BOROUGH'], palette='Set2')
plt.title("Number of Accidents in Each Borough")
plt.xlabel("Borough")
plt.ylabel("Accident Count")
plt.show()

# 📍 4. Heatmap of Accident Locations (requires lat/lon data)
if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
    import folium
    from folium.plugins import HeatMap

    accident_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)  # NYC Center
    heat_data = df[['LATITUDE', 'LONGITUDE']].dropna().values.tolist()
    HeatMap(heat_data).add_to(accident_map)
    accident_map.save("accident_hotspots.html")
    print("Heatmap saved as 'accident_hotspots.html'. Open it in your browser.")

# 🚗 5. Contributing Factors
plt.figure(figsize=(12,6))
sns.countplot(y=df['CONTRIBUTING FACTOR VEHICLE 1'], order=df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().index[:10], palette='pastel')
plt.title("Top 10 Contributing Factors in Accidents")
plt.xlabel("Accident Count")
plt.ylabel("Contributing Factor")
plt.show()

print("\n✅ Traffic accident analysis completed successfully!")

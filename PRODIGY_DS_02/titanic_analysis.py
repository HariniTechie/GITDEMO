import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = pd.read_csv("titanic.csv")  

# Display first few rows
print(df.head())  

# 1. Check Dataset Information
print("\nDataset Info:")
print(df.info())  

# 2. Check for Missing Values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())  

# 3. Summary Statistics
print("\nSummary Statistics:")
print(df.describe())  

# 4. Count Unique Values
print("\nUnique Values in Each Column:")
print(df.nunique())  

# 5a. Survival Count
sns.countplot(x="survived", data=df)
plt.title("Survival Count (0 = Died, 1 = Survived)")
plt.show()

# 5b. Survival Based on Passenger Class
sns.countplot(x="pclass", hue="survived", data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()

# 5c. Age Distribution
sns.histplot(df["age"].dropna(), bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.show()

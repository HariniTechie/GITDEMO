import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load dataset from GitHub (Replace with your actual raw GitHub link)
url = "https://raw.githubusercontent.com/HariniTechie/GITDEMO/main/PRODIGY_DS_03/bank-additional-full.csv"
df = pd.read_csv(url, sep=";")  # Dataset uses ';' as separator

# Handle missing values (if any)
df.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for inverse transformation if needed

# Define features (X) and target variable (y)
X = df.drop(columns=["y"])  # 'y' is the target (purchase decision)
y = df["y"]

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Decision Tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Print accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print detailed classification report
print(classification_report(y_test, y_pred))

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=["No Purchase", "Purchase"], filled=True)
plt.show()

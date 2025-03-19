import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('dataset for assignment 2.csv')

# --- Data Exploration and Preparation ---

# Display basic information about the dataset
print("Dataset Info:")
print(data.info())

# Display summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Handle missing values (example: impute with mean)
for column in data.columns:
    if data[column].isnull().any():
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column].fillna(data[column].mean(), inplace=True)
        else:
            data[column].fillna(data[column].mode()[0], inplace=True)

# Handle outliers (example: using IQR method)
for column in data.select_dtypes(include=np.number).columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include='object').columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# --- Feature Engineering ---

# Example: Create an 'activity_score' feature
data['activity_score'] = data['activity_level'] * data['distance_traveled']

# --- Predictive Modeling (Regression) ---

# Prepare data for regression
X_reg = data[['age', 'activity_level', 'distance_traveled', 'activity_score']] # Features
y_reg = data['calories_burned']  # Target variable

# Split data into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Train a linear regression model
model_reg = LinearRegression()
model_reg.fit(X_train_reg, y_train_reg)

# Make predictions and evaluate the model
y_pred_reg = model_reg.predict(X_test_reg)
mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
r2_reg = r2_score(y_test_reg, y_pred_reg)

print("\nRegression Model Evaluation:")
print(f"Mean Squared Error: {mse_reg}")
print(f"R-squared: {r2_reg}")

# --- Predictive Modeling (Clustering) ---

# Prepare data for clustering
X_cluster = data[['age', 'activity_level', 'distance_traveled', 'activity_score']]

# Standardize the data
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Choose the number of clusters (example: using the elbow method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_cluster_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the elbow plot, choose the optimal number of clusters (e.g., 3)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_cluster_scaled)

# Add cluster labels to the original data
data['cluster'] = clusters

# Visualize clusters (example: scatter plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='distance_traveled', y='calories_burned', hue='cluster', data=data, palette='viridis')
plt.title('Clusters of Users')
plt.show()

# --- Ethical Considerations (Example - Basic awareness) ---
print("\nEthical Considerations:")
print("Ensure data anonymization and secure storage to protect user privacy.")
print("Implement clear data usage policies and obtain informed consent.")

# --- Cultural Relevance (Example - Simple grouping) ---
print("\nCultural Relevance:")
print("Analyze user activity patterns by age and gender to identify potential differences.")
print(data.groupby('gender')['activity_level'].mean())

# --- Additional Considerations ---
# Further analysis, visualizations, and ethical/cultural considerations can be added here.
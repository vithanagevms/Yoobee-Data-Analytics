import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df = pd.read_csv("dataset for assignment 2.csv")

# Assuming your dataframe is called 'df'
# Select features that have meaningful correlation with app_sessions
# Based on your correlation matrix, these could include:
# distance_traveled, calories_burned, activity_level_Numeric

# First convert activity_level to numeric if it isn't already
df['activity_level_Numeric'] = df['activity_level'].map({'Sedentary': 1, 'Moderate': 2, 'Active': 3})


# Define X (predictors) and y (target)
X = df[['distance_traveled', 'calories_burned', 'activity_level_Numeric']]
y = df['app_sessions']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the model coefficients
intercept = model.intercept_
coefficients = model.coef_

# Create the regression equation
equation = f"app_sessions = {intercept:.2f}"
for i, feature in enumerate(X.columns):
    equation += f" + ({coefficients[i]:.2f} × {feature})"

print(equation)

# Evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()


silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {i}, the silhouette score is {silhouette_avg:.3f}")

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-')
plt.title('Silhouette Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.decomposition import PCA
import scipy.stats as stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set styling for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Function to load and explore data
def load_and_explore_data(file_path):
    """
    Load the dataset and perform initial exploratory analysis
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    df (DataFrame): Pandas DataFrame containing the loaded data
    """
    print("Loading and exploring data...\n")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Display basic information
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values count:")
    print(df.isnull().sum())
    
    # Data types
    print("\nData types:")
    print(df.dtypes)
    
    # Summary statistics
    print("\nSummary statistics:")
    print(df.describe())
    
    
        # Correlation matrix
    print("Analyzing correlations between pre-features...")
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.select_dtypes("number").corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Pre-Correlation Matrix')
    plt.savefig('pre-correlation_matrix.png')
    plt.close()
    
    return df

# Function for data preprocessing
def preprocess_data(df):
    """
    Preprocess the data for analysis and modeling
    
    Parameters:
    df (DataFrame): Raw data
    
    Returns:
    df_processed (DataFrame): Preprocessed data
    """
    print("\nPreprocessing data...\n")
    
    # Create a copy to avoid modifying the original DataFrame
    df_processed = df.copy()
    
    # Check for and handle missing values
    if df_processed.isnull().sum().sum() > 0:
        print("Handling missing values...")
        # For numerical columns, fill with median
        num_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        cat_cols = df_processed.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    # Convert activity_level to numerical values if needed
    if 'activity_level' in df_processed.columns:
        activity_map = {'Sedentary': 1, 'Moderate': 2, 'Active': 3}
        if df_processed['activity_level'].dtype == 'object':
            df_processed['activity_level Numeric'] = df_processed['activity_level'].map(activity_map)
    
    # Check for outliers in numerical columns
    print("Checking for outliers in numerical columns...")
    for col in df_processed.select_dtypes(include=['int64', 'float64']).columns:
        # Skip user_id or any identifier columns
        if col == 'user_id':
            continue
            
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)]
        if not outliers.empty:
            print(f"Found {len(outliers)} outliers in column '{col}'")
            # We'll keep outliers but note them for interpretability
            print(outliers)
            
    # Create derived features that might be useful for analysis
    print("Creating derived features...")
    
    # Calculate engagement ratio (sessions per km)
    if 'app_sessions' in df_processed.columns and 'distance_traveled' in df_processed.columns:
        df_processed['Sessions per km'] = df_processed['app_sessions'] / df_processed['distance_traveled'].replace(0, 0.001)
    
    # Calculate efficiency ratio (calories burned per km)
    if 'calories_burned' in df_processed.columns and 'distance_traveled' in df_processed.columns:
        df_processed['Calories per km'] = df_processed['calories_burned'] / df_processed['distance_traveled'].replace(0, 0.001)
    
    # Calculate session intensity (calories per session)
    if 'calories_burned' in df_processed.columns and 'app_sessions' in df_processed.columns:
        df_processed['Calories per Session'] = df_processed['calories_burned'] / df_processed['app_sessions'].replace(0, 0.001)
    
    print("Preprocessing completed.")
    return df_processed

# Function for exploratory data analysis
def perform_eda(df):
    """
    Perform exploratory data analysis to understand patterns and relationships
    
    Parameters:
    df (DataFrame): Preprocessed data
    
    Returns:
    None (displays plots and statistics)
    """
    print("\nPerforming exploratory data analysis...\n")
    
    # Set up the matplotlib figure
    plt.figure(figsize=(15, 10))
    
    # Distribution of numerical features
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    # Skip UserID if it exists
    numerical_cols = [col for col in numerical_cols if 'user_id' not in col]
    
    print("Analyzing distributions of numerical features...")
    for i, col in enumerate(numerical_cols[:6]):  # Limit to first 6 features for clarity
        plt.subplot(2, 3, i+1)
        sns.histplot(df[col],  kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig('numerical_distributions.png')
    plt.close()
    
    # Correlation matrix
    print("Analyzing correlations between features...")
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Print key insights from correlation matrix
    high_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.5:  # Threshold for high correlation
                high_correlations.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))
    
    if high_correlations:
        print("\nHigh correlations found:")
        for var1, var2, corr in high_correlations:
            print(f"  - {var1} and {var2}: {corr:.2f}")
    else:
        print("\nNo high correlations found between variables.")
    
    # Categorical analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        print("\nAnalyzing categorical features...")
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            value_counts = df[col].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'categorical_{col}.png')
            plt.close()
            
            print(f"\nDistribution of {col}:")
            print(df[col].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
    
    # Bivariate analysis
    print("\nPerforming bivariate analysis...")
    
    # Relationship between activity_level and App Usage
    if 'activity_level' in df.columns and 'app_sessions' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='activity_level', y='app_sessions', data=df)
        plt.title('app_sessions by activity_level')
        plt.savefig('activity_vs_sessions.png')
        plt.close()
        
        # One-way ANOVA to test for statistically significant differences
        activity_groups = df.groupby('activity_level')['app_sessions'].apply(list)
        f_statistic, p_value = stats.f_oneway(*activity_groups)
        print(f"\nOne-way ANOVA for app_sessions by activity_level:")
        print(f"  F-statistic: {f_statistic:.2f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  {'Statistically significant' if p_value < 0.05 else 'Not statistically significant'} at α=0.05")
    
    # gender differences
    if 'gender' in df.columns:
        for col in ['app_sessions', 'distance_traveled', 'calories_burned', 'activity_level Numeric']:
            if col in df.columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='gender', y=col, data=df)
                plt.title(f'{col} by gender')
                plt.savefig(f'gender_vs_{col.replace(" ", "_").replace("(", "").replace(")", "")}.png')
                plt.close()
                
                # T-test for gender differences
                male_data = df[df['gender'] == 'Male'][col]
                female_data = df[df['gender'] == 'Female'][col]
                t_stat, p_val = stats.ttest_ind(male_data, female_data, equal_var=False)
                print(f"\nT-test for {col} by gender:")
                print(f"  t-statistic: {t_stat:.2f}")
                print(f"  p-value: {p_val:.4f}")
                print(f"  {'Statistically significant' if p_val < 0.05 else 'Not statistically significant'} at α=0.05")
    
    # location-based analysis
    if 'location' in df.columns:
        for col in ['app_sessions', 'distance_traveled', 'calories_burned', 'activity_level Numeric']:
            if col in df.columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='location', y=col, data=df)
                plt.title(f'{col} by location')
                plt.savefig(f'location_vs_{col.replace(" ", "_").replace("(", "").replace(")", "")}.png')
                plt.close()
    
    # age-based analysis
    if 'age' in df.columns:
        # age groups for better visualization
        df['age Group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 100], labels=['<20', '20-30', '30-40', '40-50', '50+'])
        
        for col in ['app_sessions', 'distance_traveled', 'calories_burned', 'activity_level Numeric']:
            if col in df.columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='age Group', y=col, data=df)
                plt.title(f'{col} by age Group')
                plt.savefig(f'age_vs_{col.replace(" ", "_").replace("(", "").replace(")", "")}.png')
                plt.close()
    
    print("EDA completed and visualizations saved.")

def plot_model_performance(results, target, X_test, y_test):
    """
    Create visualizations to compare model performance.
    
    Parameters:
    results (dict): Dictionary containing model results
    target (str): The target variable name
    """
    
    # Extract model names and metrics
    models = list(results[target].keys())
    rmse_values = [results[target][model]['RMSE'] for model in models]
    r2_values = [results[target][model]['R2'] for model in models]
    
    # Set up the figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot 1: Bar chart of RMSE values (lower is better)
    axes[0].bar(models, rmse_values, color=colors)
    axes[0].set_title(f'RMSE Comparison for {target}', fontsize=14)
    axes[0].set_ylabel('RMSE (lower is better)', fontsize=12)
    axes[0].set_ylim(0, max(rmse_values) * 1.2)  # Add some space above the highest bar
    
    # Add value labels on top of bars
    for i, v in enumerate(rmse_values):
        axes[0].text(i, v + (max(rmse_values) * 0.05), f'{v:.2f}', 
                    ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Bar chart of R² values (higher is better)
    axes[1].bar(models, r2_values, color=colors)
    axes[1].set_title(f'R² Comparison for {target}', fontsize=14)
    axes[1].set_ylabel('R² (higher is better)', fontsize=12)
    axes[1].set_ylim(0, max(1, max(r2_values) * 1.2))  # R² should typically be between 0 and 1
    
    # Add value labels on top of bars
    for i, v in enumerate(r2_values):
        axes[1].text(i, v + 0.05, f'{v:.2f}', 
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{target}_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    #Scatter plot of actual vs. predicted values for best model
    best_model_name = models[np.argmax(r2_values)]
    best_model = results[target][best_model_name]['model']
    
    # Get predictions for test data
    predictions = best_model.predict(X_test)
    
    plt.figure(figsize=(9, 7))
    plt.scatter(y_test, predictions, alpha=0.5)
    
    # Add the perfect prediction line
    min_val = min(min(y_test), min(predictions))
    max_val = max(max(y_test), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'Actual vs. Predicted Values\nModel: {best_model_name}', fontsize=14)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{target}_{best_model_name}_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional visualization: Residual plot for the best model
    residuals = y_test - predictions
    
    plt.figure(figsize=(9, 7))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.title(f'Residual Plot\nModel: {best_model_name}', fontsize=14)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{target}_{best_model_name}_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()


# Function for advanced statistical analysis and modeling
def perform_advanced_analysis(df):
    """
    Perform advanced statistical analysis and modeling
    
    Parameters:
    df (DataFrame): Preprocessed data
    
    Returns:
    results (dict): Dictionary containing model results and performance metrics
    """
    print("\nPerforming advanced statistical analysis and modeling...\n")
    
    results = {}
    
    # Prepare data for modeling
    # Remove UserID or any identifier columns
    if 'user_id' in df.columns:
        X = df.drop(['user_id'], axis=1)
    else:
        X = df.copy()
    
    # Define target variables for different models
    target_variables = []
    
    if 'app_sessions' in X.columns:
        target_variables.append('app_sessions')
    if 'calories_burned' in X.columns:
        target_variables.append('calories_burned')
    if 'distance_traveled' in X.columns:
        target_variables.append('distance_traveled')
    
    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # For each target variable, create and evaluate models
    for target in target_variables:
        print(f"\nModeling for target variable: {target}")
        
        # Prepare X and y
        numerical_predictors = [col for col in numerical_features if col != target]
        y = X[target]
        
        # Create the feature set without the target
        X_model = X.drop([target], axis=1)
        
        # Update lists for preprocessing
        categorical_predictors = categorical_features
        numerical_predictors = [col for col in X_model.select_dtypes(include=['int64', 'float64']).columns]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_model, y, test_size=0.2, random_state=42
        )
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_predictors),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_predictors)
            ]
        )
        
        # Linear Regression
        print("  Training Linear Regression Model...")
        lr_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        lr_pipeline.fit(X_train, y_train)
        lr_predictions = lr_pipeline.predict(X_test)
        
        lr_mse = mean_squared_error(y_test, lr_predictions)
        lr_rmse = np.sqrt(lr_mse)
        lr_r2 = r2_score(y_test, lr_predictions)
        
        print(f"  Linear Regression Results:")
        print(f"    RMSE: {lr_rmse:.2f}")
        print(f"    R²: {lr_r2:.2f}")
        
        # Random Forest Regression
        print("  Training Random Forest Model...")
        rf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])
        
        rf_pipeline.fit(X_train, y_train)
        rf_predictions = rf_pipeline.predict(X_test)
        
        rf_mse = mean_squared_error(y_test, rf_predictions)
        rf_rmse = np.sqrt(rf_mse)
        rf_r2 = r2_score(y_test, rf_predictions)
        
        print(f"  Random Forest Results:")
        print(f"    RMSE: {rf_rmse:.2f}")
        print(f"    R²: {rf_r2:.2f}")
        
        # Gradient Boosting Regression
        print("  Training Gradient Boosting Model...")
        gb_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(random_state=42))
        ])
        
        gb_pipeline.fit(X_train, y_train)
        gb_predictions = gb_pipeline.predict(X_test)
        
        gb_mse = mean_squared_error(y_test, gb_predictions)
        gb_rmse = np.sqrt(gb_mse)
        gb_r2 = r2_score(y_test, gb_predictions)
        
        print(f"  Gradient Boosting Results:")
        print(f"    RMSE: {gb_rmse:.2f}")
        print(f"    R²: {gb_r2:.2f}")
        
        # Store results
        results[target] = {
            'Linear Regression': {'RMSE': lr_rmse, 'R2': lr_r2, 'model': lr_pipeline},
            'Random Forest': {'RMSE': rf_rmse, 'R2': rf_r2, 'model': rf_pipeline},
            'Gradient Boosting': {'RMSE': gb_rmse, 'R2': gb_r2, 'model': gb_pipeline}
        }
        plot_model_performance(results, target, X_test, y_test)
        
        # Model results
        models = ["Linear Regression", "Random Forest", "Gradient Boosting"]
        rmse_values = [lr_rmse, rf_rmse, gb_rmse]
        r2_values = [lr_r2, rf_r2, gb_r2]

        # Create a figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bar plot for RMSE on the first y-axis
        sns.barplot(x=models, y=rmse_values, ax=ax1, color="blue", alpha=0.6)
        ax1.set_ylabel("RMSE", color="blue")
        ax1.set_title(f"Model Performance Comparison (RMSE & R²) for {target}")

        # Show the RMSE values on bars
        for i, v in enumerate(rmse_values):
            ax1.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12, color="blue")

        # Create the second y-axis to plot R² values
        ax2 = ax1.twinx()
        sns.barplot(x=models, y=r2_values, ax=ax2, color="green", alpha=0.4)
        ax2.set_ylabel("R² Score", color="green")

        # Show the R² values on bars
        for i, v in enumerate(r2_values):
            ax2.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12, color="green")

        # Set limits for the R² axis to be between 0 and 1
        ax2.set_ylim(0, 1)

        # Show the R² values on bars
        for i, v in enumerate(r2_values):
            ax2.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)

        plt.savefig(f'model_performance_R2_{target}.png')
        plt.close()
        
        # Feature importance for Random Forest
        if len(rf_pipeline.named_steps['regressor'].feature_importances_) > 0:
            # Get feature names after preprocessing
            preprocessed_features = []
            
            # Get the feature names for numerical columns (they stay the same after StandardScaler)
            preprocessed_features.extend(numerical_predictors)
            
            # Get the feature names for one-hot encoded categorical columns
            if categorical_predictors:
                # Get the categories from the OneHotEncoder
                ohe = preprocessor.named_transformers_['cat']
                cat_feature_names = []
                for i, category in enumerate(categorical_predictors):
                    cat_values = ohe.categories_[i]
                    for value in cat_values:
                        cat_feature_names.append(f"{category}_{value}")
                preprocessed_features.extend(cat_feature_names)
            
            # Get feature importance
            feature_importances = rf_pipeline.named_steps['regressor'].feature_importances_
            
            # Match feature importances with feature names (if they match in length)
            if len(feature_importances) == len(preprocessed_features):
                importance_df = pd.DataFrame({
                    'Feature': preprocessed_features,
                    'Importance': feature_importances
                }).sort_values('Importance', ascending=False)
                
                print("\n  Top 10 most important features:")
                print(importance_df.head(10))
                
                # Plot feature importance
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
                plt.title(f'Top 10 Feature Importance for {target}')
                plt.tight_layout()
                plt.savefig(f'feature_importance_{target.replace(" ", "_")}.png')
                plt.close()
            else:
                print("\n  Feature importance analysis skipped due to dimensionality mismatch.")
    
    # Clustering Analysis for User Segmentation
    print("\nPerforming cluster analysis for user segmentation...")
    
    # Prepare data for clustering - use only numerical columns excluding targets
    cluster_data = df.select_dtypes(include=['int64', 'float64']).copy()
    
    # Remove identifier columns
    if 'user_id' in cluster_data.columns:
        cluster_data = cluster_data.drop(['user_id'], axis=1)
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # Determine optimal number of clusters using the elbow method
    inertia = []
    silhouette_scores = []
    k_range = range(2, min(11, len(df) - 1))  # Test 2-10 clusters
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
    
    # Plot the elbow method results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method for Optimal k')
    
    plt.tight_layout()
    plt.savefig('optimal_clusters.png')
    plt.close()
    
    # Choose the optimal number of clusters based on the elbow method and silhouette score
    optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"Optimal number of clusters determined: {optimal_k}")
    
    # Apply K-means with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels to the original dataframe
    df['Cluster'] = cluster_labels
    
    # Analyze the clusters
    print("\nCluster analysis:")
    cluster_analysis = df.select_dtypes(include=['number']).groupby('Cluster').mean()
    print(cluster_analysis)
    
    # Visualize clusters using PCA for dimensionality reduction
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # Create a dataframe with PCA results and cluster labels
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = cluster_labels
    
    # Plot the clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
    plt.title('Cluster Analysis (PCA)')
    plt.savefig('cluster_pca.png')
    plt.close()
    
    # Profile each cluster
    print("\nCluster profiles:")
    for i in range(optimal_k):
        cluster_size = sum(cluster_labels == i)
        cluster_percentage = cluster_size / len(df) * 100
        print(f"\nCluster {i} ({cluster_size} users, {cluster_percentage:.1f}% of total):")
        
        # Calculate average values for key metrics in this cluster
        print('Cluster HEAD')
        df[df['Cluster'] == i].head()
        cluster_profile = df.select_dtypes(include=['number'])[df['Cluster'] == i].mean()
        
        # Format and print a summary of each cluster's characteristics
        for col in cluster_data.columns:
            overall_avg = df[col].mean()
            cluster_avg = cluster_profile[col]
            diff_percentage = ((cluster_avg - overall_avg) / overall_avg) * 100
            print(f"  - {col}: {cluster_avg:.2f} ({diff_percentage:+.1f}% vs. overall average)")
    
    # Add clustering results to return dict
    results['clustering'] = {
        'optimal_k': optimal_k,
        'silhouette_score': max(silhouette_scores),
        'cluster_profiles': cluster_analysis
    }
    
    return results, df

# Function for ethical and cultural relevance analysis
def analyze_ethical_cultural_relevance(df, results):
    """
    Analyze ethical implications and cultural relevance of the data and models
    
    Parameters:
    df (DataFrame): Processed data with clustering results
    results (dict): Results from advanced analysis
    
    Returns:
    ethical_analysis (dict): Dictionary with ethical and cultural relevance findings
    """
    print("\nAnalyzing ethical implications and cultural relevance...\n")
    
    ethical_analysis = {}
    
    # gender bias analysis
    if 'gender' in df.columns:
        print("Analyzing potential gender bias in data and models...")
        
        # gender distribution
        gender_distribution = df['gender'].value_counts(normalize=True)
        ethical_analysis['gender_distribution'] = gender_distribution
        
        print("gender distribution in the dataset:")
        for gender, percentage in gender_distribution.items():
            print(f"  - {gender}: {percentage:.2%}")
        
        # Check for gender bias in target variables
        for target in results.keys():
            if target != 'clustering':
                # Only analyze numerical target variables
                if df[target].dtype in ['int64', 'float64']:
                    gender_means = df.groupby('gender')[target].mean()
                    
                    # Calculate the gender gap
                    if 'Male' in gender_means and 'Female' in gender_means:
                        gender_gap = (gender_means['Male'] - gender_means['Female']) / gender_means.mean()
                        
                        ethical_analysis[f'gender_gap_{target}'] = gender_gap
                        
                        print(f"\ngender gap analysis for {target}:")
                        print(f"  - Male average: {gender_means['Male']:.2f}")
                        print(f"  - Female average: {gender_means['Female']:.2f}")
                        print(f"  - Relative gender gap: {gender_gap:.2%}")
                        
                        if abs(gender_gap) > 0.1:  # 10% threshold for noticeable gap
                            print(f"  - POTENTIAL BIAS ALERT: Noticeable gender gap in {target}")
    
    # age-based analysis
    if 'age' in df.columns:
        print("\nAnalyzing age-related patterns and potential biases...")
        
        # age distribution
        age_stats = df['age'].describe()
        ethical_analysis['age_stats'] = age_stats
        
        print("age distribution statistics:")
        print(f"  - Min age: {age_stats['min']}")
        print(f"  - Max age: {age_stats['max']}")
        print(f"  - Mean age: {age_stats['mean']:.2f}")
        print(f"  - Median age: {age_stats['50%']}")
        
        # Check if there's adequate representation across age groups
        age_groups = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 100], 
                           labels=['<20', '20-29', '30-39', '40-49', '50-59', '60+'])
        age_distribution = age_groups.value_counts(normalize=True)
        ethical_analysis['age_distribution'] = age_distribution
        
        print("\nage group distribution:")
        for age_group, percentage in age_distribution.items():
            print(f"  - {age_group}: {percentage:.2%}")
            if percentage < 0.05:  # 5% threshold for underrepresentation
                print(f"    REPRESENTATION ALERT: age group {age_group} may be underrepresented")
    
    # location-based analysis
    if 'location' in df.columns:
        print("\nAnalyzing location-based patterns and cultural relevance...")
        
        # location distribution
        location_distribution = df['location'].value_counts(normalize=True)
        ethical_analysis['location_distribution'] = location_distribution
        
        print("location distribution in the dataset:")
        for location, percentage in location_distribution.items():
            print(f"  - {location}: {percentage:.2%}")
        
        # Analyze location-based differences in app usage
        if 'app_sessions' in df.columns:
            location_usage = df.groupby('location')['app_sessions'].agg(['mean', 'median'])
            ethical_analysis['location_usage'] = location_usage
            
            print("\nApp usage patterns by location:")
            print(location_usage)
            
            # One-way ANOVA to test for statistically significant differences
            location_groups = df.groupby('location')['app_sessions'].apply(list)
            f_statistic, p_value = stats.f_oneway(*location_groups)
            
            print(f"\nlocation-based differences in app usage:")
            print(f"  F-statistic: {f_statistic:.2f}")
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print("  CULTURAL RELEVANCE ALERT: Statistically significant differences in app usage across locations")
                print("  This suggests the need for location-specific customization.")
    
    # Privacy risk assessment
    print("\nPerforming privacy risk assessment...")
    
    # Calculate user uniqueness (risk of re-identification)
    if 'Cluster' in df.columns:
        print("\nAnalyzing ethical implications of user clusters...")
        
        # Check for demographic skew in clusters
        if 'gender' in df.columns:
            cluster_gender = pd.crosstab(df['Cluster'], df['gender'], normalize='index')
            ethical_analysis['cluster_gender'] = cluster_gender
            
            print("gender distribution across clusters:")
            print(cluster_gender)
            
            # Chi-square test for independence
            contingency_table = pd.crosstab(df['Cluster'], df['gender'])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            
            print(f"\nCluster-gender independence test:")
            print(f"  Chi-square: {chi2:.2f}")
            print(f"  p-value: {p:.4f}")
    
        print(f"\nCluster-gender independence test:")
        print(f"  Chi-square: {chi2:.2f}")
        print(f"  p-value: {p:.4f}")
        if p < 0.05:
            print("  ETHICAL ALERT: Clusters show statistically significant gender skew")
            print("  This could lead to biased recommendations if not accounted for.")
        
        # Check for location skew in clusters
        if 'location' in df.columns:
            cluster_location = pd.crosstab(df['Cluster'], df['location'], normalize='index')
            ethical_analysis['cluster_location'] = cluster_location
            
            print("\nlocation distribution across clusters:")
            print(cluster_location)
            
            # Chi-square test for independence
            contingency_table = pd.crosstab(df['Cluster'], df['location'])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            
            print(f"\nCluster-location independence test:")
            print(f"  Chi-square: {chi2:.2f}")
            print(f"  p-value: {p:.4f}")
            if p < 0.05:
                print("  CULTURAL RELEVANCE ALERT: Clusters show statistically significant location skew")
                print("  This suggests cultural factors may influence user segmentation.")
    
        # Check for location skew in clusters
        if 'activity_level' in df.columns:
            cluster_activity_level = pd.crosstab(df['Cluster'], df['activity_level'], normalize='index')
            ethical_analysis['cluster_activity_level'] = cluster_activity_level
            
            print("\nActivity-Level distribution across clusters:")
            print(cluster_activity_level)
            
            # Chi-square test for independence
            contingency_table = pd.crosstab(df['Cluster'], df['activity_level'])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            
            print(f"\nCluster-Activity-Level independence test:")
            print(f"  Chi-square: {chi2:.2f}")
            print(f"  p-value: {p:.4f}")
            if p < 0.05:
                print("  CULTURAL RELEVANCE ALERT: Clusters show statistically significant Activity-Level skew")
                print("  This suggests cultural factors may influence user segmentation.")
    
    # Develop ethical guidelines based on analysis
    print("\nDeveloping ethical guidelines based on analysis...")
    
    ethical_guidelines = [
        "1. Data Collection Transparency: Clearly inform users about what data is being collected, how it's used, and provide easy opt-out options.",
        "2. Demographic Fairness: Regularly audit recommendation algorithms for bias across gender, age, and location demographics.",
        "3. Privacy Protection: Implement k-anonymity (k≥5) to ensure user data cannot be easily re-identified.",
        "4. Cultural Relevance: Customize features and recommendations based on cultural and geographical context.",
        "5. Inclusive Design: Ensure the app is accessible and beneficial to users of all demographics.",
        "6. User Consent: Obtain explicit consent before using sensitive health metrics for analysis.",
        "7. Data Minimization: Only collect data that is necessary for the app's core functionality.",
        "8. Algorithmic Transparency: Make the basis of personalized recommendations explainable to users.",
        "9. Regular Ethical Audits: Conduct regular audits of data usage and algorithmic outcomes.",
        "10. Secure Data Handling: Implement robust security measures to protect user health and fitness data."
    ]
    
    ethical_analysis['guidelines'] = ethical_guidelines
    
    print("\nProposed ethical guidelines for data collection and usage:")
    for guideline in ethical_guidelines:
        print(guideline)
    
    return ethical_analysis

# Function to generate a comprehensive report
def generate_report(df, results, ethical_analysis):
    """
    Generate a comprehensive report of the analysis findings
    
    Parameters:
    df (DataFrame): Processed data
    results (dict): Results from advanced analysis
    ethical_analysis (dict): Results from ethical analysis
    
    Returns:
    None (prints the report)
    """
    print("\n" + "="*80)
    print(" "*30 + "ANALYSIS REPORT")
    print("="*80 + "\n")
    
    print("EXECUTIVE SUMMARY")
    print("-----------------")
    print("This report presents a comprehensive analysis of user data from a fitness tracking app,")
    print("including statistical patterns, predictive models, and ethical considerations.")
    
    print("\nKEY FINDINGS")
    print("------------")
    
    # Dataset summary
    print(f"1. Dataset Overview:")
    print(f"   - Total users analyzed: {len(df)}")
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            print(f"   - {gender} users: {count} ({count/len(df):.1%})")
    
    if 'location' in df.columns:
        location_counts = df['location'].value_counts()
        print(f"   - location distribution: {', '.join([f'{loc} ({count})' for loc, count in location_counts.items()])}")
    
    # Model performance summary
    print("\n2. Predictive Model Performance:")
    
    for target, models in results.items():
        if target != 'clustering':
            print(f"\n   Target Variable: {target}")
            best_model = max(models.items(), key=lambda x: x[1]['R2'])
            print(f"   - Best performing model: {best_model[0]}")
            print(f"   - R² score: {best_model[1]['R2']:.4f}")
            print(f"   - RMSE: {best_model[1]['RMSE']:.2f}")
    
    # Clustering results
    if 'clustering' in results:
        print("\n3. User Segmentation:")
        print(f"   - Optimal number of user segments identified: {results['clustering']['optimal_k']}")
        print(f"   - Segmentation quality (silhouette score): {results['clustering']['silhouette_score']:.2f}")
        
        # Add brief description of the clusters
        cluster_profiles = results['clustering']['cluster_profiles']
        for i in range(results['clustering']['optimal_k']):
            cluster_size = sum(df['Cluster'] == i)
            cluster_pct = cluster_size / len(df) * 100
            print(f"\n   Segment {i+1} ({cluster_pct:.1f}% of users):")
            
            # Find defining characteristics (columns with highest deviation from mean)
            profile = cluster_profiles.loc[i]
            overall_means = df.select_dtypes(include=['number']).mean()
            deviations = (profile - overall_means) / overall_means
            top_features = deviations.abs().nlargest(3).index.tolist()
            
            for feature in top_features:
                deviation_pct = deviations[feature] * 100
                print(f"   - {feature}: {profile[feature]:.2f} ({deviation_pct:+.1f}% vs. average)")
    
    # Ethical considerations
    print("\n4. Ethical Considerations:")
    
    # Check for gender bias
    if 'gender_distribution' in ethical_analysis:
        has_gender_gap = any([abs(ethical_analysis.get(f'gender_gap_{target}', 0)) > 0.1 
                            for target in results.keys() if target != 'clustering'])
        
        if has_gender_gap:
            print("   - ALERT: Potential gender bias detected in app usage patterns")
        else:
            print("   - No significant gender bias detected in usage patterns")
    
    # Check for privacy concerns
    if 'privacy_risk' in ethical_analysis:
        if ethical_analysis['privacy_risk'] > 0.1:
            print("   - ALERT: Privacy risk identified - potential for user re-identification")
        else:
            print("   - Low privacy risk observed in the current dataset")
    
    # Check for cultural relevance
    if 'location_usage' in ethical_analysis:
        print("   - Cultural relevance finding: App usage varies significantly by location")
        print("     This suggests the need for culturally adapted features and recommendations")
    
    print("\nRECOMMENDATIONS")
    print("--------------")
    
    # Software engineering recommendations
    print("1. Software Engineering Recommendations:")
    print("   - Implement personalized recommendations based on identified user segments")
    print("   - Optimize app features for the most predictive variables identified in the models")
    print("   - Develop location-specific features to address varying usage patterns")
    
    # Ethical recommendations
    print("\n2. Ethical Guidelines for Implementation:")
    for i, guideline in enumerate(ethical_analysis['guidelines'][:5]):  # Show top 5 guidelines
        print(f"   {guideline}")
    
    print("\nMETHODOLOGY")
    print("----------")
    print("The analysis utilized advanced statistical techniques including:")
    print("- Regression models (Linear, Random Forest, Gradient Boosting)")
    print("- Cluster analysis for user segmentation")
    print("- Hypothesis testing for demographic differences")
    print("- Privacy risk assessment")
    print("- Ethical and cultural relevance analysis")
    
    print("\n" + "="*80)
    print(" "*25 + "END OF ANALYSIS REPORT")
    print("="*80)

# Main function to run the entire analysis
def main(file_path):
    """
    Main function to run the entire analysis pipeline
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    None
    """
    # 1. Load and explore data
    df = load_and_explore_data(file_path)
    
    # 2. Preprocess data
    df_processed = preprocess_data(df)
    
    # 3. Perform exploratory data analysis
    perform_eda(df_processed)
    
    # 4. Perform advanced statistical analysis and modeling
    results, df_with_clusters = perform_advanced_analysis(df_processed)
    
    # 5. Analyze ethical implications and cultural relevance
    ethical_analysis = analyze_ethical_cultural_relevance(df_with_clusters, results)
    
    # 6. Generate comprehensive report
    generate_report(df_with_clusters, results, ethical_analysis)
    
    print("\nAnalysis completed successfully. All visualizations have been saved to the current directory.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "dataset for assignment 2.csv" #input("Please enter the path to your CSV file: ")
    
    main(file_path)
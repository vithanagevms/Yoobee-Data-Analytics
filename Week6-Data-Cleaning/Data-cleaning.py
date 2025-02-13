
################ Data cleaning the Iris dataset #################
from sklearn import datasets
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load iris dataset
iris = datasets.load_iris()
# Since this is a bunch, create a dataframe

iris_df=pd.DataFrame(iris.data)
iris_df['class']=iris.target

iris_df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
#### ===> TASK 1: here - add two more lines of the code to find the number and mean of missing data
sum_missing = iris_df.isnull().values.sum()
mean_missing = iris_df.isnull().values.mean()
print(f"The total missig records are {sum_missing} and mean is {mean_missing}")
cleaned_data = iris_df.dropna(how="all", inplace=True) # remove any empty lines

iris_X=iris_df.iloc[:5,[0,1,2,3]]
print(iris_X)

print(iris_X.duplicated())

### TASK2: Here - Write a short readme to explain above code and how we can calculate the corrolation amoung featuers with description

correlation_matrix = iris_df.iloc[:, :4].corr()  # Compute correlation for feature columns
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Iris Dataset Features")
plt.show()
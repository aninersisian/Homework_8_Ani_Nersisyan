from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil


# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training (80%) and inference (20%) sets
X_train, X_inference, y_train, y_inference = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to Pandas DataFrame
train_df = pd.DataFrame(X_train, columns=iris.feature_names)
train_df['target'] = y_train

inference_df = pd.DataFrame(X_inference, columns=iris.feature_names)
inference_df['target'] = y_inference

train_csv_path = 'iris_training.csv'
inference_csv_path = 'iris_inference.csv'

# Save the datasets to CSV files
train_df.to_csv('iris_training.csv', index=False)
inference_df.to_csv('iris_inference.csv', index=False)

shutil.copy(train_csv_path, '../training/' + train_csv_path)
shutil.copy(inference_csv_path, '../inference/' + inference_csv_path)

print("Datasets saved as iris_training.csv and iris_inference.csv")


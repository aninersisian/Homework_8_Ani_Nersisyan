import os

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


try:
    if not os.path.isfile('iris_training.csv'):
        raise FileNotFoundError(
            "Error: Training data file 'iris_training.csv' not found. Please make sure the file exists.")

    train_df = pd.read_csv('iris_training.csv')

    # Check if training data is empty
    if train_df.empty:
        raise ValueError("Training data is empty. Please check the data file.")

    # Check if 'target' column exists in the training data
    if 'target' not in train_df.columns:
        raise ValueError("The 'target' column is missing in the training data.")

    # Split the original training data into smaller training set and test set
    X = train_df.drop('target', axis=1).values
    y = train_df['target'].values

    # Check if there are enough samples for training
    if len(X) < 2:
        raise ValueError("There are not enough samples in the training data for training the model.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 20% of data for testing

    # Check if there are enough samples for testing
    if len(X_test) < 1:
        raise ValueError("There are not enough samples in the test data for evaluation.")

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = IrisNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test, dtype=torch.float32).clone().detach()
        test_outputs = model(test_inputs)
        _, predicted = torch.max(test_outputs.data, 1)
        correct = (predicted == torch.tensor(y_test).clone().detach()).sum().item()
        accuracy = correct / len(y_test)
        print(f'Test Accuracy: {accuracy}')

        class_report = classification_report(y_test, predicted)
        print(f'Classification Report:\n{class_report}')

    # Save the trained model
    torch.save(model.state_dict(), 'iris_model.pth')

    print("Model training and evaluation complete. Model saved as iris_model.pth")

except FileNotFoundError:
    print("Error: 'iris_training.csv' not found. Please make sure the training data file exists.")
except ValueError as ve:
    print(f"ValueError: {str(ve)}")
except Exception as e:
    print(f"An error occurred: {str(e)}")





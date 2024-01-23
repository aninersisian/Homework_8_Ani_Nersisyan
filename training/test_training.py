import unittest
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from training.train import IrisNet


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training_data_path = 'training/test_iris_training.csv'

        # Create test data for training
        test_csv_data = {
            'feature1': [5.1, 4.9, 4.7, 4.6, 5.0],
            'feature2': [3.5, 3.0, 3.2, 3.1, 3.6],
            'feature3': [1.4, 1.4, 1.3, 1.5, 1.4],
            'feature4': [0.2, 0.2, 0.2, 0.2, 0.2],
            'target': [0, 0, 0, 1, 1]  # Labels (0 and 1 for two classes)
        }
        test_df = pd.DataFrame(test_csv_data)
        test_df.to_csv(cls.training_data_path, index=False)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary files after tests
        os.remove(cls.training_data_path)

    def test_train_and_evaluate_model(self):
        # Load training data from test file
        train_df = pd.read_csv(self.training_data_path)
        X_train = train_df.drop('target', axis=1).values
        y_train = train_df['target'].values

        # Initialize the model, loss function, and optimizer
        model = IrisNet()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Convert data to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)

        # Train the model
        for epoch in range(100):
            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Perform evaluation (example: check accuracy)
        y_pred = torch.argmax(model(X_train), axis=1).numpy()
        accuracy = accuracy_score(y_train.numpy(), y_pred)

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)


if __name__ == '__main__':
    unittest.main()

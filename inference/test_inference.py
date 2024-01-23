import unittest
import os
import pandas as pd
import torch
from inference.inference import IrisNet


class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define test data paths for inference
        cls.inference_data_path = 'inference/test_iris_inference.csv'
        cls.model_path = 'inference/test_iris_model.pth'

        # Create test data for inference
        inference_data = pd.DataFrame({
            'feature1': [5.1, 4.9, 4.7],
            'feature2': [3.5, 3.0, 3.2],
            'feature3': [1.4, 1.4, 1.3],
            'feature4': [0.2, 0.2, 0.2]
        })
        inference_data.to_csv(cls.inference_data_path, index=False)

        # Create a dummy model for testing inference
        model = IrisNet()
        torch.save(model.state_dict(), cls.model_path)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary files after tests
        os.remove(cls.inference_data_path)
        os.remove(cls.model_path)

    def test_model_inference(self):
        # Load the trained model
        model = IrisNet()
        model.load_state_dict(torch.load(self.model_path))
        model.eval()

        # Load inference data
        inference_data = pd.read_csv(self.inference_data_path)

        # Convert data to PyTorch tensors
        inference_data = torch.tensor(inference_data.values, dtype=torch.float32)

        # Perform inference on the inference data
        with torch.no_grad():
            outputs = model(inference_data)
            _, predicted = torch.max(outputs, 1)

        # Check if predictions are of the correct shape
        self.assertEqual(predicted.shape[0], len(inference_data))


if __name__ == '__main__':
    unittest.main()


import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np


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
    # Check if the inference data file exists
    if not os.path.isfile('iris_inference.csv'):
        raise FileNotFoundError(
            "Error: Inference data file 'iris_inference.csv' not found. Please make sure the file exists.")

    model = IrisNet()
    model.load_state_dict(torch.load('iris_model.pth'))
    model.eval()

    # Check if the model loaded successfully
    if not model.parameters():
        raise ValueError("The model has not been trained or loaded correctly.")

    inference_data = pd.read_csv('iris_inference.csv')

    # Check if inference data is empty
    if inference_data.empty:
        raise ValueError("Inference data is empty. Please check the data file.")

    inference_data.drop(['target'], axis=1, inplace=True)
    inference_data = inference_data.apply(pd.to_numeric, errors='coerce')
    inference_data = inference_data.to_numpy(dtype=np.float32)

    # Check if there are enough samples for inference
    if len(inference_data) < 1:
        raise ValueError("There are not enough samples in the inference data for inference.")

    inference_data = torch.tensor(inference_data, dtype=torch.float32)

    predictions = []

    # Perform inference on the inference data
    with torch.no_grad():
        outputs = model(inference_data)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.tolist())

    # Save the inference results to a file
    inference_results = pd.DataFrame({'predicted_class': predictions})
    inference_results.to_csv('inference_results.csv', index=False)

    print("Inference complete. Results saved as 'inference_results.csv'")

except FileNotFoundError:
    print("Error: Model file 'iris_model.pth' or inference data file 'iris_inference.csv' not found.")
except ValueError as ve:
    print(f"ValueError: {str(ve)}")
except Exception as e:
    print(f"An error occurred: {str(e)}")




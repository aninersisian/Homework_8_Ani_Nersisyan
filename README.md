# ML Project README

Welcome to the Iris Deep Learning Model! This readme is designed to provide a structured explanation for running the model. Below, you'll find information on project setup, directory structure, and how to perform training and inference.

## Project Directory Structure

The project has the following directory structure:

```
hw_8/
├── data/
│   ├── iris_inference.csv                # Data for inference
│   ├── iris_training.csv                 # Data for training
│   ├── split_data.py                  # Script for splitting data
├── inference/
│   ├── __init__.py
│   ├── Dockerfile
│   ├── inference.py                   # Inference script
│   ├── inference_results.csv          # Results of inference
│   ├── iris_inference.csv             # Inference data
│   ├── iris_model.pth                 # Trained model
│   ├── requirements.txt               # Dependencies for inference
│   ├── test_inference.py              # Unit tests for inference
├── mlflow/
│   ├── Dockerfile
├── training/
│   ├── __init__.py
│   ├── Dockerfile
│   ├── iris_training.csv              # Training data
│   ├── requirements.txt               # Dependencies for training
│   ├── test_training.py               # Unit tests for training
│   ├── train.py                       # Training script
└── README.md                           # Project README

```

## Data

The `data` directory is responsible for data handling and preparation.

### Data Splitting

To split the data into inference and training sets and save them locally:

1. Navigate to the `data` directory.

2. Run the script `split_data.py`. This script divides the dataset into inference and training subsets.

3. Copy the resulting sets to the `inference` and `training` directories for further processing.

## Training

The `training` directory contains all the necessary components for training the model.

### Requirements

Ensure that all the required packages and dependencies for training are installed. You can find the list of dependencies in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
### Training Script
Use train.py to manage model training, splitting data into train and test sets, and evaluation.

### Dockerfile

1. Build a Docker image for training:
```bash
sudo docker build -t iris-train . 
```
2. Run the container, which saves the model as `iris_model.pth` in it and later use it in the inference part:
```bash
sudo docker run iris-train
```
3. Check the Docker image id:
```bash
sudo docker ps -a
```
4. Copy the iris_model.pth from the container to the directory inference:
```bash
sudo docker cp <image_id>:/usr/src/app/iris_model.pth ~/Desktop/hw_8/inference
```

### Test
Use the `test_training.py` to check whether the model works.

## Inference

This directory is responsible for returning results on the inference set using the model.
### Requirements

Ensure that all the required packages and dependencies for training are installed. You can find the list of dependencies in the `requirements.txt` file.
```bash
pip install -r requirements.txt
```
### Inference script
Use inference.py to obtain predictions on the inference set, and save it locally in `inference_results.csv`
1. Build a Docker image for inference:
```bash
sudo docker build -t iris-inference . 
```
2. Run the image:
```bash
sudo docker run iris-inference
```

### Test
Use the `test_inference.py` to check whether the model works.
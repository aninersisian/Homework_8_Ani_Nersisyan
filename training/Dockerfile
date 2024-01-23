# Use an official Python runtime as a parent image
FROM python:3.8

LABEL authors="ani"

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
  #  mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0


# Set the default command to execute
# when creating a new container
ENTRYPOINT ["python", "./train.py"]
CMD ["top", "-b"]

# Expose the port MLflow will run on
# EXPOSE 5000

# Command to run the MLflow server and training script
# CMD mlflow server --host 0.0.0.0 & python train.py
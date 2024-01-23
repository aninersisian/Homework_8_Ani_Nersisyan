FROM python:3.8
LABEL authors="ani"

# Set the working directory
WORKDIR /usr/src/app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt


# Run the inference script as the entry point
CMD ["python", "inference.py"]

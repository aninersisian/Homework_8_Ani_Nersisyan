FROM python:3.8
LABEL authors="ani"

RUN pip install mlflow

WORKDIR /mlflow

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0"]

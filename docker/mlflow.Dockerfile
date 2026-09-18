FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow psycopg2-binary boto3

EXPOSE 5000

ENTRYPOINT ["mlflow", "server"]
CMD ["--backend-store-uri", "postgresql+psycopg2://mlflow:mlflow@postgres:5432/mlflow", "--default-artifact-root", "s3://mlflow-artifacts", "--host", "0.0.0.0", "--port", "5000", "--allowed-hosts", "*"]
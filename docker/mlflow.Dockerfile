FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow

EXPOSE 5000

ENTRYPOINT ["mlflow", "server"]
CMD ["--backend-store-uri", "sqlite:///mlruns.db", "--default-artifact-root", "/mlruns", "--host", "0.0.0.0", "--port", "5000"]
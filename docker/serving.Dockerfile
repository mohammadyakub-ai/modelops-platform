# Multi-stage serving image.
# Stage 1 installs the exact dependency set into a clean prefix; stage 2 copies
# only that prefix plus the application, so the runtime layer stays minimal and
# the dependency layer is independently cacheable.
FROM python:3.11-slim AS deps
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1
COPY --from=deps /install/ /usr/local/
COPY src/ ./src/
COPY configs/ ./configs/
COPY deploy/ ./deploy/
COPY scripts/ ./scripts/
COPY docker/entrypoint.serving.sh ./entrypoint.serving.sh
RUN chmod +x ./entrypoint.serving.sh
EXPOSE 8000
ENTRYPOINT ["./entrypoint.serving.sh"]
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
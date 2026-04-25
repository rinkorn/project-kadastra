# MLflow tracking server with Postgres + S3 artifact-store support.
# Kept as a slim purpose-built image instead of the official ghcr.io/mlflow/mlflow
# because the official one lacks boto3 at build time of some tags.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install \
        "mlflow==2.22.0" \
        "psycopg2-binary==2.9.10" \
        "boto3==1.42.96"

EXPOSE 5000

# Command is provided via docker-compose; this image is intentionally
# a thin wrapper around mlflow so `mlflow server ...` remains visible there.
ENTRYPOINT ["mlflow"]
CMD ["server", "--help"]

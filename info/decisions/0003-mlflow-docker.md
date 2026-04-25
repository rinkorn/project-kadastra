# ADR-0003: MLflow в Docker с Postgres-бэкендом и S3-артефактами

**Статус:** Accepted
**Дата:** 2026-04-25

## Контекст

Для обучения моделей оценки кадастровой стоимости нужен трекинг экспериментов и реестр моделей. Рассматривались варианты:

- **A. File backend** (`file:./mlruns`) — ноль инфраструктуры, всё локально.
- **B. MLflow в Docker** — docker-compose с tracking-сервером + Postgres + S3 artifacts.
- **C. Без MLflow** — pickle/cbm + JSON метрик на диск.

## Решение

**B. MLflow в Docker.**

Состав стека (см. [docker-compose.mlflow.yml](../../docker-compose.mlflow.yml)):

- `mlflow-postgres` (postgres:16-alpine) — backend store (runs, params, metrics, tags).
- `mlflow-server` (кастомный образ на base python:3.11-slim + `mlflow[extras]` + `psycopg2-binary` + `boto3`, см. [infra/mlflow/Dockerfile](../../infra/mlflow/Dockerfile)).
- Artifact store — `s3://${S3_BUCKET}/mlflow-artifacts/` на нашем Synology (тот же S3 что уже используется для сырых данных).
- Tracking UI — `http://127.0.0.1:5000` (порт биндится только на loopback).
- Флаг `--serve-artifacts` — MLflow проксирует доступ к артефактам через HTTP, клиентам не нужны S3-creds отдельно.
- Постгрес-том `mlflow-pg` — именованный named volume, переживает редеплой.

## Обоснование выбора B vs A/C

- Эксперименты будут повторяться и сравниваться; без UI/реестра это ломается на 3-ем эксперименте.
- Файловый backend не переживает смену машины и не синкается — артефакты теряются.
- C (без MLflow) — нет механизма сравнения моделей и привязки метрик к артефакту.
- Docker-стоимость небольшая: один compose, два сервиса, один named volume.

## Последствия

- В [.env.example](../../.env.example) добавлены: `MLFLOW_ENABLED`, `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_DB_PASSWORD`.
- В [Settings](../../src/kadastra/config.py) добавлены: `mlflow_enabled`, `mlflow_tracking_uri`, `mlflow_experiment_name`.
- Training use case при `mlflow_enabled=True` регистрирует run в MLflow; при `False` — сохраняет артефакт на диск и JSON метрики (fallback). Это значит пилот можно катать без Docker, а полноценный трекинг включается одним флагом.
- Бакет `s3://kadastrova/mlflow-artifacts/` будет создан MLflow'ом при первом `log_artifact`.

## Открытые вопросы

- Нужен ли отдельный MLflow user/scope для артефактов (пока — те же S3-creds)?
- Прод-режим: биндить на 0.0.0.0 и закрывать файерволом/reverse-proxy с авторизацией, либо оставить на loopback + SSH-туннель для просмотра.

## Как поднять локально

```sh
cp .env.example .env
# заполнить S3_ACCESS_KEY, S3_SECRET_KEY, MLFLOW_DB_PASSWORD
docker compose -f docker-compose.mlflow.yml up -d --build
# UI: http://127.0.0.1:5000
# Остановить: docker compose -f docker-compose.mlflow.yml down
# Полностью снести с данными Postgres: docker compose -f docker-compose.mlflow.yml down -v
```

# Правила и процесс разработки

`info/` — общая база знаний для разработчика и AI-ассистента. Не проектная документация, а внутренние договорённости: правила, архитектурные принципы, процесс.

Контекст проекта, доменный стек и архитектура — в [project.md](project.md). Источники данных — в [data-sources.md](data-sources.md). Архитектурные решения — в [info/decisions/](decisions/) (ADR 0001+). Здесь — общее по разработке.

## Python

Версия: **3.13**.

## Базовый стек

| Инструмент | Назначение |
| ---------- | ---------- |
| **uv** | Пакетный менеджер, виртуальное окружение, lock-файл `uv.lock` |
| **ruff** | Линтер + форматтер (настройки в `pyproject.toml`) |
| **pyright** | Type checking (strict mode) |
| **pytest** | Тесты (`tests/unit/`, `tests/integration/`) |
| **pre-commit** | Хуки перед коммитом (ruff, pyright, commitizen) |
| **structlog** | Structured logging (JSON в проде, pretty в dev) |
| **pydantic-settings** | Типизированная конфигурация из `.env` |
| **FastAPI** | REST API (`/api/*`) + Web UI (Jinja2, без htmx — раскраска и работа с картой целиком на клиенте) |

Доменный стек (geo / ML / визуализация) — в [project.md](project.md).

## Архитектура — Hexagonal (Ports & Adapters)

Изоляция домена и use case'ов от инфраструктуры через порты и адаптеры. Зависимости только внутрь: `adapters → ports ← usecases ← api / web`.

```text
src/kadastra/
  domain/             ← Entities: ValuationObject, AssetClass; чистые классификаторы
  ports/              ← Интерфейсы (Protocol):
                        RawDataPort, CoverageReader/Store, FeatureReader/Store,
                        GoldFeatureReader/Store, NspdSilverStore,
                        ValuationObjectReader/Store, OofPredictionsReader,
                        QuartetModel, ModelLoader, ModelRegistry,
                        RoadGraph, RegionBoundary
  adapters/           ← Реализации портов (см. таблицу ниже)
  etl/                ← Чистые трансформации: H3, расстояния/доли в буфере,
                        relative ЦОФ, road graph builder, синтетика target
  ml/                 ← Модели: train, spatial K-Fold по parent H3 (ADR-0012),
                        feature columns, quartet metrics
  usecases/           ← Бизнес-логика: BuildRegionCoverage, BuildObjectFeatures,
                        BuildHexAggregates, TrainObjectValuationModel,
                        TrainQuartet, InferObjectValuation, …
  api/
    routes.py         ← FastAPI REST (/api/*: hex_aggregates, objects, market_reference, …)
    auth.py           ← BearerAuthMiddleware (cookie + bearer)
  web/
    routes.py         ← Web UI роуты (отдают карту map.html)
    templates/        ← Jinja2 шаблоны
    static/           ← Vendored CSS/JS (maplibre-gl, deck.gl)
  config.py           ← Settings (pydantic-settings, feature flags)
  composition_root.py ← Container, create_app, выбор адаптеров по флагам
```

### Адаптеры

| Порт | Адаптер | Когда используется |
| ---- | ------- | ------------------ |
| `RawDataPort` | `S3RawData` | Всегда — boto3 c `path` или `virtual` addressing |
| `CoverageReader/Store` | `ParquetCoverageStore` | Всегда — гексы покрытия региона на res 7–11 |
| `FeatureReader/Store` | `ParquetFeatureStore` | Всегда — per-hex признаки места (Слой 1) |
| `GoldFeatureReader/Store` | `ParquetGoldFeatureStore` | Всегда — per-hex gold-таблица для синтетики target |
| `NspdSilverStore` | `ParquetNspdSilverStore` | Всегда — отнормализованные NSPD-объекты |
| `ValuationObjectReader/Store` | `ParquetValuationObjectStore` | Всегда — feature matrix объектов |
| `OofPredictionsReader` | `LocalOofPredictionsReader` | Всегда — OOF-предсказания квартета |
| `QuartetModel` | `CatBoostQuartetModel`, `EbmQuartetModel`, `GreyTreeQuartetModel`, `NaiveLinearQuartetModel` | Один экземпляр на модель в квартете (ADR-0016) |
| `ModelRegistry` | `MlflowModelRegistry` | `mlflow_enabled=True` |
| `ModelRegistry` | `LocalModelRegistry` | `mlflow_enabled=False` (дефолт) |
| `ModelLoader` | `MlflowModelLoader` / `LocalModelLoader` | По тому же флагу |
| `RoadGraph` | `NetworkxRoadGraph` | Всегда (по pre-built `silver/road_graph/edges.parquet`) |
| `RegionBoundary` | `LocalGeoJsonRegionBoundary` | Всегда (boundary GeoJSON в `data/raw/regions/`) |

API-роуты тонкие: парсят запрос, вызывают use case, формируют ответ. Никакой бизнес-логики в `routes.py`.

### Feature flags

Включение/выключение компонентов через `Settings` ([src/kadastra/config.py](../src/kadastra/config.py)). При отключённом компоненте его настройки не обязательны. `Container` в [composition_root.py](../src/kadastra/composition_root.py) выбирает адаптеры на основе флагов.

| Флаг / поле | По умолчанию | Что контролирует |
| ----------- | ------------ | ---------------- |
| `auth_token` | `None` | `None` — auth выключен; задан — `BearerAuthMiddleware` ставится поверх всего, кроме `/health`, `/login`, `/logout`, `/favicon.ico` |
| `mlflow_enabled` | `False` | MLflow vs локальный реестр моделей (диск) |
| `pull_data_on_start` | `False` | На старте контейнера фоном пуллить `s3://$S3_BUCKET/Kadatastr/{gold,models,silver}/` в `/app/data` |
| `quartet_parallel_folds` | `True` | Спатиальные фолды квартета фитятся параллельно (joblib) |
| `quartet_skip_final_simplifier_fits` | `True` | Пропускать full-data refit EBM/Grey/Naive (никто их `*_model.pkl` не читает; OOF достаточно) |

## Dependency Injection

Без фреймворка. Constructor injection + composition root.

- Use case получает порты через `__init__`.
- Сборка всех зависимостей — в одном месте ([composition_root.py](../src/kadastra/composition_root.py), класс `Container`).
- FastAPI роутеры создаются фабричными функциями, принимающими use cases.

## Конфигурация

- **pydantic-settings** — все настройки типизированы в [Settings](../src/kadastra/config.py).
- Обязательные поля — для компонентов, которые включены всегда (S3, пути к store'ам, region settings).
- Optional — для компонентов под флаг (MLflow, auth).
- **.env** обязателен для запуска (даже локально).
- **.env.example** — в репозитории, **.env** — в `.gitignore`.

## Type Hints

Обязательны. Проверяются pyright в strict mode. Все функции, методы, аргументы, возвращаемые значения — с типами. На границах модулей DataFrame'ы — типизированные обёртки или явные схемы (контракты Слой1↔Слой2 в `etl/hex_aggregation.py`, например).

## Качество кода — SOLID и best practices

**Главный принцип: писать правильно, а не как проще.**

- Соблюдать SOLID. Никаких хаков, обходных путей и «быстрых» решений.
- Если есть правильный способ и простой способ — выбирать правильный, даже если он сложнее.
- Не срезать углы: типизированные API, явные контракты портов, корректные spatial-паттерны (UTM-перепроекция для площадных операций; haversine только для радиальных дистанций).
- Каждое архитектурное решение должно быть обоснованным, а не «так проще написать». Существенные решения — отдельным ADR в [info/decisions/](decisions/).

## TDD

Строгий TDD: тесты → стабы → реализация → повторять → docs → чистка.

Цикл на каждую единицу функциональности:

1. **Тест + стаб** — тест + порт-протокол + use case-заглушка (`raise NotImplementedError`). Стаб нужен чтобы pyright не блокировал коммит — но не содержит логики.
2. **Коммит** — `test: add failing tests for Foo` — тест падает с `NotImplementedError`.
3. **Реализация** — заменить `raise NotImplementedError` на рабочий код.
4. **Коммит** — `feat: implement Foo`.
5. **Повторять** — следующая единица.
6. **Docs** — обновить `info/`, при существенных решениях — новый ADR в `info/decisions/`.
7. **Чистка** — pyright, pytest — всё зелёное.

**Порядок строго соблюдать.** Стаб ≠ реализация. Реализация никогда не идёт в одном коммите с тестом.

- **Unit-тесты** — бизнес-логика (use cases), domain entities, чистые трансформации на маленьких фикстурах. Fake-порты вместо реальных адаптеров.
- **Integration-тесты** — адаптеры с реальными сервисами (S3, MLflow, Postgres) через testcontainers / docker compose.
- **Data-тесты** — проверка контрактов схем (диапазоны, NaN, типы) на синтетических батчах (см. `tests/unit/test_*_features.py`).

Код без тестов не мержится.

## Git

### Branching

```text
main ← dev-stage ← dev ← feature/xxx
```

- **main** — production. Прямые коммиты запрещены. Мерж только через PR из dev-stage. Push в main → CI (lint+test). Автодеплой prod ещё не подключён — сейчас только dev-stage задеплоен на VM.
- **dev-stage** — staging. Прямые коммиты допускаются для инфра-фиксов; для feature-работы — мерж из dev. **Push в dev-stage → deploy-dev-stage** (rsync + .env + `docker compose up -d --build` на VM).
- **dev** — интеграционная. Прямые коммиты запрещены. Мерж только из feature-веток. Push в dev → CI (lint+test).
- **feature/xxx** — ветки под задачу, создаются от dev. **НЕ удаляются после merge** без явной просьбы.

**Именование веток:** название должно конкретно описывать задачу.

| Плохо | Хорошо |
| ----- | ------ |
| `feature/etl` | `feature/h3-aggregate-transactions` |
| `feature/model` | `feature/catboost-baseline-valuation` |
| `feature/api` | `feature/predict-endpoint` |
| `feature/map` | `feature/hex-tile-viewport-endpoint` |

Ветка = одна конкретная задача. Если название можно применить к десятку разных задач — оно слишком абстрактное.

**Одна ветка = фича целиком.** Тесты, код, обновления `info/` и ADR — всё в одной feature-ветке.

**Коммитить постоянно.** Написал файл → сразу коммит. Не писать 2+ файла перед коммитом. Не стэшить и откладывать.

### Commits — Angular-стиль

Формат: `type(optional scope): short summary`

```text
feat: add H3 aggregation for transactions
fix: handle empty hex cells in valuation predict
perf: switch ETL transform from pandas to polars
build: pin catboost version
ci: add pytest stage
docs: describe ETL layer contracts
style: ruff format
refactor: extract FeatureStorePort
test: add unit tests for hex aggregator
chore: update ruff config
```

| Тип | Назначение | Версия (PSR) |
| --- | ---------- | ------------ |
| `feat` | Новая функциональность | MINOR |
| `fix` | Исправление бага | PATCH |
| `perf` | Оптимизация | PATCH |
| `build` / `ci` / `docs` / `style` / `refactor` / `test` / `chore` | — | — |

Breaking changes: `feat!:` или `BREAKING CHANGE:` в footer → MAJOR.

Версия хранится в `pyproject.toml`, бампится автоматически при мерже в main (Python Semantic Release).

### Pre-commit hooks

Блокируют коммит если не прошли:

- ruff (lint + format)
- pyright (type check)
- commitizen (формат коммита)

## CI/CD

GitHub Actions, GitHub-hosted runner (`ubuntu-latest`). Workflow: [.github/workflows/ci-cd.yml](../.github/workflows/ci-cd.yml).

### Триггеры

| Событие | CI (lint + test) | Deploy |
| ------- | ---------------- | ------ |
| push в `dev` | да | — |
| push в `dev-stage` | да | **deploy-dev-stage** |
| push в `main` | да | — (prod не подключён) |
| PR в `dev` / `dev-stage` / `main` | да | — |

> Feature-ветки сами CI не триггерят. Локальные проверки — через pre-commit hooks.

### Pipeline деплоя

```text
feature/xxx → merge в dev → push в dev-stage → CI (lint+test) → deploy-dev-stage
```

Сейчас feature-ветки чаще мержат напрямую в `dev-stage` для интеграционной проверки на VM, потом — в `main` через PR. После подключения prod-деплоя поток будет строже: feature → dev → PR в dev-stage → PR в main.

### Деплой механизм (deploy-dev-stage)

```text
checkout → SSH setup → rsync → generate .env on VM → docker compose up -d --build → healthcheck /health
```

1. **SSH setup** — `DEV_STAGE_SSH_KEY` записывается в `~/.ssh/deploy_key`, `ssh-keyscan` в `known_hosts`.
2. **rsync** — синхронизация исходников на VM (исключаются `.git`, `.venv`, кэши, `mlruns`, `site`, `.env`). `data/` НЕ исключается целиком: внутри есть whitelist'нутый region-boundary GeoJSON, нужный для билда образа.
3. **Генерация `.env`** — собирается из GitHub Variables (не секретные) и Secrets (секретные) heredoc'ом на VM.
4. **`docker compose up -d --build --remove-orphans`** на `docker-compose.dev-stage.yml` (project name `kadastra-dev-stage`).
5. **Healthcheck** — 30 попыток с 5-секундным интервалом до `http://localhost:$DEV_STAGE_INTERNAL_PORT/health`. При фейле — выгружает `docker compose logs --tail=200` в stderr GHA.

### Окружения

| Параметр | Local | Dev-Stage | Production |
| -------- | ----- | --------- | ---------- |
| Хост | `127.0.0.1` | внутренняя сеть | TBD |
| Compose | `docker-compose.yml` | `docker-compose.dev-stage.yml` (project `kadastra-dev-stage`) | TBD |
| Том `data/` | bind mount хоста | named volume `kadastra_data` | — |
| Cold-start data sync | вручную | `PULL_DATA_ON_START=true` (фоновый рsync с S3) | — |
| Auth | `auth_token=None` (выключен) | `DEV_STAGE_AUTH_TOKEN` | — |
| MLflow | opt-in (`docker-compose.mlflow.yml`) | выключен (`MLFLOW_ENABLED=false`) | — |

### GitHub Variables и Secrets

Префикс окружения — `DEV_STAGE_` (после подключения prod добавится `PROD_`).

**Variables** (не секретные):
`DEPLOY_USER`, `DEV_STAGE_HOST`, `DEV_STAGE_PORT` (SSH), `DEV_STAGE_DEPLOY_PATH`, `DEV_STAGE_INTERNAL_PORT`, `DEV_STAGE_PULL_DATA_ON_START`, `DEV_STAGE_S3_ENDPOINT_URL`, `DEV_STAGE_S3_BUCKET`, `DEV_STAGE_S3_REGION`, `DEV_STAGE_S3_ADDRESSING_STYLE`.

**Secrets**: `DEV_STAGE_SSH_KEY`, `DEV_STAGE_S3_ACCESS_KEY`, `DEV_STAGE_S3_SECRET_KEY`, `DEV_STAGE_AUTH_TOKEN`.

## Docker

| Файл | Назначение |
| ---- | ---------- |
| [Dockerfile](../Dockerfile) | Multi-stage: `base` (Python 3.13-slim + system geo-libs: GDAL, GEOS, PROJ, spatialindex, osmium-tool) → `deps` (uv sync без проектного кода) → `runtime` (исходники + entrypoint) → `scripts` (тот же образ, но `ENTRYPOINT=uv run python` для запуска одиночных скриптов через профиль) |
| [docker-compose.yml](../docker-compose.yml) | Локальная разработка. `./data` — bind mount, изменения parquet'ов / моделей видны без ребилда. Профиль `scripts` для разовых ETL-скриптов |
| [docker-compose.dev-stage.yml](../docker-compose.dev-stage.yml) | Dev-Stage VM. `kadastra_data` — named volume (переживает `compose down`). Healthcheck на `/health` (30s/5s/5retries) |
| [docker-compose.mlflow.yml](../docker-compose.mlflow.yml) | Опциональный стек: `mlflow-postgres` (бэкенд-стор) + MLflow tracking server. Артефакты — в `s3://$S3_BUCKET/mlflow-artifacts/`. Поднимается отдельной командой |
| [entrypoint.sh](../entrypoint.sh) | На старте опционально (`PULL_DATA_ON_START=true`) фоном пуллит `s3://$S3_BUCKET/Kadatastr/{gold,models,silver}/` в `/app/data` (раскладка с восстановлением `gold/`/`models/`/`silver/` префиксов), затем — `scripts/serve.py`. Pull идёт фоном, чтобы `/health` отвечал в первые ~150 секунд healthcheck'а CI |

## Аутентификация

`BearerAuthMiddleware` ([src/kadastra/api/auth.py](../src/kadastra/api/auth.py)) ставится поверх всего приложения, если задан `auth_token` в `Settings`.

- **Bearer**: `Authorization: Bearer <token>` для API-клиентов.
- **Cookie**: после `/login` ставится cookie с тем же токеном — браузер ходит без заголовков.
- **Public-endpoints** (без проверки): `/health`, `/login`, `/logout`, `/favicon.ico`. Запрос на закрытый ресурс без cookie/header — редирект `302 → /login`.
- **Локально**: `auth_token=None` (дефолт) → middleware не подключается, никакой авторизации не нужно.
- **Dev-Stage**: токен из `DEV_STAGE_AUTH_TOKEN` инжектируется в `.env` при деплое.

## Документация

- **`info/`** — внутренняя база знаний (этот файл, [project.md](project.md), [grid-rationale.md](grid-rationale.md), [h3-primer.md](h3-primer.md), [hex-feature-layers.md](hex-feature-layers.md), [data-sources.md](data-sources.md), [nspd-api.md](nspd-api.md)).
- **`info/decisions/`** — Architectural Decision Records (ADR), нумерованные `0001+`. Что решили, почему, какие альтернативы рассматривали. На текущий момент 0001–0025; новые добавляются для существенных архитектурных шагов.
- **MkDocs** — пока не подключён (нет `mkdocs.yml`/`docs/`). Если когда-то понадобится наружная документация — поднимаем mkdocs Material с mkdocstrings.

---

## Рабочий процесс

### 0. Первоначальная настройка (один раз на свежий клон)

1. `uv sync`
2. `uv run pre-commit install`
3. `uv run pre-commit install --hook-type commit-msg`
4. `cp .env.example .env` — заполнить S3-ключи (минимум для локальной разработки)
5. (опц.) `docker compose -f docker-compose.mlflow.yml up -d` — поднять MLflow, если нужен
6. Проверить: `uv run ruff check .`, `uv run pyright .`, `uv run pytest`

### 1. Взять задачу

1. `git checkout dev && git pull`
2. `git checkout -b feature/описание-задачи`

### 2. Цикл разработки (повторять пока задача не готова)

**2a.** Написать тест

1. Создать/открыть файл в `tests/unit/` или `tests/integration/`.
2. Написать тест на ожидаемое поведение.
3. `uv run pytest tests/unit/test_xxx.py` — убедиться что тест падает.

**2b.** Написать код

1. Реализация чтобы тест прошёл.
2. Новая зависимость: `uv add пакет`.
3. Существенное архитектурное решение → добавить ADR в `info/decisions/`.

**2c.** Проверить всё

1. `uv run pytest`
2. `uv run ruff check . --fix`
3. `uv run ruff format .`
4. `uv run pyright .`
5. Что-то упало? → исправить → снова с 2c.1.

**2d.** Закоммитить

1. `git add <конкретные файлы>` — **не** `git add .`.
2. `git commit -m "type(scope): описание"`.
3. Хук упал → исправить → `git add` → **новый** `git commit` (не `--amend`).
4. К следующему тесту.

### 3. Завершить задачу

1. `uv run pytest` — все тесты проходят.
2. **Ждать подтверждения** перед мержем.
3. `git checkout dev && git pull && git merge feature/xxx --no-ff && git push`.

### 4. Релиз

```text
dev → merge в dev-stage → push в dev-stage → CI → deploy-dev-stage → проверка на VM
                        ↓
                      (после prod-подключения) PR dev-stage → main → deploy-prod
```

1. Слить feature-ветку в `dev` (через PR или локальный `merge --no-ff`).
2. Влить `dev → dev-stage`, запушить — CI прогонит lint+test, при успехе deploy-dev-stage задеплоит на VM.
3. Проверить вручную на dev-stage.
4. (когда подключим prod) PR `dev-stage → main` — deploy-prod бампит версию через PSR, генерирует tag.

---

## Справочник команд

### Линтеры

- `uv run ruff check .` — проверка
- `uv run ruff check . --fix` — автоисправление
- `uv run ruff format .` — форматирование
- `uv run pyright .` — типы

### Pre-commit

- `uv run pre-commit run --all-files`

### Зависимости

- `uv add пакет` — основная (с версией)
- `uv add --group dev пакет` — dev
- Коммитить и `pyproject.toml`, и `uv.lock`

### Docker (команды)

- `docker compose up -d` — локальное приложение
- `docker compose --profile scripts run scripts <script.py>` — одноразовый ETL-скрипт
- `docker compose -f docker-compose.mlflow.yml up -d` — MLflow стек
- `docker compose -f docker-compose.dev-stage.yml up -d --build` — dev-stage стек локально (отладка деплоя)
- `docker compose down` — остановить

### CI

- `gh run list --branch dev-stage --limit 5` — последние раны на dev-stage
- `gh run watch <run_id> --exit-status` — следить за раном до завершения

---

## Чего НЕ делать

- Не коммитить напрямую в `main` (защищён). В `dev-stage` — только инфра-фиксы, для feature-работы мерж из `dev`.
- Не использовать `git add .` или `git add -A`.
- Не добавлять Co-Authored-By в коммиты.
- Не удалять feature-ветки после merge (`git branch -d` запрещён без явной просьбы).
- Не пропускать хуки (`--no-verify`).
- Не добавлять пакеты без версий в `pyproject.toml` (только через `uv add`).
- Не мержить feature-ветку в dev без явного подтверждения.
- Не вносить изменения в файлы пока идёт обсуждение — ждать явного подтверждения.
- Не пушить в `dev-stage` без необходимости — каждый push триггерит реальный деплой на VM с rebuild и `--remove-orphans`.

Доменно-специфичные «нельзя» — в [project.md](project.md).

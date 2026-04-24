# Правила и процесс разработки

`info/` — общая база знаний для разработчика и AI-ассистента. Не проектная документация, а внутренние договорённости: правила, архитектурные принципы, процесс.

Контекст проекта, доменный стек и архитектура — в [project.md](project.md). Здесь — общее по разработке.

## Python

Версия: **3.13**.

## Базовый стек

| Инструмент | Назначение |
|------------|------------|
| **uv** | Пакетный менеджер, виртуальное окружение |
| **ruff** | Линтер + форматтер (настройки в pyproject.toml) |
| **pyright** | Type checking (strict mode) |
| **pytest** | Тесты (pytest-asyncio при появлении async кода) |
| **pre-commit** | Хуки перед коммитом |
| **structlog** | Structured logging (JSON в проде, pretty в dev) |
| **pydantic-settings** | Типизированная конфигурация из .env |
| **MkDocs** | Документация (Material theme, mkdocstrings) |

Доменный стек (геопространство, ML, сервинг) — в [project.md](project.md).

## Архитектура — Hexagonal (Ports & Adapters)

Изоляция домена и use case'ов от инфраструктуры через порты и адаптеры.

Зависимости только внутрь: `adapters → ports ← usecases ← api / web`.

- **Use case** знает про **Port** (интерфейс), не про **Adapter** (реализацию).
- **Adapter** реализует Port и инкапсулирует работу с внешним сервисом (S3, БД, MLflow и т.п.).
- **API/Web роуты** — тонкие: парсят запрос, вызывают use case, формируют ответ.

Конкретное наполнение портов и адаптеров под kadastra — в [project.md](project.md).

### Feature flags

Включение/выключение компонентов через `Settings` (pydantic-settings).
При отключённом компоненте его настройки не обязательны (Optional).
`Container` в `composition_root.py` выбирает адаптеры на основе флагов.
Дефолт флага — `True` для backward compatibility, `False` для новых опциональных интеграций.

## Dependency Injection

Без фреймворка. Constructor injection + composition root.

- Use case получает порты через `__init__`.
- Сборка всех зависимостей — в одном месте (`composition_root.py`, класс `Container`).
- FastAPI роутеры создаются фабричными функциями, принимающими use cases.

## Конфигурация

- **pydantic-settings** — все настройки типизированы.
- Обязательные поля — для компонентов, которые включены всегда.
- Optional — для компонентов под feature flag (обязательны когда флаг `=True`).
- **.env** обязателен для запуска (даже локально).
- **.env.example** — в репозитории, **.env** — в `.gitignore`.

## Type Hints

Обязательны. Проверяются pyright в strict mode.
Все функции, методы, аргументы, возвращаемые значения — с типами.
Для DataFrame'ов на границах модулей — типизация через схемы (pandera) или явные dataclass-обёртки.

## Качество кода — SOLID и best practices

**Главный принцип: писать правильно, а не как проще.**

- Соблюдать SOLID. Никаких хаков, обходных путей и «быстрых» решений.
- Если есть правильный способ и простой способ — выбирать правильный, даже если он сложнее.
- Не срезать углы: типизированные API, явные контракты портов, корректные async/spatial-паттерны.
- Каждое архитектурное решение должно быть обоснованным, а не «так проще написать».

## TDD

Строгий TDD: тесты → стабы → реализация → повторять → docs → чистка.

Цикл на каждую единицу функциональности:

1. **Тест + стаб** — тест + порт-протокол + use case-заглушка (`raise NotImplementedError`). Стаб нужен чтобы pyright не блокировал коммит — но не содержит логики.
2. **Коммит** — `test: add failing tests for Foo` — тест падает с `NotImplementedError`.
3. **Реализация** — заменить `raise NotImplementedError` на рабочий код.
4. **Коммит** — `feat: implement Foo`.
5. **Повторять** — следующая единица.
6. **Docs** — обновить `docs/`, `info/`.
7. **Чистка** — pyright, pytest, mkdocs build — всё зелёное.

**Порядок строго соблюдать.** Стаб ≠ реализация. Реализация никогда не идёт в одном коммите с тестом.

- **Unit-тесты** — бизнес-логика (use cases), domain entities, чистые трансформации на маленьких фикстурах. Fake-порты вместо реальных адаптеров.
- **Integration-тесты** — адаптеры с реальными сервисами через testcontainers / docker compose.
- **Data-тесты** — проверка контрактов схем (диапазоны, NaN, типы) на синтетических батчах.

Код без тестов не мержится.

## Git

### Branching

```text
main ← stage ← dev ← feature/xxx
```

- **main** — production. Прямые коммиты запрещены. Мерж только через PR из stage.
- **stage** — staging. Прямые коммиты запрещены. Мерж только через PR из dev.
- **dev** — интеграционная. Прямые коммиты запрещены. Мерж только из feature-веток.
- **feature/xxx** — ветки под задачу, создаются от dev. **НЕ удаляются после merge** без явной просьбы.

> На MVP пока живём в `main` локально. Полный flow поднимаем когда появится удалёнка и CI.

**Именование веток:** название должно конкретно описывать задачу.

| Плохо | Хорошо |
| ----- | ------ |
| `feature/etl` | `feature/h3-aggregate-transactions` |
| `feature/model` | `feature/catboost-baseline-valuation` |
| `feature/api` | `feature/predict-endpoint` |
| `feature/map` | `feature/hex-tile-viewport-endpoint` |

Ветка = одна конкретная задача. Если название можно применить к десятку разных задач — оно слишком абстрактное.

**Одна ветка = фича целиком.** Тесты, код, обновления docs/ и info/ — всё в одной feature-ветке.

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

> **TBD** — пока нет удалённого репо и сервера. Поднимаем когда будут готовы окружения.

Ориентир: GitHub Actions, push в `dev`/`stage`/`main` → CI; push в `stage`/`main` → деплой по rsync + `docker compose up -d --build`. `.env` генерируется на сервере из GitHub Variables (не секретные) и Secrets (секретные).

## Docker

> **TBD** — поднимаем когда появится сервинг или нужна локальная инфра (MinIO, PostGIS, MLflow).

Ожидаемый набор:

- `Dockerfile` — multi-stage (base → deps → runtime), uv.
- `docker-compose.yml` — локальная dev-инфра, опциональные сервисы через profiles.
- `docker-compose.prod.yml` — staging/prod (внешние сервисы из `.env`).
- `docker-compose.test.yml` — интеграционные тесты (tmpfs, отдельные порты).

## Документация

MkDocs + Material for MkDocs.

- Проектная документация в `docs/`.
- Автогенерация API из docstrings (mkdocstrings).
- Архитектурные решения — в `info/decisions/` (ADR-стиль): что решили, почему, какие альтернативы рассматривали.

---

## Рабочий процесс

### 0. Первоначальная настройка (один раз на свежий клон)

1. `uv sync`
2. `uv run pre-commit install`
3. `uv run pre-commit install --hook-type commit-msg`
4. `cp .env.example .env` — заполнить (когда появится)
5. (опц.) `docker compose up -d` — поднять инфраструктуру
6. Проверить: `uv run ruff check .`, `uv run pyright .`, `uv run pytest`

### 1. Взять задачу

1. `git checkout dev && git pull` (когда появится dev)
2. `git checkout -b feature/описание-задачи`

### 2. Цикл разработки (повторять пока задача не готова)

**2a.** Написать тест

1. Создать/открыть файл в `tests/unit/` или `tests/integration/`.
2. Написать тест на ожидаемое поведение.
3. `uv run pytest tests/unit/test_xxx.py` — убедиться что тест падает.

**2b.** Написать код

1. Реализация чтобы тест прошёл.
2. Новая зависимость: `uv add пакет`.

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
2. `uv run mkdocs build --strict` — документация собирается.
3. **Ждать подтверждения** перед мержем.
4. `git checkout dev && git pull && git merge feature/xxx --no-ff && git push`.

### 4. Релиз

```text
dev → PR → stage (deploy-stage) → PR → main (deploy-prod)
```

(когда появится CI/CD)

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

### MkDocs

- `uv run mkdocs serve` — предпросмотр на <http://127.0.0.1:8000>
- `uv run mkdocs build --strict`

### Docker (когда появится)

- `docker compose up -d` — минимальный режим
- `docker compose --profile postgis --profile mlflow up -d` — полная dev-инфраструктура
- `docker compose -f docker-compose.test.yml up -d` — тестовая инфраструктура
- `docker compose down`

---

## Чего НЕ делать

- Не коммитить напрямую в main, stage или dev (когда появятся — защищены).
- Не использовать `git add .` или `git add -A`.
- Не добавлять Co-Authored-By в коммиты.
- Не удалять feature-ветки после merge (`git branch -d` запрещён без явной просьбы).
- Не пропускать хуки (`--no-verify`).
- Не добавлять пакеты без версий в pyproject.toml (только через `uv add`).
- Не мержить feature-ветку в dev без явного подтверждения.
- Не вносить изменения в файлы пока идёт обсуждение — ждать явного подтверждения.

Доменно-специфичные «нельзя» — в [project.md](project.md).

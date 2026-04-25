# ADR-0005: Baseline-модель CatBoost + Spatial K-fold CV + двухдорожечный реестр

**Статус:** Accepted
**Дата:** 2026-04-25

## Контекст

Есть синтетический таргет ([ADR-0004](0004-synthetic-target.md)) и gold-фичи на res=8 (Татарстан, 85 674 ячейки, 10 числовых фичей). Нужна базовая модель и понятный путь к продакшну: воспроизводимый train с метриками и сохранением артефакта.

Решения, которые надо зафиксировать:

1. **Какой алгоритм baseline.** CatBoost / LightGBM / линейная регрессия / kNN.
2. **Как валидировать.** Случайный train/test, K-fold или пространственный K-fold.
3. **Как хранить запуски и модели.** MLflow (см. [ADR-0003](0003-mlflow-docker.md)) или файловый fallback.

## Решения

### 1. Алгоритм: CatBoost

- Хорош «из коробки» на табличных данных без ручной нормализации.
- Поддерживает категориальные признаки нативно (пригодится когда добавим `municipality_id`, `district_id` и т.п.).
- API стабильный, экспорт `.cbm` компактный.
- LightGBM — альтернатива с близким качеством, но CatBoost фиксируем как дефолт; смена — через новый ADR.

Параметры baseline (см. [Settings](../../src/kadastra/config.py)):

| Параметр | Дефолт | Зачем |
|----------|--------|-------|
| `catboost_iterations` | 500 | Достаточно для baseline; на полном датасете ~2-3 сек CPU |
| `catboost_learning_rate` | 0.05 | Стабильно на регрессии без раннего стопа |
| `catboost_depth` | 6 | Стандарт для табличных данных |
| `catboost_seed` | 42 | Воспроизводимость |

### 2. Валидация: Spatial K-fold по родителям res=6

Согласно [project.md](../project.md), запрещён случайный train/test split на пространственных данных — соседние гексы делят рыночный контекст, и random K-fold даёт оптимистичную метрику (утечка через соседство).

`spatial_kfold_split` ([src/kadastra/ml/spatial_kfold.py](../../src/kadastra/ml/spatial_kfold.py)) группирует гексы по `cell_to_parent` и кладёт каждого родителя строго в одну фолд-партицию. Тесты гарантируют:

- val-родители не пересекаются с train-родителями;
- каждый индекс попадает ровно в одну val-партицию;
- детерминизм по seed.

Дефолты:

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| `train_n_splits` | 5 | Стандарт для табличной регрессии |
| `train_parent_resolution` | 6 | Один res=6 ≈ 36 км² ≈ район/группа районов; крупнее res=7 (5 км²) → меньше утечки через соседство в Казани |

### 3. Реестр: двухдорожечный

- **Dev/local (default):** `LocalModelRegistry` ([adapters/local_model_registry.py](../../src/kadastra/adapters/local_model_registry.py)) — пишет `params.json`, `metrics.json`, `model.cbm` в `data/models/{run_name}_{utc_timestamp}/`. Никаких внешних зависимостей.
- **Prod/team:** `MLflowModelRegistry` ([adapters/mlflow_model_registry.py](../../src/kadastra/adapters/mlflow_model_registry.py)) — `mlflow.start_run + log_params + log_metrics + log_model`. Tracking URI и эксперимент берутся из Settings.

Контейнер выбирает реализацию по `mlflow_enabled` в `Settings`. Тесты MLflow-адаптера используют file-backend (`file:tmp_path/mlruns`) — не нужен docker-стек для unit-тестов.

## Sanity check (zoom-out)

Запуск на реальном Татарстане (res=8, 85 674 ячейки, 10 фичей, 5-fold spatial CV):

```text
mean_mae  = 3 012 ₽/м²
mean_rmse = 3 910 ₽/м²
mean_mape = 2.88 (288%)
```

Контекст таргета (см. [ADR-0004](0004-synthetic-target.md)): mean=6 568, median=2 690, max=177 479. MAE ~46% от mean — для baseline без явной координаты центра города это разумно. Высокий MAPE — ожидаем: на периферии target ≈ 0, и любая малая абсолютная ошибка даёт огромный относительный процент.

Важно: `kazan_distance_km` лежит в target-таблице, но НЕ в gold-фичах, поэтому модель не выучивает таргет тривиально. Если бы добавили — MAE упал бы ~до 0 (формула target явно зависит от distance). Это сделано намеренно: baseline должен учиться предсказывать стоимость через косвенные признаки (метро, плотность застройки, дороги), как и реальная модель когда-нибудь.

## Последствия

- В Settings: `model_registry_path`, `catboost_*`, `train_n_splits`, `train_parent_resolution`. Все с дефолтами — пилот стартует без `.env`-настроек тренировки.
- `Container.build_train_valuation_model()` — точка входа из CLI и тестов.
- CLI: [scripts/train_valuation_model.py](../../scripts/train_valuation_model.py).
- Артефакты в `data/models/` (gitignore). При `mlflow_enabled=True` — в S3 через MLflow.

## Открытые вопросы

- Раннее стоп (`early_stopping_rounds`) — не включён в baseline, чтобы избежать неявной валидации внутри fold. Решить отдельно при гиперпараметрической оптимизации.
- На сельских гексах target ≈ 0 → стоит ли учить отдельно «экстенсивную» модель (zero-vs-positive) и «интенсивную» (only positive) → откладываем до реального таргета.
- SHAP / feature importance — на этом этапе не делаем, добавим когда будут реальные деньги.
- Регистрация модели в production-stage — не сейчас (одна модель, одна стадия).

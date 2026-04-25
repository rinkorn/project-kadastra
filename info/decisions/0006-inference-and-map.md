# ADR-0006: Инференс модели в parquet-снимок и отображение на карте

**Статус:** Accepted
**Дата:** 2026-04-25

## Контекст

Есть обученная baseline-модель ([ADR-0005](0005-baseline-training.md)). Нужно отобразить её предсказания на той же карте, где уже видны фичи из gold ([project.md](../project.md), web/map). Решения, которые надо зафиксировать:

1. **Где хранить предсказания.** Inline в gold или отдельная таблица.
2. **Как сервить.** Тот же `/api/hex_features` или отдельный эндпоинт.
3. **Как выбирать модель для инференса.** Жёстко прибитый run_id, последняя по prefix или production-стадия в MLflow.

## Решения

### 1. Отдельная parquet-таблица предсказаний

```text
data/gold/predictions/region={code}/resolution={r}/data.parquet
columns: h3_index, resolution, predicted_value
```

Хранится через `ParquetGoldFeatureStore` (тот же адаптер, другой `base_path` из `predictions_store_path` в Settings).

Почему отдельно, а не в gold:

- gold пересобирается редко (новый источник данных), predictions — после каждой переучённой модели; разные lifecycle.
- удобно сравнивать несколько моделей: `predictions_v1/`, `predictions_v2/` без перезаписи gold.
- API/карта читают прозрачно через ту же абстракцию (`GoldFeatureReaderPort`).

### 2. Тот же эндпоинт `/api/hex_features?feature=predicted_value`

`GetHexFeatures` принимает второй опциональный `prediction_reader: GoldFeatureReaderPort | None`. Когда `feature == "predicted_value"`, читает из него; иначе — из gold-reader. Если `prediction_reader is None` и запросили `predicted_value` → `KeyError`.

`predicted_value` добавлен **первым** в `DEFAULT_FEATURES` ([web/routes.py](../../src/kadastra/web/routes.py)), чтобы карта стартовала на нём.

Альтернатива (отдельный `/api/predictions` эндпоинт) отклонена: рендер карты — это «выбрать колонку и раскрасить», стоимость лишнего эндпоинта = ноль выгоды.

### 3. Выбор модели: latest by run_name_prefix, override через env

`InferValuation` имеет параметр `run_name_prefix: str = "catboost-baseline-res"`. На запуске:

- если передан `run_id` явно (env `INFER_RUN_ID` в CLI) — используется он;
- иначе зовётся `model_loader.find_latest_run_id(f"{prefix}{resolution}")`.

`ModelLoaderPort` ([ports/model_loader.py](../../src/kadastra/ports/model_loader.py)) — два метода:

```python
def load(self, run_id: str) -> CatBoostRegressor: ...
def find_latest_run_id(self, run_name_prefix: str) -> str: ...
```

Реализации:

- `LocalModelLoader` — сканирует `model_registry_path`, использует лексикографический порядок (timestamp-суффикс ISO-8601 → лексикографический порядок == хронологический).
- `MLflowModelLoader` — `MlflowClient.search_runs(filter_string="tags.mlflow.runName LIKE 'prefix%'", order_by=["start_time DESC"], max_results=1)`.

Production-стадии (`staging`/`production`) пока не используем — одна модель, нет промоушена. Перейдём, когда появятся 2+ моделей.

## Sanity check (zoom-out)

```text
rows=85674
predicted_value:
  mean   ≈ 6 569 ₽/м²    (target mean = 6 568)
  median ≈ 2 403 ₽/м²    (target median = 2 690)
  max    ≈ 173 363 ₽/м²  (target max = 177 479)
  min    ≈ 1 755 ₽/м²    (target min = 0; модель не может предсказать 0 без знания координаты)
```

Топ-5 предсказанных гексов совпадают с топ-5 по таргету (центр Казани). Карта на http://127.0.0.1:8000/ показывает predicted_value по умолчанию: тёмно-красный кластер на Казани, ровный жёлтый по периферии.

API контракт-проверка:

```sh
curl -s 'http://127.0.0.1:8000/api/hex_features?resolution=8&feature=predicted_value'
# → {"region": "RU-TA", "resolution": 8, "feature": "predicted_value", "data": [{"hex": "...", "value": 1861.6}, ...]}
```

## Последствия

- В Settings: `predictions_store_path: Path = Path("data/gold/predictions")`.
- Container: `build_model_loader()`, `build_infer_valuation()`, `build_get_hex_features()` теперь передаёт prediction_reader.
- CLI: [scripts/infer_predictions.py](../../scripts/infer_predictions.py) — env `INFER_RUN_ID` для override.
- Web map: `predicted_value` в `DEFAULT_FEATURES` первым → карта стартует на нём.

## Открытые вопросы

- Нужно ли фиксировать гарантии "модель совместима с gold-схемой" (snapshot feature_columns в реестре + проверка на инференсе)? Сейчас порядок колонок берётся `[c for c in df.columns if c not in {h3_index, resolution}]` — детерминирован, но при изменении схемы gold молча сломается. Решить отдельно когда добавим версионирование gold.
- Кэш предсказаний для viewport-эндпоинта (если/когда будем отдавать тайлы) — пока не нужно: всё помещается в один JSON, ~85k гексов = ~5 МБ.
- Промоушен `staging`/`production` в MLflow — отложено до момента, когда появятся ≥ 2 моделей.

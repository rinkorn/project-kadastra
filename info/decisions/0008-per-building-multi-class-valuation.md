# ADR-0008: Per-building multi-class valuation поверх hex-пайплайна

**Статус:** Accepted; уточнено в [ADR-0010](0010-methodology-compliance-roadmap.md) — pipeline и доменные сущности этого ADR корректны, но формулировка цели «оценка кадастровой стоимости» поправлена на «модель рынка недвижимости», а текущий target (`synthetic_target_rub_per_m2` ← `cost_index` из ЕГРН по [ADR-0009](0009-real-cadastre-target-via-nspd.md)) считается **interim placeholder** до появления сделочных данных.
**Дата:** 2026-04-25
**Дополняет:** [ADR-0004](0004-synthetic-target.md), [ADR-0005](0005-baseline-training.md), [ADR-0006](0006-inference-and-map.md)

## Контекст

После прогона baseline на res=11 ([ADR-0007](0007-kazan-agglomeration-scope.md)) выяснилось: гексагональная подача врёт визуально — на res=11 (~28 м) одна ячейка часто попадает в пустой двор между домами или на торец здания, и модель честно предсказывает «этой точке низкая стоимость». Глаз видит «дома есть, а оценки нет», ухо слышит «модель сломалась». На самом деле сломалась семантика: гекс — это случайная точка в пространстве, а пользователь хочет оценить **объект** (квартиру, дом, коммерцию).

Дополнительно конечная задача — кадастровая стоимость **квартир, частных домов, коммерческой недвижимости и земельных участков**. Один таргет «₽/м² гекса» не различает класс объекта; модель выучивает усреднённую химеру.

## Решение

**Параллельный per-object пайплайн** поверх существующего hex-стека, не заменяющий его. Hex продолжает работать как region-overview инструмент; объектный — как кадастровая оценка.

### Asset classes (3 из имеющихся данных)

```python
class AssetClass(StrEnum):
    APARTMENT = "apartment"
    HOUSE = "house"
    COMMERCIAL = "commercial"
```

Маппинг OSM `building=*` → `AssetClass`:

| OSM tag | AssetClass | OSM объектов в агломерации |
|---------|------------|----------------------------|
| `apartments` | APARTMENT | 7 214 |
| `house`, `detached`, `semidetached_house`, `terrace`, `bungalow` | HOUSE | 48 758 |
| `retail`, `commercial`, `supermarket`, `kiosk`, `shop`, `office`, `industrial`, `warehouse` | COMMERCIAL | 2 507 |
| `yes`, `garage`, `shed`, `barn`, ... | — (отбрасываем) | ~120 000 |

`LandPlot` (земельные участки) **отложен** — для него нужен парсер ГАР `AS_STEADS` (~10 ГБ XML в S3), это отдельный слайс.

### Доменные сущности

- [`AssetClass`](../../src/kadastra/domain/asset_class.py) — `StrEnum` с тремя членами; единый ключ для маршрутизации (формула таргета, run_name модели, файловая партиция).
- [`ValuationObject`](../../src/kadastra/domain/valuation_object.py) — `frozen + slots` dataclass с валидацией `__post_init__`. Используется для одиночных операций (per-object inference), массовая ETL идёт через polars.
- [`classify_asset_class(tag)`](../../src/kadastra/domain/classify_asset_class.py) — чистая функция, регистронезависимая, обрабатывает `None`.

### ETL transforms (чистые polars-функции)

| Функция | Что делает |
|---------|------------|
| [`assemble_valuation_objects`](../../src/kadastra/etl/valuation_objects.py) | OSM CSV → `(object_id, asset_class, lat, lon, levels, flats)`; через `replace_strict` без Python-цикла на ~180k строк |
| [`compute_object_metro_features`](../../src/kadastra/etl/object_metro_features.py) | `dist_metro_m`, `dist_entrance_m`, `count_stations_1km`, `count_entrances_500m`. Полный haversine-матрица, пустой набор станций → 1e9 м sentinel |
| [`compute_object_road_features`](../../src/kadastra/etl/object_road_features.py) | `road_length_500m`. H3 res=9 bucketing — иначе 58k×100k сегментов не помещаются в RAM. K-ring подобран по радиусу |
| [`compute_object_neighbor_features`](../../src/kadastra/etl/object_neighbor_features.py) | `count_apartments_500m`, `count_houses_500m`, `count_commercial_500m`. Исключает self из кандидатов до haversine-фильтра |
| [`compute_object_synthetic_target`](../../src/kadastra/etl/object_synthetic_target.py) | Per-class `synthetic_target_rub_per_m2`, через `pl.when().then()` без Python-цикла |

### Per-class synthetic target — три формулы

Базовая идея: каждый класс реагирует на разные комбинации соседства.

| Параметр | apartment | house | commercial |
|----------|-----------|-------|------------|
| BASE ₽/м² | 80 000 | 50 000 | 100 000 |
| Метро (затухание) | `1 + 0.6·exp(-d/800)` | `1 + 0.3·exp(-d/5000)` | `1 + 1.0·exp(-d/1500)` |
| Плотность | `1 + 0.05·log1p(apt+comm)` | `1 + 0.04·log1p(houses)` (плюс `crowding` штраф за apartments вокруг) | `1 + 0.1·log1p(apt+houses+comm)` |
| Дороги | `1 + 1e-4 · road_500m` | `1 + 5e-5 · road_500m` | `1 + 2e-4 · road_500m` |
| σ шума | 10 000 | 8 000 | 15 000 |

Финал клипается к `≥ 0`. Параметры — синтетические, не калиброванные; пилот всё ещё synthetic proxy.

### Storage и порты

- Партиционирование: `data/gold/valuation_objects/region={code}/asset_class={class}/data.parquet` (в одной партиции живут и featured-объекты, и target — каждый шаг переписывает файл с дополнительной колонкой).
- Предсказания: отдельный путь `data/gold/object_predictions/...` с тонкой схемой `(object_id, asset_class, lat, lon, predicted_value)`.
- Порты: [`ValuationObjectReaderPort`](../../src/kadastra/ports/valuation_object_reader.py), [`ValuationObjectStorePort`](../../src/kadastra/ports/valuation_object_store.py). Адаптер один — [`ParquetValuationObjectStore`](../../src/kadastra/adapters/parquet_valuation_object_store.py) — реализует оба.

### Use cases

```text
BuildValuationObjects        OSM CSV → per-class партиции
BuildObjectFeatures          объединить классы → metro/road/neighbor → разделить → сохранить
BuildObjectSyntheticTarget   per-class формула → +synthetic_target колонка
TrainObjectValuationModel    spatial K-fold + final fit; run_name=catboost-object-{class}
InferObjectValuation         latest run по классу → predicted_value parquet
GetObjectPredictions         API-friendly список {object_id, lat, lon, value}
```

### Spatial CV для объектов

Объект не имеет h3-индекса как «родного» ключа. CV получает h3 из lat/lon при `cell_resolution = max(parent_resolution+1, 10)`, группирует по `parent_resolution=7` (как для hex). lat/lon **исключены из feature_columns** — иначе модель восстановит kazan-distance, что мы явно убрали из синтетического таргета. Это даёт честные метрики: модель учится через признаки, а не через координаты.

### Map UI

Селектор **Mode**:
- `Hex feature` — поведение из ADR-0006 (`H3HexagonLayer` поверх `/api/hex_features`).
- `Objects: apartment | house | commercial` — `ScatterplotLayer` поверх `/api/object_predictions?asset_class=…`, точка радиусом 18 м, цвет по `predicted_value` через ту же log/p98/linear color-ramp.

API: `GET /api/object_predictions?asset_class={apartment|house|commercial}` → `{region, asset_class, data: [{object_id, lat, lon, value}]}`. Неизвестный класс → 400, отсутствующая партиция → 404 (то же поведение, что у `/api/hex_features`).

## Sanity check (zoom-out)

Прогон на агломерации (synthetic_target_seed=42, catboost iters=500, lr=0.05, depth=6, n_splits=5, parent_res=7):

| Класс | n объектов | predicted ₽/м² min/median/max | mean MAE | mean MAPE |
|-------|------------|--------------------------------|----------|-----------|
| apartment | 7 214 | 79 034 / 135 015 / 274 711 | 8 327 | 6.6% |
| house | 48 758 | 50 292 / 64 216 / 100 018 | 6 420 | 10.1% |
| commercial | 2 507 | 98 185 / 309 282 / 731 278 | 14 340 | 5.4% |

Числа правдоподобные:
- Apartment median 135к ₽/м² — реальные новостройки центральной Казани сегодня в этом коридоре.
- House < apartment, commercial > apartment — соответствует и формуле, и интуиции.
- MAPE везде 5–10% — модель уверенно ловит свои синтетические закономерности; на реальном таргете будет хуже.

API-чек:

```sh
curl 'http://127.0.0.1:15777/api/object_predictions?asset_class=apartment'
# → {"region": "RU-KAZAN-AGG", "asset_class": "apartment", "data": [
#     {"object_id": "way/103502712", "lat": 55.83, "lon": 48.67, "value": 114722.41}, ...]}
```

## Последствия

- В [Settings](../../src/kadastra/config.py) добавлены: `valuation_object_store_path`, `object_predictions_store_path`, `object_neighbor_radius_m=500`, `object_road_radius_m=500`.
- Container получил пять новых билдеров (`build_valuation_objects`, `build_object_features`, `build_object_synthetic_target`, `build_train_object_valuation_model`, `build_infer_object_valuation`) + `build_get_object_predictions` для API.
- 5 CLI-скриптов в `scripts/` зеркалят имена hex-пайплайна.
- Hex-пайплайн остаётся живым и не удалён — он полезен как region-overview и как сравнительный baseline.

## Открытые вопросы

- **`LandPlot` через ГАР `AS_STEADS`** — следующий слайс. Нужен XML-парсер; кадастр участков — реестровый, а не OSM.
- **Реальный таргет**. Synthetic proxy остаётся; реальные кадастровые/сделочные цены подключатся отдельным слоем `targets_real/`.
- **Площадь объекта**. Сейчас CSV хранит только centroid + `levels`+`flats`, без footprint. Для оценки **полной** стоимости (а не ₽/м²) нужна площадь — она есть в `building` polygon в OSM PBF, но мы при конверсии оставили только centroid. Когда дойдём до полной оценки — пересоберём CSV с дополнительной колонкой `footprint_m2`.
- **Cross-class context vs target snapshotting**. `BuildObjectFeatures` сейчас читает все классы целиком в RAM (~58k объектов). При увеличении до 4 классов с участками (~миллион) понадобится chunked-обработка или PostGIS spatial index.
- **Inference latency**. На 58k объектов CatBoost predict ≈ 100 мс — приемлемо для batch. Per-object online inference (ad-hoc запрос «оцените мой адрес») потребует отдельного online-эндпоинта `POST /api/predict_object` — отложено до момента запроса.

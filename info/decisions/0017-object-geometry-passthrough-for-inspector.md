# ADR-0017: Сквозной пропуск полигональной геометрии объекта silver→gold→API→UI

**Статус:** Accepted
**Дата:** 2026-04-26
**Дополняет:** [ADR-0009](0009-real-cadastre-target-via-nspd.md) (NSPD silver-схема), [ADR-0008](0008-per-building-multi-class-valuation.md) (per-object pipeline + map UI)
**Источник методологии:** [info/grid-rationale.md](../grid-rationale.md), §6 (объектные ЦОФ — геометрия объекта).

## Контекст

В silver-слое NSPD ([ADR-0009](0009-real-cadastre-target-via-nspd.md)) у каждого здания и участка лежит `polygon_wkt_3857: pl.Utf8` — реальная геометрия объекта в EPSG:3857 (web-mercator метры). Однако `AssembleNspdValuationObjects._OUTPUT_SCHEMA` (block 4 после ADR-0015) колонку дропает: gold содержит только `lat/lon` центроид. Дальше по pipeline'у `BuildObjectFeatures` геометрии не видит и видеть не может.

Эффект на UX inspector'а ([ADR-0008](0008-per-building-multi-class-valuation.md), `map.html`):

- Карточка объекта показывает только координату + рассчитанные ЦОФ.
- На карте — точка `ScatterplotLayer`, не полигон.
- Невозможно глазом отличить «крошечный гараж» от «панельной 9-этажки» от «промзоны 50×200 м». А это базовый sanity-check для модели: «куда конкретно пришла оценка ₽/м²».
- Невозможно визуально сравнить два объекта по форме/размеру при перекликивании в side panel.

Эта дыра не методологическая (это не ЦОФ), а **product-уровневая**: для понимания и валидации модели человеком геометрия нужна как часть inspector'а, не как ML-вход.

## Решение

**Прокинуть `polygon_wkt_3857` через всю цепочку silver → gold → API → UI как passthrough-колонку:**

1. **Schema (`AssembleNspdValuationObjects._OUTPUT_SCHEMA`)** — добавить `"polygon_wkt_3857": pl.Utf8`. Полностью passthrough из silver: ни регенерация, ни re-projection в assemble не делается. Силверный формат WKT(3857) сохраняется до API edge.
2. **`BuildObjectFeatures`** — *никаких изменений*. Use case работает на полном DataFrame через `with_columns`/joins, не делает финального `.select(specific_cols)`, поэтому новая колонка едет вместе со всеми остальными автоматически (это явно проверено в коде — `_to_valuation_objects_buildings/landplots` вернёт всё, что в schema; `BuildObjectFeatures.execute` записывает обратно `enriched.filter(...)` без проектирования).
3. **API edge (`/api/inspection/{object_id}`)** — конверсия WKT(3857) → GeoJSON(WGS84) **в момент ответа**, не в gold. Используем `shapely.from_wkt` + `shapely.transform` поверх векторного `pyproj.Transformer.from_crs(3857, 4326, always_xy=True)`. Сырой WKT снимается с ответа; в JSON летит `geometry: {"type": "Polygon", "coordinates": [[[lon, lat], ...]]}` или `null`.
4. **API list (`/api/inspection`)** — геометрию **не отдаём**. `LoadObjectInspection.list_for_map` уже выбирает только 7 колонок (`object_id, lat, lon, y_true, y_pred_oof, residual, fold_id`) — geometry в этот whitelist не попадает. Это сознательно: 290k полигонов × ~500 байт WKT = ~150 МБ payload только на список — недопустимо.
5. **UI (`map.html`)** — при клике по точке/гексу подгружается detail; если в нём есть `geometry`, в state кладётся `selectedGeometry`, и при следующем `setProps` поверх базового слоя (Hex / Scatter) добавляется `deck.PolygonLayer` с этим одним полигоном (амбер-заливка 27 % + амбер-обводка 90 %, lineWidthMinPixels=2). Полигон остаётся видимым при смене mode/scale, до клика по другому объекту.

### Что **не** делаем в этой итерации

- **Геометрические объектные ЦОФ** (площадь полигона, perimeter, orientation относительно дорог, IoU с red lines) — отдельный блок методологии (§6 grid-rationale.md). Сейчас *passthrough only*; полигон **не входит** в `feature_columns`. Введение каждой derived-фичи — TDD-итерация под отдельный ADR с замером эффекта на CatBoost / White Box.
- **Все полигоны на карту сразу.** Render-стоимость PolygonLayer на 290k полигонов в браузере значительная (deck.gl tessellation), а методически добавляет визуального шума — большинство объектов выглядят как точки на zoom < 13. Подход «селект-on-demand» эквивалентен по информации и кратно дешевле.
- **Конверсия в WGS84 в gold.** Держим silver-формат (3857) до API edge. Причина — pyproj-вызов на 290k объектов делается один раз в API-роуте под один объект (один WKT, одно reproject), а не на полный gold-build. Затраты на конверсию амортизируются по запросам.
- **Геометрия в `LoadObjectInspection.list_for_map` payload.** См. п. 4 — даже компактный GeoJSON на 290k объектов раздуется. Если когда-то понадобится «все полигоны в viewport» — сделаем отдельный endpoint с bbox-фильтром и tile-стилем.

## Архитектура

```text
silver/nspd/region={code}/source={buildings|landplots}/data.parquet
  └─ polygon_wkt_3857: Utf8 (EPSG:3857)
            │
            ▼ AssembleNspdValuationObjects (passthrough)
gold/valuation_objects/region={code}/asset_class={class}/data.parquet
  └─ polygon_wkt_3857: Utf8 (EPSG:3857)
            │
            ▼ BuildObjectFeatures (passthrough — не дропает)
gold/valuation_objects/... (overwrite, теперь с feature-колонками + polygon_wkt_3857)
            │
            ▼ /api/inspection/{object_id} (api/routes.py: WKT→GeoJSON конверсия)
JSON: {"data": {..., "geometry": {"type": "Polygon", "coordinates": [...]} | null}}
            │
            ▼ map.html (selectionOverlayLayers + PolygonLayer)
deck.gl renders one PolygonLayer over base layer
```

## Реализация (в этой ветке `feature/object-geometry-passthrough`)

| Слой | Изменение | Тесты |
| --- | --- | --- |
| Schema | `_OUTPUT_SCHEMA["polygon_wkt_3857"] = pl.Utf8` | unit: passthrough для buildings + landplots в gold |
| API | `_convert_wkt_3857_to_geojson_wgs84()` в `routes.py` через `shapely.from_wkt` + `shapely.transform(coords→pyproj.transform)` + `mapping()`; `inspection_detail` снимает сырой WKT и кладёт `geometry`; `inspection_list` остаётся слим | integration: GeoJSON polygon с координатами в Kazan-диапазоне, `null`-graceful, list не несёт geometry |
| UI | `selectedGeometry` в JS state; `selectionOverlayLayers()` возвращает `PolygonLayer` или `[]`; `redrawWithSelection()` пересобирает слои; обновление `renderHex` + `renderObjects`; в side panel — строка `geometry: Polygon (N vertices)` | manual smoke (HTML/JS не покрыты unit-тестами в проекте) |

## Эмпирический эффект

Не методологический ADR — на ML-метрики **влияние нулевое** (полигон не идёт в feature_columns). Эффект — **product-уровневый**: при клике по любому объекту в карте теперь видно его реальную форму и габариты. Это unblock-условие для будущих per-object ЦОФ из §6 grid-rationale (площадь, ориентация, IoU с дорогами) — они смогут читать `polygon_wkt_3857` уже из gold, без re-load silver.

## Открытые вопросы

- **Темпоральная стабильность WKT.** Snapshot NSPD от 2026-04-25 фиксирует геометрии. Как и для `cost_index` ([ADR-0009](0009-real-cadastre-target-via-nspd.md)) и для остальных ЦОФ — refresh-стратегия отдельный ops-вопрос.
- **MultiPolygon в NSPD.** Текущая UI-обвязка предполагает `Polygon`. На датасете Kazan агломерации все известные `polygon_wkt_3857` — простые `POLYGON ((...))`, MultiPolygon не встречен. Если когда-то проявится — `selectionOverlayLayers()` нужно расширить на `MultiPolygon` (deck.gl `PolygonLayer` принимает массив колец, надо разобрать `coordinates` для multi-).
- **Производительность WKT-парсинга в API.** На один detail-запрос: один `shapely.from_wkt` + один `shapely.transform` + один `mapping`. Это секунды на N=1, незаметно. Если ipocard когда-то понадобится batch (например, 100 polygon'ов в одном HTTP-запросе для viewport) — можно vectorize через `shapely.from_wkt` на список + `shapely.transform` на один объект `GeometryCollection`.

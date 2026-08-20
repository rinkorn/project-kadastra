# ADR-0024: Продвинутые road-network ЦОФ

**Статус:** Proposed
**Дата:** 2026-04-26
**Реализует:** [info/grid-rationale.md §7](../grid-rationale.md) (Дистанционные ЦОФ — графовые), [§8](../grid-rationale.md) (Зональные).
**Опирается на:** [ADR-0011](0011-graph-based-distance-features.md) (road graph уже построен, путевые расстояния от метро), [ADR-0019](0019-poi-distances-and-zonal-counts.md) (POI distance pattern).

## Контекст

В [ADR-0011](0011-graph-based-distance-features.md) построен граф пешеходных дорог Татарстана ([data/silver/road_graph/edges.parquet](../../data/silver/road_graph/edges.parquet)) и используется для walking-distance до метро. Сейчас граф **используется для одного применения** — networkx Dijkstra на каждом запросе пешеходной дистанции.

Из того же графа можно вытащить ещё несколько типов сигналов:

1. **Класс ближайшей дороги** — `motorway/trunk/primary/secondary/tertiary/residential/service` из OSM. Сильно влияет на:
   - Шум/трафик-штраф для жилья (apartment/house рядом с motorway дешевле).
   - Видимость/трафик-премию для commercial (магазин у primary road дороже).
   - Доступность для landplot (участок без подъезда дешевле).
2. **Дистанция до road-class N** — параллельно: «5 м до residential» одно, «5 м до motorway» совсем другое.
3. **Walking-isochrone enrichment** — за 15 минут пешком сколько POI/населения доступно. Это «15-min city» концепция.
4. **Centrality** — насколько улица объекта центральна в локальном road-network'е (proxy для пешеходного потока). Самый продвинутый — на первой итерации не делаем.

Граф уже в проекте — добавление этих фич **требует только новых extracts из того же raw**. Новых внешних зависимостей нет.

## Решение

Ввести 3 группы признаков из существующего road graph. Все вычисляются один раз на регион и сохраняются в silver, чтобы по объекту просто LEFT JOIN.

### Группа 1: Класс ближайшей дороги + дистанции по классам

Для каждой OSM-категории `highway`:

| фича | смысл | релевантно для |
| --- | --- | --- |
| `nearest_road_class` (cat) | класс ближайшего edge | все классы |
| `dist_to_motorway_m` | euclidean до ближайшего `motorway`/`trunk` | apartment, house (штраф), commercial (премия для logistic) |
| `dist_to_primary_m` | euclidean до `primary` | apartment, house |
| `dist_to_secondary_m` | до `secondary` | все |
| `dist_to_residential_m` | до `residential` | landplot (подъезд) |
| `dist_to_pedestrian_m` | до `pedestrian`/`footway` | apartment, commercial (пешеходная улица — премия) |

Вычисление: один проход по `edges.parquet`, фильтр по `highway` категориям, `KDTree.query` per-object. O(N + M log M).

### Группа 2: 15-min walking isochrone enrichment

Для каждого объекта строим 15-минутную пешеходную изохрону через `networkx.single_source_dijkstra` от ближайшего road node, веса = distance, отсечение на `15 × 80 м/мин = 1200 м`. В получившемся подграфе считаем:

| фича | смысл |
| --- | --- |
| `iso15_pop_count` | сумма населения в hex-ячейках, пересекающих изохрону (нужен population grid — может быть из EMISS [ADR-0022](#)) |
| `iso15_amenity_count` | сумма всех POI ([ADR-0019](#)) внутри изохроны |
| `iso15_metro_reach` | флаг «достал ли до хотя бы одной станции метро» |

Нужно: `compute_object_isochrone_features` per-object. Дороже, чем Group 1: ~50–200 ms на объект (Dijkstra на ограниченном радиусе быстр), 197k landplot × 100 ms = **5.5 ч**. На первой итерации — кэш на hex-cell (resolution 11) и LEFT JOIN объект → hex → cached isochrone.

### Группа 3: Centrality (отложено)

Локальная betweenness/closeness для node street network — proxy для трафика. Требует пересчёта centrality на уровне всего graph (минуты), но смысл маргинальный. **Не делаем в этой итерации**, оставляем на возможный вкладыш в ADR-0024-extension.

### Применимость по классам

- **apartment**: Group 1 (штраф motorway, премия pedestrian) + Group 2 (15-min reach как «премиум-стиль жизни»).
- **house**: Group 1, Group 2 — особенно `iso15_metro_reach` для пригорода.
- **commercial**: Group 1 (motorway/primary как трафик), Group 2 (`iso15_amenity_count` как proxy потока).
- **landplot**: Group 1 (`dist_to_residential` как подъезд), Group 2 маргинально.

### Что **не** делаем в этой итерации

- **Driving-isochrone** (15 мин на машине) — нужно учесть скорости road class, светофоры. Дорого, точность невысокая. Для commercial был бы полезен, отложим.
- **Public transit isochrone** — нужно GTFS-расписание Казанского транспорта (есть, но требует отдельного парса). Очень полезно для apartment — отдельная отложенная ADR.
- **Centrality** (Group 3 выше).
- **Дороги с разделителем / с трамваем** (визуальные/шумовые подкатегории) — отдельным потенциальным улучшением, не по факту.

## Архитектура

```text
data/silver/road_graph/edges.parquet            ← существующий граф
            │
            ├──► scripts/build_nearest_road_features.py (новый)
            │    data/silver/road_class_per_object/region={code}/data.parquet
            │      └─ object_id, nearest_road_class, dist_to_{class}_m × 6
            │
            └──► scripts/build_isochrone_cache_per_hex.py (новый)
                 data/silver/isochrone_cache/region={code}/h3_p={11}/data.parquet
                   └─ h3_index, iso15_pop_count, iso15_amenity_count, iso15_metro_reach

src/kadastra/etl/object_road_class_features.py   ← per-object KDTree query
src/kadastra/etl/object_isochrone_features.py    ← LEFT JOIN per-hex cache
```

Новые зависимости — нет (`networkx` + `polars` уже есть).

### TDD

| Уровень | Что покрывается |
| --- | --- |
| unit / `test_build_nearest_road_features.py` | Synthetic edges (3 motorway + 5 residential) → ожидаемые `nearest_road_class` для тест-точек. |
| unit / `test_object_road_class_features.py` | Synthetic objects + synthetic road_class_per_object → join produces expected columns. |
| unit / `test_object_isochrone_features.py` | Synthetic objects + synthetic isochrone_cache → join produces expected columns + null fallback. |

### Settings

```python
isochrone_walking_speed_m_per_min: float = 80.0
isochrone_walking_time_min: int = 15
isochrone_cache_resolution: int = 11
nearest_road_classes: list[str] = ["motorway", "trunk", "primary",
                                   "secondary", "tertiary",
                                   "residential", "pedestrian"]
```

## Эмпирический эффект (гипотеза)

- **apartment**: Δ MAPE −0.5…−1.5 пп. Mostly from `dist_to_motorway` (штраф) и `iso15_amenity_count` (премия за «walkability»).
- **house**: Δ −1…−2 пп. `dist_to_residential_m`, `iso15_metro_reach` для пригорода.
- **commercial**: Δ −1…−3 пп. `dist_to_primary_m` + `iso15_amenity_count` — основные сигналы трафика/потока.
- **landplot**: Δ −0.5…−1 пп. Подъезд решает.

Группа 1 даёт большую часть выигрыша, Группа 2 — добавочный +30–50% эффекта от Group 1 для apartment/commercial.

## Открытые вопросы

- **Качество road graph для Татарстана.** Сейчас он — extract из `data/raw/tatarstan_major_roads/tatarstan_major_roads.json` ([config.py:26](../../src/kadastra/config.py#L26)). Содержит ли он residential/pedestrian — надо сверить. Если только major — нужно перевыкачать через OSM extract с расширенным фильтром.
- **Население на hex-ячейку.** Для `iso15_pop_count` нужен population grid. EMISS даёт население на OKTMO ([ADR-0022](#)) — можно равномерно распределить по hex-ячейкам OKTMO как первое приближение. Но это не отдельный source.
- **Размер isochrone cache.** Hex p11 на Казанскую агломерацию ~50k гексов. Каждый гекс — Dijkstra на subgraph. ~50k × 50 ms = 40 минут single-thread. С joblib parallel — 5–10 минут. Однократно.

## Аудит данных (2026-08-20, закрывает «Открытые вопросы»)

1. **Граф без highway-классов.** `data/silver/road_graph/edges.parquet` (403 928 рёбер) хранит только `(from_lat, from_lon, to_lat, to_lon, length_m)` — классов нет. Исходный raw `tatarstan_major_roads.json` содержит только major-классы (motorway..tertiary + `*_link`, 15 404 ways). Для minor-классов подготовлен новый raw `data/raw/osm/kazan-agg-minor_road_ways.parquet` (69 827 ways Казанской агломерации: `osm_id, highway, coords_json` с `[lon, lat]`-парами; service 37 962, footway 17 134, residential 12 731, unclassified 1 918, pedestrian 47, living_street 35; полный geojsonseq — `s3://…/Kadatastr/raw/osm/kazan-agg-minor_roads.geojsonseq`). **Выбор:** Группа 1 считается не из `edges.parquet`, а из объединения двух raw (major JSON через S3 RawDataPort + minor parquet), с нормализацией классов (`*_link` → базовый класс) в `normalize_highway_class`.
2. **pedestrian/living_street почти отсутствуют (47+35 ways).** Для `dist_to_pedestrian_m` (и значения `pedestrian` в `nearest_road_class`) объединяем pedestrian + living_street + footway в «пешеходную инфраструктуру» — иначе фича шумовая.
3. **Население для `iso15_pop_count`.** Первое приближение из спеки реализовано буквально: население ОКТМО (`oktmo_population` в gold, ADR-0022) равномерно распределено по res-11 ячейкам, **содержащим valuation-объекты этого ОКТМО** (октмо-полигонов в репозитории нет, поэтому «ячейки ОКТМО» определяются через объекты). Ячейки без объектов получают 0 — смещение в сторону заселённых ячеек осознанное (меряем достижимое население), задокументировано как ограничение.
4. **POI для `iso15_amenity_count`** — точечные слои ADR-0019 (`walk_dist_layer_names`, 11 слоёв: school..railway_station), 8 918 точек.
5. **Метро для `iso15_metro_reach`** — те же 11 станций, что в ADR-0011 (`metro_stations.csv` через S3 RawDataPort).
6. **Размер кэша.** Реальных res-11 ячеек с объектами — 78 719 (не ~50k). Считаем кэш **только для ячеек с объектами** (а не для всех 1 240 001 ячеек покрытия res 11) — кэш существует ради LEFT JOIN от объектов, объект-less ячейки никогда не читаются. Фактическое время: ~4 мс на ячейку, 31 с на 10 процессах (spawn, своя копия графа ~1.6 ГБ на воркера).

## Отличия реализации от спеки

- **STRtree вместо KDTree (Группа 1).** Вместо `KDTree.query` по вершинам рёбер использован существующий паттерн ADR-0019 (`object_geom_distance_features`): shapely `STRtree.nearest` + `shapely.distance` в EPSG:32639 (UTM-39N). Это **истинное** расстояние точка→полилиния в метрах (KDTree по вершинам систематически завышал бы дистанцию до половины длины сегмента), тот же класс сложности, ноль новых зависимостей, единообразие с соседним кодом.
- **Без флага `nearest_road_classes` в Settings.** Список классов и группы дистанций — константы модуля `etl/object_road_class_features.py` (`NEAREST_ROAD_CLASSES`, `DIST_CLASS_GROUPS`): настраивать их в рантайме незачем, а согласованность нормализации и дист-групп важнее гибкости.
- **RAW_OBJECT_SCHEMA не расширялся.** Ловушка со сбросом фрейма в `RAW_OBJECT_SCHEMA` решается не добавлением колонок в схему, а паттерном ADR-0022/0023: обе группы джойнятся из silver **после** сброса (`join_road_class_features` / `join_isochrone_features`), поэтому перезапуск пересчитывает их из silver, а не теряет.
- **Порт `RoadGraphPort` расширен** тремя методами (`snap_node`, `node_coord`, `reachable_nodes_within_m`) — изохроне нужны cutoff-Dijkstra и снэпы POI; реализация в `NetworkxRoadGraph`, фейки в тестах дополнены.
- **Группа 3 (centrality) не делалась** — как и предписано спекой.

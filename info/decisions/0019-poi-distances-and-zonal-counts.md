# ADR-0019: POI-расстояния и зональные счётчики (OSM)

**Статус:** In progress (poly-distance готов; точечные POI/extracts ждут)
**Дата:** 2026-04-26
**Реализует:** [info/grid-rationale.md §7](../grid-rationale.md), [§8](../grid-rationale.md) (Дистанционные и Зональные ЦОФ).
**Опирается на:** [ADR-0011](0011-graph-based-distance-features.md) (graph-distance pipeline для метро/road), [ADR-0013](0013-zonal-density-features-multi-radius.md) (zonal counts).

## Контекст

Сейчас покрыты только метро (станции, входы) как точечные POI и agg-кольца по зданиям/дорогам. Социальная инфраструктура — школы, детсады, поликлиники, торговые точки, остановки наземного транспорта — **в фичи не входит**. Это самая дешёвая дыра: пайплайны `compute_object_zonal_features` ([ADR-0013](0013-zonal-density-features-multi-radius.md)) и `compute_object_metro_features` ([ADR-0011](0011-graph-based-distance-features.md)) уже умеют брать слой OSM-точек и считать ближайшую дистанцию + counts в кольцах. Достаточно расширить список слоёв.

Кадастровая стоимость жилья и коммерции в реальности завязана на «5 минут пешком до садика», «остановка трамвая под окном», «10 минут до Парка Горького». Без этих сигналов модель компенсирует через косвенные прокси (lat/lon × OKATO), что плохо генерализуется на новые районы и новые регионы.

## Решение

Расширить ETL-список OSM-слоёв и считать для каждого нового слоя стандартный пакет фич: `dist_*_m`, `count_*_500m`, `count_*_1km`. Где имеет смысл — добавить и линейные/полигональные дистанции (вода, парк, промзона как **distance**, а не только `share` из [ADR-0014](0014-poly-area-buffer-features.md)).

### Точечные POI (новые слои в `zonal_layer_names`)

| OSM-тег | человеческое имя | релевантно для |
| --- | --- | --- |
| `amenity=school` | школа | apartment, house |
| `amenity=kindergarten` | детсад | apartment, house |
| `amenity=clinic`/`amenity=doctors` | поликлиника | apartment, house |
| `amenity=hospital` | больница | apartment, house |
| `amenity=pharmacy` | аптека | все жилые |
| `shop=supermarket`/`shop=mall` | супермаркет/торгцентр | apartment, house, commercial |
| `amenity=cafe`/`amenity=restaurant` | кафе/ресторан | apartment (стрит-фуд premium), commercial |
| `highway=bus_stop` | автобус | все классы |
| `railway=tram_stop` | трамвай | apartment, house |
| `railway=station` | ж/д вокзал | apartment, commercial |

Для каждого: per-object `dist_<layer>_m`, `count_<layer>_500m`, `count_<layer>_1km`.

### Полигональные/линейные distance (дополняют существующие share-фичи)

| источник | новая фича | смысл |
| --- | --- | --- |
| `data/raw/osm/kazan-agg-water.geojsonseq` | `dist_water_m` | premium for view (Казанка, Кремлёвская, Лебяжье) |
| `data/raw/osm/kazan-agg-park.geojsonseq` | `dist_park_m` | «у парка» — отдельный сигнал от share |
| `data/raw/osm/kazan-agg-industrial.geojsonseq` | `dist_industrial_m` | штраф за близость к промке (помимо share) |
| `data/raw/osm/kazan-agg-cemetery.geojsonseq` | `dist_cemetery_m` | штраф за вид/соседство |
| `landuse=landfill` (новый extract) | `dist_landfill_m` | мощный штраф за свалку |
| `power=line` (новый extract) | `dist_powerline_m` | санитарно-защитная зона ЛЭП |
| `railway=rail` (новый extract) | `dist_railway_m` | шум от ж/д |

### Применимость по классам

- **apartment / house** — всё.
- **commercial** — большинство, кроме садиков/школ (для коммерции это нерелевантно). Но `dist_bus_stop`, `dist_road_class_X`, `dist_powerline` критичны.
- **landplot** — почти ничего из POI (для участка важна VRI и распложение, не школы), но негативные externalities (`dist_landfill`, `dist_industrial`, `dist_cemetery`, `dist_railway`) актуальны.

Селектор фич для каждого класса не делаем — модели сами разберутся. Если для класса фича всегда ноль/null или константа — CatBoost проигнорирует, EBM выкинет в `feature_importance ≈ 0`.

### Что **не** делаем в этой итерации

- **Walking-isochrone enrichment** через road_graph (пешеходная доступность за 15 мин). Технически возможно — у нас уже есть road graph ([ADR-0011](0011-graph-based-distance-features.md)) — но это перепрыгивает в [ADR-0024](0024-road-network-advanced-features.md) (road-network advanced). Здесь — euclidean dist + radius counts.
- **Качество POI** (рейтинг, площадь, бренд). У OSM это есть редко, не унифицировано.
- **Density gradients per OSM-классу** (например, восстановление «коммерческой улицы» через kernel density). Перебор — счётчики в кольцах справляются.

## Архитектура

```text
data/raw/osm/
  ├─ kazan-agg-school.geojsonseq          ← новые extract'ы
  ├─ kazan-agg-kindergarten.geojsonseq
  ├─ kazan-agg-clinic.geojsonseq
  ├─ ...
  ├─ kazan-agg-bus-stop.geojsonseq
  ├─ kazan-agg-tram-stop.geojsonseq
  ├─ kazan-agg-railway-station.geojsonseq
  └─ kazan-agg-{landfill|powerline|railway}.geojsonseq

scripts/extract_osm_polygons.py (расширить):
  --layers school kindergarten clinic hospital pharmacy supermarket
           cafe restaurant bus-stop tram-stop railway-station
           landfill powerline railway

silver/zonal_features/region={code}/layer={X}/data.parquet  ← parallel pattern
silver/poly_distance_features/region={code}/layer={X}/data.parquet  ← новый dist-pattern

src/kadastra/etl/object_zonal_features.py (без изменений — уже generic).
src/kadastra/etl/object_poly_distance_features.py (новый, параллельно poly_area_share).
```

Cofig `Settings`:

```python
zonal_layer_names: list[str] = [
    "stations", "entrances", "apartments", "houses", "commercial",
    # новые в этой ADR
    "school", "kindergarten", "clinic", "hospital", "pharmacy",
    "supermarket", "cafe", "restaurant",
    "bus_stop", "tram_stop", "railway_station",
]
poly_distance_layer_paths: dict[str, str] = {
    "water":      "data/raw/osm/kazan-agg-water.geojsonseq",
    "park":       "data/raw/osm/kazan-agg-park.geojsonseq",
    "industrial": "data/raw/osm/kazan-agg-industrial.geojsonseq",
    "cemetery":   "data/raw/osm/kazan-agg-cemetery.geojsonseq",
    "landfill":   "data/raw/osm/kazan-agg-landfill.geojsonseq",
    "powerline":  "data/raw/osm/kazan-agg-powerline.geojsonseq",
    "railway":    "data/raw/osm/kazan-agg-railway.geojsonseq",
}
```

## Эмпирический эффект (гипотеза до замера)

- **apartment**: Δ MAPE −1…−3 пп. Школы/садики/трамваи добавляют сильный signal в дополнение к существующему метро.
- **house**: Δ −2…−4 пп. House шире разбросан по периферии, где метро нет, и шум от транспорта/торговли решает.
- **commercial**: Δ −1…−2 пп. Bus-stop count, supermarket dist, dist-to-railway-station — основные.
- **landplot**: Δ ≈ 0…−0.5 пп. Negative externalities (`dist_landfill`, `dist_industrial`) дадут что-то, остальное нерелевантно.

## Открытые вопросы

- **Полнота OSM в Татарстане.** OSM в Казани заполнен сравнительно неплохо (mappers активны), но в пригороде — местами дыры. Если на этапе extract увидим систематические лакуны (например, school-points отсутствуют в посёлках N km от Казани) — нужен fallback на 2GIS/Wikimapia. Решается по факту.
- **Bus stop coverage.** OSM имеет `highway=bus_stop` для большей части городских маршрутов, но не пригородных. Точность для apartment в центре высокая, для house на периферии — деградирует.
- **Дублирование** между `dist_water_m` и существующим `share_water_500m`. Гипотетически распался бы CatBoost-importance между ними; EBM покажет, какая полезнее. На больших датасетах CatBoost устойчив к этому.

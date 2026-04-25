# ADR-0009: Замена synthetic-таргета реальной кадастровой стоимостью из НСПД, добавление LandPlot

**Статус:** Accepted частично — pipeline и схема корректны, **выбор target пересмотрен в [ADR-0010](0010-methodology-compliance-roadmap.md)**: государственная кадастровая стоимость из ЕГРН (`cost_index`) — это broken state output, который сам по себе подлежит пересчёту. Используется как **interim placeholder** до появления сделочных данных (track 1, заблокирован). NSPD как источник объектов, парсинг, silver-store и feature engineering — остаются valid'ными.
**Дата:** 2026-04-25
**Заменяет:** synthetic-таргет из [ADR-0004](0004-synthetic-target.md) (для всех 4 классов)
**Дополняет:** [ADR-0007](0007-kazan-agglomeration-scope.md), [ADR-0008](0008-per-building-multi-class-valuation.md)
**Уточняется:** [ADR-0010](0010-methodology-compliance-roadmap.md) (target reframe)

## Контекст

В пилот мы пришли с synthetic proxy ([ADR-0004](0004-synthetic-target.md)), потому что в S3-бакете `s3://kadastrova/Kadatastr/` не было реальных кадастровых стоимостей. На синтетических данных модель училась воспроизводить нашу же формулу, что давало эффектные графики, но никакой реальной валидации.

Параллельно [ADR-0008](0008-per-building-multi-class-valuation.md) запланировал четвёртый класс `LandPlot`, оставив его «до момента, когда найдём источник». В ноутбуке этот «момент» обрисовывался как доступ к ГАР `AS_STEADS` или ПКК — оба варианта в нашей разведке оказались тупиками (см. [info/nspd-api.md](../nspd-api.md), раздел «Зачем это всё»).

В разведке 2026-04-25 нашли третий путь: **анонимный bulk-канал НСПД (Национальная Система Пространственных Данных)**, который отдаёт по фильтру адреса публичные слои ЕГРН с полигонами, кадастровыми номерами и **реальной кадастровой стоимостью** (`cost_value` ₽ и `cost_index` ₽/м²). Скачали:

- `Kadatastr/nspd/buildings-kazan/` — 91 864 здания (L36049), 204 МБ.
- `Kadatastr/nspd/landplots-kazan/` — 199 819 участков (L36048), 514 МБ.

Покрытие `cost_index` в выборке зданий — **99.9 %**, в выборке участков — **100 %**.

## Решение

**1. Источник таргета меняется со synthetic на реальный `cost_index` из ЕГРН** — для всех 4 классов одновременно, не по этапам. Synthetic-формулы и use case `BuildObjectSyntheticTarget` остаются в репозитории как референс, но новый пайплайн их не зовёт.

**2. Добавляется четвёртый класс `AssetClass.LANDPLOT`** для земельных участков. Это пустой пока что в плане feature engineering, но полноценный по таргету — `cost_index` из L36048 заполнен у всех.

**3. NSPD становится первичным источником объектов застройки.** OSM-выгрузка `osm_buildings_kazan_agglomeration.csv` остаётся в S3 для совместимости и как источник дополнительных тегов, но на новом пути обучения мы переключаемся на NSPD-объекты — у них больше атрибутов (`floors`, `year_built`, `materials`, `purpose`, `quarter_cad_number`) и есть реальный таргет.

### Маппинг NSPD `options.purpose` → `AssetClass`

| Значение NSPD | AssetClass | Доля в Казани (sample 4600) |
|---|---|---|
| `Многоквартирный дом` | `APARTMENT` | 2.3 % |
| `Жилой дом` | `HOUSE` | 51 % |
| `Жилое` | `HOUSE` | 2.4 % |
| `Садовый дом` | `HOUSE` | 0.4 % |
| `Нежилое` | `COMMERCIAL` | 42 % |
| `Гараж` | — (отбрасываем) | 1.8 % |
| (прочее, в т.ч. None) | — (отбрасываем) | <0.1 % |

«Нежилое» в ЕГРН — это широкий класс: офисы, магазины, склады, инфраструктура. Совпадает по широте с тем, что в OSM-варианте мы называли COMMERCIAL (`retail,commercial,supermarket,kiosk,office,industrial,warehouse`).

«Гараж» отдельно — это, как правило, маленькие отдельностоящие постройки, мы их не оцениваем (в OSM-варианте `garage` тоже отбрасывался).

«Садовый дом» классифицируется как `HOUSE`, потому что это та же типология (домик с участком), просто с дачным происхождением.

### Маппинг для участков (`AssetClass.LANDPLOT`)

Все объекты слоя L36048 → `AssetClass.LANDPLOT`. Внутреннюю классификацию (категория земель × вид разрешённого использования) откладываем — на пилоте достаточно одного класса для всей таблицы участков.

### Контракт нового источника

Парсинг сырых page-NNNN.json в нормализованный polars-фрейм со схемой:

```python
NSPD_OBJECT_SCHEMA = {
    "object_id": pl.Utf8,        # "nspd-building/<geom_data_id>" / "nspd-landplot/<geom_data_id>"
    "asset_class": pl.Utf8,      # apartment/house/commercial/landplot
    "cad_num": pl.Utf8,          # "16:50:010406:40"
    "lat": pl.Float64,           # центроид полигона, WGS84
    "lon": pl.Float64,
    "area_m2": pl.Float64,       # build_record_area / specified_area / specified_area
    "cost_value_rub": pl.Float64,    # ₽ полная
    "cost_index_rub_per_m2": pl.Float64,  # ₽/м² (это и есть таргет)
    "year_built": pl.Int64,      # для зданий, иначе null
    "floors": pl.Int64,          # для зданий, иначе null
    "underground_floors": pl.Int64,
    "materials": pl.Utf8,        # «Деревянные», «Кирпичные», ...
    "purpose": pl.Utf8,          # сырое значение purpose из NSPD
    "land_record_category_type": pl.Utf8,  # для участков, иначе null
    "land_record_subtype": pl.Utf8,
    "ownership_type": pl.Utf8,
    "registration_date": pl.Date,
    "readable_address": pl.Utf8,
    "polygon_wkt": pl.Utf8,      # геометрия в WKT для последующего spatial-фильтра
}
```

EPSG:3857 → EPSG:4326 (lat/lon) делается в момент чтения.

### Spatial postfilter

Фильтр `address ⊂ "Казань"` ловит лишние объекты (где «Казань» в адресе по другим причинам — например, «улица Казанская» в другом городе). После парсинга проверяем, что centroid попадает в [`data/raw/regions/kazan-agglomeration.geojson`](../../data/raw/regions/kazan-agglomeration.geojson) (тот же 30 км буфер, что и в [ADR-0007](0007-kazan-agglomeration-scope.md)). Лишние выбрасываем.

### Storage

```text
data/silver/nspd/region={code}/source={buildings|landplots}/data.parquet
```

Промежуточный clean-слой между сырым JSON и feature-таблицей. Партиционирован по региону (`RU-KAZAN-AGG`) и источнику (`buildings` vs `landplots`), чтобы было удобно делать union без потери трассировки.

### Use cases

```text
LoadNspdRawObjects        Kadatastr/nspd/{buildings,landplots}-kazan/page-*.json → silver parquet
                            (включает spatial postfilter + классификацию по purpose)
BuildObjectFeatures       (как есть в ADR-0008, добавляется LANDPLOT в маршрутизацию)
BuildRealCostTarget       silver/nspd/... → real target column в feature-таблице
                            (заменяет BuildObjectSyntheticTarget на новом пути обучения)
TrainObjectValuationModel (как есть, но читает реальный таргет)
InferObjectValuation      (как есть, но обучен на реальных данных)
```

### Map UI

Селектор Mode в [src/kadastra/web/templates/map.html](../../src/kadastra/web/templates/map.html) расширяется на четвёртый пункт: `Objects: landplot`. API-эндпоинт `/api/object_predictions?asset_class=landplot` отдаёт landplot-предсказания тем же контрактом, что и три существующих класса.

## Почему не сохранить и synthetic, и real одновременно

- Synthetic учил модель сам себе. Метрики MAPE 5–10 % были артефактом совпадения формулы и таргета, не реальной точности.
- Cross-validation на реальных данных будет давать другие, скорее всего хуже, цифры — это полезно, потому что наконец можно говорить о реальном качестве модели.
- Параллельная двухветочная архитектура удваивает поверхность и требует синхронизации между synthetic-формулой и реальной шкалой. Лучше переключиться полностью.

Старые synthetic-таргеты остаются на диске (`data/gold/targets/...` и `data/gold/valuation_objects/...`) — не удаляем, чтобы не сломать историю экспериментов.

## Почему OSM-источник остаётся

- OSM-выгрузка покрывает агломерацию полигонально (180k объектов), NSPD — фильтром по адресу (~92k для зданий ⊂ «Казань»). Это **разное покрытие**, и OSM в сумме шире.
- Но у OSM нет реального таргета и беднее набор атрибутов.
- На текущем слайсе мы переключаемся на NSPD как primary, но OSM-выгрузку оставляем в S3 для будущей пере-инженировки (например, joinить OSM по spatial proximity, получать ширину покрытия + богатство атрибутов NSPD).

## Что закрывается этим решением

- **Open question из [ADR-0004](0004-synthetic-target.md):** *«синтетический proxy будет заменён реальными кадастровыми/сделочными ценами»* — закрыт.
- **Open question из [ADR-0007](0007-kazan-agglomeration-scope.md):** *«реальный таргет»* — закрыт для кадастровой стоимости. Сделочные цены остаются открытыми.
- **Open question из [ADR-0008](0008-per-building-multi-class-valuation.md):** *«LandPlot через ГАР AS_STEADS»* — закрыт через NSPD; ГАР XML остаётся как fallback / источник адресной иерархии.

## Что остаётся открытым

- **Сделочные цены ≠ кадастр.** Кадастровая стоимость государственная, рыночная — другая. Пилот дойдёт до реального коммерческого качества, только когда подтянем выгрузки сделок (в перспективе — через тот же договор с ППК Роскадастр).
- **Гранулярность LANDPLOT.** Все 200k участков в одной корзине. Если понадобится более тонкая модель (urban / commercial / agricultural), маппинг через `land_record_category_type` уже доступен.
- **Refresh-стратегия.** ЕГРН обновляется. Сейчас выгрузка разовая. Когда дойдём до прода — нужен инкремент по `cost_application_date` или `cost_registration_date` (фильтруемые `intervalDate`).
- **OSM ↔ NSPD spatial join.** Пока NSPD primary; вернёмся к джойну если выяснится, что 92k зданий ⊂ «Казань» сильно недопокрывают агломерацию по сравнению с 180k OSM.

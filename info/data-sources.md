# Источники данных

## S3 бакет проекта

```text
endpoint: https://s3.ohnice.synology.me   (path-style addressing)
bucket:   kadastrova
prefix:   Kadatastr/
```

Креды — в `.env` (см. [.env.example](../.env.example)). Доступ — `aws --endpoint-url ... s3 ls` или `boto3` с `addressing_style="path"`.

### Содержимое `s3://kadastrova/Kadatastr/`

| Префикс | Что | Комментарий |
| ------- | --- | ----------- |
| `gar_xml/16/` | **ГАР Татарстана**: адресные объекты, дома, квартиры, парцеллы, машиноместа, иерархия адм. и муниципальная, история. Снапшот 2026-04-06. | ~10 ГБ. Основной источник реестровых данных. Файлы: `AS_ADDR_OBJ`, `AS_HOUSES`, `AS_APARTMENTS`, `AS_STEADS`, `AS_ADM_HIERARCHY`, `AS_MUN_HIERARCHY`, `AS_*_PARAMS`, `AS_REESTR_OBJECTS`, `AS_CHANGE_HISTORY`. |
| `gar_xml/01..89/` | ГАР по другим субъектам РФ | Для пилота не нужно. |
| `osm/volga-fed-district-latest.osm.pbf` | OSM-выгрузка Поволжского ФО | 725 МБ, сырой PBF. Татарстан — внутри; вырезается через `osmium extract`. |
| `osm/osm_buildings_kazan_agglomeration.csv` | Здания Казанской агломерации (≈30 км буфер) из OSM | 13.3 МБ, ~179 133 строки. Сделан скриптом [scripts/buildings_geojsonseq_to_csv.py](../scripts/buildings_geojsonseq_to_csv.py) из PBF, обрезанного по [scripts/build_kazan_agglomeration_boundary.py](../scripts/build_kazan_agglomeration_boundary.py). Используется по умолчанию (`buildings_key` в [Settings](../src/kadastra/config.py)). |
| `rosreestr/osm_buildings_kazan.csv` | Здания Казани из OSM (CSV) | 4.3 МБ. **Не Росреестр**, имя префикса вводит в заблуждение. Заменён на `osm/osm_buildings_kazan_agglomeration.csv` ([ADR-0007](decisions/0007-kazan-agglomeration-scope.md)). |
| `rosreestr/osm_buildings_kazan_raw.json` | То же в сыром JSON | 16 МБ. |
| `metro/metro_stations.csv` | Станции метро Казани (11 шт), WGS84 | См. [README в S3](s3://kadastrova/Kadatastr/metro/README.md) — содержит схему CSV и предложенные ML-фичи (`dist_metro_m`, `count_stations_1km`, …). |
| `metro/metro_entrances.csv` | Входы метро (68 шт) | |
| `metro/metro_kazan_raw.json` | Сырой ответ Overpass API | |
| `tatarstan_major_roads/tatarstan_major_roads.json` | Крупные дороги Татарстана (Overpass JSON) | 16 МБ. |
| `nspd/buildings-kazan/page-NNNN.json` | **Здания ЕГРН (NSPD L36049), фильтр адрес ⊂ «Казань»** | 460 страниц по 200 объектов + 1 хвост (64 объекта) = **91 864 здания**. ~204 МБ. Сырая GeoJSON-выдача bulk attrib-search v3 + `_state.json` с прогрессом. См. [info/nspd-api.md](nspd-api.md) — там полный гайд по тому, как мы это получили. |
| `nspd/landplots-kazan/page-NNNN.json` | **Земельные участки ЕГРН (NSPD L36048), фильтр адрес ⊂ «Казань»** | 1000 страниц + 1 хвост (19 объектов) = **199 819 участков**. ~514 МБ. Каждый объект содержит полигон в EPSG:3857, кадастровый номер, площадь и **реальную кадастровую стоимость** (`cost_value` ₽ и `cost_index` ₽/м²). |

## Внешние источники

| Источник | Что берём | Использование |
| -------- | --------- | ------------- |
| **geoBoundaries gbOpen** ([commit 9469f09](https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/RUS/ADM1/)) | Граница Татарстана (RUS ADM1, поле `shapeISO == "RU-TA"`) | Скачивается через [scripts/download_geoboundaries.py](../scripts/download_geoboundaries.py) в `data/raw/regions/`. Закреплён на конкретном коммите → воспроизводимо. С [ADR-0007](decisions/0007-kazan-agglomeration-scope.md) пилот сужен до агломерации, граница Татарстана остаётся как референс. |
| **Kazan agglomeration buffer** | Граница пилотного региона (`shapeISO == "RU-KAZAN-AGG"`) — 30 км буфер вокруг центра Казани, UTM zone 39N | Генерируется [scripts/build_kazan_agglomeration_boundary.py](../scripts/build_kazan_agglomeration_boundary.py) в `data/raw/regions/kazan-agglomeration.geojson`. Параметры (центр, радиус) фиксированы в скрипте. |

## Чего не хватает

- ~~**Целевая переменная (реальная).** Кадастровая стоимость или рыночные сделки в бакете отсутствуют. На пилот принят синтетический proxy — см. [ADR-0004](decisions/0004-synthetic-target.md). Реальный таргет будет подключён отдельным слоем `data/gold/targets_real/...`, когда найдём источник.~~ **Закрыто 2026-04-25:** в `nspd/buildings-kazan/` и `nspd/landplots-kazan/` лежат `cost_value` и `cost_index` из ЕГРН — реальная кадастровая стоимость по 91 864 зданиям и 199 819 участкам Казанской агломерации. Подробности — [info/nspd-api.md](nspd-api.md). ETL для перевода JSON → parquet и переключения моделей с synthetic на real target — следующий слайс.
- **Рыночные сделки.** Кадастровая ≠ рыночная. Источник для рыночных цен пока не найден. Возможные варианты — Циан/Авито парсинг (ToS-чувствительно) либо Росреестр-выгрузки сделок (через тот же договор с ППК Роскадастр, что нужен для платных слоёв NSPD).

## Локальные пути по умолчанию

См. [src/kadastra/config.py](../src/kadastra/config.py).

| Путь | Что | В git? |
| ---- | --- | ------ |
| `data/raw/` | Сырые входы (geoBoundaries, S3-выгрузки) | нет (gitignore) |
| `data/silver/coverage/region={code}/resolution={r}/data.parquet` | Покрытие региона H3-сеткой, hive-партицировано | нет (gitignore) |
| `data/silver/features/...` | Hex-агрегаты признаков | нет (gitignore) |
| `data/gold/features/region={code}/resolution={r}/data.parquet` | Финальная feature-таблица для модели | нет (gitignore) |
| `data/gold/targets/region={code}/resolution={r}/data.parquet` | Синтетический таргет ([ADR-0004](decisions/0004-synthetic-target.md)) | нет (gitignore) |
| `data/models/{run_name}_{ts}/` | Локальные артефакты тренировки baseline ([ADR-0005](decisions/0005-baseline-training.md)) — `params.json`, `metrics.json`, `model.cbm`. Используется когда `MLFLOW_ENABLED=False`. | нет (gitignore) |
| `data/gold/predictions/region={code}/resolution={r}/data.parquet` | Снимок предсказаний модели на сетке ([ADR-0006](decisions/0006-inference-and-map.md)). Колонки: `h3_index`, `resolution`, `predicted_value`. | нет (gitignore) |

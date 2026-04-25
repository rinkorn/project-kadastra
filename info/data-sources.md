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
| `rosreestr/osm_buildings_kazan.csv` | Здания Казани из OSM (CSV) | 4.3 МБ. **Не Росреестр**, имя префикса вводит в заблуждение. |
| `rosreestr/osm_buildings_kazan_raw.json` | То же в сыром JSON | 16 МБ. |
| `metro/metro_stations.csv` | Станции метро Казани (11 шт), WGS84 | См. [README в S3](s3://kadastrova/Kadatastr/metro/README.md) — содержит схему CSV и предложенные ML-фичи (`dist_metro_m`, `count_stations_1km`, …). |
| `metro/metro_entrances.csv` | Входы метро (68 шт) | |
| `metro/metro_kazan_raw.json` | Сырой ответ Overpass API | |
| `tatarstan_major_roads/tatarstan_major_roads.json` | Крупные дороги Татарстана (Overpass JSON) | 16 МБ. |

## Внешние источники

| Источник | Что берём | Использование |
| -------- | --------- | ------------- |
| **geoBoundaries gbOpen** ([commit 9469f09](https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/RUS/ADM1/)) | Граница Татарстана (RUS ADM1, поле `shapeISO == "RU-TA"`) | Скачивается через [scripts/download_geoboundaries.py](../scripts/download_geoboundaries.py) в `data/raw/regions/`. Закреплён на конкретном коммите → воспроизводимо. |

## Чего не хватает

- **Целевая переменная (реальная).** Кадастровая стоимость или рыночные сделки в бакете отсутствуют. На пилот принят синтетический proxy — см. [ADR-0004](decisions/0004-synthetic-target.md). Реальный таргет будет подключён отдельным слоем `data/gold/targets_real/...`, когда найдём источник.

## Локальные пути по умолчанию

См. [src/kadastra/config.py](../src/kadastra/config.py).

| Путь | Что | В git? |
| ---- | --- | ------ |
| `data/raw/` | Сырые входы (geoBoundaries, S3-выгрузки) | нет (gitignore) |
| `data/silver/coverage/region={code}/resolution={r}/data.parquet` | Покрытие региона H3-сеткой, hive-партицировано | нет (gitignore) |
| `data/silver/features/...` | Hex-агрегаты признаков | нет (gitignore) |
| `data/gold/features/region={code}/resolution={r}/data.parquet` | Финальная feature-таблица для модели | нет (gitignore) |
| `data/gold/targets/region={code}/resolution={r}/data.parquet` | Синтетический таргет ([ADR-0004](decisions/0004-synthetic-target.md)) | нет (gitignore) |

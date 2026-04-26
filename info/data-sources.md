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
| `osm/kazan_walking_network.json` | **Сырая Overpass JSON-выгрузка пешеходной сети Казанской агломерации** (`highway=*` без motorway/trunk/construction/proposed; bbox ≈ 30 км буфер). Источник для графа путевых расстояний ([ADR-0011](decisions/0011-graph-based-distance-features.md)). | ~50 МБ; запрос делается [scripts/download_walking_network.py](../scripts/download_walking_network.py); затем [scripts/build_road_graph_artifact.py](../scripts/build_road_graph_artifact.py) превращает в parquet edges-таблицу. |
| `nspd/buildings-kazan/page-NNNN.json` | **Здания ЕГРН (NSPD L36049), фильтр адрес ⊂ «Казань»** | 460 страниц по 200 объектов + 1 хвост (64 объекта) = **91 864 здания**. ~204 МБ. Сырая GeoJSON-выдача bulk attrib-search v3 + `_state.json` с прогрессом. См. [info/nspd-api.md](nspd-api.md) — там полный гайд по тому, как мы это получили. |
| `nspd/landplots-kazan/page-NNNN.json` | **Земельные участки ЕГРН (NSPD L36048), фильтр адрес ⊂ «Казань»** | 1000 страниц + 1 хвост (19 объектов) = **199 819 участков**. ~514 МБ. Каждый объект содержит полигон в EPSG:3857, кадастровый номер, площадь и **реальную кадастровую стоимость** (`cost_value` ₽ и `cost_index` ₽/м²). |
| `emiss/31452/raw/raw_{YYYY-MM-DD}.xls` | **ЕМИСС / fedstat.ru, индикатор #31452** — «Средняя цена 1 кв.м. общей площади квартир на рынке жилья» (по регионам в целом, **без разреза на города-центры**). Pivot-таблица .xls (3 991 строк × 31 колонка), **CC-BY 3.0**. Покрытие: **2000-2025 (26 лет)**, 114 регионов/округов, квартально (I-IV), 2 рынка (первичный/вторичный), 5 типов квартир (низкое/среднее/улучшенное/элитное/все). | Скачивается curl: `https://fedstat.ru/indicator/data.do?id=31452&format=excel`. Прим.: тот же приказ Росстата №572 от 12.08.2022 (форма 1-РЖ), но индикатор-«ствол» — длинный исторический ряд региональных средних. |
| `emiss/31452/silver/data.parquet` | Long-format unpivot (73 682 строк × 17 колонок): indicator_id, region_okato/name, period_code/name/quarter, rynzhel_code/name, tipkvartir_code/name, year, period_label (`YYYY-Qn`), value_rub_per_m2 + пустые `mestdom_*` (для совместимости со схемой #61781). | См. [scripts/parse_emiss_xls_to_parquet.py](../scripts/parse_emiss_xls_to_parquet.py). Используется как **долгий калибровочный временной ряд ₽/м² по всей РФ** — для темпоральных ЦОФ (grid-rationale §10) и контр-проверки результатов пересчёта кадастра по годам. |
| `emiss/61781/raw/raw_{YYYY-MM-DD}.xls` | **ЕМИСС / fedstat.ru, индикатор #61781** — «Средняя цена 1 кв.м. общей площади квартир на рынке жилья **по центрам субъектов РФ**» (т.е. _только_ административный центр субъекта, например Казань для Татарстана, Иркутск для Иркутской обл.). Pivot-таблица .xls (2 477 × 11), **CC-BY 3.0**. Покрытие: **2021-2025 (5 лет)**, 80 центров субъектов, квартально, 2 рынка, 5 типов квартир. | Скачивается curl: `https://fedstat.ru/indicator/data.do?id=61781&format=excel`. Молодая методология: приказ Росстата №572 от 12.08.2022 (форма 1-РЖ), city-level разрез появился именно с этой ревизии. |
| `emiss/61781/silver/data.parquet` | Long-format unpivot (10 952 строк × 17 колонок) — аналогичная схема #31452, но `mestdom_name = "Центр субъекта Российской Федерации"`. | См. тот же [scripts/parse_emiss_xls_to_parquet.py](../scripts/parse_emiss_xls_to_parquet.py) (общий парсер для обоих индикаторов). Используется как **city-level калибровочный якорь** для конкретных городов (Казань 2025: ~192K ₽/м², Иркутск 2025: ~147K ₽/м²). |
| `listings-mvp/raw/{source}_{city}/page-NNN.html` | **MVP-выгрузка листингов** трёх площадок (Yandex Realty / CIAN / Avito) для сравнения источников и обучения моделей цены ₽/м² **на квартирах**. Текущий снимок Казани (2026-04-26): Yandex 99 страниц (auto-stop на SmartCaptcha), **CIAN 183 страницы** (естественный потолок выдачи через `cat.php?...&region=4777&p=N`), Avito 4 страницы (упёрся в IP-блок «Доступ ограничен: проблема с IP» на p=5, доскачка отложена до снятия блока). Иркутск: Yandex 28 страниц (partial). Suммарно ~2.4 ГБ HTML. Все через `patchright + system Chrome + persistent profile` (headed). **Не cron**: research-snapshot, ToS у всех трёх запрещает automated сбор. | См. [info/listings-scraping-mvp.md](listings-scraping-mvp.md) (полное описание: антибот, URL-ловушки CIAN, region_id-справочник). Скрипт [scripts/download_listings_mvp.py](../scripts/download_listings_mvp.py) — инкрементальный, auto-stop при 3 мелких страницах подряд. |
| `listings-mvp/silver/{source}_{city}.parquet` + `all.parquet` | **Извлечённые объявления**, дедуплицированы по `id`. Per-source-city (native схема): `cian_kazan` 1268×28 (богатая мета — build_year, material_type, lat/lon в 100 %), `yandex_realty_kazan` 651×12, `yandex_realty_irkutsk` 441×12, `avito_kazan` 196×15. Объединённый `all.parquet` 2556×18 с общей ML-схемой (listing_id, source, city, price_rub, total_area_m2, rooms, floor, floors_count, floor_share, build_year, material_type, lat, lon, url, page_file, price_per_sqm_rub). Медианы ₽/м² — CIAN Казань 221K, Yandex Казань 226K, Avito Казань 198K, Yandex Иркутск 159K. | Парсер [scripts/extract_listings_mvp.py](../scripts/extract_listings_mvp.py) — три стратегии: CIAN раскодирует встроенный `"offers":[…]` JSON, Avito парсит JSON-LD `AggregateOffer.offers[]`, Yandex Realty — regex по DOM-карточкам `data-test="OffersSerpItem"`. Сравнительная аналитика по полноте полей и anchor-delta (vs ЕМИСС #61781) — [scripts/compare_listings_sources.py](../scripts/compare_listings_sources.py). |

## Внешние источники

| Источник | Что берём | Использование |
| -------- | --------- | ------------- |
| **geoBoundaries gbOpen** ([commit 9469f09](https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/RUS/ADM1/)) | Граница Татарстана (RUS ADM1, поле `shapeISO == "RU-TA"`) | Скачивается через [scripts/download_geoboundaries.py](../scripts/download_geoboundaries.py) в `data/raw/regions/`. Закреплён на конкретном коммите → воспроизводимо. С [ADR-0007](decisions/0007-kazan-agglomeration-scope.md) пилот сужен до агломерации, граница Татарстана остаётся как референс. |
| **Kazan agglomeration buffer** | Граница пилотного региона (`shapeISO == "RU-KAZAN-AGG"`) — 30 км буфер вокруг центра Казани, UTM zone 39N | Генерируется [scripts/build_kazan_agglomeration_boundary.py](../scripts/build_kazan_agglomeration_boundary.py) в `data/raw/regions/kazan-agglomeration.geojson`. Параметры (центр, радиус) фиксированы в скрипте. |

## Чего не хватает

- **Рыночные сделки — основной target обучения.** Конечный **продукт** проекта — кадастровая оценка нового образца, считаемая из обученной рыночной модели через методические указания ГБУ ([ADR-0010](decisions/0010-methodology-compliance-roadmap.md)). Текущий ЕГРН `cost_index` сам по себе broken и подлежит пересчёту, поэтому **обучаемся на рынке** (сделки), а **выдаём кадастр** (deliverable). Сделочных (объект-уровневых) данных у нас пока нет. Что проверено и **не подходит**:
  - **Каталог квартир ЕИСЖС / наш.дом.рф** — это прейскурант застройщиков, не сделки. Дополнительно: каталог покрывает в основном крупные московские/краснодарские застройщики; региональные подключены слабо (Казань — 0, Иркутск — десятки, остальная РФ — фрагментарно). Скрейпинг и попытка использования отброшены ([reverse-engineering remained on disk через `git log`/прошлые сессии] — код и snapshot удалены).
  - **PRO.Дом.рф** — закрытый коммерческий API ЕИСЖС с фактическими ценами ДДУ, 350K–1М ₽/год — исключено.

  Кандидаты, проверенные **частично**:
  - **Yandex Realty / CIAN / Avito** — обход антибот-защиты подтверждён через `patchright + system Chrome + persistent profile`. MVP-выгрузка по Казани (~2 600 уникальных квартир из 3 источников) лежит в `listings-mvp/silver/` — детально см. [info/listings-scraping-mvp.md](listings-scraping-mvp.md). **Это листинги (asks), не сделки** — цены завышены относительно фактических ДКП на 5-15 %, и это всё ещё ToS-violation у всех трёх площадок. Production-loader / cron **не реализуется** — нужен легитимный канал (партнёрский API с одной из площадок; именно для выбора партнёра и нужен MVP-сравнительный анализ).

  Кандидаты, которые ещё **не пробовали**:
  - Росреестр-выгрузки сделок через ППК (закрытое API, нужен договор).
  - Региональные опендата по сделкам (open.tatarstan.ru — данные мёртвые, проверено).

  До появления сделок трек 1 заблокирован, проект движется по треку 2 (методологическое соответствие, см. ADR-0010). В качестве калибровочного якоря — `emiss/61781/silver/data.parquet`: квартальные ср. цены ₽/м² по 79 регионам РФ, 2021-Q1..2025-Q4 (с расширением вглубь по мере доступности).

- **Кадастровая стоимость из ЕГРН — interim placeholder, не цель.** В `nspd/buildings-kazan/` и `nspd/landplots-kazan/` лежат `cost_value` и `cost_index` по 91 864 зданиям и 199 819 участкам Казанской агломерации (подробности — [info/nspd-api.md](nspd-api.md)). Используется как временный target в обучении, чтобы можно было катать методологические feature engineering'овые фичи (граф дорог, относительные ЦОФ и т.д.) до появления сделочных данных. **Метрики на этом target измеряют, насколько модель воспроизводит государственную кадастровую стоимость, которая сама по себе сейчас broken и подлежит пересчёту** — это анти-цель в долгую. Подробнее — [ADR-0009](decisions/0009-real-cadastre-target-via-nspd.md) (источник объектов остаётся valid'ным) и [ADR-0010](decisions/0010-methodology-compliance-roadmap.md) (target reframe).

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
| `data/raw/osm/kazan_walking_network.json` | Сырая Overpass-выгрузка пешеходной сети ([ADR-0011](decisions/0011-graph-based-distance-features.md)). Локально кэшируется, чтобы не дёргать Overpass на каждый rebuild. | нет (gitignore) |
| `data/silver/road_graph/edges.parquet` | Edges-таблица графа `(from_lat, from_lon, to_lat, to_lon, length_m)`. Грузится eager-loadом в `NetworkxRoadGraph` через `Settings.road_graph_edges_path`. | нет (gitignore) |

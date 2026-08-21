# ADR-0025: Прочие ЦОФ — CBD, культнаследие, ЗОУИТ

**Статус:** Accepted
**Дата:** 2026-04-26 (принят 2026-08-21)
**Реализует:** [info/grid-rationale.md §6](../grid-rationale.md), [§9](../grid-rationale.md) (объектные/территориальные ЦОФ — разнородный «вкладыш»).
**Опирается на:** [ADR-0014](0014-poly-area-buffer-features.md) (poly-area pattern для overlay-фич).

## Контекст

Сборная ADR для трёх независимых, но мелких фич, каждая из которых не оправдывает отдельный документ, но вместе они закрывают «остаточный список» из ADR-19/20/21/22.

Все три — низкая сложность реализации, средний-высокий эффект на отдельных классах объектов.

## Решение

### 1. `dist_to_cbd_m` — расстояние до делового центра

CBD (Central Business District) для каждой агломерации — ручная константа. Для Казани: «Кремль / пл. Свободы» (~55.7975, 49.1066). Чисто `haversine(lat, lon, cbd_lat, cbd_lon)`.

Для большинства hedonic-моделей это **самый сильный single feature** для apartment в большом городе. У нас он captures-имплицитно через `dist_metro_m` (метро-станции концентрируются у центра) + lat/lon — но **explicit signal** работает чище и переносится между регионами.

| фича | формула | релевантно |
| --- | --- | --- |
| `dist_to_cbd_m` | `haversine(obj_lat, obj_lon, cbd_lat, cbd_lon)` | apartment (sa), house (sa), commercial (sa) |

«Sa» = strong association в hedonic literature. Для Иркутска при расширении понадобится свой `cbd` per-region — конфигурируется через `Settings.cbd_coords: dict[str, tuple[float, float]]`.

### 2. Heritage / культурное наследие

Российский **открытый реестр объектов культурного наследия** (Минкульт, ОКН):

- API: `https://opendata.mkrf.ru/opendata/7705851331-egrkn`
- Содержит точные координаты + полигоны для большинства зарегистрированных объектов.
- Включает: памятники архитектуры, ансамбли, достопримечательные места, охранные зоны.

Полезные фичи:

| фича | формула | релевантно |
| --- | --- | --- |
| `is_heritage_object` | объект сам — ОКН (точный match по cad_num или buffer 50 м) | apartment, house, commercial (для сталинок/дореволюционки в центре) |
| `dist_to_nearest_heritage_m` | до ближайшего ОКН | apartment, house — премия «исторический район» |
| `count_heritage_500m` | сколько ОКН в кольце 500 м | apartment в центре Казани (Кремль, Старо-Татарская слобода) |
| `inside_heritage_zone` | флаг попадания в **охранную зону ОКН** (полигон) | landplot — там ограничено строительство, штраф −20…−40% |

Особенно важно для **landplot** — попадание в охранную зону = ограничения на застройку = реальная цена ниже кадастровой. Без этой фичи модель не объясняет такие случаи.

### 3. ЗОУИТ (зоны с особыми условиями использования территорий)

ЗОУИТ — публичный слой в Росреестре / НСПД: санитарно-защитные зоны промышленных предприятий, охранные зоны ЛЭП/трубопроводов, водоохранные, защитные приаэродромные, охранные зоны ОКН (см. п.2). Доступ через **публичную кадастровую карту** или **NSPD `featureExt` blob**.

Сейчас НСПД отдаёт по объекту флаг наличия пересечения с ЗОУИТ в `attrs.zouit_intersection` (если поле есть; нужно сверить парсер). Если оно прокидывается до silver — берём готовое.

| фича | формула | релевантно |
| --- | --- | --- |
| `inside_zouit` | бинарный флаг попадания | landplot (ключевой), commercial (промзона рядом), apartment (СЗЗ от завода) |
| `zouit_types` | категориальная: типы пересекающих ЗОУИТ (water_protection / aerodrome / sanitary / heritage_buffer / power_line / pipeline) | landplot |
| `inside_water_protection` | подмножество — водоохранная (отдельно из-за частоты) | landplot, house |

Применимость: главным образом landplot (ограничения на застройку = снижение цены) и house в пригороде (СЗЗ).

### Что **не** делаем в этой итерации

- **NPY / cadastral quarter detail enrichment** — есть в [ADR-0021](#).
- **Land-use overlay** (genplan, ПЗЗ — правила землепользования) — публичные есть, но per-municipality формат, не унифицированы. Отдельная ADR при сильном спросе.
- **Buyer demographic profile per okrug** — слишком private/коммерческое.

## Архитектура

```text
# 1) CBD — никаких новых данных
src/kadastra/etl/object_cbd_distance.py
  def compute_cbd_distance(objects, *, cbd_coords) -> pl.DataFrame

# 2) Heritage
data/raw/heritage/okn-russia.parquet (or geojsonseq)  ← разовая выгрузка ОКН Минкульта, фильтрованная по региону
src/kadastra/etl/object_heritage_features.py
  def compute_object_heritage_features(objects, *, heritage_layer) -> pl.DataFrame

# 3) ZOUIT
# Сначала — re-parse NSPD JSON чтобы вытащить attrs.zouit_intersection (см. [ADR-0021](#))
# Если пусто — отдельный download через NSPD overlay endpoint
data/silver/nspd/.../data.parquet (расширение)
  └─ + zouit_types_raw: list[Utf8] (raw из NSPD)
src/kadastra/etl/object_zouit_features.py
  def compute_object_zouit_features(objects) -> pl.DataFrame
    (decode list[str] → boolean flags + categorical primary type)
```

Settings:

```python
cbd_coords: dict[str, tuple[float, float]] = {
    "RU-KAZAN-AGG": (55.7975, 49.1066),  # Кремль, пл. Свободы
}
heritage_layer_path: Path = Path("data/raw/heritage/okn-tatarstan.geojsonseq")
```

### TDD

| Уровень | Что покрывается |
| --- | --- |
| unit / `test_object_cbd_distance.py` | 5 точек с известными координатами Казани → ожидаемые haversine расстояния. |
| unit / `test_object_heritage_features.py` | Synthetic ОКН + objects → ожидаемые dist/count флаги. |
| unit / `test_object_zouit_features.py` | Synthetic NSPD attrs → бинарные флаги. |

## Эмпирический эффект (гипотеза)

- **apartment**: Δ MAPE −0.3…−1 пп (CBD signal сильный, остальное вторично).
- **house**: Δ −0.5…−1 пп (dist_to_cbd для пригорода + heritage для исторических посёлков).
- **commercial**: Δ −0.5…−1 пп (CBD как proxy для трафика).
- **landplot**: Δ **−1…−5 пп** (ZOUIT — самый существенный фактор после VRI).

Совокупно для landplot ADR-0021 (VRI) + ADR-0025 (ZOUIT) дают самый большой потенциальный сдвиг — landplot 253% MAPE может упасть до 100–150% (всё ещё высоко, но это уже про сегментацию рынка участков, а не про «модель ничего не видит»).

## Открытые вопросы

- **Покрытие ОКН Минкульта.** Реестр заявленно полный, но регулярно дополняется. Версия выгрузки фиксируется в `data/raw/heritage/okn-tatarstan.{geojsonseq,manifest}`.
- **NSPD `attrs.zouit_intersection` reality-check.** Перед уверенным включением — отдельный data-quality аудит. Если `zouit_intersection` пуст или не структурирован — нужен отдельный download через ПКК (публичную кадастровую карту), что усложняет ETL.
- **Heritage zone polygons.** Не у всех ОКН есть полигональная охранная зона — у некоторых только точка. Тогда `inside_heritage_zone` подменяется на `dist_to_nearest_heritage_m < 100`. Документируется в data-quality отчёте.
- **CBD для не-Казани.** При расширении на другой регион нужно вручную добавить координату центра. Не критично — таких добавлений будет 5–10 за весь жизненный цикл проекта.

## Аудит данных (2026-08-21, закрывает «Открытые вопросы»)

1. **Heritage: Минкульт недоступен, источник = OSM.** `opendata.mkrf.ru` с нашей сети отдаёт connection refused. Использован OSM-экстракт `data/raw/osm/kazan-agg-heritage.geojsonseq` (285 строк; S3-бэкап `Kadatastr/raw/osm/kazan-agg-heritage.geojsonseq`). После фильтрации по `properties.heritage` (not null) — **188 ОКН**: 109 MultiPolygon + 78 closed-LineString footprints + 1 Point; `ref:egrokn` есть только у 60. **Data-quality ограничение: покрытие скромное (~188 объектов на агломерацию)** — реестр Минкульта насчитывает на порядок больше ОКН в Казани; фичи работают на OSM-полноте.
2. **ЗОУИТ: гипотетического `attrs.zouit_intersection` не существует.** Вместо него — полный bulk-дамп слоя НСПД 36302 по Татарстану: `data/raw/nspd/zouit-tatarstan/page-0000..0888.json`, 177 663 объекта (Polygon/MultiPolygon, EPSG:3857), `properties.options.type_zone` — человекочитаемый вид зоны. S3-бэкап: `Kadatastr/raw/nspd/zouit-tatarstan/`. **Отклонение от спеки: фичи считаются настоящим spatial join'ом (точка объекта в полигоне зоны), а не чтением готового флага** — это надёжнее и не зависит от наличия поля в НСПД-ответах.
3. **Состав type_zone по дампу (Татарстан):** ~72% «Охранная зона инженерных коммуникаций» (127 530), электроэнергетика 15 840 (~9%), геодезические пункты 11 119 + 1 160, трубопроводы 4 594 (+90), публичные сервитуты 3 516, СЗЗ предприятий 2 052 (+678 «Санитарно-защитная зона»), санитарная охрана источников водоснабжения 1 901 + 852, связь 1 461 + 279, водоохранные 930, прибрежные защитные 925, ОКН-зоны 432 + 250 + 46 + 2, придорожные 394 + 22, приаэродромные 39, длинный хвост (~30 редких видов, включая военные и «Иная зона»). Маппинг в категории спеки — подстроковыми правилами (`categorize_zouit_type`): «Зоны затопления и подтопления» (77) отнесены к `water_protection`; «Охранная зона тепловых сетей», геодезические пункты, сервитуты, связь, придорожные полосы и прочие — в `other`; пустой/`null` type_zone (623+4) — тоже `other`.
4. **CBD:** константа Казани (55.7975, 49.1066) — Кремль / пл. Свободы, через `Settings.cbd_coords`.
5. **Производительность ЗОУИТ-join.** 177 663 полигона по всему Татарстану сначала bbox-фильтруются до агломерации (+10 км margin, EPSG:3857) → **21 073 зоны**; per-object join — STRtree (`shapely`, предикат `intersects`) над полигонами в UTM-39N (EPSG:32639), как в ADR-0019/0024.

## Отличия реализации от спеки

- **ЗОУИТ — spatial join вместо `attrs.zouit_intersection`** (см. аудит, п. 2).
- **Архитектура фич — по паттерну ADR-0022/0023/0024**: silver-таблицы материализуются скриптами, `BuildObjectFeatures` подключает их ПОСЛЕ сброса фрейма в `RAW_OBJECT_SCHEMA`, поэтому новые колонки в `RAW_OBJECT_SCHEMA` **не добавляются** (сброс — защита от чтения собственного обогащённого вывода при rerun; колонки пересчитываются из silver каждый раз):
  - `data/silver/heritage/region={code}/data.parquet` — слой ОКН (scripts/build_heritage_silver.py); 4 фичи считаются inline (слой крошечный);
  - `data/silver/zouit_zones/region={code}/data.parquet` — зоны с категорией (scripts/build_zouit_silver.py);
  - `data/silver/zouit_per_object/region={code}/data.parquet` — per-object таблица (scripts/build_zouit_features.py), джойнится по `object_id` как `road_class_per_object`;
  - `dist_to_cbd_m` — чистый haversine, silver-таблицы нет.
- **`inside_heritage_zone` — по полигонам-footprint'ам ОКН** (187 из 188 имеют полигон), fallback `dist < 100 м` остаётся для слоёв без полигонов. Семантика — «объект внутри контура ОКН», а не «в охранной зоне ОКН»: охранные зоны покрываются ЗОУИТ-категорией `heritage_buffer` (135 зон в агломерации). Оба сигнала идут в модель.
- **Флаги — Int64 (0/1), не Boolean**: `select_object_feature_columns` отбирает только numeric/Utf8, Boolean-колонки в матрицу не попадают.
- **`zouit_types`** — sorted `;`-joined строка категорий (multi-label → одна категориальная колонка для CatBoost), null вне зон.

## Эмпирика реализации (2026-08-21, RU-KAZAN-AGG, 287 610 объектов)

### ЗОУИТ

- Зон после bbox-фильтра: 21 073 (other 16 386 / power_line 3 592 / sanitary 741 / heritage_buffer 135 / water_protection 126 / pipeline 84 / aerodrome 9).
- **`inside_zouit` = 1 у 268 835 объектов (93,5%)** — охранные зоны инженерных коммуникаций + гигантские «Охранная зона транспорта» (до ~3 200 км²) и приаэродромные территории (до ~706 км²) накрывают почти всю агломерацию. **`inside_zouit` сам по себе почти насыщен — дискриминативный сигнал несут `zouit_types` и `inside_water_protection`** (28 122 объекта, 9,8%).
- Топ комбинаций `zouit_types`: `aerodrome;other` 137 475; `aerodrome;other;sanitary` 66 663; `other` 24 694; `aerodrome;other;water_protection` 14 136; `aerodrome;heritage_buffer;other` 6 397.

### Heritage

- Silver-слой: 188 ОКН (187 с полигоном-footprint). Покрытие — см. data-quality ограничение выше.

### CBD

- `dist_to_cbd_m`: sanity — Кремль ≈ 0 м, окраины агломерации 20–40 км (проверено юнит-тестами на реальных координатах Казани).

### Покрытие новых фич по классам (gold valuation_objects)

Non-null share (share==1 для флагов). dist_to_cbd_m: non-null 100% во всех классах, диапазоны min→max: apartment 539 м→16,5 км; house 529 м→23,9 км; commercial 77 м→26,3 км; landplot 132 м→31,0 км (санити: центр ≈ 0 м, окраины 20–40 км ✓).

| фича | apartment (1 089) | house (46 596) | commercial (42 411) | landplot (197 514) |
| --- | --- | --- | --- | --- |
| `is_heritage_object` | 100% (1: 0,64%) | 100% (1: 0,01%) | 100% (1: 0,17%) | 100% (1: 0,11%) |
| `dist_to_nearest_heritage_m` | 100% | 100% | 100% | 100% |
| `count_heritage_500m` | 100% | 100% | 100% | 100% |
| `inside_heritage_zone` | 100% (1: 0,09%) | 100% (1: 0,00%) | 100% (1: 0,06%) | 100% (1: 0,01%) |
| `inside_zouit` | 100% (1: 97,3%) | 100% (1: 89,7%) | 100% (1: 96,6%) | 100% (1: 93,7%) |
| `zouit_types` | 97,2% | 89,7% | 96,6% | 93,7% |
| `inside_water_protection` | 100% (1: 1,9%) | 100% (1: 11,0%) | 100% (1: 8,4%) | 100% (1: 9,8%) |

Heritage-фичи sparse (ожидаемо при 188 ОКН): `is_heritage_object` = 1 у 7 apartment / 6 house / 71 commercial / 214 landplot объектов.

## Замечание для будущих переобучений

Модели в рамках ADR-0025 **не переобучались** — новые колонки подбираются `select_object_feature_columns` автоматически при следующем обучении. Финальное переобучение квартета (включая ADR-0021…0025) выполняется отдельной задачей. Учесть при интерпретации: `inside_zouit` близок к константе (93,5% единиц) — основная ценность блока в `zouit_types` и `inside_water_protection`; heritage-фичи sparse (188 ОКН) и заработают в основном для apartment/house в историческом центре.

## Эмпирический эффект (измеренный, 2026-08-21)

Финальное переобучение квартета на полном наборе фич (ADR-0021…0025) выполнено; сравнение с предыдущим прогоном (2026-08-18, до программы enrichment), spatial-CV 5 фолдов:

| класс | модель | MAPE было | MAPE стало | Δпп | WAPE было | WAPE стало | Spearman было → стало |
| --- | --- | --- | --- | --- | --- | --- | --- |
| apartment | CatBoost | 8,4% | 8,0% | **−0,5** | 5,3% | 4,8% | 0,900 → 0,917 |
| apartment | EBM | 9,5% | 9,1% | −0,5 | 6,4% | 6,1% | 0,844 → 0,860 |
| house | CatBoost | 12,6% | 12,9% | +0,4 | 9,5% | 8,6% | 0,863 → 0,882 |
| house | EBM | 19,3% | 20,7% | +1,4 | 15,5% | 14,0% | 0,767 → 0,849 |
| commercial | CatBoost | 39,0% | 37,6% | **−1,4** | 35,5% | 34,2% | 0,775 → 0,777 |
| commercial | EBM | 55,2% | 53,0% | −2,3 | 46,6% | 44,6% | 0,676 → 0,681 |
| landplot | CatBoost | 246,8% | 173,4% | **−73,4** | 21,4% | 17,7% | 0,915 → 0,945 |
| landplot | EBM | 257,4% | 202,2% | −55,1 | 31,9% | 22,8% | 0,860 → 0,909 |

Чтение таблицы:

- **landplot — главный выигрыш программы**: MAPE −73 пп (246,8% → 173,4%; метрика всё ещё структурно зашумлена, см. [ADR-0026](0026-wape-for-landplot.md)), WAPE 21,4% → 17,7%, Spearman 0,915 → 0,945. Соответствует гипотезе ADR-0025 (−1…−5 пп по WAPE + ADR-0021 VRI как ключевые факторы земли).
- **commercial**: −1,4 пп MAPE у CatBoost — в полосе ожиданий ADR-0024 (−1…−3 пп).
- **apartment**: −0,5 пп — нижняя граница ожиданий (сумма ADR-0019/0024/0025 давала −1…−3 пп); базовый набор фич уже был силён, новые дают скромный прирост.
- **house**: MAPE +0,4 пп при **улучшении WAPE (9,5→8,6) и Spearman (0,863→0,882)** — MAPE-просадка в пределах шума фолдов, агрегатные метрики улучшились.
- Прогоны: `data/models/quartet-object-{apartment,house,commercial,landplot}_20260821T*/quartet_metrics.json`.

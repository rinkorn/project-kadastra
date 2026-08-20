# ADR-0021: Богатые ЕГРН/НСПД-атрибуты (VRI, материалы, этажность)

**Статус:** Accepted
**Дата:** 2026-04-26
**Принят:** 2026-08-20 (реализация + эмпирика — см. §«Результат реализации»)
**Реализует:** [info/grid-rationale.md §6](../grid-rationale.md) (Объектные ЦОФ — характеристики объекта).
**Опирается на:** [ADR-0009](0009-real-cadastre-target-via-nspd.md) (NSPD как источник target и базовых атрибутов), [ADR-0017](0017-object-geometry-passthrough-for-inspector.md) (полигон уже тащим).

## Контекст

Сейчас из НСПД ([silver/nspd/...](../../data/silver/nspd)) мы вытаскиваем минимум: `cad_num`, `area_m2`, `year_built`, `levels`, `flats`, `polygon_wkt_3857`, базовый territorial-блок. Из реально доступных полей в JSON-выдаче НСПД **остаются за бортом**:

- **VRI** (вид разрешённого использования) — для landplot **критическая** фича. Без VRI модель не может различить «ИЖС-участок 6 соток в дачном товариществе» (~0.5 млн ₽) и «промышленность под складскую застройку 6 соток на трассе» (~30 млн ₽). 253% MAPE на landplot — следствие.
- **`category_zem`** (категория земель: с/х, населённого пункта, промышленности, ООПТ, лесфонд, водфонд, запас) — спутник VRI, чуть грубее, всегда заполнен.
- **Материал стен** для зданий: `walls_material` (кирпич/панель/монолит/деревянный/смешанный/прочее) — мощный signal для apartment (кирпич премиум +20%, панель базовая, монолит средне-премиум).
- **`floors_total`** (этажей в здании всего) — у нас есть `levels` для квартир (этажей в здании), но для **квартир** интересно ещё и:
  - **`floor_in_building`** (этаж самой квартиры) — первый и последний этажи дешевле на 5–10%. Может не быть в НСПД для квартир (нужно сверить).
- **`ceiling_height_m`** — высота потолков, premium-фича.
- **`is_emergency`** (аварийный фонд) — пытались вытащить ранее, нашли пустоту в текущей выгрузке. Перепроверить парсер: возможно, поле в другом ключе JSON.

## Решение

Расширить ЕТЛ-парсеры НСПД (silver-уровень), чтобы прокидывать в gold следующие колонки. Для каждого класса актуальны разные подмножества — пайплайн отдаёт всё, селектор фич сам разберётся.

### Признаки для зданий (apartment / house / commercial)

Сверка со спекой при реализации (2026-08-20): фактические ключи raw JSON
отличаются от предполагавшихся, а `walls_material` и `floors_total`
оказались **уже в пайплайне** под именами `materials` и `levels`
(silver-парсер читал их с ADR-0009). Существующие колонки не
переименовывались; в gold добавлены только реально новые поля.

| колонка в gold | тип | источник в NSPD JSON | статус реализации |
| --- | --- | --- | --- |
| `materials` (= `walls_material` спеки) | Utf8 (cat) | `options.materials` | уже была; silver coverage 99.7% |
| `levels` (= `floors_total` спеки) | Int64 | `options.floors` | уже была; silver coverage 98.0% |
| `underground_floors` | Int64 | `options.underground_floors` | **добавлен passthrough в gold** (в silver была); coverage 12–78% по классам |
| `kadnum_quarter` | Utf8 (cat) | `options.quarter_cad_number` | **добавлена** (silver + gold), 100% non-null |
| ~~`floor_in_building`~~ | — | — | **known gap**: ключа нет в выгрузке (apartment-уровня в НСПД-слое зданий нет) |
| ~~`ceiling_height_m`~~ | — | — | **known gap**: ключа нет в выгрузке |
| ~~`is_emergency`~~ | — | — | **known gap**: ключа нет в выгрузке (ни `emergency`, ни `condition`) |

### Признаки для landplot

| колонка в gold | тип | источник (фактический) | статус реализации |
| --- | --- | --- | --- |
| **`vri`** | Utf8 (cat) | `options.permitted_use_established_by_document` (название VRI; ключа `permitted_use_name` в выгрузке нет) | **добавлена**, 99.9% non-null в silver |
| ~~`vri_code`~~ | — | — | **known gap**: ключа `permitted_use_code` в выгрузке нет, кодированного VRI-классификатора НСПД не отдаёт |
| `category_zem` | Utf8 (cat) | gold-алиас silver-колонки `land_record_category_type` | **добавлена**, ~100% non-null |
| `kadnum_quarter` | Utf8 (cat) | `options.quarter_cad_number` (готовое поле выгрузки, не derive из `cad_num`) | **добавлена**, 100% non-null |

`kadnum_quarter` — ультрадешёвая категориальная: «кадастровый квартал» как proxy уровня района/посёлка. CatBoost любит такие mid-cardinality cat-фичи.

### Применимость по классам — селектор не настраиваем

Каждый класс получает все поля; где non-null процент низкий, там модель просто видит null. Это безопасно для CatBoost native-NA-handling и EBM с per-feature missing-bin.

### Что **не** делаем в этой итерации

- **Лиц.собственника / правообладатель** — публично не отдаётся (Росреестр требует выписку с подписью). Не источник для open-data пайплайна.
- **История переходов прав / обременения** — то же.
- **Кадастровая стоимость прошлых туров** — это *target*, не feature.
- **Отдельный NLP по `readable_address`** — адрес уже разбит на territorial-блок ([ADR-0015](0015-territorial-features-via-gar.md)). Дальнейший парс не оправдан.

## Архитектура

```text
data/raw/nspd/{buildings,landplots}-kazan/*.json     ← raw JSON с фичами
            │
            ▼ scripts/load_nspd_raw_objects.py (парсеры расширены)
data/silver/nspd/region={code}/source={...}/data.parquet
  └─ + kadnum_quarter (buildings, landplots), vri (landplots)
     (walls_material/floors_total уже были как materials/floors)
            │
            ▼ AssembleNspdValuationObjects (passthrough, RAW_OBJECT_SCHEMA +4)
gold/valuation_objects/...
  └─ + underground_floors, kadnum_quarter, vri, category_zem
            │
            ▼ BuildObjectFeatures (selector подхватывает автоматически)
```

Изменения:
1. **`src/kadastra/etl/parse_nspd_feature.py`** — парсеры читают
   `quarter_cad_number` (оба источника) и
   `permitted_use_established_by_document` → `vri` (landplots).
   Литерал `"None"` (так НСПД сериализует отсутствие значения)
   мапится в null для новых колонок; поведение legacy-колонок не
   менялось. Схемы — в **`src/kadastra/etl/read_nspd_dir.py`**.
2. **silver-схема расширяется** — rebuild silver/nspd из сохранённого
   raw JSON ([data/raw/nspd/](../../data/raw/nspd/)), перевыкачка не
   нужна.
3. **gold-схема расширяется** — `RAW_OBJECT_SCHEMA` +4 колонки
   (`underground_floors`, `kadnum_quarter`, `vri`, `category_zem`);
   `AssembleNspdValuationObjects` тащит их passthrough'ом
   (`category_zem` — алиас `land_record_category_type`). Добавление в
   `RAW_OBJECT_SCHEMA` обязательно: `BuildObjectFeatures` при reruns
   сбрасывает фрейм к этой схеме (идемпотентность, эпик 001).
4. **Selector** ([object_feature_columns.py](../../src/kadastra/ml/object_feature_columns.py))
   подхватывает новые колонки без изменений: `underground_floors` —
   numeric, `kadnum_quarter`/`vri`/`category_zem` — categorical; ни
   одна не входит в `_NON_FEATURE_COLUMNS`.

### TDD

| Уровень | Что покрывается |
| --- | --- |
| unit / `tests/unit/test_parse_nspd_buildings.py` | Синтетический NSPD-JSON → ожидаемый extract `walls_material`, `floors_total` etc. |
| unit / `tests/unit/test_parse_nspd_landplots.py` | Синтетический NSPD-JSON → ожидаемый extract `vri`, `vri_code`, `category_zem`. |
| data-quality / отдельный отчёт | После rebuild silver — non-null %, top-N значений по категориальным. Sanity check: `vri` имеет ~50–100 уникальных, не миллион. |

## Эмпирический эффект (гипотеза)

- **landplot**: Δ MAPE **−50…−150 пп** ожидаемо. Без VRI 253% — артефакт смешения «жилая ИЖС» / «коммерческая под застройку» / «сельхоз» в одной модели. С VRI как cat-feature CatBoost разделит распределения, EBM покажет per-VRI shape.
  - Если эффект окажется ≤ −20 пп — значит VRI распределён очень неравномерно (большинство объектов одного типа), либо `cost_value_rub` в ЕГРН для landplot жёстко смещён независимо от VRI (тогда сама ЕГРН-таргет «сломан» для участков, и это — отдельный вывод для ADR-0010).
- **apartment**: Δ −1…−3 пп. Материал стен и floor_in_building — мощные сигналы.
- **house**: Δ −0.5…−2 пп. Материал стен влияет; floor_in_building не применим.
- **commercial**: Δ −0.5…−1 пп. Floors_total как proxy типа здания.

## Открытые вопросы (закрыты при реализации 2026-08-20)

- ~~**Полнота NSPD JSON для VRI.**~~ — закрыт: `permitted_use_established_by_document`
  заполнен у 99.9% участков; 225 записей с литералом `"None"` мапятся в null.
- ~~**Codebook для `vri_code`.**~~ — неактуально: кодированного VRI
  (`permitted_use_code`) в выгрузке нет, `vri` — свободный текст. Его
  нормализация (13k уникальных, дубли по регистру/формулировке —
  «Садоводство» / «ведение садоводства») — кандидат на отдельную
  итерацию, не блокер.
- ~~**Объём rebuild silver/NSPD.**~~ — закрыт: rebuild из локального raw
  занял минуты (silver ~3 мин, assemble — секунды), перевыкачка не
  потребовалась.
- ~~**`is_emergency` пустой.**~~ — закрыт как known gap: ключи
  `is_emergency`/`ceiling_height`/`floor_in_building` в текущей выгрузке
  отсутствуют физически (проверено по всем 91 864 зданиям), не
  препятствие к merge.

## Результат реализации (2026-08-20)

Реализовано в ветке `feature/nspd-rich-attributes`:

- **Парсеры.** `src/kadastra/etl/parse_nspd_feature.py`: buildings +
  `kadnum_quarter`; landplots + `vri`, `kadnum_quarter`. Новый хелпер
  `_to_str_na` глушит литерал `"None"` только для новых колонок.
  Схемы — `src/kadastra/etl/read_nspd_dir.py`.
- **Silver rebuild** (`scripts/load_nspd_raw_objects.py`, без
  перевыкачки): buildings 91 743 строки, landplots 198 789 строк.
  Coverage в silver: `kadnum_quarter` 100%/100%, `vri` 99.9%
  (13 171 уникальных), `land_record_category_type` 100% (7 значений),
  `materials` 99.7% (198 значений), `floors` 98.0%,
  `underground_floors` 12.7%.
- **Gold.** `RAW_OBJECT_SCHEMA` +4 (`underground_floors`,
  `kadnum_quarter`, `vri`, `category_zem`) + passthrough в
  `AssembleNspdValuationObjects`. Полный прогон
  `scripts/build_object_features.py` на всех 4 классах (2026-08-20) —
  колонки пережили enrichment (230 колонок в gold), coverage:

| asset_class | объектов | vri | category_zem | kadnum_quarter | underground_floors | materials | levels |
| --- | --- | --- | --- | --- | --- | --- | --- |
| apartment | 1 089 | — (null) | — (null) | 100% (273 уник.) | 78.5% | 100% (25 уник.) | 92.7% |
| house | 46 596 | — (null) | — (null) | 100% (2 608) | 12.0% | 99.8% (144) | 100% |
| commercial | 42 411 | — (null) | — (null) | 100% (2 231) | 12.0% | 99.6% (115) | 96.0% |
| landplot | 197 514 | 100% (13 084 уник.) | 100% (6 уник.) | 100% (3 648) | — (null) | — (null) | — (null) |

  Top-5 значений (silver после пространственного фильтра):
  - `vri`: Садоводство (25 074), Гараж боксового типа (17 924),
    Индивидуальный жилой дом (13 528), Индивидуальное жилищное
    строительство (8 248), Индивидуальный гараж (7 270).
  - `category_zem`: Земли населенных пунктов (196 120), Земли
    сельскохозяйственного назначения (2 417), Земли промышленности…
    (137), Категория не установлена (83), Земли лесного фонда (25).
  - `materials`: Кирпичные (47 835), Деревянные (15 343), Из прочих
    материалов (8 709), Смешанные (6 229), Бетонные (2 894).

  Sanity: `vri` имеет 13k уникальных (свободный текст с дублями по
  регистру), `category_zem` — 7 значений, `kadnum_quarter` — 3.6k
  кварталов. Распределения соответствуют ожиданиям по Татарстану
  (доминируют садоводство/гаражи/ИЖС и земли населённых пунктов).

- **Тесты.** +7 unit-тестов: `test_parse_nspd_feature.py` (extract +
  `"None"`→null + missing→null), `test_read_nspd_dir.py` (схема),
  `test_assemble_nspd_valuation_objects.py` (passthrough per class).
  Всего 648 passed, ruff/pyright чисто.

### Замечание для будущих переобучений

Новые колонки добавлены в gold **после** последнего обучения моделей;
модели в `data/models/` их не видели. При следующем переобучении
(`train_object_models`) селектор подхватит автоматически:
`underground_floors` — как numeric, `kadnum_quarter` / `vri` /
`category_zem` — как categorical (проверено
`select_object_feature_columns` на свежем gold). Учесть:
- `vri` — high-cardinality свободный текст (13k уникальных) с дублями
  по регистру; CatBoost справится, но нормализация VRI (кластеризация
  формулировок) может дать дополнительный эффект — отдельная задача.
- `vri`/`category_zem` для зданий и `underground_floors`/`materials`/
  `levels` для landplot — полностью null (класс-специфика по дизайну),
  модели увидят константный missing.
- Ожидаемый эффект — см. §«Эмпирический эффект»: основной на landplot
  (VRI + кадастровый квартал), минус десятки пп MAPE.

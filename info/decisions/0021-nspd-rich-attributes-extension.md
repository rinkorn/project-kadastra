# ADR-0021: Богатые ЕГРН/НСПД-атрибуты (VRI, материалы, этажность)

**Статус:** Proposed
**Дата:** 2026-04-26
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

| колонка | тип | источник в NSPD JSON | ожидаемый non-null % |
| --- | --- | --- | --- |
| `walls_material` | Utf8 (cat) | `attrs[*].materialOfStructures` или похоже | ~80% (apartment), ~50% (house) |
| `floors_total` | Int64 | `attrs[*].floorsCount` (всего в здании) | ~95% (apartment), ~80% (house), ~70% (commercial) |
| `floor_in_building` | Int64 | для apartment — `attrs[*].floor` | требует проверки |
| `ceiling_height_m` | Float64 | `attrs[*].ceilingHeight` | низкий, ~30% (но где есть — сильный signal) |
| `is_emergency` | Bool | `attrs[*].emergency` или производное от `condition` | требует data-quality аудита |

### Признаки для landplot

| колонка | тип | источник | актуальность |
| --- | --- | --- | --- |
| **`vri`** | Utf8 (cat) | `attrs[*].permitted_use_name` (название VRI) | ~95% non-null |
| `vri_code` | Utf8 (cat) | `attrs[*].permitted_use_code` (классификатор) | ~95% non-null |
| `category_zem` | Utf8 (cat) | `attrs[*].land_category` | ~99% non-null |
| `kadnum_quarter` | Utf8 (cat) | первые 11 символов `cad_num` (16:00:N:...) | 100% non-null, derivable |

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
            ▼ scripts/build_nspd_silver.py (расширить парсер)
data/silver/nspd/region={code}/source={...}/data.parquet
  └─ + walls_material, floors_total, floor_in_building, ceiling_height_m,
     is_emergency, vri, vri_code, category_zem, kadnum_quarter
            │
            ▼ AssembleNspdValuationObjects (passthrough)
gold/valuation_objects/...
            │
            ▼ BuildObjectFeatures (selector подхватывает автоматически)
```

Изменения:
1. **`src/kadastra/etl/parse_nspd_*`** — расширить парсер: новые поля прочитать из JSON, добавить в полярную схему. Если поле отсутствует — null. Существующие колонки трогать не надо.
2. **silver-схема расширяется** — потребует rebuild silver/nspd, который требует сохранённый raw JSON. Raw сохранён ([data/raw/nspd/](../../data/raw/nspd/)).
3. **gold-схема расширяется** — после rebuild silver, `AssembleNspdValuationObjects` тащит новые поля без изменений.
4. **Selector** ([object_feature_columns.py](../../src/kadastra/ml/object_feature_columns.py)) сам подхватит numeric/categorical новые колонки.

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

## Открытые вопросы

- **Полнота NSPD JSON для VRI.** Парсер расследует raw JSON: какие поля реально присутствуют, в каких ключах, какова кардинальность `vri_code`. Этап 0 ADR-0021 = data-quality аудит существующего raw без перевыкачки.
- **Codebook для `vri_code`.** ВРИ имеет официальный классификатор (приказ Минэкономразвития №540 + переход на ОК 011-2024). Если в NSPD приходит код — можно дешифровать в человеко-читаемое имя для `vri` (если оно отсутствует). Это deterministic mapping из публичного reference (нужен один parquet `data/raw/reference/vri_codebook.parquet`).
- **Объём rebuild silver/NSPD.** Сейчас raw JSON лежит локально — rebuild парсера должен быть быстрым (минуты). Перевыкачка не нужна.
- **`is_emergency` пустой.** Если после re-parse поле всё ещё null для всех — отдельным data-quality репортом обозначить как «known gap, не препятствие к merge».

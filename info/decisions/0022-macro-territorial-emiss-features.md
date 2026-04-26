# ADR-0022: Макро-территориальные ЦОФ (EMISS / Росстат)

**Статус:** Proposed
**Дата:** 2026-04-26
**Реализует:** [info/grid-rationale.md §9](../grid-rationale.md) (Территориальные ЦОФ — социально-экономический контекст).
**Опирается на:** [ADR-0010 §«Эмпирическое подтверждение тезиса»](0010-methodology-compliance-roadmap.md) (EMISS silver уже в проекте), [ADR-0015](0015-territorial-features-via-gar.md) (`oktmo_full`/`okato` присоединяются к объекту через GAR).

## Контекст

Сейчас в gold-схеме у каждого объекта есть `oktmo_full` и `okato` через [ADR-0015](0015-territorial-features-via-gar.md). Но как **признаки модели** мы их используем только в виде категориальных id (`intra_city_raion`, `mun_okrug_name`). CatBoost с ними справляется в пределах обученного множества районов, но:

1. **Не генерализуется** на новые территории. Когда регион расширится с Татарстана на Иркутскую агломерацию (как закладывается долгосрочно) — модель встретит unseen `okato` и упадёт в global-mean.
2. **Не отражает динамику.** «Раион N был в 2019 году дешёвый, к 2025 догнал центр» — сигнал, который cat-id не передаёт. Нужны числовые per-OKATO признаки, которые **меняются по времени**.
3. **EMISS уже подключён** ([ADR-0010, добавление 2026-04-26](0010-methodology-compliance-roadmap.md)). Сейчас читаем `#61781` (apartment market reference) для inspector. Тот же пайплайн (`scripts/parse_emiss_xls_to_parquet.py`) тривиально расширяется на другие indicator'ы.

Источник — fedstat.ru (EMISS / Росстат) — публичный, бесплатный, обновляется регулярно.

## Решение

Подгружать набор per-OKATO/OKTMO indicator'ов из EMISS, превращать в silver-таблицы по образцу `silver/emiss/61781/`, и через `compute_object_municipality_features` ([uses GAR-derived oktmo_full](../../src/kadastra/etl/object_municipality_features.py)) делать enrichment объекта.

### Indicator'ы из EMISS

Стартовый набор (расширяемо):

| EMISS код | название | гранула | per-class |
| --- | --- | --- | --- |
| **57792** | Среднемесячная начисленная заработная плата работников организаций по муниципальным образованиям | OKTMO, год | apartment, house, commercial |
| **31074** | Численность населения на 1 января | OKTMO, год | все |
| **34466** | Объём ввода в эксплуатацию жилья (м²) | OKTMO/субъект, год | apartment, house |
| **44164** | Уровень безработицы | OKTMO/субъект, год | все |
| **40464** | Объём оборота розничной торговли | OKTMO/субъект, год | commercial |
| 61781 | (уже в проекте) рыночная цена жилья м² | город, квартал | apartment |

Конкретные коды могут уточниться при сверке в fedstat — выбираем те, что покрывают именно муниципалитет (OKTMO), а не только субъект РФ.

### Производные фичи (per-object)

После джойна по `oktmo_full` (или `okato`, если EMISS даёт только okato) и года target:

| фича | формула | смысл |
| --- | --- | --- |
| `oktmo_avg_salary_rub` | `salary_57792` напрямую | покупательная способность населения |
| `oktmo_population` | `pop_31074` | масштаб муниципалитета |
| `oktmo_population_density` | `pop_31074 / oktmo_area_km2` | плотность (площадь из GAR) |
| `oktmo_housing_volume_5y_m2` | сумма `34466` за последние 5 лет | темп новой застройки |
| `oktmo_unemployment_pct` | `44164` напрямую | депрессивность территории |
| `oktmo_retail_turnover_per_capita` | `40464 / pop_31074` | экономическая активность |

Все числовые. Категориальный `okato` остаётся параллельно — для capture через CatBoost точечных эффектов.

### Year alignment

EMISS — temporal data. Объект имеет неявную «дату оценки» = `cost_value_year` или `year_built` или дата выгрузки кадастра. Привязываем EMISS-индикаторы по последнему доступному году ≤ target_year. На первой итерации:

```
target_year = settings.cadastre_target_year  # default = 2024 (последний полный год до текущей выгрузки)
```

— берём для всех объектов один год, не персонализируем по объекту. Если потом ЕГРН-таргет станет panel-data (несколько лет на объект), тогда индекс по `(oktmo_full, target_year)` уже подготовлен.

### Что **не** делаем в этой итерации

- **Per-quarter/monthly** EMISS-индикаторы. Стартовая грануларность — год.
- **Прогнозирование** EMISS на год вперёд (некоторые EMISS имеют lag). Берём last-available-year.
- **Per-объект temporal taxes / commercial property tax rates** — НСПД отдаёт это для landplot, но это уже не EMISS, отдельная задача.
- **Сравнение Татарстана с РФ-medium**. Дельты вида «зарплата OKTMO − средняя по РФ» — кандидат на следующую итерацию, когда покрытие выйдет за один регион.

## Архитектура

```text
data/raw/emiss/{indicator_id}.xlsx           ← скачано через fedstat
        │
        ▼ scripts/parse_emiss_xls_to_parquet.py (общий)
data/silver/emiss/{indicator_id}/data.parquet
        │
        ▼ scripts/build_macro_oktmo_features.py (новый)
data/silver/macro_oktmo_features/region={code}/year={Y}/data.parquet
  └─ oktmo_full, oktmo_avg_salary_rub, oktmo_population, ... (joined wide)
        │
        ▼ BuildObjectFeatures.execute(...)
          └─ compute_object_macro_features(...)  ← новый шаг
gold/valuation_objects/...
  └─ + 6 oktmo_*_* колонок
```

Новый модуль:

```text
src/kadastra/etl/object_macro_features.py
  def compute_object_macro_features(
      objects: pl.DataFrame,
      *,
      macro_table: pl.DataFrame,
      target_year: int,
  ) -> pl.DataFrame
```

`macro_table` — wide-формат `silver/macro_oktmo_features/region={code}/year={Y}/data.parquet`. Левый join по `oktmo_full`. Где no-match — null.

### TDD

| Уровень | Что покрывается |
| --- | --- |
| unit / `test_parse_emiss_*` | Расширение существующих парсеров для каждого indicator'а — образец из ADR-0010 EMISS-bricks. |
| unit / `test_object_macro_features.py` | Синтетический objects + synthetic macro_table → join produces expected columns. Edge cases: oktmo не найден, year > доступного. |
| integration | Не нужна — работа полностью in-memory polars. |

### Settings

```python
emiss_indicators_yearly: list[str] = ["57792", "31074", "34466", "44164", "40464"]
cadastre_target_year: int = 2024
```

## Эмпирический эффект (гипотеза)

- **apartment**: Δ MAPE −0.5…−1.5 пп. Зарплата + плотность населения дают чёткий sort по премиальности района.
- **house**: Δ −1…−2 пп. Для пригорода и сельских территорий зарплата OKTMO — основной маркер «дорогая периферия vs депрессивное село».
- **commercial**: Δ −1…−3 пп. Сильные сигналы: розничный оборот на душу, безработица.
- **landplot**: Δ −0.5…−1 пп. Меньше — у участка цена сильнее завязана на VRI ([ADR-0021](#)) и кадастровый квартал.

Сильнее всего эффект ожидается для **новых регионов** при расширении (модель станет переносимой), а не для текущего Татарстана, где CatBoost уже выучил локальные паттерны через `okato` cat-id.

## Открытые вопросы

- **Доступность OKTMO-уровня** для всех indicator'ов. Не каждый EMISS-indicator детализирован до муниципалитета; некоторые только до субъекта РФ. Если только до субъекта — для Татарстана получим константу для всех объектов (нет signal). Перед инвестицией в indicator проверяем глубину детализации.
- **Совпадение OKTMO в EMISS и в GAR.** EMISS использует OKTMO-classifier, GAR использует свой ID. У нас уже есть `oktmo_full` через GAR ([ADR-0015](0015-territorial-features-via-gar.md)) — на этапе integration убедиться, что коды побайтно совпадают (или нормализовать через codebook).
- **Поправка на инфляцию.** `oktmo_avg_salary_rub` — номинальная. На объект 2018 года target — это номинальная зарплата 2018 года, и тогда OK. На «текущую переоценку» нужна реальная (с CPI-нормировкой). На первой итерации — номинал, далее по ситуации.

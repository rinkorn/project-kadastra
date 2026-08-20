# ADR-0022: Макро-территориальные ЦОФ (EMISS / Росстат / БД ПМО)

**Статус:** Accepted
**Дата:** 2026-04-26
**Принят:** 2026-08-20 (реализация + эмпирика — см. §«Результат реализации»)
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

### Indicator'ы

Стартовый набор из таблицы ниже при реализации **не подтвердился**:
проверка на fedstat (2026-08-20) показала, что коды 57792 / 31074 /
34466 / 40464 / 44164 либо относятся к другим показателям (57792 —
объём бытовых услуг, 31074 — ИПЦ), либо не существуют; муниципального
(ОКТМО) разреза по зарплате, рознице и вводу жилья на fedstat нет
вовсе, а единственный ОКТМО-индикатор населения (31557) покрывается
полнее и стабильнее из **БД ПМО** (зеркало Росстата на tochno.st,
лицензия CC-BY, снапшот `data_bdmo_118_v20250918`).

Финальный состав (реализован):

| источник | код | название | гранула | годы (Татарстан) |
| --- | --- | --- | --- | --- |
| БД ПМО | **8112027** | Оценка численности населения на 1 января | ОКТМО (МО верхнего уровня), год | 2009–2025 |
| БД ПМО | **8213002** | Среднемесячная номинальная начисленная зарплата работников крупных и средних предприятий ГО/МР | ОКТМО, год | 2008–2024 |
| БД ПМО | **8010001** | Введено в действие жилых домов, м² | ОКТМО, год | 2006–2022 |
| БД ПМО | **8401003** | Оборот розничной торговли, тыс. руб | ОКТМО, год | 2017–2024 |
| БД ПМО | **8006001** | Общая площадь земель МО, га | ОКТМО, год | 2006–2023 |
| fedstat | **43062** | Уровень безработицы (методология МОТ), % | **субъект РФ**, год | 2000–2026 |
| fedstat | 61781 | (уже в проекте) рыночная цена жилья м² | город, квартал | — |

Безработицы в муниципальном разрезе нет ни на fedstat, ни в БД ПМО —
используем субъектный уровень 43062 как константу по Татарстану
(signal только по времени, не по территории).

### Производные фичи (per-object)

После джойна по `oktmo_full` (или `okato`, если EMISS даёт только okato) и года target:

| фича | формула | смысл |
| --- | --- | --- |
| `oktmo_avg_salary_rub` | `bdmo_8213002` напрямую | покупательная способность населения |
| `oktmo_population` | `bdmo_8112027` | масштаб муниципалитета |
| `oktmo_population_density` | `bdmo_8112027 / (bdmo_8006001_га / 100)` | плотность (площадь из БД ПМО, не GAR) |
| `oktmo_housing_volume_5y_m2` | сумма `bdmo_8010001` за последние 5 лет | темп новой застройки |
| `oktmo_unemployment_pct` | `43062` (субъект РФ, broadcast на все ОКТМО) | депрессивность территории |
| `oktmo_retail_turnover_per_capita` | `bdmo_8401003 × 1000 / bdmo_8112027` (руб/чел) | экономическая активность |

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
data/raw/bdmo/data_Y4{код}_112_v20250918.csv   ← БД ПМО (tochno.st)
data/raw/emiss/43062/raw_*.xls                 ← fedstat downloadData.do
        │
        ▼ scripts/parse_bdmo_csv_to_parquet.py / parse_emiss_subject_xls.py
data/silver/emiss/{bdmo_код|43062}/data.parquet   (generic long)
        │
        ▼ scripts/build_macro_oktmo_features.py
data/silver/macro_oktmo_features/region={code}/year={Y}/data.parquet
  └─ oktmo, oktmo_avg_salary_rub, oktmo_population, ... (joined wide)
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
emiss_indicators_yearly: list[str] = [
    "bdmo_8112027",  # численность населения на 1 января
    "bdmo_8213002",  # среднемесячная зарплата работников организаций ГО/МР
    "bdmo_8010001",  # ввод в действие жилых домов, м²
    "bdmo_8401003",  # оборот розничной торговли, тыс. руб
    "bdmo_8006001",  # общая площадь земель МО, га
    "43062",  # уровень безработицы по методологии МОТ (субъект РФ)
]
cadastre_target_year: int = 2024
```

## Эмпирический эффект (гипотеза)

- **apartment**: Δ MAPE −0.5…−1.5 пп. Зарплата + плотность населения дают чёткий sort по премиальности района.
- **house**: Δ −1…−2 пп. Для пригорода и сельских территорий зарплата OKTMO — основной маркер «дорогая периферия vs депрессивное село».
- **commercial**: Δ −1…−3 пп. Сильные сигналы: розничный оборот на душу, безработица.
- **landplot**: Δ −0.5…−1 пп. Меньше — у участка цена сильнее завязана на VRI ([ADR-0021](#)) и кадастровый квартал.

Сильнее всего эффект ожидается для **новых регионов** при расширении (модель станет переносимой), а не для текущего Татарстана, где CatBoost уже выучил локальные паттерны через `okato` cat-id.

## Открытые вопросы (закрыты при реализации 2026-08-20)

- ~~**Доступность OKTMO-уровня**~~ — закрыт: на fedstat муниципальный
  разрез есть фактически только у населения (31557); зарплата, розница,
  ввод жилья и площадь взяты из БД ПМО (ОКТМО), безработица осталась
  субъектной константой.
- ~~**Совпадение OKTMO в EMISS и в GAR**~~ — закрыт: ОКТМО БД ПМО
  (МО верхнего уровня, Татарстан) побайтно совпали с
  `gar_lookup.mun_okrug_oktmo` — 45 из 45.
- **Поправка на инфляцию.** `oktmo_avg_salary_rub` — номинальная. На объект 2018 года target — это номинальная зарплата 2018 года, и тогда OK. На «текущую переоценку» нужна реальная (с CPI-нормировкой). На первой итерации — номинал, далее по ситуации.

## Результат реализации (2026-08-20)

Реализовано в ветке `feature/emiss-macro-features`:

- **Скачивание.** fedstat отдаёт данные только через SPA →
  `scripts/download_emiss_indicator.py` (patchright, persistent profile
  `data/raw/emiss-profile/`, POST `/indicator/downloadData.do` со
  struts-токеном). БД ПМО — прямые per-indicator zip с
  storage.yandexcloud.net/tochno-st-catalog.
- **Парсеры.** `scripts/parse_bdmo_csv_to_parquet.py` (CSV БД ПМО →
  generic long silver; фильтры: префикс ОКТМО 92, МО верхнего уровня,
  без агрегатных строк «Городские округа ...», per-indicator фильтры
  extra-измерений mest/okved2/mosh) и `scripts/parse_emiss_subject_xls.py`
  (строка субъекта из downloadData.do-xls; layout без кодов территорий).
- **Silver.** `data/silver/emiss/{bdmo_8112027, bdmo_8213002,
  bdmo_8010001, bdmo_8401003, bdmo_8006001, 43062}/data.parquet` —
  generic long (indicator_id, territory_code, year, value), все 45
  ОКТМО Татарстана верхнего уровня.
- **Wide.** `data/silver/macro_oktmo_features/region=RU-KAZAN-AGG/
  year={2006..2025}/data.parquet` — 819 строк (45 ОКТМО), 6 фич.
  Coverage по строкам wide: salary 93%, population 93%, density 82%
  (площадь до 2023), housing_5y 73% (ввод жилья до 2022), unemployment
  100% (broadcast), retail_per_capita 44% (розница с 2017).
- **Enrichment.** `compute_object_macro_features` + wiring в
  `BuildObjectFeatures` (шаг после municipality-блока, до geometry);
  join по `oktmo_full[:8]`, per-feature last-available year ≤
  `cadastre_target_year=2024`. Прогон `scripts/build_object_features.py`
  на всех 4 классах (2026-08-20) — все 6 колонок записаны в gold,
  coverage одинаков для всех фич (ограничен наличием `oktmo_full`):

| asset_class | объектов | non-null oktmo_* | уникальных ОКТМО8 |
| --- | --- | --- | --- |
| apartment | 1 089 | 82.1% | 2 |
| house | 46 596 | 56.4% | 4 |
| commercial | 42 411 | 31.3% | 7 |
| landplot | 197 514 | 23.8% | 12 |

  Spot-check (apartment, Казань 92701000): salary 94 673 руб (2024),
  population 1 318 604 (2024), density 2050 чел/км² (площадь 2023),
  housing_5y 4.83M м² (до 2022), unemployment 1.9% (2024) — значения
  сходятся с публичной статистикой.

- **Тесты.** `tests/unit/test_object_macro_features.py` (8 тестов) +
  3 теста wiring в `test_build_object_features.py` — зелёные.

### Замечание для будущих переобучений

Новые колонки добавлены в gold **после** последнего обучения моделей.
При переобучении (train_object_models) 6 `oktmo_*` колонок подхватятся
как числовые фичи автоматически. Учесть: Казанская агломерация
сконцентрирована в 2–12 уникальных ОКТМО на класс (apartment — 2,
house — 4, commercial — 7, landplot — 12), поэтому на текущих данных
макрофичи близки к константам и ΔMAPE, скорее всего, минимальна;
основной эффект ожидается при расширении на новые регионы (см.
§«Эмпирический эффект»). `oktmo_unemployment_pct` — субъектная
константа, территориального сигнала не даёт вовсе.

# ADR-0023: Топографические ЦОФ (DEM)

**Статус:** Accepted
**Дата:** 2026-04-26 (реализовано 2026-08-20)
**Реализует:** [info/grid-rationale.md §6](../grid-rationale.md) (Объектные ЦОФ — рельеф/крутизна, явно отмечено как «требует DEM»).
**Опирается на:** ничего из существующего пайплайна — это новый источник данных.

## Контекст

Высота, крутизна склона, относительный рельеф — стандартные географические ЦОФ. В ЕГРН-сегменте и hedonic-моделях дают:

- **«Видовой» эффект** — склон с видом на воду/город дороже плоского.
- **Премия за нагорье / штраф за низину** — значимый для Сочи/Иркутска/Владивостока, маргинальный для Казани.
- **Косогорные landplot** — официально дешевле (труднее застройка, водо-оползневые риски).
- **Подтопление** — низины у воды штрафуются в страховке/кадастре.

В нашем пилоте (Казанская агломерация) рельеф **слабый сигнал** — Казань на плато с относительно небольшими перепадами (Кремлёвский холм vs Заречье — 30–50 м). Для Иркутской агломерации (запланировано в [info/project.md](../project.md) как этап 2) эффект будет в разы сильнее: Ангарск, Шелехов, нагорные посёлки.

Решение первоначально откладывалось; реализовано 2026-08-20 в рамках
программы расширения ЦОФ (вместе с ADR-0021/0022/0024/0025) перед
финальным переобучением quartet — дешёвый слой, инфраструктура
пригодится и для горных регионов.

## Решение

Подключить DEM как новый raw-слой и считать 3 фичи per-object через `rasterio.sample`.

### Источник DEM

| опция | разрешение | покрытие | стоимость |
| --- | --- | --- | --- |
| **Copernicus GLO-30** (выбран) | 30 м | глобально, open license (требует указания © DLR) | 0₽, ~115 MB на пилотное окно (4 тайла) |
| SRTM 1 arc-second (fallback) | 30 м | глобально, бесплатно | 0₽, схожее |
| ASTER GDEM v3 | 30 м | глобально | 0₽, схожее |
| Российский DTM (Роскартография) | 5 м | РФ, по запросу | требует лицензии или платной выгрузки |
| Open-source LiDAR | <1 м | редко покрывает РФ | n/a для пилота |

**Выбор: Copernicus GLO-30, а не SRTM** (первоначальный план). Причина —
доступность и лицензия: тайлы GLO-30 скачиваются без токена/учётки и
покрываются открытой лицензией с атрибуцией © DLR, тогда как доступ к
SRTM через Earthdata требует учётной записи. Для пайплайна это drop-in
замена (те же 30 м, GeoTIFF, EPSG:4326). Sanity по факту: Казанский
Кремль ~84 м, берег Волги в Заречье ~49 м — сходится с реальностью.
SRTM остаётся fallback-опцией. Если на Иркутске понадобится точнее —
переходим на 5-метровый DTM как drop-in замену, остальной пайплайн не
меняется.

Raw: 4 GeoTIFF-тайла (`N55/N56 × E048/E049`, ~115 MB суммарно) в
`data/raw/dem/`. В отличие от первоначального плана «один файл»,
`BuildDemSilver` делает mosaic (`rasterio.merge`) всех `*.tif` в
директории, затем перепроекцию в UTM-зону центра мозаики (метрическая
сетка ~29.5 м) — slope/relief требуют пиксели в метрах (dev-rules:
UTM-перепроекция для площадных операций).

### Признаки

| фича | формула | смысл |
| --- | --- | --- |
| `elevation_m` | `dem.sample(lon, lat)` | абсолютная высота |
| `slope_deg_local` | `arctan(grad_norm) × 180/π`, где `grad` берётся из 3×3 окна вокруг точки | локальная крутизна (градусы) |
| `relative_relief_500m_m` | `max − min` высоты в 500-м кольце | амплитуда рельефа в окрестности |

`slope_deg_local` и `relative_relief` — производные второго порядка от DEM, считаются один раз на регион через `numpy.gradient` + сепарабельное скользящее окно (`richdem` **не** добавляем — numpy достаточно, лишняя compiled-зависимость не нужна). Не за каждый объект — препроцессим DEM в три растра, потом `sample` per-object.

### Применимость по классам

- **landplot**: ключевое — `slope_deg_local` влияет на стоимость застройки.
- **house**: видовой/подтопляемый эффект — `elevation_m` + `relative_relief`.
- **apartment**: маргинально (внутри здания DEM не различает этажи).
- **commercial**: маргинально, кроме «торгцентр у трассы на возвышенности» — но это уже captures через distance к road class.

### Что **не** делаем в этой итерации

- **Aspect** (направление склона: север/юг/восток/запад). Для солнца/освещения важно в коттеджных оценках, но требует ещё одного DEM-derived layer. Отложим.
- **3D-поверхность здания + угол падения солнца** (insolation index) — слишком дорого, специфично.
- **Затопление по моделям** (Wat. flood-risk). Платные данные/долгий ETL.
- **Радоновая активность / геопатогенные зоны** — псевдо-наука для feature engineering.

## Архитектура

```text
data/raw/dem/*.tif                            ← raw GLO-30, 4 тайла (~115 MB, © DLR)
            │
            ▼ scripts/build_dem_silver.py    ← mosaic → UTM reproject → производные
data/silver/dem/region={code}/               ← EPSG:32639, float32, NaN nodata
  ├─ elevation.tif
  ├─ slope_deg.tif
  └─ relative_relief_500m.tif
            │
            ▼ BuildObjectFeatures.execute(...)
              └─ compute_object_dem_features(...)  ← новый шаг (после age, до relative)
gold/valuation_objects/...
  └─ + elevation_m, slope_deg_local, relative_relief_500m_m
```

Новые зависимости: **никаких** — `rasterio` уже был в проекте
(`pyproject.toml`, >=1.5), `richdem` сознательно не добавляем.

Новый порт + adapter:

```text
src/kadastra/ports/dem_sampler.py
  class DemSamplerPort(Protocol):
      def sample_elevation(self, *, lat, lon) -> float | None: ...
      def sample_slope_deg(self, *, lat, lon) -> float | None: ...
      def sample_relative_relief(self, *, lat, lon) -> float | None: ...

src/kadastra/adapters/rasterio_dem_sampler.py
  class RasterioDemSampler:
      def __init__(self, *, elevation_path, slope_path, relief_path) -> None: ...
```

Расхождение со спекой-планом: у `sample_relative_relief` **нет**
аргумента `radius_m` — relief-растр препроцессится на один фиксированный
радиус (`dem_relief_radius_m`), и сэмплинг с другим радиусом молча читал
бы не тот слой.

Use case:

```text
src/kadastra/usecases/build_dem_silver.py — preprocess GLO-30 → 3 derived layers
src/kadastra/etl/dem_derivatives.py       — slope (np.gradient) + relief (rolling window)
src/kadastra/etl/object_dem_features.py   — per-object sampling
```

**RAW_OBJECT_SCHEMA не расширяется.** Первоначальная гипотеза «колонки
потеряются при сбросе фрейма в RAW_OBJECT_SCHEMA» не подтвердилась:
`BuildObjectFeatures` сбрасывает фрейм **до** вычисления производных
шагов, а DEM-фичи считаются внутри того же `execute()` после сброса —
как и все прочие производные ЦОФ (metro/geometry/age). Расширять raw
схему пришлось бы ещё и в `AssembleNspdValuationObjects` (null-литералы
для derived-колонок) — концептуально неверно: это не passthrough из
silver, а вычисляемые фичи. Идемпотентность покрыта тестом
`test_idempotent_rerun_replaces_columns`.

### TDD

| Уровень | Что покрывается |
| --- | --- |
| unit / `test_object_dem_features.py` | FakeDemSampler + objects → ожидаемые колонки. |
| unit / `test_rasterio_dem_sampler.py` | Маленький synthetic GeoTIFF (10×10), sampling в известных точках. |
| data-quality / отчёт | После rebuild — non-null %, диапазоны (по факту: `elevation_m` 47.7–216.2 м по объектам, `slope` p50 ~1.7°) — см. «Результаты реализации». |

### Settings

```python
dem_raw_dir: Path = Path("data/raw/dem")
dem_silver_base_path: Path = Path("data/silver/dem")
dem_relief_radius_m: float = 500.0
dem_features_enabled: bool = True
```

Composition root передаёт `dem_sampler=None` (шаг пропускается), когда
silver-слои для региона ещё не построены или `dem_features_enabled=False`
— тот же opt-in паттерн, что у gar-lookup и macro-oktmo.

## Эмпирический эффект (гипотеза)

- **landplot**: Δ MAPE −0.5…−2 пп. На участках в Татарстане эффект ограничен, но реален (склоны вдоль Волги/Камы).
- **house**: Δ −0.3…−1 пп. Видовые посёлки на холмах.
- **apartment**: Δ ≈ 0…−0.3 пп. Маргинально.
- **commercial**: Δ ≈ 0. Не основной фактор.

При переходе на Иркутскую агломерацию (плановое расширение) ожидание возрастает в 3–5 раз: сложный рельеф = реальный price driver.

## Результаты реализации (2026-08-20)

- **Raw.** 4 тайла Copernicus GLO-30 (`Copernicus_DSM_COG_10_N55/N56_00_E048/E049_DEM.tif`, ~115 MB) в `data/raw/dem/`. Заметка: у GLO-30 на этих широтах lon-постинг 1.5″ (2400 px/°) при lat-постинге 1″ — merge + репроекция это поглощают.
- **Silver.** `data/silver/dem/region=RU-KAZAN-AGG/{elevation,slope_deg,relative_relief_500m}.tif` — EPSG:32639 (UTM 39N, выбран по центру мозаики), сетка 7668×4447, пиксель 29.49 м, float32, NaN nodata (6.3% NaN — углы репроекции за пределами мозаики). Диапазоны silver: elevation 45.5–281.5 м (медиана 134.6), slope 0–61.7° (медиана 1.8°), relief 0–185.4 м (медиана 32.5). Окно 2×2° уже, чем весь Татарстан, поэтому max 281 м < 380 м высшей точки республики (она юго-восточнее окна).
- **Sanity-точки.** Казанский Кремль 84.1 м (ожидалось ~83), берег Волги в Заречье 49.5 м (ожидалось ~49), пойма Казанки 70.4 м — сходится с реальностью.
- **Enrichment.** `compute_object_dem_features` + wiring в `BuildObjectFeatures` (шаг после age-фич, до relative). Прогон `scripts/build_object_features.py` на всех 4 классах (2026-08-20) — три колонки записаны в gold, coverage 100% везде (вся выборка внутри DEM-окна):

| asset_class | объектов | non-null (все 3 фичи) | elevation_m min/p50/max | slope_deg_local p50/max | relative_relief_500m_m p50/max |
| --- | --- | --- | --- | --- | --- |
| apartment | 1 089 | 100.0% | 52.2 / 86.6 / 134.9 | 2.0 / 18.9 | 30.5 / 75.8 |
| house | 46 596 | 100.0% | 49.5 / 86.5 / 176.5 | 1.6 / 26.9 | 33.5 / 93.9 |
| commercial | 42 411 | 100.0% | 48.8 / 81.7 / 200.8 | 1.7 / 21.5 | 29.7 / 99.7 |
| landplot | 197 514 | 100.0% | 47.7 / 84.0 / 216.2 | 1.7 / 26.9 | 31.1 / 119.3 |

- **Тесты.** `tests/unit/test_rasterio_dem_sampler.py` (5), `tests/unit/test_object_dem_features.py` (6), `tests/unit/test_build_dem_silver.py` (2), +2 wiring-теста в `test_build_object_features.py` — зелёные (663 passed суммарно).

### Замечание для будущих переобучений

Новые колонки добавлены в gold **после** последнего обучения моделей;
модели НЕ переобучались — финальное переобучение quartet на полном
наборе фич будет отдельным шагом в конце программы расширения ЦОФ.
При переобучении 3 DEM-колонки подхватятся как числовые фичи
автоматически (`select_object_feature_columns`). Ожидание по ΔMAPE —
скромное для Казани (см. §«Эмпирический эффект»): вариативность
elevation внутри агломерации мала (p50 ≈ 84–87 м во всех классах),
основной сигнал — slope/relief для landplot/house у речных долин.

## Открытые вопросы

- ~~**Сроки.**~~ Реализовано (2026-08-20) в рамках программы расширения
  ЦОФ перед финальным переобучением quartet — см. «Результаты реализации».
- **Размер raw DEM.** По факту ~115 MB (4 тайла GLO-30 на окно
  N55–57/E048–50), а не ~5 GB — проблемы хранения нет, raw лежит в
  `data/raw/dem/` локально (не в git).
- **Точность GLO-30 в зонах со зданиями.** GLO-30 — DSM: измеряет «верх
  кроны/крыши», а не голую землю. В плотной застройке elevation смещён
  вверх на высоту здания. Для городских apartment это шум; для landplot
  за городом — достоверно. Если важно — переходим на `BareEarth DTM`
  (Copernicus GLO-30 имеет отдельный продукт DTE/HA, либо FABDEM).

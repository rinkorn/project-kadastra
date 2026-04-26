# ADR-0023: Топографические ЦОФ (DEM)

**Статус:** Proposed
**Дата:** 2026-04-26
**Реализует:** [info/grid-rationale.md §6](../grid-rationale.md) (Объектные ЦОФ — рельеф/крутизна, явно отмечено как «требует DEM»).
**Опирается на:** ничего из существующего пайплайна — это новый источник данных.

## Контекст

Высота, крутизна склона, относительный рельеф — стандартные географические ЦОФ. В ЕГРН-сегменте и hedonic-моделях дают:

- **«Видовой» эффект** — склон с видом на воду/город дороже плоского.
- **Премия за нагорье / штраф за низину** — значимый для Сочи/Иркутска/Владивостока, маргинальный для Казани.
- **Косогорные landplot** — официально дешевле (труднее застройка, водо-оползневые риски).
- **Подтопление** — низины у воды штрафуются в страховке/кадастре.

В нашем пилоте (Казанская агломерация) рельеф **слабый сигнал** — Казань на плато с относительно небольшими перепадами (Кремлёвский холм vs Заречье — 30–50 м). Для Иркутской агломерации (запланировано в [info/project.md](../project.md) как этап 2) эффект будет в разы сильнее: Ангарск, Шелехов, нагорные посёлки.

Решение откладываем до фактического расширения проекта на горный регион **или** до момента, когда [ADR-0019](#) + [ADR-0021](#) выйдут к плато по MAPE и нужно искать новые сигналы.

## Решение

Подключить DEM как новый raw-слой и считать 3 фичи per-object через `rasterio.sample`.

### Источник DEM

| опция | разрешение | покрытие | стоимость |
| --- | --- | --- | --- |
| **SRTM 1 arc-second** (~30 м) | 30 м | глобально, бесплатно | 0₽, ~5 GB на регион |
| ASTER GDEM v3 | 30 м | глобально | 0₽, схожее |
| Российский DTM (Роскартография) | 5 м | РФ, по запросу | требует лицензии или платной выгрузки |
| Open-source LiDAR | <1 м | редко покрывает РФ | n/a для пилота |

Стартуем с SRTM 30 м — open license, достаточно для регионального уровня. Если на Иркутске понадобится — переходим на 5-метровый DTM как drop-in замену, остальной пайплайн не меняется.

### Признаки

| фича | формула | смысл |
| --- | --- | --- |
| `elevation_m` | `dem.sample(lon, lat)` | абсолютная высота |
| `slope_deg_local` | `arctan(grad_norm) × 180/π`, где `grad` берётся из 3×3 окна вокруг точки | локальная крутизна (градусы) |
| `relative_relief_500m_m` | `max − min` высоты в 500-м кольце | амплитуда рельефа в окрестности |

`slope_deg_local` и `relative_relief` — производные второго порядка от DEM, считаются один раз на регион через `richdem` или `numpy.gradient`. Не за каждый объект — препроцессим DEM в три массива, потом `sample` per-object.

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
data/raw/dem/srtm-tatarstan.tif              ← raw DEM (~5 GB)
            │
            ▼ scripts/build_dem_silver.py    ← preprocess: elevation/slope/relief grids
data/silver/dem/region={code}/
  ├─ elevation.tif
  ├─ slope_deg.tif
  └─ relative_relief_500m.tif
            │
            ▼ BuildObjectFeatures.execute(...)
              └─ compute_object_dem_features(...)  ← новый шаг
gold/valuation_objects/...
  └─ + elevation_m, slope_deg_local, relative_relief_500m_m
```

Новые зависимости:

```toml
# pyproject.toml
rasterio = "^1.4"
# опционально, для красивого slope/aspect:
richdem = "^2.3"
```

Новый порт + adapter:

```text
src/kadastra/ports/dem_sampler.py
  class DemSamplerPort(Protocol):
      def sample_elevation(self, lat: float, lon: float) -> float | None: ...
      def sample_slope_deg(self, lat: float, lon: float) -> float | None: ...
      def sample_relative_relief(self, lat: float, lon: float, radius_m: float) -> float | None: ...

src/kadastra/adapters/rasterio_dem_sampler.py
  class RasterioDemSampler:
      def __init__(self, *, elevation_path, slope_path, relief_path) -> None: ...
```

Use case:

```text
src/kadastra/usecases/build_dem_silver.py — preprocess SRTM → 3 derived layers
src/kadastra/etl/object_dem_features.py    — per-object sampling
```

### TDD

| Уровень | Что покрывается |
| --- | --- |
| unit / `test_object_dem_features.py` | FakeDemSampler + objects → ожидаемые колонки. |
| unit / `test_rasterio_dem_sampler.py` | Маленький synthetic GeoTIFF (10×10), sampling в известных точках. |
| data-quality / отчёт | После rebuild — non-null %, диапазоны (`elevation_m` Татарстан: 50–250 м, `slope` 0–15°). |

### Settings

```python
dem_silver_base_path: Path = Path("data/silver/dem")
dem_relief_radius_m: float = 500.0
```

## Эмпирический эффект (гипотеза)

- **landplot**: Δ MAPE −0.5…−2 пп. На участках в Татарстане эффект ограничен, но реален (склоны вдоль Волги/Камы).
- **house**: Δ −0.3…−1 пп. Видовые посёлки на холмах.
- **apartment**: Δ ≈ 0…−0.3 пп. Маргинально.
- **commercial**: Δ ≈ 0. Не основной фактор.

При переходе на Иркутскую агломерацию (плановое расширение) ожидание возрастает в 3–5 раз: сложный рельеф = реальный price driver.

## Открытые вопросы

- **Сроки.** Эта ADR — кандидат на отложенную реализацию (ROI слабый для Татарстана). Имеет смысл прокатить вместе с расширением региона.
- **Размер raw DEM.** SRTM на Татарстан целиком ~5 GB. Для GitHub repo — большой, S3-совместимый (как у нас уже [s3 настроен](../../src/kadastra/config.py#L15-L20)) подойдёт.
- **Точность SRTM в зонах со зданиями.** SRTM измеряет «верх кроны/крыши», а не голую землю. В плотной застройке elevation смещён вверх на высоту здания. Для городских apartment это шум; для landplot за городом — достоверно. Если важно — переходим на `BareEarth DTM`.

# ADR-0025: Прочие ЦОФ — CBD, культнаследие, ЗОУИТ

**Статус:** Proposed
**Дата:** 2026-04-26
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

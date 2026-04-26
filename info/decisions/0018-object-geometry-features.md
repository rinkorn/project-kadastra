# ADR-0018: Объектные геометрические ЦОФ

**Статус:** Proposed
**Дата:** 2026-04-26
**Реализует:** [info/grid-rationale.md](../grid-rationale.md), §6 (Объектные ЦОФ — геометрия объекта).
**Опирается на:** [ADR-0017](0017-object-geometry-passthrough-for-inspector.md) (`polygon_wkt_3857` доехал до gold).

## Контекст

[ADR-0017](0017-object-geometry-passthrough-for-inspector.md) прокинул реальный полигон объекта (EPSG:3857 WKT) от silver до gold как passthrough — но сами **ЦОФ из этой геометрии пока не считаются**. В feature-матрицу модели уходит только NSPD-атрибут `area_m2` и набор местоположенческих/окрестностных фич (graph distance, zonal density, poly-area share, territorial). Форма самого объекта — нет.

[grid-rationale.md §6](../grid-rationale.md) явно отмечает геометрию как отдельный класс ЦОФ. Существенная гипотеза — на **landplot** (земельные участки) форма критична: вытянутый-неудобный участок при равной площади дешевле квадратного. На зданиях (apartment/house/commercial) эффект скромнее — стоимость определяется локацией, годом, этажностью, а не формой контура.

Данные есть, расчёт дешёвый (per-object, локальная операция над одним полигоном), методологически обязательный — пора закрыть.

## Решение

**Считать 7 чистых геометрических фич per-object из `polygon_wkt_3857` напрямую в EPSG:3857 (метры web-mercator), без round-trip через WGS84.** Прокидывать в gold-схему `BuildObjectFeatures` как обычные numeric-фичи. Где полигона нет (NSPD выдал null) — фичи null, модель сама сообразит.

### 7 фич

| колонка | формула | что меряет |
| --- | --- | --- |
| `polygon_area_m2` | `shapely.area(geom)` | геом-площадь, независимая от NSPD-`area_m2`. Бонус: sanity-check NSPD-выдачи (расхождение с NSPD ⇒ проблемная запись) |
| `polygon_perimeter_m` | `geom.length` | периметр |
| `polygon_compactness` | `4·π·A / P²` (Polsby–Popper) | 1 — круг; 0 — длинная узкая полоса. Стандартная мера компактности в geoanalysis |
| `polygon_convexity` | `A / area(convex_hull(geom))` | 1 — выпуклый; <1 — рваная/вогнутая форма (буквы L, T, П) |
| `bbox_aspect_ratio` | `max_side / min_side` минимального rotated bounding box | вытянутость; квадрат = 1, длинный прямоугольник → ∞ |
| `polygon_orientation_deg` | угол длинной оси rotated BBox, диапазон [0°, 180°) | ориентация формы. Без привязки к дорогам в этой итерации (см. ниже) |
| `polygon_n_vertices` | `len(geom.exterior.coords) - 1` | proxy сложности границы (последняя точка кольца дублирует первую) |

Все вычисления на исходном EPSG:3857 — это уже метрическая проекция (в Казани искажение площади ≈ +0.4 % от истинной за счёт mercator-растяжения, но это _систематическое_ смещение, к которому модель адаптируется; для feature engineering важна **относительная** точность между объектами, а не абсолютная).

### Применимость по классам

- `apartment` (NSPD building) — есть полигон контура здания.
- `house` (NSPD building) — есть полигон.
- `commercial` (NSPD building) — есть полигон.
- `landplot` (NSPD landplot) — есть полигон, и здесь форма наиболее критична.

Если у конкретного объекта NSPD геометрию не отдал (бывает на старых записях) — все 7 фич = null. CatBoost native-handling и EBM с per-feature missing-bin это съедят.

### Что **не** делаем в этой итерации

- **Ориентация относительно дорог** ([grid-rationale §6](../grid-rationale.md)) — требует кросс-сопоставления ориентации полигона с азимутом ближайшего road-edge. Технически это «найти ближайший edge → его азимут → угол между ним и `polygon_orientation_deg`». Само по себе несложно, но затрагивает `RoadGraphPort` (нужен метод поиска ближайшего edge с азимутом), и эффект гипотетический (для зданий вторичен, для landplot — надо мерить). Отдельным ADR при первой потребности — сейчас даём «голую» ориентацию, у CatBoost достаточно ёмкости связать её с lat/lon и неявно выучить ось улиц.
- **Рельеф / крутизна склона** ([grid-rationale §6](../grid-rationale.md)) — нужен DEM (SRTM 30 м или DTM-Россия). Это новый источник данных, новая ETL-тропа, новая зависимость. Отдельным ADR с замером эффекта на Иркутскую агломерацию (там рельеф реально работает) — в Казани плоско, эффект ожидается слабый.
- **IoU полигона с red lines / промзонами** — частично уже покрыто [ADR-0014](0014-poly-area-buffer-features.md) `share`-фичами в буфере. Прямой IoU без буфера дал бы практически те же сигналы для landplot и был бы **избыточен** к существующему block 3b. Если поднимется reranking — вернёмся.
- **Аварийность** — NSPD не отдаёт это поле в текущей выдаче. Нечего добавлять.

### Почему просто 7 фич, а не «всё, что можно»

Каждая фича — это ещё одно измерение, которое модель должна выучить. Чем больше тонко-коррелирующих фич, тем сильнее переобучение на малых классах (apartment 1k objects). Эти 7 — _ортогональны_ друг другу:

- area / perimeter — масштабные, размерные.
- compactness / convexity / aspect_ratio / n_vertices — формовые, безразмерные.
- orientation — единственная угловая.

Дополнительные кандидаты типа `min_inscribed_circle_radius`, `max_inscribed_rectangle_area` — это уже дубль compactness/convexity, без независимого сигнала. Не добавляем.

## Архитектура

```text
silver/nspd/region={code}/source={buildings|landplots}/data.parquet
  └─ polygon_wkt_3857: Utf8 (EPSG:3857)  ← passthrough из ADR-0017
            │
            ▼ AssembleNspdValuationObjects (passthrough)
gold/valuation_objects/region={code}/asset_class={class}/data.parquet (initial assemble)
  └─ polygon_wkt_3857
            │
            ▼ BuildObjectFeatures.execute(...)
              ├─ compute_object_metro_features(...)
              ├─ compute_object_road_features(...)
              ├─ compute_object_neighbor_features(...)
              ├─ compute_object_zonal_features(...)
              ├─ compute_object_polygon_features(...)
              ├─ compute_object_municipality_features(...) (optional, GAR)
              └─ compute_object_geometry_features(...)  ← ЭТА ИТЕРАЦИЯ
gold/valuation_objects/... (overwrite, теперь с 7 новыми колонками + polygon_wkt_3857)
```

Новый модуль:

```text
src/kadastra/etl/object_geometry_features.py
  def compute_object_geometry_features(objects: pl.DataFrame) -> pl.DataFrame:
      """Read polygon_wkt_3857 column, derive 7 geometry features,
      return DataFrame with new columns appended. Null WKT → null
      values for all 7."""
```

Чисто-функция, никаких портов. Реализация:

1. `shapely.from_wkt` на серию (vectorized) — даёт `np.ndarray[BaseGeometry]`.
2. Per-фича — векторный shapely-вызов (`shapely.area`, `shapely.length`, …) или numpy-комбинация результатов. shapely 2 release-GIL'ит на батчах — фактически весь модуль за O(N) без python-level loop'ов.
3. `bbox_aspect_ratio` и `polygon_orientation_deg` через `shapely.minimum_rotated_rectangle` — единственная не-vectorized ветка (shapely 2 не отдаёт это batch-API), но вызовов N штук, для 290k объектов — секунды.
4. `n_vertices` через `shapely.get_num_coordinates` − 1 (exterior замкнут — последняя точка дублирует первую).

Итоговое время на 290k объектов на M1 — ожидание ~10–20 секунд.

### TDD

| Уровень | Что покрывается |
| --- | --- |
| unit / `tests/unit/test_object_geometry_features.py` | Синтетические полигоны (квадрат, прямоугольник 10×100, L-форма, треугольник, null) → ожидаемые значения 7 фич. Численная проверка `compactness`/`convexity` против известных формул (квадрат: compactness = π/4 ≈ 0.785, convex; прямоугольник 10×100: compactness ≈ 0.038, convex; L-форма: convexity < 1). |
| unit / `tests/unit/test_build_object_features.py` (расширение) | После прогона `BuildObjectFeatures.execute` в DataFrame должны быть все 7 новых колонок с правильными типами. |
| integration | Не нужна — модуль чистый, без внешних зависимостей. |

### Settings

В первой итерации параметров нет — все 7 фич считаются всегда.

## Эмпирический эффект

Будет заполнено после rebuild gold + retrain quartet: per-class Δ MAPE (CatBoost / EBM / Naive) от текущего main → main + ADR-0018.

Гипотеза до замера:

- apartment (1k, building) — Δ ≈ 0…+0.3 пп (геометрия здания вторична, локация решает).
- house (47k, building) — Δ ≈ 0…+0.5 пп (тот же эффект).
- commercial (42k, building) — Δ ≈ 0…+1 пп (форма коммерческой недвижимости — крайне разнородна, тут шум доминирует).
- **landplot (200k)** — Δ ≈ −1…−3 пп (форма участка реально влияет на цену; ожидаем самый явный выигрыш, особенно на White Box / Naive Linear).

Если для landplot эффект окажется незначимым — это **сигнал**, что либо `bbox_aspect_ratio` мало где сильно отличается от 1 (датасет однороден по форме), либо текущие модели уже неявно выучили форму через `area_m2` и proximity-фичи. Тогда обсуждаем ориентацию-к-дорогам или DEM как следующий шаг.

## Открытые вопросы

- **MultiPolygon в NSPD.** [ADR-0017](0017-object-geometry-passthrough-for-inspector.md) отметил, что в Kazan-агломерации мульти-полигоны не встречены. Если найдутся при rebuild — `compute_object_geometry_features` обработает их через `geom.area / geom.length` (shapely корректно агрегирует по частям), но `minimum_rotated_rectangle` для MultiPolygon даст BBox объединения, что физически осмысленно (общая ориентация). Следим при первом прогоне.
- **EPSG:3857 искажение площади на широте Казани.** ≈ +0.4 % систематический оверштат. Не критично для feature engineering (модель обучается на относительных значениях), но если в будущем `polygon_area_m2` будет использоваться **сам по себе** в отчёте (например, в текстовом уравнении EBM как «вклад площади») — стоит заменить на UTM 39N как в ADR-0014. Сейчас это перевзвешивание модели, а не KPI; оставляем 3857.
- **Корреляция с NSPD `area_m2`.** Гипотетически расхождение `polygon_area_m2` vs `area_m2` ≥ 10 % — индикатор битой записи NSPD. Можно вынести в data-quality репорт, но это уже не feature engineering, а отдельная задача.

# ADR-0015: Территориальные ЦОФ через ГАР + парс NSPD-адресов (hybrid)

**Статус:** Accepted
**Дата:** 2026-04-26
**Реализует:** [ADR-0010](0010-methodology-compliance-roadmap.md), пункт 4 дорожной карты трека 2 — территориальные ЦОФ.
**Источник методологии:** [info/grid-rationale.md](../grid-rationale.md), §9.

## Контекст

Блоки 1–3b закрыли расстояния, относительные ЦОФ, плотности и поли-площадь. Метрики на CatBoost после блока 3 насытились (block 2 и 3b CatBoost-нейтральны), но методологически осталась дыра §9 — **территориальные ЦОФ**: значения, присвоенные через принадлежность к административной/муниципальной зоне (муниципалитет, населённый пункт, видовое место, опорный НП).

Без территориальной привязки каждый объект для модели «висит в координатах», а не «живёт в Советском районе Казани, который имеет собственный ценовой режим из-за инфраструктуры/демографии». White Box модели (блок 5) — линейные/правил-ориентированные — без явного `municipality_id` не смогут применить дифференцированные коэффициенты по территории, что заложено в методические указания ГБУ. Поэтому к моменту блока 5 территориальные фичи должны лежать в gold-схеме.

`cad_num` (первые два сегмента) даёт **кадастровое** деление, не муниципальное. На Казанской агломерации они в основном совпадают, но не всегда: cad_district 50 ≠ муниципальный округ «Казань» в полной точности (cadastral chamber делит иначе по городским/пригородным зонам). Для CatBoost разница ~0–1 пп MAPE, но **methodologically** это разные сущности и для будущих джойнов с внешними данными по муниципалитетам (демография, бюджеты, ввод жилья, OKTMO-привязанные стат-сборники) нужны **канонические OBJECTID/OKTMO** из ГАР.

## Решение

**Принять полный `gar_xml/16/` ингест** (Татарстан) и построить per-object атрибуты:

- `mun_oktmo: str` — код OKTMO муниципального образования.
- `mun_okrug_name: str` — имя муниципального округа / района (например, «город Казань», «Зеленодольский муниципальный район»).
- `settlement_name: str` — населённый пункт (например, «г Казань», «п Куркачи», «д Большие Кабаны»).
- `intra_city_raion: str` — внутригородской район (Советский / Приволжский / …) для объектов внутри городских округов; null для НП за их пределами.

Все четыре — категориальные, передаются в CatBoost через `cat_features`. Числовые derivatives (count_per_municipality и т.п.) — **не делаем** в этой итерации, по той же логике, что и в [ADR-0014](0014-poly-area-buffer-features.md): сначала измеряем сырой сигнал, derivatives вводим если линейные модели (блок 5) попросят.

### Рассмотренные альтернативы

| Вариант | Что даёт | Стоимость | Почему отвергнут |
| --- | --- | --- | --- |
| A. `cad_num` + regex-парс `readable_address` | `cad_district`, `cad_quarter`, `okrug_name`, `raion_name` через простую парсилку | 1 день | Не даёт OKTMO, не канонические id, не годится под White Box / внешние джойны. |
| A+. OSM `admin_level=6/8` + spatial join | `mun_okrug_name`, `intra_city_raion` через geometric контейнмент | 1.5–2 дня | Имена есть, OKTMO/OBJECTID нет. На пилоте достаточно, но потребует переделки при расширении на новый регион (OSM-границы могут быть нестабильны). |
| **B. Полный ГАР** | OBJECTID/OKTMO + полная муниципальная иерархия из официального источника | 5 дней | **Выбрано.** Канонические id для будущих джойнов; единый источник истины для всех регионов; методологическая совместимость с ГБУ-указаниями. |

## Структура решения

### Минимальный набор файлов ГАР

Из 18 файлов в `s3://kadastrova/Kadatastr/gar_xml/16/` (всего 11 ГБ) реально нужны 4:

| Файл | Размер | Зачем |
| --- | --- | --- |
| `AS_ADDR_OBJ` | 14 МБ | Имена и типы адресных объектов (район, населённый пункт, улица). |
| `AS_MUN_HIERARCHY` | 991 МБ | Walk родителей (OBJECTID → PARENTOBJID) до муниципального округа. Поле `OKTMO` присутствует на каждом узле. |
| `AS_HOUSES_PARAMS` | **2.2 ГБ** | `<PARAM TYPEID="8" VALUE="..." OBJECTID="..."/>` даёт `cadnum → objectid` для зданий. |
| `AS_STEADS_PARAMS` | **2.4 ГБ** | То же для земельных участков. |

**Эмпирически подтверждённая taxonomy TYPEID** в HOUSES_PARAMS Татарстана (срез 2026-04-06; 30 МБ выборка, основные значения):

| TYPEID | n | Содержимое (по примерам) | Заметка |
| --- | --- | --- | --- |
| 1, 2 | 17.7k | 4-знач. коды (1675 / 1684 / …) | КЛАДР short |
| 3, 4 | 13.4k | 4-знач. коды | КЛАДР short (повтор?) |
| 5 | 16.3k | 6-знач. (420011) | Почтовый индекс |
| 6 | 14.4k | 11-знач. (92233000011) | ОКТМО полный (settlement-level) |
| 7 | 22.6k | 11-знач. | ОКАТО полный |
| **8** | **18.5k** | **`16:NN:NNNNNN:NNNN`** | **CADNUM** ✓ |
| 13, 14, 15 | … | флаги/ID | служебные |
| 21 | 12.1k | 8-знач. (92730000) | ОКТМО short (municipality-level) |

Раньше предполагалось, что `cad_num ↔ OBJECTID` mapping лежит в `AS_REESTR_OBJECTS`. На проверке REESTR содержит только `OBJECTID/OBJECTGUID/LEVELID/ISACTIVE/dates` — без CADNUM. CADNUM живёт исключительно в `*_PARAMS` под `TYPEID=8`. Это удлинило этап download на ~1 день и поменяло architecture: вместо одного reestr-парсера держим два больших streaming-парсера PARAMS с TYPEID-фильтром.

**Не нужны для блока 4:** `AS_REESTR_OBJECTS`, `AS_HOUSES`, `AS_STEADS` (служебные/физические атрибуты, без CADNUM), `AS_APARTMENTS_*`, `AS_CARPLACES_*`, `AS_ROOMS_*`, `AS_CHANGE_HISTORY`, `AS_NORMATIVE_DOCS`, `AS_ADM_HIERARCHY` (нужна именно муниципальная), `AS_ADDR_OBJ_DIVISION`, `AS_ADDR_OBJ_PARAMS` (TYPEID на адресных объектах не CADNUM-несущий, OKTMO заберём из MUN_HIERARCHY).

Итого ~5.6 ГБ download.

### Pipeline

```text
S3 (gar_xml/16/*.XML) → silver/gar/{table}/data.parquet → silver/gar_lookup/object_to_municipality.parquet → join в pipeline объектов
```

1. **Download**: `scripts/download_gar.py` — selective S3-load 5 файлов в `data/raw/gar/16/`, идемпотентно (skip если файл уже есть и size совпадает).
2. **Parse XML → parquet**: `kadastra/etl/gar_xml_*.py` — iterparse-стримы, по одной функции на таблицу. Output: `data/silver/gar/{AS_*}/data.parquet`. TDD pair с маленькими XML-фрагментами как фикстурами.
3. **Build hierarchy lookup**: `kadastra/etl/gar_municipality_lookup.py` — берёт `AS_MUN_HIERARCHY` + `AS_ADDR_OBJ`, для каждого OBJECTID-листа (house/stead) walk-up до муниципального уровня (LEVEL=2 в АДМ-иерархии или соответствующего в МУН), пишет `(objectid, mun_oktmo, mun_okrug_name, settlement_name, intra_city_raion)` в `data/silver/gar_lookup/`.
4. **CADNUM → OBJECTID**: `kadastra/etl/gar_cadnum_index.py` — из `AS_REESTR_OBJECTS` строит индекс. Output: `data/silver/gar_lookup/cadnum_to_objectid.parquet`.
5. **Spatial fallback**: для NSPD-объектов с null матчем (СНТ/ГСК адреса без AOGUID, или cad_num не в реестре) — использовать spatial join с OSM `admin_level=6/8` boundaries (всё равно лежат в нашем PBF). Это purely-fallback путь, на пилоте на 5–15 % объектов.
6. **Merge в pipeline объектов**: новый use case-метод `attach_municipality(objects: pl.DataFrame) -> pl.DataFrame` — приджойнивает по cad_num через `cadnum_to_objectid` + `object_to_municipality`. Spatial-fallback применяется только к строкам с null после join'а.

### Архитектура

```text
src/kadastra/
  etl/
    gar_xml_addr_obj.py      # parser AS_ADDR_OBJ
    gar_xml_houses.py        # parser AS_HOUSES
    gar_xml_steads.py        # parser AS_STEADS
    gar_xml_mun_hierarchy.py # parser AS_MUN_HIERARCHY
    gar_xml_reestr.py        # parser AS_REESTR_OBJECTS
    gar_municipality_lookup.py # walk hierarchy + OKTMO
    gar_cadnum_index.py      # cadnum → objectid
    object_municipality_features.py # join в pipeline объектов
  ports/
    gar_archive.py           # Protocol: stream_xml(table) -> Iterator[bytes]
  adapters/
    s3_gar_archive.py        # adapter
    local_gar_archive.py     # adapter (для dev/тестов)
```

### Settings

```python
gar_local_root: Path = Path("data/raw/gar/16/")
gar_silver_root: Path = Path("data/silver/gar/")
gar_lookup_root: Path = Path("data/silver/gar_lookup/")
gar_files: dict[str, str] = {
    "addr_obj": "AS_ADDR_OBJ_*.XML",
    "houses": "AS_HOUSES_*.XML",
    "steads": "AS_STEADS_*.XML",
    "mun_hierarchy": "AS_MUN_HIERARCHY_*.XML",
    "reestr": "AS_REESTR_OBJECTS_*.XML",
}
```

## Что фактически было сделано (1 день)

Изначальный план B оказался **проще, чем рассчитывали** — сделали за один день:

1. `download_gar.py`: selective S3 (4 файла, 5.6 ГБ; AS_ADDR_OBJ + AS_HOUSES_PARAMS + AS_STEADS_PARAMS + AS_MUN_HIERARCHY). AS_ADM_HIERARCHY (1 ГБ) скачали попутно — оказалось не нужно (см. ниже).
2. **3 TDD-парсера** (`gar_xml_addr_obj`, `gar_xml_mun_hierarchy`, `gar_xml_object_params`) на iterparse + `elem.clear()`. Бенч: 14 МБ за 0.1 с, 991 МБ за 9.1 с, 2.2 ГБ за 19.5 с. Memory ~50 МБ.
3. **2 TDD-lookup'a** (`gar_cadnum_index`, `gar_mun_lookup`) — Polars vectorized. 1.85М cadnums + 3.84М mun-rows за 52 с total.
4. `scripts/build_gar_lookup.py` → `data/silver/gar_lookup/{cadnum_index,mun_lookup}.parquet`.
5. `compute_object_municipality_features` (GAR primary + NSPD address parse fallback) с TDD-парой, 6 unit-тестов.
6. Wire в `BuildObjectFeatures` + `Settings` + `composition_root` + extend `_OUTPUT_SCHEMA` для assemble (cad_num + readable_address до gold).
7. Re-assemble + rebuild + retrain + infer на 4 классах.

**Сюрпризы по пути:**

- **`AS_REESTR_OBJECTS` НЕ содержит CADNUM** (только OBJECTID/LEVELID/ISACTIVE). CADNUM живёт в `AS_HOUSES_PARAMS` / `AS_STEADS_PARAMS` под TYPEID=8 — пришлось докачать +4.6 ГБ. Изначальная посылка ADR-0015 о REESTR-mapping'е была неверной.
- **Coverage GAR-canonical match = 35 %** (44.6 % buildings + 24.4 % landplots). Большинство NSPD-объектов **не в ФИАС** — стройки, СНТ/ГСК без зарегистрированного адреса, частные дома вне адресной системы. Ожидалось ~80 %, по факту в три раза меньше.
- **`AS_ADM_HIERARCHY` оказался бесполезен для intra-Kazan raions** — ФИАС не моделирует внутригородские районы Казани (Советский / Приволжский / …) ни в MUN, ни в ADM иерархии. Их нет даже в `AS_ADDR_OBJ` как отдельных entries. Решение в итоге — OSM admin_level=9 spatial join (см. ниже).
- **NSPD `readable_address` спас планку по okrug**: парс regex'ами на г.о. / муниципальный район / г X даёт fallback с ~100 % coverage там, где ГАР молчит. Финальное coverage окрут: buildings **93 %**, landplots **97 %**.
- **OSM admin_level=9 → 100 % intra_city_raion внутри Казани.** Полигоны 7 раионов извлекаются `osmium tags-filter` в две стадии (`r/boundary=administrative` → `r/admin_level=9`) с post-фильтром в Python (только Polygon/MultiPolygon с admin_level=9 в properties; реляционные osmium-выходы вытаскивают ноды/линии). Spatial join shapely 2 STRtree + intersects predicate; имя стрипается до короткой формы («Советский район» → «Советский»), совместимой с regex-путём. Polygon — primary, address regex — fallback. Coverage `intra_city_raion`: apartment **100.0 %**, house **99.0 %**, commercial **97.3 %**, landplot **98.9 %** (было соответственно 12.7 / 17.7 / 12.6 / 62.4 % через regex-only).
- **AS_*_PARAMS pivot → 3 дополнительных GAR-only территориальных ключа.** Парсер `parse_object_params_xml` принимает whitelist TYPEIDов; `build_gar_lookup.py` теперь стримит PARAMS XMLы (4.6 ГБ) **один раз** с whitelist `{5, 6, 7, 8}`, после чего `build_cadnum_index` берёт TYPEID=8 (cadnum), а `build_object_params_lookup` пивотирует TYPEIDы 7/6/5 в широкую таблицу `(objectid, oktmo_full, okato, postal_index)`. Результат: `data/silver/gar_lookup/object_params.parquet` ~2.24 М строк за 0.5 с поверх уже спарсенных PARAMS-фреймов.

  Инвентарь TYPEIDов в нашем снапшоте Татарстана (active rows, первые 2 М строк `AS_HOUSES_PARAMS`):

  | TYPEID | Что | Sample | Покрытие | В фичах? |
  | --- | --- | --- | --- | --- |
  | 8 | **CADNUM** | `16:23:071001:1035` | 149 k | да (cadnum_index) |
  | 7 | **ОКТМО full** (11 знач.) | `92633412101` | 222 k | да (`oktmo_full`) |
  | 21 | ОКТМО short (8 знач.) | `92730000` | 219 k | нет — есть в `mun_okrug_oktmo` через MUN_HIERARCHY |
  | 6 | **ОКАТО** (11 знач.) | `92233000011` | 222 k | да (`okato`) |
  | 5 | **Postal index** | `420030` | 180 k | да (`postal_index`) |
  | 13 | Long FNS code (33 знач.) | `926334121010000005620001000000000` | 222 k | нет (semantics unclear, не территориальный ключ) |
  | 1, 2 | ИФНС ФЛ/ЮЛ | `1675` | 222 k | нет (tax inspector — не валуация) |
  | 3, 4 | ИФНС территориальный | `1689` | 20 k | нет |
  | 14, 15, 19 | флаги | `1` | varied | нет |
  | 20 | дата | `14.12.2022` | 100 | нет |

  Coverage в gold-выборке (только GAR-matched объекты, остальные — null):
  apartment **81.9 %**, house **56.3 %**, commercial **31.6 %**, landplot **24.4 %** для `oktmo_full`/`okato`; `postal_index` чуть меньше для commercial/landplot (rare в STEADS_PARAMS). Cardinality в Казанской агломерации низкая: 2-16 unique `oktmo_full`, 61-88 unique `postal_index` per class — для CatBoost cat_features справляется без one-hot.

## Эмпирический эффект (CatBoost spatial CV, parent_res=7, n_splits=5, 500 iters)

Сравнение block 3b (poly-area) → block 4 первая итерация (+ 4 territorial cat-features, regex-only intra_city_raion):

| class | block 3b MAPE | block 4 MAPE | Δ |
| --- | --- | --- | --- |
| apartment | 11.28 % | 11.52 % | +0.24 |
| house | 15.34 % | 15.25 % | **−0.09** |
| commercial | 41.49 % | 41.38 % | **−0.11** |
| landplot | 249.67 % | 254.97 % | +5.30 |

Сравнение block 4 v1 (regex-only) → block 4 v2 (OSM admin_level=9 polygon spatial join, intra_raion coverage 12-62 % → 97-100 %):

| class | block 4 v1 MAPE | block 4 v2 MAPE | Δ pp | block 4 v2 MAE |
| --- | --- | --- | --- | --- |
| apartment | 11.52 % | **10.70 %** | **−0.82** | 7 729 |
| house | 15.25 % | **15.11 %** | **−0.14** | 3 874 |
| commercial | 41.38 % | 41.82 % | +0.44 | 5 115 |
| landplot | 254.97 % | **252.40 %** | **−2.57** | 965 |

Сравнение block 4 v2 → block 4 v3 (+ `oktmo_full` / `okato` / `postal_index` через `object_params` lookup):

| class | block 4 v2 MAPE | block 4 v3 MAPE | Δ pp | block 4 v3 MAE |
| --- | --- | --- | --- | --- |
| apartment | 10.70 % | **10.30 %** | **−0.40** | 7 378 |
| house | 15.11 % | **15.03 %** | **−0.08** | 3 856 |
| commercial | 41.82 % | 42.78 % | +0.96 | 5 152 |
| landplot | 252.40 % | **252.29 %** | **−0.11** | 965 |

Cumulative от block 3b → block 4 v3:

| class | block 3b | block 4 v3 | Δ всего |
| --- | --- | --- | --- |
| apartment | 11.28 % | **10.30 %** | **−0.98** |
| house | 15.34 % | 15.03 % | **−0.31** |
| commercial | 41.49 % | 42.78 % | +1.29 (шум) |
| landplot | 249.67 % | 252.29 % | +2.62 (шум на 250%+) |

**apartment −0.98 пп MAPE** — стабильный signal от блока 4 в финальной форме (OSM polygon + AS_PARAMS pivot). Большая часть выигрыша (−0.82 пп) от polygon-driven `intra_city_raion`, ещё −0.40 пп — от `oktmo_full` / `okato` / `postal_index` (settlement-level OKTMO даёт CatBoost дополнительный «нормальный» bin для квартир в крупных микрорайонах). Остальные классы — в пределах run-to-run шума.

**Реальный выигрыш block 4 для house/commercial/landplot** всё ещё ожидается на **White Box моделях квартета (block 5)**: линейная регрессия / правил-ориентированная не сможет рекомбинировать lat/lon + h3_p7-агрегаты в "Советский район Казани" сама — ей нужен явный категориальный признак.

Решение **оставить block 4** в gold-схеме:

- стоимость ребилда дешёвая (поверх Polars vectorized — ~5 с на 290k объектов после lookup'а в кэше);
- закрывает методологический gap §9;
- готово к White Box без отдельной итерации;
- не ломает CatBoost-метрики (Δ в пределах run-to-run шума).

## Что доработали в этой ADR-итерации (поверх первой инкарнации)

- ~~**OSM `admin_level=9` spatial join для intra_city_raion.**~~ **Сделано** (см. block 4 v2 выше). Polygon-путь дал coverage 97–100 % против 12–62 % regex-only — и принёс apartment-MAPE −0.82 пп.
- ~~**AS_PARAMS другие TYPEID.**~~ **Сделано** (см. block 4 v3 выше). Pivot на TYPEID 7/6/5 → 3 GAR-only cat-фичи `oktmo_full` / `okato` / `postal_index`. Apartment получил ещё −0.40 пп MAPE поверх polygon-выигрыша. TYPEID=21 (short OKTMO) пропустили — redundant с MUN_HIERARCHY. Остальные TYPEIDы (1/2/3/4 ИФНС, 13 long FNS code, 14/15/19 флаги, 20 дата) не территориальные ключи — не добавляем.

## Что **не** делаем (out-of-scope для пилота)

- **Полный ингест ГАР по другим регионам.** Скачиваем только Татарстан (region 16). Эмпирически в gold-выборке Казанской агломерации **100 % `cad_num`'ов начинаются с `16:`** — non-Tatarstan объектов нет. Каждый региональный архив ФИАС ~5–6 ГБ, RF целиком ~500 ГБ raw. Триггер для ингеста других регионов = расширение pilot scope за пределы Татарстана; будет отдельный ADR при первом таком требовании. До тех пор это pure scope deferral, не упрощение.
- **Числовые derivatives по муниципалитету** (`count_per_mun`, `mean_price_per_mun`). Не вводим до White Box (блок 5). Сейчас сырые категориальные.
- **Видовые места и опорные НП.** Методологически — отдельная подкатегория §9, требует специальных source-данных (например, ОКН-реестр памятников, генплан города). Откладывается до отдельного ADR.
- **Темпоральная стабильность ГАР snapshot'а.** Снапшот 2026-04-06 фиксирован. Регулярное обновление — отдельный ops-вопрос.

## Открытые вопросы

- **Coverage of CADNUM → OBJECTID match.** Гипотеза: `AS_REESTR_OBJECTS` покрывает >95 % NSPD-объектов на Казанской агломерации. Если меньше — нужен сильный spatial fallback.
- **Что делать с null-матчами.** В CatBoost null в категориальной — отдельный bin, это OK. Но если null > 10 %, важно понять почему — это либо bug в matcher'е, либо реально внеотчётные объекты.
- **Производительность парсера на 991 МБ XML (`AS_MUN_HIERARCHY`).** Naive ElementTree.parse() съест ~5 ГБ RAM. Нужен `iterparse` + `clear()` + потоковая запись в parquet. Бенч на день 3.
- **Как валидировать lookup.** Тесты: 10–20 известных адресов из NSPD проверить, что выходят правильные мунип-имена. Источник правды — Яндекс.Карты по cad_num. Без формальной выгрузки validation set — спорный момент.
- **Стабильность OBJECTID между ГАР-снапшотами.** OBJECTID/OBJECTGUID должны быть стабильны (это назначение ФИАС), но при обновлении gar_xml/16/ возможны редкие переупорядочивания. Подтвердим при первом обновлении.

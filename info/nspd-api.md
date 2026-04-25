# NSPD API — разведка для LandPlot

Результаты разведки публичного API НСПД (Национальная Система Пространственных Данных, преемник ПКК) для извлечения данных земельных участков ЕГРН.

**Дата разведки:** 2026-04-25
**База:** `https://nspd.gov.ru`
**Авторизация:** анонимный доступ с домашнего RU IP, без cookies/токенов.

## Ключевой вывод

**`cost_index` (₽/м²) и `cost_value` (₽) из ЕГРН — это РЕАЛЬНЫЙ кадастровый таргет**, а не synthetic proxy. Это закрывает «открытый вопрос» из [ADR-0004](decisions/0004-synthetic-target.md), [ADR-0007](decisions/0007-kazan-agglomeration-scope.md), [ADR-0008](decisions/0008-per-building-multi-class-valuation.md): *«Synthetic proxy остаётся; реальные кадастровые цены подключатся отдельным слоем `targets_real/`»*.

Поля доступны и для **участков** (L36048), и для **зданий** (L36049). То есть NSPD potentially даёт реальный таргет не только для нового LandPlot-слоя, но и ретроспективно для apartment/house/commercial.

## Технические детали

### TLS

Сертификаты NSPD/Rosreestr подписаны российским Минцифры (`Russian Trusted Sub CA`), которого нет в `certifi`-бандле Python. Варианты:

1. **dev/probing:** `verify=False` (только для разведки, не для прода).
2. **прод:** установить Russian Trusted Root CA из gosuslugi.ru/crt в собственный CA-bundle и пользовать через `REQUESTS_CA_BUNDLE` или `verify=/path/to/bundle.pem`.

### Сеть

С домашнего RU IP (89.179.127.188) идёт напрямую (split-tunnel мимо VPN). VPN-выход (149.50.212.241 US) блокируется TCP-уровнем — добавлять exit-IP в исключения **не нужно**, нужно держать `nspd.gov.ru` в исключениях VPN.

Проверка маршрута:

```bash
echo | openssl s_client -connect nspd.gov.ru:443 -servername nspd.gov.ru 2>/dev/null \
  | openssl x509 -noout -issuer -subject
```

### WAF

Часть эндпоинтов режется WAF на нашем RU IP:

| Эндпоинт | Поведение |
|----------|-----------|
| `GET /api/geoportal/v2/search/geoportal?query=...` | **403 Rule: 697093d72eea83106f88c559** |
| `GET /api/wfs/v2?service=WFS&...` | **403 Forbidden** |

Остальные эндпоинты доходят до приложения.

## Слои (layers)

Layer ID можно получить через `GET /api/geoportal/v1/layers/{id}` (анонимно, без auth для большинства).

Прочёсано: `36000–36100`. Ключевые слои:

| ID | Имя | Тип | Категория | Открыт? |
|----|-----|-----|-----------|---------|
| **36048** | **Росреестр: Земельные участки ЕГРН** | wms | 36368 | **да** |
| **36049** | Росреестр: Здания ЕГРН | wms | 36369 | да |
| 36070 | ЕГРН. Кадастровые районы | wms | 36382 | да |
| 36071 | ЕГРН. Кадастровые кварталы | wms | 36381 | да |
| 36050+ | Ортофотопланы регионов | wmts | — | да |
| 36473 | Участки, образуемые по проекту межевания (полигональный) | — | — | да (упомянут в settings) |

Слои 870186 и подобные в верхнем диапазоне ID — **403/auth-walled**, требуют SSO с Client ID/Secret от ППК Роскадастр (бундл прямо это говорит).

## Рабочие эндпоинты

### 1. WMS GetFeatureInfo (single-point lookup) ⭐

**Ключевой канал для извлечения данных одного объекта по координате.**

```
GET /api/aeggis/v4/{layerId}/wms?
    SERVICE=WMS&VERSION=1.3.0&REQUEST=GetFeatureInfo&
    LAYERS={layerId}&QUERY_LAYERS={layerId}&STYLES=&CRS=EPSG:3857&
    BBOX={minx},{miny},{maxx},{maxy}&
    WIDTH={W}&HEIGHT={H}&I={x_pixel}&J={y_pixel}&
    FORMAT=image/png&INFO_FORMAT=application/json
```

`STYLES=` обязателен (пустой). `INFO_FORMAT=application/json` — иначе вернётся XML.

**Пример (центр Казани → участок):**

```python
import math, requests
def lonlat_to_3857(lon, lat):
    x = lon * 20037508.342789244 / 180
    y = math.log(math.tan((90+lat) * math.pi / 360)) * 20037508.342789244 / math.pi
    return x, y

x, y = lonlat_to_3857(49.1221, 55.7887)
buf = 50
url = (f"https://nspd.gov.ru/api/aeggis/v4/36048/wms?"
       f"SERVICE=WMS&VERSION=1.3.0&REQUEST=GetFeatureInfo&"
       f"LAYERS=36048&QUERY_LAYERS=36048&STYLES=&CRS=EPSG:3857&"
       f"BBOX={x-buf},{y-buf},{x+buf},{y+buf}&"
       f"WIDTH=101&HEIGHT=101&I=50&J=50&FORMAT=image/png&INFO_FORMAT=application/json")
r = requests.get(url, verify=False)
# 200 OK, GeoJSON FeatureCollection с 1 объектом
```

**Ответ (для L36048):**

```json
{
  "type": "FeatureCollection",
  "features": [{
    "id": 38385610,
    "type": "Feature",
    "geometry": {
      "type": "Polygon",
      "coordinates": [[[5468254.6, 7516527.1], ...]],
      "crs": {"type": "name", "properties": {"name": "EPSG:3857"}}
    },
    "properties": {
      "cadastralDistrictsCode": 16,
      "category": 36368,
      "descr": "16:50:010406:40",
      "externalKey": "16:50:010406:40",
      "geom_data_id": 38385610,
      "label": "16:50:010406:40",
      "options": {
        "cad_num": "16:50:010406:40",
        "status": "Учтенный",
        "subtype": "Землепользование",
        "specified_area": 1011.09,
        "cost_value": 16428454.11,
        "cost_index": 16248.2609,
        "ownership_type": "Частная",
        "registration_date": "2009-02-16",
        "readable_address": "Республика Татарстан, г Казань, Вахитовский район, ул Университетская, дом 12/23",
        "land_record_type": "Земельный участок",
        "previously_posted": "Учтенный"
      }
    }
  }]
}
```

**FEATURE_COUNT не помогает** — независимо от значения возвращается 1 объект (та точка, на которую попал I/J).

### 2. Layer Object lookup by internal ID

Расширенная карточка объекта по `geom_data_id` (внутреннему ID NSPD, не кадастровому номеру).

```
GET /api/geoportal/v1/layers/{layerId}/object?id={internalId}
```

Возвращает FeatureCollection с одним объектом. Дополнительные поля по сравнению с WMS GetFeatureInfo:

```json
"options": {
  "cost_application_date": "2024-01-01",
  "cost_determination_date": "2022-01-01",
  "cost_registration_date": "2023-02-12",
  "determination_couse": "Акт об утверждении результатов определения кадастровой стоимости...",
  "land_record_area_verified": 1011.09,
  "land_record_category_type": "Земли населенных пунктов",
  "land_record_subtype": "Землепользование",
  "interactionId": 38356313
}
```

Полезно если `geom_data_id` уже известен (например, после первичного prefilter через bbox-сканирование).

### 3. WMS GetMap (raster tiles)

Растровые PNG-тайлы для визуализации:

```
GET /api/aeggis/v4/{layerId}/wms?
    SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={layerId}&STYLES=&
    CRS=EPSG:3857&BBOX={minx},{miny},{maxx},{maxy}&
    WIDTH=256&HEIGHT=256&FORMAT=image/png&TRANSPARENT=TRUE
```

Не даёт вектор/атрибуты, но полезно для тайлового оверлея на нашей карте.

### 4. WMS GetCapabilities

```
GET /api/aeggis/v4/{layerId}/wms?service=WMS&request=GetCapabilities&version=1.3.0
```

XML с метаданными слоя. Подтверждает поддержку `GetMap` / `GetFeatureInfo` / `queryable`.

### 5. Layer info

```
GET /api/geoportal/v1/layers/{layerId}
```

JSON с метаданными слоя: имя, категория, тип (wms/wmts), bbox, options (cache/format/queryable).

### 6. Attribute search settings

```
GET /api/geoportal/v3/page-attrib-search-settings?pageCode=geoportal
```

Возвращает поддерживаемые фильтры для каждого слоя. Для L36048:

| Поле | Тип | Filter | Rules |
|------|-----|--------|-------|
| `options.readable_address` | string | textQueryAttrib | — |
| `options.area` | number | exactNumber, intervalNumber | must, must_not, gt/gte/lt/lte |
| `options.cost_value` | **number** | **exactNumber, intervalNumber** | **must, must_not, gt/gte/lt/lte** |
| `options.ownership_type` | string | textQueryAttrib | — |
| `options.land_record_reg_date` | string | exactDate, intervalDate | gt/gte/lt/lte |
| `options.permitted_use_established_by_document` | string | textQueryAttrib | — |

Для каждого слоя `count` = max объектов на запрос (обычно 40).

### 7. Bulk attribute search ⭐⭐ (главный канал для сбора)

```
POST /api/geoportal/v3/geoportal/{layerId}/attrib-search?page={N}&count={M}&withTotalCount=true
```

**Body** — словарь `{filter_name: [filter_object, ...]}`. Несколько фильтров комбинируются по AND. Структуры filter_object извлечены из минифицированного `index-BGHZh2Td.js` (ключевая ошибка предыдущих попыток — `attribsID` с большой `D`, не `attribsId`):

| filter | Структура объекта | Пример |
|--------|-------------------|--------|
| `textQueryAttrib` | `{keyName, value}` или `{keyName, attribsID, value}` | `{"keyName": "options.readable_address", "value": "Казань"}` |
| `exactNumber` | `{keyName, rule: "must"\|"must_not", value}` | `{"keyName": "options.area", "rule": "must", "value": 1011.09}` |
| `intervalNumber` | `{keyName, gt\|gte\|lt\|lte: value}` | `{"keyName": "options.area", "gt": 1000}` |
| `exactDate` | `{keyName, rule, value}` | `{"keyName": "options.land_record_reg_date", "rule": "must", "value": "2009-02-16"}` |
| `intervalDate` | `{keyName, gt\|gte\|lt\|lte: value}` | `{"keyName": "options.land_record_reg_date", "gte": "2020-01-01"}` |
| `existVal` | `{rule: "must_not", keyName}` (есть значение) | `{"rule": "must_not", "keyName": "options.area"}` |
| `textAttribValsList` | `{keyName, rule, values: [v]}` | `{"keyName": "options.ownership_type", "rule": "must", "values": ["Частная"]}` |

**Ответ:**

```json
{
  "data": {
    "type": "FeatureCollection",
    "features": [
      {"id": ..., "geometry": {...}, "properties": {..., "options": {"cad_num": ..., "cost_index": ..., ...}}}
    ]
  },
  "meta": [{"totalCount": 199819, "categoryId": 36368}]
}
```

**Реальные параметры (на 2026-04-25):**

| Запрос | totalCount |
|--------|-----------|
| `textQueryAttrib options.readable_address ⊂ "Казань"` (L36048 — участки) | **199 819** |
| `+ intervalNumber options.area > 1000` | 7 |
| count limit per page | **минимум 200** (выше дефолтных 40 из settings) |

**Bulk-стратегия для агломерации:**
- 199 819 объектов / 200 на страницу = **1000 запросов**
- rate-limit 1 req/sec → **~17 минут** на полную выгрузку
- postfilter по полигону `data/raw/regions/kazan-agglomeration.geojson` (geopandas spatial filter) → отсечь не-агломерационные

**Минимальный воркер:**

```python
def fetch_page(layer_id: int, body: dict, page: int, count: int = 200):
    url = f"https://nspd.gov.ru/api/geoportal/v3/geoportal/{layer_id}/attrib-search"
    params = {"page": page, "count": count, "withTotalCount": "true"}
    r = requests.post(url, json=body, params=params, verify=False, timeout=30)
    r.raise_for_status()
    j = r.json()
    return j["data"]["features"], j["meta"][0]["totalCount"]

body = {"textQueryAttrib": [{"keyName": "options.readable_address", "value": "Казань"}]}
features_page0, total = fetch_page(36048, body)
# total ≈ 199819, features_page0 ≈ 200
# Pagination: page=0..total//count
```

### Не вскрытые эндпоинты

### intersects (geometry-based bulk lookup)

В JS-бандле:

```js
// .../categories/{categoryId}/geom/geojson
intersect: { path: "api/geoportal/v3/intersects" },
intersectionFinder: { path: "api/geoportal/v3/intersects" },
```

Все попытки `POST /api/geoportal/v3/intersects` → **404**. Возможно полный путь `/api/{base}/v3/{prefix}/intersects` использует промежуточный сегмент, который не извлёкся из минификации.

В JS вызов выглядит так:

```js
this.intersectionViewer.find({
  excludedGeomId: [g.id],
  object: { geom: h, categories: m },
  typeIntersect: qa  // qa — переменная, значение в минификации не нашлось
})
```

`typeIntersect` — это значение из enum, не строковый литерал.

### download/intersections/csv

```
api/geoportal/v3/download/intersections/csv
```

Упоминается в JS-бандле, не тестировалось — потенциально bulk-download через CSV. Та же зависимость от правильного тела запроса.

## Стратегия bulk-сбора (ранжированы по реалистичности)

**1. Reverse-engineer attrib-search через браузер.** Открыть `nspd.gov.ru/map`, выполнить **атрибутный поиск** на слое «Земельные участки» (фильтр по адресу/площади/стоимости), скопировать **успешный** запрос из DevTools Network как cURL. Из него:
- точное URL и тело
- значение `typeIntersect` enum
- структура `attribsId` / `attributes` массива

После этого можно постранично выкачивать все участки по фильтру bbox/cost_value/area. Лимит 40 на страницу × N страниц.

**2. Сетка точек GetFeatureInfo (медленно, но работает).** Для агломерации Казани (30km × 30km, π·30² ≈ 2 827 км²) при шаге 50м это 600×600 ≈ 360k запросов. При rate-limit 1 req/sec → 100 часов; даже 5 req/sec не безопасно (риск блокировки IP).

Можно ускорить через **OSM-driven seeding**: у нас уже есть 179k OSM-зданий с координатами. Прогон 1 GetFeatureInfo на здание = 179k запросов = 50 часов на 1/sec, 10 часов на 5/sec. Это даёт участки ПОД зданиями, без пустых дворов и улиц. Покроет наш use case.

**3. Cadastral-quarter-driven.** L36071 (Кадастровые кварталы) — границы квартала. Для каждого квартала нашей агломерации сначала fetch квартальной геометрии, потом — равномерная сетка ВНУТРИ квартала с минимальным шагом. Меньше холостых запросов, чем (1).

**4. Подождать формального доступа** через ППК Роскадастр (Client ID/Secret). Реалистично: недели. Открывает все эндпоинты включая bulk и расширенные слои.

## Rate-limit и анти-бан

Параметры разведки (что не вызвало проблем):

- ~50 запросов в течение 30 минут с домашнего RU IP — нормально.
- WMS GetFeatureInfo, WMS GetCapabilities, layer info — 100% success rate.
- Single-IP, простой `User-Agent`, без cookies — работает.

**Что НЕ делать:**
- Sweep слоёв ID 1000+ диапазонами без пауз → быстрый sweep на ~100 layer ID одним скриптом ~3 секунды отработал, но это близко к границе.
- Конкурентные запросы.
- Активничать на v2/search (она режется WAF, риск получить расширенный бан).

**Что пробовать в проде:**
- Глобальный rate-limit ≤ 1 req/sec (полит-секунды).
- Exponential backoff при 429/503.
- Респектовать `Retry-After`.
- Логировать `requestId` для дебаггинга.
- Делать выгрузку батчами по нескольку часов с перерывами.

## Запросы для повторной проверки

Все пробы лежат во временных скриптах `/tmp/probe_*.py`. Сохраняемая выжимка:

```python
# Минимальный probe — проверить что канал жив
import math, urllib3, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def lonlat_to_3857(lon, lat):
    x = lon * 20037508.342789244 / 180
    y = math.log(math.tan((90+lat) * math.pi / 360)) * 20037508.342789244 / math.pi
    return x, y

def fetch_parcel(lat, lon):
    x, y = lonlat_to_3857(lon, lat)
    buf = 50
    url = (f"https://nspd.gov.ru/api/aeggis/v4/36048/wms?"
           f"SERVICE=WMS&VERSION=1.3.0&REQUEST=GetFeatureInfo&"
           f"LAYERS=36048&QUERY_LAYERS=36048&STYLES=&CRS=EPSG:3857&"
           f"BBOX={x-buf},{y-buf},{x+buf},{y+buf}&"
           f"WIDTH=101&HEIGHT=101&I=50&J=50&FORMAT=image/png&INFO_FORMAT=application/json")
    r = requests.get(url, verify=False, timeout=15)
    return r.json()
```

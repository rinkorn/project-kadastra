# NSPD API — открытый канал к реальным кадастровым данным

> Файл написан по итогам разведки 2026-04-25. Цель — задокументировать, как мы можем легально и анонимно получать данные ЕГРН (земельные участки, здания) с публичного картографического сервиса НСПД, чтобы в проекте появился настоящий кадастровый таргет вместо синтетического прокси.

## Зачем это всё

В пилоте до сих пор использовался **synthetic proxy** ([ADR-0004](decisions/0004-synthetic-target.md)) — формула, которая делает вид, что предсказывает кадастровую стоимость. Это нужно было, потому что в S3-бакете проекта (`s3://kadastrova/Kadatastr/`) лежат только OSM-выгрузки и ГАР XML, а в ГАР `AS_STEADS` нет ни координат, ни цен — только идентификаторы. Реальной цены на квадратный метр у нас не было.

Параллельно в [ADR-0008](decisions/0008-per-building-multi-class-valuation.md) мы запланировали четвёртый класс — `LandPlot` — но «отложили его до момента, когда найдём источник с геопривязкой и кадастром».

Разведка НСПД (Национальная Система Пространственных Данных, gov-преемник «публичной кадастровой карты» pkk.rosreestr.ru) показала: и геопривязка, и кадастровые номера, и **реальные цены ЕГРН** доступны анонимно. Это закрывает оба открытых вопроса разом — для всех четырёх классов, не только для участков.

## Что мы получили в итоге одним POST-запросом

Пример выдачи на одну точку в центре Казани:

```json
{
  "cad_num": "16:50:010406:40",
  "specified_area": 1011.09,
  "cost_value": 16428454.11,
  "cost_index": 16248.2609,
  "ownership_type": "Частная",
  "land_record_category_type": "Земли населенных пунктов",
  "land_record_subtype": "Землепользование",
  "land_record_reg_date": "2009-02-16",
  "readable_address": "Республика Татарстан, г Казань, Вахитовский район, ул Университетская, дом 12/23",
  "geometry": "Polygon EPSG:3857 [...]"
}
```

`cost_value` (16,4 млн ₽) — полная кадастровая стоимость, `cost_index` (16 248 ₽/м²) — удельная. Это **ровно тот таргет**, который мы пытались синтезировать через формулы в [`compute_synthetic_target`](../src/kadastra/etl/synthetic_target.py) и [`compute_object_synthetic_target`](../src/kadastra/etl/object_synthetic_target.py). Теперь его можно просто скачать.

Для зданий (слой L36049) набор полей ещё богаче: `floors`, `year_built`, `materials`, `purpose` («Жилой дом», «Многоквартирный дом», ...), `quarter_cad_number`, `right_type`. Это **реальные ML-фичи**, которых не было в OSM (там только centroid, building tag и иногда `levels`).

## Где именно живёт нужный нам слой

В НСПД сотни «слоёв» (`layers`), у каждого свой числовой ID и свои уровни доступа. Большинство закрыто: попытка прочитать `/api/geoportal/v1/layers/{id}` для крупных ID типа `870186` возвращает `forbidden код=3` — за этой стеной живёт **формальное соглашение с ППК Роскадастр** (Client ID + Client Secret), которое нам не выдадут без юр.договора. Бандл сайта прямо это говорит цитатой: *«Обратитесь в ППК «Роскадастр» для заключения соглашения об информационном взаимодействии с ФГИС ЕЦП НСПД»*.

Но **базовые слои ЕГРН открыты** — их видит обычный пользователь, заходящий на nspd.gov.ru/map. Прочесав диапазон ID от 36000 до 36100, нашли:

| ID | Название | Тип | Что это |
|----|----------|-----|---------|
| **36048** | Росреестр: Земельные участки ЕГРН | wms | то, что нужно для класса LandPlot |
| **36049** | Росреестр: Здания ЕГРН | wms | реальный таргет для apartment/house/commercial |
| 36070 | ЕГРН. Кадастровые районы | wms | админ-разбиение (контекст) |
| 36071 | ЕГРН. Кадастровые кварталы | wms | мельче чем районы (потенциальный фильтр) |
| 36473 | Земельные участки в межевании (полигональный) | — | в работе у Росреестра |
| 36050+ | Ортофотопланы регионов | wmts | растровая подложка под карту |

С этого момента мы говорим только про L36048 и L36049 — это анонимные слои с реальными атрибутами и геометрией.

## Технические подробности доступа

### TLS — почему `verify=False` это не «костыль»

Сертификат `nspd.gov.ru` подписан **«Russian Trusted Sub CA»** (CA Минцифры РФ). Это легитимный российский корневой сертификат, но его **нет в `certifi`-бандле**, который Python подгружает по умолчанию. Поэтому `requests` (или `httpx`) с дефолтной верификацией падает с «self-signed certificate in certificate chain» — это не атака, а просто неизвестный CA.

Варианты для прода:
1. Скачать `Russian Trusted Root CA` с `gu-st.ru/content/Other/doc/russian_trusted_root_ca.cer`, добавить в собственный CA-bundle, передавать через `REQUESTS_CA_BUNDLE` или `verify="/path/to/bundle.pem"`.
2. Альтернатива — `truststore` package (использует системный keychain macOS, который часто уже знает Минцифровый CA).

Для разовой выгрузки публичных данных мы используем `verify=False`. Скрипт прода переключим на bundle.

### Сеть — почему VPN не мешает

Я попробовал три IP:
- **VPN-выход US Datacamp (149.50.212.241)** — TCP timeout. NSPD блокирует не-российские IP уже на сетевом уровне.
- **Домашний RU IP (89.179.127.188) с дефолтной маршрутизацией VPN** — то же самое (трафик уходил через VPN-выход).
- **Домашний RU IP с `nspd.gov.ru` в исключениях VPN (split-tunnel)** — работает.

Это видно прямо в ответе WAF: при попытках со split-tunnel сервер возвращает `Client IP: 89.179.127.188` — наш реальный домашний адрес. То есть нужно держать `nspd.gov.ru` в exclude-списке конкретного VPN-клиента.

### WAF — что блокируется и почему это не страшно

Часть эндпоинтов NSPD режется WAF rule'ами **независимо от IP**:
- `GET /api/geoportal/v2/search/geoportal?query=...` → 403, rule `697093d72eea83106f88c559`
- `GET /api/wfs/v2?...` → 403 Forbidden

Эти эндпоинты — публичный текстовый поиск и WFS — закрыты от автоматизации, видимо чтобы не давали скрейпить «по подсказкам в строке поиска». **Нам они и не нужны** — у нас есть нормальный фильтрованный канал (см. ниже).

## Каналы данных

### Канал 1: WMS GetFeatureInfo — 1 точка, 1 объект

Это лендингный канал: даёт по координате (lon/lat) ровно 1 объект под этой точкой со всеми атрибутами и геометрией. Полезен для ad-hoc проверок («что лежит на адресе X»), не для bulk.

```
GET /api/aeggis/v4/{layerId}/wms?
    SERVICE=WMS&VERSION=1.3.0&REQUEST=GetFeatureInfo&
    LAYERS={layerId}&QUERY_LAYERS={layerId}&STYLES=&CRS=EPSG:3857&
    BBOX={minx},{miny},{maxx},{maxy}&
    WIDTH=101&HEIGHT=101&I=50&J=50&FORMAT=image/png&INFO_FORMAT=application/json
```

Особенности:
- `STYLES=` обязателен (даже пустой), иначе 500.
- `INFO_FORMAT=application/json` — иначе вернётся XML `ServiceException`.
- `I=, J=` — пиксельные координаты внутри плитки `WIDTH×HEIGHT` пикселей. На центр кладём `I=W/2, J=H/2`.
- `BBOX` — в EPSG:3857 (Web Mercator), не в WGS84. Конвертация через стандартную формулу.
- `FEATURE_COUNT` не помогает — ответ всегда 1 объект (тот, на который попал I/J).

Возвращает `FeatureCollection` с одним `Feature`. Геометрия — `Polygon` или `MultiPolygon` в EPSG:3857. Все интересные поля внутри `properties.options`.

### Канал 2: Object lookup by internal ID — расширенная карточка

```
GET /api/geoportal/v1/layers/{layerId}/object?id={geom_data_id}
```

Принимает внутренний ID (`geom_data_id` из ответа GetFeatureInfo, не кадастровый номер!). Возвращает то же самое, но с **дополнительными полями**: `cost_application_date`, `cost_determination_date`, `cost_registration_date`, `determination_couse` (sic — это опечатка в API), `interactionId`. Полезно для углублённой выгрузки конкретного объекта; для bulk не подходит — нужно сначала узнать `geom_data_id`.

### Канал 3: attrib-search v3 — bulk по фильтру (наш основной канал) ⭐

```
POST /api/geoportal/v3/geoportal/{layerId}/attrib-search?page={N}&count={M}&withTotalCount=true
Content-Type: application/json
```

Это тот самый bulk-канал, на котором держится вся наша выгрузка. Был не очевиден потому, что:
- В JS-бандле сайта функция-обёртка называется `Q5` (минификация);
- Из её сигнатуры `Q5({layerId, attributes, count})` я сначала отправлял body `{attributes: [...]}` — отдавался `code:1010 attribsId or keyName was not set`;
- В минифицированном коде сборки body есть строка `attribsID:e.attribsId` — то есть API ждёт **`attribsID` с большой `D`**, а не `attribsId`. Это уже совсем не очевидно по сообщению ошибки.

После реверса оказалось, что body — это **словарь, ключи которого — имена фильтров**, а значения — массивы конкретных условий. Несколько разных фильтров комбинируются по AND.

#### Поддерживаемые фильтры

Для слоя L36048 (участки), запрос `GET /api/geoportal/v3/page-attrib-search-settings?pageCode=geoportal` возвращает:

| keyName | type | Доступные filter-методы |
|---|---|---|
| `options.readable_address` | string | `textQueryAttrib`, `existVal` |
| `options.area` | number | `exactNumber`, `intervalNumber`, `existVal` |
| `options.ownership_type` | string | `textQueryAttrib` |
| `options.cost_value` | number | `exactNumber`, `intervalNumber`, `existVal` |
| `options.land_record_reg_date` | string (date) | `exactDate`, `intervalDate`, `existVal` |
| `options.permitted_use_established_by_document` | string | `textQueryAttrib`, `existVal` |

Для L36049 (здания) набор похожий, плюс `build_record_area`, `build_record_registration_date`.

#### Структура body для каждого filter-метода

| filter | Объект-условие |
|---|---|
| `textQueryAttrib` | `{"keyName": "...", "value": "..."}` |
| `exactNumber` | `{"keyName": "...", "rule": "must"\|"must_not", "value": N}` |
| `intervalNumber` | `{"keyName": "...", "gt"\|"gte"\|"lt"\|"lte": N}` |
| `exactDate` | `{"keyName": "...", "rule": "must"\|"must_not", "value": "YYYY-MM-DD"}` |
| `intervalDate` | `{"keyName": "...", "gt"\|"gte"\|"lt"\|"lte": "YYYY-MM-DD"}` |
| `textAttribValsList` | `{"keyName": "...", "rule": "must"\|"must_not", "values": [v]}` |
| `existVal` | `{"rule": "must_not", "keyName": "..."}` (= это поле непусто) |

Пример рабочего body — взять все участки с адресом «Казань», у которых площадь больше 1000 м²:

```json
{
  "textQueryAttrib": [
    {"keyName": "options.readable_address", "value": "Казань"}
  ],
  "intervalNumber": [
    {"keyName": "options.area", "gt": 1000}
  ]
}
```

#### Структура ответа

```json
{
  "data": {
    "type": "FeatureCollection",
    "features": [
      {
        "id": 38385610,
        "geometry": {"type": "Polygon", "coordinates": [...]},
        "properties": {
          "cad_num": "...",
          "category": 36368,
          "options": {"cad_num": "...", "cost_value": ..., "cost_index": ..., ...}
        }
      },
      ...
    ]
  },
  "meta": [{"totalCount": 199819, "categoryId": 36368}]
}
```

`meta[0].totalCount` — общее число объектов для фильтра, по нему мы и пагинируем.

#### Параметры пагинации

- `page=N` — номер страницы, начиная с 0.
- `count=M` — сколько объектов на странице. Settings по умолчанию советуют 40, но фактический лимит выше — мы успешно выкачивали по 200 за страницу. Это в 5 раз сокращает число запросов.
- `withTotalCount=true` — заставляет включать `totalCount` в каждый ответ (иначе вернётся только в первом).

## Реальные масштабы (агломерация Казани)

| Запрос | totalCount |
|---|---|
| L36048, адрес ⊂ «Казань» | **199 819** участков |
| L36049, адрес ⊂ «Казань» | **91 864** зданий |
| L36048, «Казань» + площадь > 1000м² | 7 (sanity-check, что AND работает) |

199 819 / 200 на страницу = **1 000 запросов** для всей выгрузки участков. При rate-limit 2 сек/запрос с jitter — **~33–42 минуты**. Для зданий 460 страниц — **~17–20 минут** (по факту вышло 38 минут с одним промежуточным 502, см. ниже).

Фильтр `адрес ⊂ "Казань"` шире самой агломерации — он захватит и участки в других населённых пунктах Татарстана с похожим адресом. После выгрузки у нас есть `kazan-agglomeration.geojson` (буфер 30 км вокруг центра, [ADR-0007](decisions/0007-kazan-agglomeration-scope.md)), которым можно отфильтровать на этапе ETL.

## Стратегия безопасной выгрузки

NSPD нигде явно не публикует rate-limit, и публичных RPS-гайдлайнов у Минцифры нет. Поэтому действуем по принципу «не быть похожим на бота, который качает в продуктив»:

- **1 TCP-сессия** через `httpx.Client()` (keepalive). Никакого параллелизма с одного IP.
- **2 секунды + jitter ±0.5 сек** между запросами. Это **в два раза медленнее минимума** (1 RPS), который сам по себе уже консервативен.
- **Реалистичный User-Agent** (Chrome 131 на macOS) и `Referer: https://nspd.gov.ru/map` — как будто из браузера на самой карте.
- **Backoff** на 429 (Too Many Requests), 502 (Bad Gateway), 503 (Service Unavailable), 504 (Gateway Timeout): 30 сек → 90 сек → 5 мин, потом фейл.
- **Hard-stop** на 403: если зацепили WAF rule, продолжать молотить — гарантированно получить более жёсткий бан.
- **Resume**: каждая страница — отдельный JSON на диске. Если упали посередине, повторный запуск пропустит уже скачанные.

Скрипт-выгрузчик: [scripts/download_nspd_layer.py](../scripts/download_nspd_layer.py).

### Опыт первой полной выгрузки (2026-04-25)

Слой L36049 (здания), 460 страниц:
- На странице 220 (47%) пришёл `502 Bad Gateway`. Скрипт первой версии знал только про 429/503 и упал.
- Пофиксил retry-список (`429, 502, 503, 504`), перезапустил — pickup с 220, прошёл до конца за ещё ~13 минут.
- Итого 38 минут на 91 864 объекта. Размер на диске — **204 МБ** сырого JSON.
- Никаких 403 / IP-блокировок не было.

Никаких побочных эффектов на нашем IP я тоже не заметил — после выгрузки одиночные пробы продолжали возвращать 200 OK как раньше.

## Минимальный пример своими руками

```python
import httpx
import math

def fetch_page(layer_id: int, body: dict, page: int = 0, count: int = 200) -> dict:
    url = f"https://nspd.gov.ru/api/geoportal/v3/geoportal/{layer_id}/attrib-search"
    params = {"page": page, "count": count, "withTotalCount": "true"}
    headers = {
        "User-Agent": "Mozilla/5.0 (...) Chrome/131.0.0.0 Safari/537.36",
        "Referer": "https://nspd.gov.ru/map",
        "Origin": "https://nspd.gov.ru",
    }
    with httpx.Client(verify=False, timeout=30.0, headers=headers) as client:
        r = client.post(url, json=body, params=params)
        r.raise_for_status()
        return r.json()

body = {"textQueryAttrib": [{"keyName": "options.readable_address", "value": "Казань"}]}
data = fetch_page(36048, body)
print(data["meta"])           # [{'totalCount': 199819, 'categoryId': 36368}]
print(len(data["data"]["features"]))  # 200
```

## Что **не** работает (чтобы не тратить время)

| Эндпоинт | Что вернул | Вывод |
|---|---|---|
| `POST /api/geoportal/v1/intersects` | 400 `field:typeIntersect, rule:required;` | Эндпоинт жив, но `typeIntersect` — минифицированная переменная, которую я не нашёл в бандле (`qa`). Reverse-engineering можно продолжить, но смысла нет — атрибутный поиск перекрывает |
| `POST /api/geoportal/v3/intersects` | 404 | Не существует на этом пути |
| `POST /api/geoportal/v3/geoportal/{id}/attrib-search` с body `{"keyName": ..., "value": ...}` (плоско) | 400 `attribsId or keyName was not set` | Body должен быть **словарь по типу фильтра**, не плоский. Пока это не понял — никакой keyName API не примет |
| `GET /api/aeggis/v4/{id}/wms?REQUEST=GetFeatureInfo` без `STYLES=` | 500 `field:styles, rule:required` | Параметр обязателен, даже если пустой |
| `GET /api/geoportal/v2/search/geoportal?query=...` | 403 WAF | Не пытаться |

## Открытые направления

- **`/api/geoportal/v1/download/intersections/csv`** упоминается в JS-бандле, не пробовался. Возможно — дополнительный канал bulk-выгрузки в CSV-формате, что упростит ETL (без парсинга GeoJSON). Стоит попробовать после того как закончится текущая выгрузка.
- **`intersects` с правильным `typeIntersect`** — даст пространственную фильтрацию (бэкэнд считает пересечения с присланным полигоном). Альтернатива нашему postfilter'у через `kazan-agglomeration.geojson`. Не критично, потому что postfilter работает.
- **Refresh-стратегия**: ЕГРН обновляется регулярно. Сейчас выгрузка — разовая. Когда дойдём до прода, нужен будет инкремент по `cost_application_date` или `cost_registration_date` (оба поля — вполне фильтруемые `intervalDate`).

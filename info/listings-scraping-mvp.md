# MVP-выгрузка листингов (Yandex Realty / CIAN / Avito)

## Зачем

Конечная цель — кадастровая оценка квартир по-новому, через обученную
рыночную модель ([ADR-0010](decisions/0010-methodology-compliance-roadmap.md)).
Для этого нужен **источник цен квартир**. Сделок (Росреестр/ППК) у нас пока
нет — это закрытое API. Из публично доступного остаются три листинговых
площадки: **Yandex Realty**, **CIAN**, **Avito**.

ToS у всех трёх **запрещает automated сбор**. Поэтому задача MVP — не
production-cron, а **research-сравнение**: понять, у кого данные больше и
точнее, и **с кем выгоднее заключать договор о легитимном доступе**.

Пилот — **Казань** (текущий regiond MVP-проекта kadastra, см.
[ADR-0007](decisions/0007-kazan-agglomeration-scope.md)). Иркутск запланирован вторым после расширения скоупа.

## Что собрано (на 2026-04-26)


| Источник      | Город   | Страниц HTML | Уникальных карточек | Метод                          |
| ------------- | ------- | ------------ | ------------------- | ------------------------------ |
| Yandex Realty | Казань  | **99**       | **651**             | DOM-regex                      |
| CIAN          | Казань  | **183**      | **1 268**           | JSON `"offers":[...]`, cat.php |
| Avito         | Казань  | 4            | 196                 | JSON-LD AggregateOffer         |
| Yandex Realty | Иркутск | 28           | 441                 | DOM-regex (partial)            |


**Всего:** 2 556 уникальных квартир (продажа, всех типов: вторичка +
новостройка-ДДУ + студии). Размер raw HTML на S3 — ~2.4 ГБ.

Следующие шаги: доскачка Avito Казань после снятия IP-блока + Иркутск
(Yandex+CIAN+Avito) после probe `region_id` для CIAN.

## S3 и локальные пути

```
s3://kadastrova/Kadatastr/listings-mvp/
  raw/
    {source}_{city}/page-NNN.html       # 315 файлов
    _manifest.json                      # ok/err/skipped по запускам
  silver/
    cian_kazan.parquet                  # native CIAN-schema (28 кол)
    yandex_realty_kazan.parquet         # native Yandex (12 кол)
    yandex_realty_irkutsk.parquet
    avito_kazan.parquet                 # native Avito (15 кол)
    all.parquet                         # унифицированный long (18 кол)
    _summary.json                       # rows/dedup на источник-город
```

Локально те же пути под `data/raw/listings-mvp/` и `data/silver/listings-mvp/`
(оба в `.gitignore` — большие, восстанавливаются с S3 / повторного scrape).

## Антибот: что работает

Все три площадки активно блокируют automated-сбор. Что прошло:


| Площадка      | Защита                                                  | Решение                                                               |
| ------------- | ------------------------------------------------------- | --------------------------------------------------------------------- |
| Yandex Realty | Yandex SmartCaptcha (на curl/headless), TLS-fingerprint | `patchright + channel="chrome" + headless=False + persistent profile` |
| CIAN          | DataDome-class WAF («Кажется, у вас включён VPN»)       | то же                                                                 |
| Avito         | PerimeterX-class антибот, прогрессивный challenge       | то же                                                                 |


`headless=True`, `httpx`/`curl` без браузера и любые «одиночные сессии без
profile» режутся у всех трёх.

**Ограничения**, всплывшие при MVP-выгрузке:

- **Yandex SmartCaptcha** на ~p=99 — стабильно. После триггера 3 страницы
подряд возвращают ~18 КБ HTML с заголовком «Вы не робот?». Auto-stop
detector в [scripts/download_listings_mvp.py](../scripts/download_listings_mvp.py)
(3 страницы < 200 КБ подряд) ловит это автоматически и переходит к
следующему источнику.
- **Avito IP-block** — мгновенный. После 4 нормальных страниц на 5-й
выдаёт 30 КБ HTML «Доступ ограничен: проблема с IP». Снимается при
смене IP / спустя ~30-60 минут. В рамках MVP отложено.
- **CIAN** не словил блок ни разу — самый «толерантный» к скрейпу из
трёх (но это тоже ToS-violation).

## CIAN URL-ловушка (важный фикс)

Subdomain-URL `https://{city}.cian.ru/kupit-kvartiru/?p=N` **не
пагинирует**: возвращает один и тот же топ-28 листингов на любом `p`.
Это «sticky list», а не пагинация. Ушло 1.6 ГБ HTML / 500 страниц
впустую, прежде чем это обнаружили (после dedup осталось всего 66
уникальных).

**Правильный URL** — `cat.php` с явным `region_id`:

```
https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&offer_type=flat&region={REGION_ID}&p={N}
```

Probe-проверка показала: на этом URL `p=1, 2, 3, 5, 10` дают **полностью
разные** 28 ID на каждой странице. После переключения на cat.php дошли
до естественного потолка выдачи: ~p=183 даёт 1 268 уникальных по Казани
(CIAN сам перестаёт показывать новые объявления после ~5-7K серпа).

### Справочник `region_id` для CIAN


| Город / регион    | region_id | Статус                              |
| ----------------- | --------- | ----------------------------------- |
| Казань            | **4777**  | ✓ валидирован probe                 |
| Иркутская область | TODO      | нужен probe перед запуском Иркутска |
| Москва            | 1         | (известно из docs)                  |
| Санкт-Петербург   | 2         | (известно из docs)                  |


Для новых городов — сделать probe: открыть `https://www.cian.ru/regions/`
в headed Chrome → выбрать регион → скопировать `region` из URL после
редиректа. Внести в `PLAN` в [download_listings_mvp.py](../scripts/download_listings_mvp.py).

## Pipeline

Три скрипта, последовательно:

1. **Download** — [scripts/download_listings_mvp.py](../scripts/download_listings_mvp.py)
  - Аргументы: `--n-pages N` (потолок), `--cities {kazan, irkutsk}`,
   `--sources {yandex_realty, cian, avito}`.
  - Пример: `uv run --with patchright python scripts/download_listings_mvp.py --n-pages 500 --cities kazan --sources yandex_realty cian`.
  - Инкрементальный: уже скачанные файлы > 100 КБ пропускает (повторный
  запуск после crash подхватывает).
  - Auto-stop: 3 подряд страницы < 200 КБ → выдача исчерпана / антибот.
  - Persistent profile **отдельно для Yandex** (`data/raw/yandex-realty-probe/profile/`)
  и **общий для CIAN+Avito** (`data/raw/cian-avito-probe/profile/`).
  Профили — в `.gitignore`, обновляются при каждом запуске.
2. **Extract** — [scripts/extract_listings_mvp.py](../scripts/extract_listings_mvp.py)
  - Самодостаточный: парсеры всех трёх источников встроены.
  - Дедуп по `id` (CIAN/Avito/Yandex все возвращают numeric id).
  - Пишет per-source-city parquet (native schema) + унифицированный
  `all.parquet`.
3. **Compare** — [scripts/compare_listings_sources.py](../scripts/compare_listings_sources.py)
  - Считает: volume, schema richness, % completeness ключевых полей,
   медианы цен ₽ и ₽/м², отклонение от ЕМИСС #61781 (city-level якорь
   Росстата), pagination dedup ratio.
  - Выход: `data/silver/listings-paginated/_compare.csv` + markdown.

## Сравнение источников (Казань 2026-04-26)

Богатство схемы и полнота ключевых полей:


| Поле                      | Yandex Realty | CIAN                  | Avito |
| ------------------------- | ------------- | --------------------- | ----- |
| price_rub                 | 100 %         | 100 %                 | 100 % |
| total_area_m2             | 100 %         | 91 %                  | 84 %  |
| floor / floors_count      | 100 %         | 100 %                 | 84 %  |
| rooms                     | 99 %          | 91 %                  | 84 %  |
| **build_year**            | —             | **41 %** (757 / 1268) | —     |
| **material_type**         | —             | **93 %**              | —     |
| **lat / lon**             | —             | **100 %**             | —     |
| Размер per row (МБ)       | 0.1           | 4.0                   | 0.15  |
| Pagination dedup на 3 стр | 30 %          | 50 % (sticky promo)   | 0 %   |


**Преимущества** каждого:

- **CIAN** — единственный даёт **lat/lon без геокодера**, плюс
`build_year`, `material_type`, `kitchen_area`, `living_area`,
`deadline_year/quarter` (для новостроек). Готовые ML-фичи.
- **Avito** — самый высокий **уник-на-страницу** (50 без dedup), но
скудная мета (тип квартиры в `name`, без адресов и координат). И
быстрее всех IP-блокирует.
- **Yandex** — средняя позиция: координат нет, но `raw_text` карточки
содержит **полный адрес с ЖК и улицей** (геокодабельно через
Nominatim). Стабильно проходит до ~99 страниц.

**Anchor delta** (медиана источника vs ЕМИСС #61781 2025-Q4 «Центр
субъекта, вторичка»): по Казани все три источника завышают на 25-37 %
относительно якоря Росстата (157 268 ₽/м²). Это ожидаемо — листинги это
asks, а не сделки. Но **относительный** ранг источников по точности —
это и есть сигнал для выбора партнёра.

## Что НЕ собрано

Текущая MVP-выгрузка покрывает только **квартиры в продажу** (URL-фильтр
`/kupit/kvartira/`, `?offer_type=flat`, `/kvartiry/prodam`). Для других
сегментов нужны отдельные scrape-сессии:

- **Земля / участки** — `/kupit-uchastok/` (Yandex), `?offer_type=land`
(CIAN), `/zemelnye_uchastki/` (Avito).
- **Частные дома / таунхаусы** — `/kupit-dom/` etc.
- **Коммерческая недвижимость** — отдельный раздел.
- **Аренда** — `?deal_type=rent`.

В рамках kadastra-проекта приоритет — квартиры (это и есть основной  
объект кадастра, по которому строится H3-сетка цен). Земля/здания  
целиком — у нас уже есть как реальный кадастр из NSPD  
(91 864 здания + 199 819 участков, см. [info/nspd-api.md](nspd-api.md));  
листинги тут не заменяют, а **дополняют** как рыночный сигнал по  
квартирам внутри этих зданий.




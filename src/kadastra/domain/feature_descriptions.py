"""Human-readable descriptions for hex/object feature columns.

Single source of truth for the per-feature tooltips shown in the map UI.
Lives in `domain/` — feature names belong to the domain, and the JS layer
just renders whatever the API hands it.

Two-tier lookup:
- explicit dict for base/non-obvious features (counts, medians, age, …);
- regex patterns covering the regular families (dist_to_X_m,
  X_share_500m, X_within_500m, count_X_500m, dominant_X) so a newly
  added POI doesn't need a manual entry to get a usable hint.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# Tokens the patterns surface as POI names. Keeping this map small
# and obvious — anything outside the dict falls back to a humanized
# version of the raw token (underscores → spaces).
_POI_RU: dict[str, str] = {
    "water": "водоём",
    "park": "парк",
    "forest": "лес",
    "industrial": "промзона",
    "cemetery": "кладбище",
    "landfill": "свалка",
    "powerline": "ЛЭП",
    "railway": "ж/д",
    "railway_station": "ж/д станция",
    "school": "школа",
    "kindergarten": "детский сад",
    "clinic": "поликлиника",
    "hospital": "больница",
    "pharmacy": "аптека",
    "supermarket": "супермаркет",
    "cafe": "кафе",
    "restaurant": "ресторан",
    "bus_stop": "автобусная остановка",
    "tram_stop": "трамвайная остановка",
    "stations": "станции метро",
    "entrances": "входы в метро",
    "apartments": "жилые дома (МКД)",
    "houses": "ИЖС",
    "commercial": "коммерческие объекты",
}
# Distance-to interpretations: positive = closer is better,
# negative = closer hurts. Empty string for neutral / unknown so the
# template line just gets dropped.
_POSITIVE_DISTANCE = frozenset(
    {
        "park",
        "forest",
        "water",
        "school",
        "kindergarten",
        "clinic",
        "hospital",
        "pharmacy",
        "supermarket",
        "cafe",
        "restaurant",
        "bus_stop",
        "tram_stop",
        "railway_station",
    }
)
_NEGATIVE_DISTANCE = frozenset({"industrial", "cemetery", "landfill", "powerline", "railway"})


def _humanize_poi(token: str) -> str:
    return _POI_RU.get(token, token.replace("_", " "))


def _interpret_distance(token: str) -> str:
    if token in _POSITIVE_DISTANCE:
        return " Чем меньше расстояние — тем привлекательнее локация (положительный фактор цены)."
    if token in _NEGATIVE_DISTANCE:
        return (
            " Чем меньше расстояние — тем сильнее негативный фактор (промзона / шум / экология): обычно снижает цену."
        )
    return ""


def _agg_suffix(mean_prefix: str | None) -> str:
    return " Усреднено по объектам внутри гекса." if mean_prefix else ""


_EXPLICIT: dict[str, str] = {
    "count": (
        "Сколько объектов попало в этот гекс. Косвенный индикатор плотности"
        " застройки выбранного класса. В пустых гексах остальные показатели"
        " смотреть бессмысленно."
    ),
    "median_target_rub_per_m2": (
        "Медианная цена ₽/м² (из ЕГРН) среди объектов гекса. Главный показатель"
        " «средней стоимости в районе». Медиана, а не среднее, чтобы один"
        " дорогой объект не искажал картину."
    ),
    "median_pred_oof_rub_per_m2": (
        "Медианный прогноз выбранной модели по объектам гекса. Сравните с"
        " median_target_rub_per_m2, чтобы увидеть, где модель в среднем"
        " завышает или занижает цену."
    ),
    "median_residual_rub_per_m2": (
        "Медианный остаток (прогноз − факт) по гексу. Положительное значение —"
        " модель в среднем переоценивает район, отрицательное — недооценивает."
        " Чисто диагностический слой: ноль здесь — идеал."
    ),
    "mean_levels": (
        "Средняя этажность зданий в гексе. Низкая (1–3) — частный сектор, высокая (15+) — современные жилые комплексы."
    ),
    "mean_flats": "Среднее число квартир на здание в гексе. Маркер крупности жилых комплексов.",
    "mean_area_m2": "Средняя площадь объекта (квартиры / дома / участка / помещения) в гексе.",
    "mean_year_built": (
        "Средний год постройки зданий в гексе. Помогает увидеть «возраст»"
        " района: довоенная застройка, послевоенная, новостройки."
    ),
    "mean_age_years": (
        "Средний возраст зданий в гексе (на текущий год). То же, что"
        " mean_year_built, но в количестве лет — иногда удобнее интерпретировать."
    ),
    "area_m2": "Площадь конкретного объекта, м².",
    "levels": "Число этажей в здании.",
    "flats": "Число квартир в доме (для apartment-объектов).",
    "year_built": "Год постройки здания.",
    "age_years": "Возраст здания на текущий год.",
    "mean_road_length_500m": (
        "Сумма длин всех дорог (любого класса) в радиусе 500 м от объекта,"
        " усреднённая по гексу. Прокси для транспортной связности района."
    ),
    "road_length_500m": (
        "Сумма длин всех дорог в радиусе 500 м от объекта. Чем выше — тем плотнее уличная сеть вокруг."
    ),
    "age_years_sq": (
        "Квадрат возраста здания (age_years²). Позволяет модели поймать"
        " нелинейную зависимость цены от возраста: эффект «старости» часто"
        " ускоряется со временем."
    ),
    # ADR-0018 — геометрия объекта.
    "polygon_area_m2": (
        "Площадь полигона объекта (здания/участка), м². Для участков (landplot)"
        " это главный ценообразующий фактор: чем больше площадь — тем дороже"
        " объект в абсолютных деньгах (но, как правило, дешевле за м²)."
    ),
    "polygon_perimeter_m": (
        "Периметр полигона объекта, м. Характеризует размер и «изрезанность»"
        " границы: чем больше периметр при той же площади — тем сложнее форма"
        " и тем длиннее фасад/межа."
    ),
    "polygon_compactness": (
        "Компактность формы: 4π·площадь / периметр². 1 — идеальный круг,"
        " близко к 0 — сильно вытянутая или изрезанная форма. Компактные участки"
        " обычно удобнее в использовании и ценятся выше."
    ),
    "polygon_convexity": (
        "Выпуклость полигона: площадь / площадь выпуклой оболочки. 1 — выпуклый,"
        " меньше 1 — форма с «вмятинами». Невыпуклая геометрия участка часто"
        " снижает его полезность."
    ),
    "bbox_aspect_ratio": (
        "Соотношение сторон описанного прямоугольника (длина/ширина). 1 —"
        " квадратное, заметно больше 1 — вытянутое здание или участок."
        " Узкие вытянутые участки обычно менее удобны."
    ),
    "polygon_orientation_deg": (
        "Ориентация главной оси полигона, в градусах (0–180). Характеризует"
        " «повёрнутость» здания/участка относительно сторон света."
    ),
    "polygon_n_vertices": (
        "Число вершин полигона. Простой прямоугольник — 4 вершины; чем больше — тем сложнее и изрезаннее форма объекта."
    ),
    # Категориальные — строительные и административные.
    "materials": (
        "Материал стен здания (кирпич, панель, монолит, дерево…). Категориальный:"
        " материал напрямую влияет на долговечность, теплоизоляцию и, как"
        " следствие, на рыночную стоимость."
    ),
    "era_category": (
        "Категория эпохи постройки (довоенная, советская, постсоветская, новая…)."
        " Категориальный: объединяет год постройки в укрупнённые периоды,"
        " у каждого из которых свой типовой уровень качества и цены."
    ),
    "mun_okrug_name": (
        "Название муниципального округа — административной единицы внутри города."
        " Категориальный: через него модель улавливает устойчивую разницу в ценах"
        " между районами и микрорайонами."
    ),
    "mun_okrug_oktmo": (
        "Код ОКТМО муниципального округа. Категориальный идентификатор той же"
        " территории, что и mun_okrug_name, но в машинном виде — устойчив"
        " к переименованиям и совпадениям названий."
    ),
    "settlement_name": (
        "Название населённого пункта (город/посёлок/деревня), где находится объект."
        " Категориальный: несёт «премию» или «дисконт» населённого пункта —"
        " областной центр обычно дороже пригорода."
    ),
    "intra_city_raion": (
        "Внутригородской район (например, Советский или Вахитовский в Казани)."
        " Категориальный — один из ключевых пространственных факторов цены:"
        " районы систематически различаются по стоимости."
    ),
    "oktmo_full": (
        "Полный код ОКТМО территории — иерархия «регион → город → район»."
        " Категориальный идентификатор административной привязки объекта."
    ),
    "okato": (
        "Код ОКАТО — классификатор административно-территориального деления."
        " Категориальный; исторически частично дублирует ОКТМО."
    ),
    "postal_index": (
        "Почтовый индекс. Категориальный — прокси локации и района, часто коррелирует с престижностью территории."
    ),
}


def _describe_dist_to(m: re.Match[str]) -> str:
    mean_prefix, token = m.group(1), m.group(2)
    head = f"Расстояние до ближайшего объекта «{_humanize_poi(token)}», в метрах."
    return head + _interpret_distance(token) + _agg_suffix(mean_prefix)


def _describe_dist_metro(m: re.Match[str]) -> str:
    base = (
        "Расстояние до ближайшей станции метро, в метрах. Один из самых"
        " сильных факторов цены в крупных городах: близость к метро поднимает"
        " стоимость."
    )
    return base + _agg_suffix(m.group(1))


def _describe_dist_entrance(m: re.Match[str]) -> str:
    base = (
        "Расстояние до ближайшего входа в метро, в метрах. Уточнение"
        " dist_metro_m: важно реальное пешее расстояние до входа, а не до"
        " центра станции."
    )
    return base + _agg_suffix(m.group(1))


def _describe_share(m: re.Match[str]) -> str:
    mean_prefix, token, radius = m.group(1), m.group(2), m.group(3)
    head = (
        f"Доля площади «{_humanize_poi(token)}» в круге радиуса {radius} м"
        " вокруг объекта (0 — нет совсем, 1 — круг полностью покрыт)."
    )
    return head + _agg_suffix(mean_prefix)


def _describe_within(m: re.Match[str]) -> str:
    mean_prefix, token, radius = m.group(1), m.group(2), m.group(3)
    head = f"Сколько объектов «{_humanize_poi(token)}» попадает в круг радиуса {radius} м вокруг объекта."
    return head + _agg_suffix(mean_prefix)


def _describe_count(m: re.Match[str]) -> str:
    mean_prefix, token, n, unit = m.group(1), m.group(2), m.group(3), m.group(4)
    head = f"Сколько объектов «{_humanize_poi(token)}» в радиусе {n} {unit} вокруг объекта."
    return head + _agg_suffix(mean_prefix)


def _describe_dominant(m: re.Match[str]) -> str:
    suffix = m.group(1).replace("_", " ")
    return (
        f"Доминирующее значение административного признака «{suffix}» в гексе."
        " Категориальный признак — раскраска по дискретным значениям, а не по"
        " числовой шкале."
    )


def _describe_relative(m: re.Match[str]) -> str:
    """ADR-0012 relative-to-parent feature: F__rel_p{R}_{stat}."""
    base, res, stat = m.group(1), m.group(2), m.group(3)
    stat_ru = {
        "diff_med": "отклонение от медианы родительского гекса (в единицах признака)",
        "ratio_med": "отношение к медиане родительского гекса (во сколько раз)",
        "z_iqr": "робастный z-score: на сколько межквартильных размахов выше/ниже медианы родительского гекса",
    }[stat]
    return (
        f"Признак «{base.replace('_', ' ')}» относительно родительского гекса H3"
        f" разрешения {res}: {stat_ru}. Ловит «дороже/больше, чем соседи», а не"
        " абсолютный уровень признака."
    )


def _describe_count_p(m: re.Match[str]) -> str:
    res = m.group(1)
    return (
        f"Сколько объектов этого класса попадает в родительский гекс H3"
        f" разрешения {res}. Укрупнённый счётчик плотности — «сколько всего"
        " вокруг», а не «насколько близко»."
    )


_PATTERNS: list[tuple[re.Pattern[str], Callable[[re.Match[str]], str]]] = [
    (re.compile(r"^(.+)__rel_p(\d+)_(diff_med|ratio_med|z_iqr)$"), _describe_relative),
    (re.compile(r"^count_p(\d+)$"), _describe_count_p),
    (re.compile(r"^(mean_)?dist_metro_m$"), _describe_dist_metro),
    (re.compile(r"^(mean_)?dist_entrance_m$"), _describe_dist_entrance),
    (re.compile(r"^(mean_)?dist_to_(.+?)_m$"), _describe_dist_to),
    (re.compile(r"^(mean_)?(.+?)_share_(\d+)m$"), _describe_share),
    (re.compile(r"^(mean_)?(.+?)_within_(\d+)m$"), _describe_within),
    (re.compile(r"^(mean_)?count_(.+?)_(\d+)(km|m)$"), _describe_count),
    (re.compile(r"^dominant_(.+)$"), _describe_dominant),
]


def describe_feature(name: str) -> str | None:
    """Return a human-readable description for a feature column name,
    or None if neither the explicit dict nor the patterns recognise it."""
    if name in _EXPLICIT:
        return _EXPLICIT[name]
    for pattern, render in _PATTERNS:
        m = pattern.match(name)
        if m is not None:
            return render(m)
    return None

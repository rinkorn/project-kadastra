# ADR-0016: Параллельный квартет моделей и измерение потерь при упрощении

**Статус:** Accepted
**Дата:** 2026-04-26
**Реализует:** [ADR-0010](0010-methodology-compliance-roadmap.md), пункт 5 дорожной карты трека 2 — Black/Grey/White Box квартет.
**Источник методологии:** [info/grid-rationale.md](../grid-rationale.md), §13.2.

## Контекст

К моменту блока 5 закрыты блоки 1–4: путевой граф ([ADR-0011](0011-graph-based-distance-features.md)), относительные ЦОФ через H3 parent-агрегаты ([ADR-0012](0012-relative-features-via-h3-parent-aggregations.md)), зональная плотность в 4 радиусах ([ADR-0013](0013-zonal-density-features-multi-radius.md)) + поли-площадь ([ADR-0014](0014-poly-area-buffer-features.md)), территориальные ЦОФ ([ADR-0015](0015-territorial-features-via-gar.md)). На каждой итерации мы обучали **одну модель** (CatBoost per-class) и смотрели её MAPE по spatial-CV.

Этого недостаточно по двум причинам, явно зафиксированным в [grid-rationale.md §13.2](../grid-rationale.md):

1. **Методические указания ГБУ требуют интерпретируемой формулы.** Сдаётся не CatBoost, а White Box (линейная/правил-ориентированная). На неё уходит налог. Чтобы понимать **сколько мы теряем при упрощении** до White Box, нужна параллельно сложная Black Box модель как верхняя планка.
2. **Часть введённых ЦОФ методологически нужна, но на CatBoost-метриках околонулевая.** Это явно отмечено в ADR-0012 (relative features), ADR-0014 (poly-area share), частично ADR-0015 (для house/commercial/landplot). Реальный выигрыш этих фичей **ожидается на White Box** — деревья и без того справляются с пространственным split'ом по lat/lon, а линейная регрессия не может рекомбинировать `lat/lon + parent_h3_p7` в «Советский район Казани» сама. Без White Box модели мы **не можем подтвердить или опровергнуть** обоснованность этих блоков.

Дополнительный мотив — **наивная нижняя планка**: насколько разрыв между Black Box и линейной регрессией без feature-engineering'а большой. Это даёт абсолютную шкалу для всех остальных моделей.

## Решение

**Обучать четыре модели параллельно на одних и тех же фолдах spatial-CV, тех же `(X, y)` per-class, и логировать сравнительную таблицу MAPE/MAE/RMSE + перцентильную асимметрию + ранговую корреляцию.**

### 4 модели

| Модель | Что | Реализация | Назначение |
| --- | --- | --- | --- |
| **Black Box** | Сложная неинтерпретируемая | CatBoost per-class (текущий) | Верхняя планка качества — она же оценка что «можно достичь» на текущих фичах |
| **White Box** | Интерпретируемая | EBM (`interpret-ml ExplainableBoostingRegressor`) per-class | То, что фактически сдаётся ГБУ. Аддитивная по shape-функциям (interpret-ml) — даёт интерпретируемый вклад каждой фичи и пар взаимодействий, при этом регуляризованная нелинейная |
| **Grey Box** | Аппроксиматор Black Box | Глубокая Decision Tree (`sklearn.tree.DecisionTreeRegressor`, `max_depth=8–12`), обучённая на `y_pred_oof` от Black Box | Измеряет: насколько простая модель может **аппроксимировать** Black Box, без потери на самой y. Fidelity = R² Grey vs Black на held-out fold |
| **Naive Linear** | Нижняя планка | `sklearn.LinearRegression` поверх `OneHotEncoder` для cat-фич + raw numeric (без feature engineering, без relative/zonal/poly-area/territorial-derivatives) | Абсолютная шкала. Если Naive ≈ Black Box, значит вся feature-engineering инфраструктура зря (сюрприз, но измерять надо) |

### Что измеряем

Для каждой модели на тех же 5 spatial-CV folds (parent_resolution=7, как у CatBoost):

1. **MAPE / MAE / RMSE** — стандартная регрессия на `y_true`.
2. **Spearman ρ** — ранговая корреляция predicted vs true. Для кадастра ранг важнее абсолютной точности (налог пропорционален стоимости; перепутанный ранг = перепутанные суммы налогов).
3. **Перцентильная асимметрия** — насколько модель «сжимает» хвосты:
   - Δ медиан p10/p25/p50/p75/p90 предсказаний vs истинных значений.
   - «Дешёвые становятся дороже / дорогие дешевле» из [grid-rationale.md §13.2](../grid-rationale.md): доля объектов с `y_pred > y_true` среди bottom-decile (по y_true) и `y_pred < y_true` среди top-decile.
4. **Fidelity Grey vs Black** — отдельная метрика для Grey Box: R² на `(black_y_pred_oof, grey_y_pred_oof)` на тех же fold'ах. Это не про точность относительно y, а про то насколько Grey ловит паттерн Black.
5. **Loss on simplification** — Δ MAPE между Black Box и White Box, между Black Box и Naive. Главная **отчётная** цифра: «насколько White Box хуже Black Box» = тот самый налог на интерпретируемость.

Артефакт: `quartet_metrics.json` рядом с CatBoost-`run_dir`'ом (тот же model_registry, отдельный run_name `quartet-object-{class}_<ts>`). Содержит per-fold + aggregate цифры по всем 4 моделям + per-model `oof_predictions.parquet` (для будущих сравнений и отчётов).

### Какие фичи

Black Box, White Box, Grey Box получают **тот же X**, что у CatBoost сейчас — ровно то, что отдаёт `select_object_feature_columns` + `build_object_feature_matrix`. Это закрывает sub-вопрос «помогли ли relative/zonal/poly-area/territorial фичи на White Box»: если EBM с этими фичами лучше EBM без них — фичи работают, как методология и обещала.

Naive Linear получает **редуцированный X**: только raw `lat`, `lon`, `area_m2`, `levels`, `flats`, `year_built` + raw cat (asset_class через one-hot, остальные cat-фичи **не передаются** — это и есть «без feature engineering»). Опционально (отдельным запуском): тот же reduced X + relative-features из ADR-0012, чтобы измерить вклад каждого блока изолированно. Не делаем в первой итерации.

### Per-fold синхронизация

Все четыре модели должны увидеть **одни и те же train/val индексы** на каждом fold'е. Это даёт честное per-fold сравнение «модели A на fold k vs модели B на fold k» (фолды spatial — на каждом fold'е разная распределение признаков, поэтому без синхронизации сравнение зашумлено).

Технически: единый вызов `spatial_kfold_split(...)` на верхнем уровне use case, list of `(train_idx, val_idx)` пробрасывается во все 4 trainer-функции.

### Grey Box на y_pred_oof Black Box

Grey Box — Decision Tree, обученная на **Black Box OOF predictions**, не на y_true. Это критично:

- На y_true Decision Tree обучилась бы на ту же задачу что Black Box → не аппроксимирует Black Box, а соревнуется с ним на тех же данных.
- На y_pred_oof Decision Tree пытается **повторить функцию Black Box**: `Tree(X) ≈ CatBoost(X)`. Это позволяет измерить: насколько Black Box объяснима простой деревовидной моделью, и какая часть его сложности — реально нужная (не аппроксимируется), а какая — оверфит.

Использование OOF, а не in-sample, нужно чтобы Grey Box не учила «утечку» CatBoost'а на train-rows.

## Рассмотренные альтернативы

| Вариант | Что | Почему отвергнут |
| --- | --- | --- |
| Только Black + White (без Grey, без Naive) | 2 модели | Не даёт ответа «что упрощается без потери в Grey Box» (методология явно требует) и нет нижней планки. |
| EBM-only (без White vs Black) | Одна интерпретируемая модель | Не измеряет налог на интерпретируемость → нельзя оценить готовность White Box к сдаче ГБУ. |
| Линейная + полиномиальные взаимодействия как White Box | Без EBM | Линейные с явными interactions — это уже не интерпретируемая «формула», и эффект interactions сложно объяснить в отчёте. EBM даёт shape-функции и pairwise — методологически чище. |
| Symbolic regression (`gplearn` / `pysr`) для White Box | Аналитическая формула | Дорого по времени (обучение часы на 200k объектов × 50+ фичей), сильно зависит от гиперпараметров, плохо обобщается на новый регион. Возможно отдельным ADR на следующем туре, не сейчас. |
| Neural net как Black Box вместо CatBoost | DNN | Black Box у нас уже есть (CatBoost). Менять верхнюю планку без явной причины ≠ цель блока 5. |
| Один common train/val split вместо spatial-CV для Naive | Random split | Утечка через пространственное соседство; делает Naive ложно лучше. Spatial-CV на тех же fold'ах что у Black Box — единственная честная альтернатива. |

## Структура решения

### Архитектура

```text
src/kadastra/
  ml/
    train.py                       # already: spatial CV + CatBoost
    train_ebm.py                   # new: EBM per fold + final
    train_grey_tree.py             # new: DT trained on Black OOF
    train_naive_linear.py          # new: LinReg + OneHot
    quartet_metrics.py             # new: spearman, percentile-asym, simplification loss
  ports/
    quartet_model.py               # Protocol: fit / predict / serialize
  adapters/
    catboost_quartet_model.py      # wrap existing CatBoostRegressor
    ebm_quartet_model.py           # ExplainableBoostingRegressor
    grey_tree_quartet_model.py     # DecisionTreeRegressor on black OOF
    naive_linear_quartet_model.py  # LinearRegression + OneHotEncoder
  usecases/
    train_quartet.py               # new: top-level use case
```

`QuartetModelPort` — общий Protocol с `fit(X, y, *, cat_features) -> None`, `predict(X) -> np.ndarray`, `serialize() -> bytes`. Каждая модель имеет свой адаптер. `TrainQuartet` use case: load object-gold per class → spatial_kfold_split (один раз) → для каждой из 4 моделей: per-fold fit + predict → собрать OOF + per-fold metrics → final fit на всём → log_run в model_registry с артефактами.

### Зависимости

- `interpret` (или `interpret-ml`) — для EBM. Тяжёлая, но самодостаточная. Будет в `pyproject.toml`.
- `scikit-learn` — уже есть, используется для `LinearRegression`, `DecisionTreeRegressor`, `OneHotEncoder`.

### Settings

```python
# Block 5 (ADR-0016)
quartet_enabled: bool = True
ebm_max_bins: int = 256
ebm_interactions: int = 10  # number of pair-wise interactions in EBM
grey_tree_max_depth: int = 10
naive_linear_only_raw_numerics: bool = True
```

### Pipeline

```text
gold per-class object table → TrainQuartet(class) → quartet-object-{class}_<ts>/
  ├ catboost_model.cbm
  ├ ebm_model.pkl
  ├ grey_tree.pkl
  ├ naive_linear.pkl
  ├ quartet_metrics.json
  └ {catboost,ebm,grey_tree,naive_linear}_oof_predictions.parquet
```

`quartet_metrics.json` schema:

```json
{
  "asset_class": "apartment",
  "n_samples": 65000,
  "n_splits": 5,
  "parent_resolution": 7,
  "models": {
    "catboost":     {"mean_mae": ..., "mean_rmse": ..., "mean_mape": ..., "mean_spearman": ...},
    "ebm":          {"mean_mae": ..., "mean_rmse": ..., "mean_mape": ..., "mean_spearman": ...},
    "grey_tree":    {"mean_mae": ..., "mean_rmse": ..., "mean_mape": ..., "mean_spearman": ..., "fidelity_r2_to_catboost": ...},
    "naive_linear": {"mean_mae": ..., "mean_rmse": ..., "mean_mape": ..., "mean_spearman": ...}
  },
  "loss_on_simplification": {
    "catboost_minus_ebm_mape_pp":    -1.23,
    "catboost_minus_naive_mape_pp":  -8.40
  },
  "percentile_asymmetry": {
    "catboost":     {"p10_pred_minus_true_rub_per_m2": ..., "frac_overpredicted_in_bottom_decile": ..., ...},
    "ebm":          {...},
    "naive_linear": {...}
  }
}
```

`*_oof_predictions.parquet` schema:

```text
object_id (str), lat (float64), lon (float64),
fold_id (int64), y_true (float64), y_pred_oof (float64)
```

— тот же layout что `oof_predictions.parquet` у CatBoost (ADR-0015 / inspector). Это позволяет инспектору в будущей итерации показывать предсказания всех 4 моделей рядом.

## Что **не** делаем (out-of-scope для блока 5)

- **Inference-серверный путь.** Сейчас в проде только CatBoost (через `InferObjectValuation`). White/Grey/Naive — пока только для отчёта, не для серва. После того как White Box по метрикам сравнится с Black — решим отдельным ADR, какую модель катить в инференс.
- **Гиперпараметрический подбор** (Optuna для EBM/Grey/Naive). Стартовые значения из дефолтов interpret-ml + `max_depth=10` для Grey + `LinearRegression` без regularization. Если White Box сильно проседает по MAPE — отдельным ADR подбираем гиперы.
- **Темпоральные срезы** ([ADR-0010](0010-methodology-compliance-roadmap.md), пункт 6). Блок 5 на текущем target (interim cost_index ЕГРН), темпоральная часть — после трека 1.
- **Symbolic regression White Box.** Отдельным ADR на следующем туре.
- **Per-fold модели в model registry.** Сохраняем только final fit; per-fold нужны только для metric'ов и пишутся в OOF parquet.
- **Уравнения регрессии в текстовом виде.** Полезно для отчёта, но это отдельная задача поверх EBM (вытащить shape-функции в форму «коэффициент × фича»). Не в этой итерации.

## Открытые вопросы

- **Как именно мерить «дешёвые становятся дороже».** Перцентильная асимметрия это направление, но конкретная агрегация (Δ медиан / KS / EMD по перцентилям) не зафиксирована. На первой итерации беру Δ медиан p10/25/50/75/90 + frac_overpredicted_in_bottom_decile. Если в отчёте окажется неинформативно — пересмотрим.
- **MAPE на landplot (~250 %)** структурно сломан (мелкий знаменатель ₽/м²). Для landplot отчётная цифра — MAE и Spearman ρ. Стоит ли вообще считать MAPE — отдельный вопрос. Пока считаем, но в отчёте помечаем «не репрезентативна».
- **EBM на 50+ фичах + 16 категорий + 5 fold × 4 классов.** Ожидаемо тяжелее CatBoost (минуты на класс). Пиковая память тоже выше — interpret-ml не такой эффективный как CatBoost. Бюджет — отдельный замер.
- **Как распределить compute между 4 моделями.** Sequential — самое простое, ожидаемо ~10-15 минут per-class. Параллельно через `concurrent.futures` — есть смысл если bottleneck CPU (а не GIL): CatBoost + EBM в потоках не дают линейного speedup. Решение — оставить sequential на первой итерации, оптимизация при необходимости.
- **Что с `mun_source` и другими provenance/identity колонками** при naive linear с raw fields. Уже отфильтровано в `select_object_feature_columns` (ADR-0015), но Naive Linear работает с **редуцированным** X — `select_naive_columns` будет отдельной функцией. Уточняется при реализации.

"""MVP-моделирование цен листингов: CatBoost-регрессия log(price_per_sqm)
на трёх источниках (Yandex / CIAN / Avito) для сравнения, какой источник
даёт **лучшую предсказательную модель** — это главный критерий выбора
партнёра для production-кадастровой оценки.

Считаем три набора моделей:

1. **Per-source-city** (до 6 моделей) — baseline на «общих» фичах, какие
   есть у всех трёх (area, floor, floors_count, floor_share, rooms,
   city). Это нижняя планка: если у источника даже на этом наборе
   модель плохая — данные шумные.

2. **CIAN-rich** — CIAN + специфичные фичи (build_year, material_type,
   lat, lon). Это верхняя планка: насколько богатая мета помогает.

3. **Cross-source** — обучить на одном источнике, протестировать на
   другом (тот же город). Нужно для понимания, насколько источники
   согласованы и можно ли смешивать.

Метрики: R², MAPE, MAE на 5-fold CV (для мелких выборок) либо 80/20
hold-out (для крупных).

Артефакты:
* `data/models/listings-mvp/{run_name}/model.cbm` — CatBoost модели.
* `data/models/listings-mvp/_metrics.json` — все метрики, для сравнения.
* `data/models/listings-mvp/_report.md` — markdown-отчёт.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, train_test_split

INP = Path("data/silver/listings-mvp/all.parquet")
OUT = Path("data/models/listings-mvp")
OUT.mkdir(parents=True, exist_ok=True)

COMMON_FEATURES = ["total_area_m2", "floor", "floors_count", "floor_share", "rooms", "city"]
COMMON_CAT = ["city"]

CIAN_RICH_FEATURES = [*COMMON_FEATURES, "build_year", "material_type", "lat", "lon"]
CIAN_RICH_CAT = [*COMMON_CAT, "material_type"]


def clean(df: pl.DataFrame, features: list[str], target: str = "price_per_sqm_rub") -> pl.DataFrame:
    """Отфильтровать невалидные строки + цены за м² в разумном диапазоне."""
    return df.filter(
        pl.col(target).is_not_null()
        & (pl.col(target) > 30_000)
        & (pl.col(target) < 1_000_000)
        & pl.col("total_area_m2").is_not_null()
        & (pl.col("total_area_m2") > 15)
        & (pl.col("total_area_m2") < 500)
    ).with_columns(
        [
            pl.col(target).log().alias("log_target"),
        ]
    )


def cat_indices(features: list[str], cat: list[str]) -> list[int]:
    return [features.index(c) for c in cat if c in features]


def _to_pandas_with_cat_strings(df: pl.DataFrame, features: list[str], cat: list[str]):
    pdf = df.select(features).to_pandas()
    for c in cat:
        if c in pdf.columns:
            pdf[c] = pdf[c].fillna("__missing__").astype(str)
    return pdf


def fit_eval(
    X_train, y_train, X_test, y_test, cat_idx: list[int], iterations: int = 600
) -> tuple[CatBoostRegressor, dict[str, float]]:
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        cat_features=cat_idx,
    )
    model.fit(Pool(X_train, y_train, cat_features=cat_idx))
    preds_log = model.predict(X_test)
    preds = np.exp(preds_log)
    actual = np.exp(y_test)
    metrics = {
        "r2_log": float(r2_score(y_test, preds_log)),
        "r2": float(r2_score(actual, preds)),
        "mape": float(mean_absolute_percentage_error(actual, preds)),
        "mae": float(mean_absolute_error(actual, preds)),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
    return model, metrics


def cv_eval(df: pl.DataFrame, features: list[str], cat: list[str], k: int = 5, iterations: int = 600) -> dict[str, Any]:
    pdf = _to_pandas_with_cat_strings(df, features, cat)
    y = df.select("log_target").to_numpy().ravel()
    cat_idx = cat_indices(features, cat)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics: list[dict[str, float]] = []
    for fold_i, (tr, te) in enumerate(kf.split(pdf)):
        _, m = fit_eval(pdf.iloc[tr], y[tr], pdf.iloc[te], y[te], cat_idx, iterations)
        m["fold"] = fold_i
        fold_metrics.append(m)
    agg = {
        "r2_mean": float(np.mean([m["r2"] for m in fold_metrics])),
        "r2_std": float(np.std([m["r2"] for m in fold_metrics])),
        "mape_mean": float(np.mean([m["mape"] for m in fold_metrics])),
        "mape_std": float(np.std([m["mape"] for m in fold_metrics])),
        "mae_mean": float(np.mean([m["mae"] for m in fold_metrics])),
        "n_total": len(y),
        "folds": fold_metrics,
    }
    return agg


def hold_out_train_full(
    df: pl.DataFrame, features: list[str], cat: list[str], run_name: str, iterations: int = 600
) -> tuple[CatBoostRegressor, dict[str, Any]]:
    pdf = _to_pandas_with_cat_strings(df, features, cat)
    y = df.select("log_target").to_numpy().ravel()
    cat_idx = cat_indices(features, cat)
    X_tr, X_te, y_tr, y_te = train_test_split(pdf, y, test_size=0.2, random_state=42)
    model, metrics = fit_eval(X_tr, y_tr, X_te, y_te, cat_idx, iterations)
    run_dir = OUT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(run_dir / "model.cbm"))
    feat_imp = dict(zip(features, model.feature_importances_, strict=False))
    metrics["feature_importance"] = {k: float(v) for k, v in feat_imp.items()}
    return model, metrics


def cross_source_eval(
    all_df: pl.DataFrame, features: list[str], cat: list[str], iterations: int = 600
) -> list[dict[str, Any]]:
    """Train on src_a, test on src_b, same city."""
    out: list[dict[str, Any]] = []
    sources = ["yandex_realty", "cian", "avito"]
    cities = all_df.select("city").unique().to_series().to_list()
    for src_train in sources:
        for src_test in sources:
            if src_train == src_test:
                continue
            for city in cities:
                df_tr = all_df.filter((pl.col("source") == src_train) & (pl.col("city") == city))
                df_te = all_df.filter((pl.col("source") == src_test) & (pl.col("city") == city))
                df_tr = clean(df_tr, features)
                df_te = clean(df_te, features)
                if df_tr.shape[0] < 30 or df_te.shape[0] < 30:
                    continue
                pdf_tr = _to_pandas_with_cat_strings(df_tr, features, cat)
                pdf_te = _to_pandas_with_cat_strings(df_te, features, cat)
                y_tr = df_tr.select("log_target").to_numpy().ravel()
                y_te = df_te.select("log_target").to_numpy().ravel()
                cat_idx = cat_indices(features, cat)
                _, m = fit_eval(pdf_tr, y_tr, pdf_te, y_te, cat_idx, iterations)
                m.update(
                    {
                        "src_train": src_train,
                        "src_test": src_test,
                        "city": city,
                    }
                )
                out.append(m)
    return out


def main() -> int:
    if not INP.exists():
        sys.exit(f"{INP} not found — run extract_listings_mvp.py first")
    all_df = pl.read_parquet(INP)
    print(f"loaded {INP}: {all_df.shape}", flush=True)

    metrics: dict[str, Any] = {}

    # === 1. Per-source-city baseline (общие фичи) ===
    print("\n=== Per-source-city baseline (общие фичи) ===", flush=True)
    per_sc: list[dict[str, Any]] = []
    for src in ("yandex_realty", "cian", "avito"):
        for city in all_df.select("city").unique().to_series().to_list():
            sub = all_df.filter((pl.col("source") == src) & (pl.col("city") == city))
            sub = clean(sub, COMMON_FEATURES)
            if sub.shape[0] < 50:
                print(f"  SKIP {src}/{city}: too few rows ({sub.shape[0]})", flush=True)
                continue
            method = "cv" if sub.shape[0] < 500 else "holdout"
            if method == "cv":
                m = cv_eval(sub, COMMON_FEATURES, COMMON_CAT)
            else:
                _, m = hold_out_train_full(sub, COMMON_FEATURES, COMMON_CAT, run_name=f"per_sc_{src}_{city}")
            m.update({"source": src, "city": city, "method": method, "feature_set": "common"})
            per_sc.append(m)
            r2 = m.get("r2_mean", m.get("r2"))
            mape = m.get("mape_mean", m.get("mape"))
            n = m["n_total"] if "n_total" in m else m["n_train"] + m["n_test"]
            print(
                f"  {src:14}/{city:8} ({method:8} n={n:>4}): R²={r2:.3f}  MAPE={mape:.3f}",
                flush=True,
            )
    metrics["per_source_city"] = per_sc

    # === 2. CIAN-rich (доп. фичи build_year/material_type/lat/lon) ===
    print("\n=== CIAN с rich-фичами ===", flush=True)
    cian_rich: list[dict[str, Any]] = []
    for city in all_df.select("city").unique().to_series().to_list():
        sub = all_df.filter((pl.col("source") == "cian") & (pl.col("city") == city))
        sub = clean(sub, CIAN_RICH_FEATURES)
        if sub.shape[0] < 50:
            continue
        method = "cv" if sub.shape[0] < 500 else "holdout"
        if method == "cv":
            m = cv_eval(sub, CIAN_RICH_FEATURES, CIAN_RICH_CAT)
        else:
            _, m = hold_out_train_full(sub, CIAN_RICH_FEATURES, CIAN_RICH_CAT, run_name=f"cian_rich_{city}")
        m.update({"source": "cian", "city": city, "method": method, "feature_set": "cian_rich"})
        cian_rich.append(m)
        r2 = m.get("r2_mean", m.get("r2"))
        mape = m.get("mape_mean", m.get("mape"))
        n = m.get("n_total", m.get("n_train", 0) + m.get("n_test", 0))
        print(f"  cian-rich/{city:8} ({method:8} n={n:>4}): R²={r2:.3f}  MAPE={mape:.3f}", flush=True)
    metrics["cian_rich"] = cian_rich

    # === 3. Cross-source ===
    print("\n=== Cross-source (train on A, test on B, same city) ===", flush=True)
    cs = cross_source_eval(all_df, COMMON_FEATURES, COMMON_CAT)
    for r in cs:
        print(
            f"  {r['src_train']:14} → {r['src_test']:14} / {r['city']:8} "
            f"(n_train={r['n_train']:>4} n_test={r['n_test']:>4}): "
            f"R²={r['r2']:.3f}  MAPE={r['mape']:.3f}",
            flush=True,
        )
    metrics["cross_source"] = cs

    (OUT / "_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=> metrics: {OUT / '_metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

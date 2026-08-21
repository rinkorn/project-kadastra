"""Build the gold cell-valuation layer (ADR-0029).

Scores every Слой 1 anchor cell (res 10) with the class's EBM quartet
model: reference-object price (product 1) and location_score (product
2, EBM additive decomposition over locational terms). Writes
``data/gold/cell_valuation/region=…/asset_class=…/data.parquet`` plus a
``_meta.json`` sidecar per class (reference-object composition, model
run id, sanity metrics, representativeness linkage).

Also verifies the scoring matrix layout against the training run's
``params.json`` — a schema drift between gold and the stored run would
silently permute columns.

Запуск:
    uv run python scripts/build_cell_valuation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from kadastra.adapters.parquet_valuation_object_store import ParquetValuationObjectStore
from kadastra.composition_root import Container
from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass
from kadastra.ml.object_feature_columns import select_object_feature_columns

_TARGET_COLUMN = "synthetic_target_rub_per_m2"


def _latest_run(models_path: Path, asset_class: AssetClass) -> str:
    prefix = f"quartet-object-{asset_class.value}_"
    runs = sorted(d.name for d in models_path.iterdir() if d.is_dir() and d.name.startswith(prefix))
    for run in reversed(runs):
        if (models_path / run / "ebm_model.pkl").is_file():
            return run
    raise FileNotFoundError(f"no quartet run with ebm_model.pkl for {asset_class.value}")


def _check_feature_layout(models_path: Path, run: str, gold: pl.DataFrame, asset_class: AssetClass) -> None:
    params = json.loads((models_path / run / "params.json").read_text())
    trained = params["feature_columns_full"]
    numeric, categorical = select_object_feature_columns(gold.drop_nulls(subset=[_TARGET_COLUMN]))
    current = numeric + categorical
    if trained != current:
        raise RuntimeError(
            f"feature layout mismatch for {asset_class.value}: "
            f"trained {len(trained)} cols vs current {len(current)}; "
            f"first diff: {next((a, b) for a, b in zip(trained, current, strict=False) if a != b)}"
        )


def _repr_summary(representativeness_path: Path, region_code: str, resolution: int, asset_class: AssetClass) -> dict:
    path = representativeness_path / f"region={region_code}" / f"resolution={resolution}" / "data.parquet"
    if not path.is_file():
        return {}
    rep = pl.read_parquet(path).filter(pl.col("segment") == asset_class.value)
    if rep.is_empty():
        return {}
    verdicts = rep.group_by("verdict").len().to_dicts()
    return {
        "representativeness_verdicts": {row["verdict"]: row["len"] for row in verdicts},
        "representativeness_coverage": float(rep["coverage"].mean()),
    }


def main() -> int:
    settings = Settings()
    container = Container(settings)

    store = ParquetValuationObjectStore(settings.valuation_object_store_path)
    classes = list(AssetClass)
    runs = {ac: _latest_run(settings.model_registry_path, ac) for ac in classes}
    for ac, run in runs.items():
        _check_feature_layout(settings.model_registry_path, run, store.load(settings.region_code, ac), ac)
        print(f"=> {ac.value}: model run {run} — feature layout OK", flush=True)

    usecase = container.build_cell_valuation()
    print(f"Building cell valuation: region={settings.region_code} classes={[c.value for c in classes]}", flush=True)
    results = usecase.execute(settings.region_code, classes)

    out_base = settings.cell_valuation_store_path / f"region={settings.region_code}"
    for ac, res in results.items():
        partition = out_base / f"asset_class={ac.value}"
        out = pl.read_parquet(partition / "data.parquet")
        meta = {
            "asset_class": ac.value,
            "model_run": runs[ac],
            "n_cells": res.n_cells,
            "n_rows": out.height,
            "reference_objects": [
                {"variant": ref.variant, "attributes": ref.attributes} for ref in res.reference_objects
            ],
            "sanity": {
                "cbd_correlation_location_score": res.cbd_correlation,
                "covered_share": res.covered_share,
                "reference_rub_per_m2_quantiles": {
                    q: out["reference_rub_per_m2"].quantile(q) for q in (0.05, 0.5, 0.95)
                },
            },
            **_repr_summary(
                settings.representativeness_path,
                settings.region_code,
                settings.cell_tsorf_resolution,
                ac,
            ),
        }
        (partition / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print(
            f"=> {ac.value}: cells={res.n_cells:,} covered={res.covered_share:.1%} "
            f"corr(location, cbd_dist)={res.cbd_correlation}",
            flush=True,
        )
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

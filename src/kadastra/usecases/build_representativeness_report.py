"""Build the representativeness report (эпик 001, этап 5).

Compares the distribution of every Слой 1 ЦОФ over the full territory
(the res-10 grid, population) against the distribution over the
training sample (valuation objects). Methodology §2 п.5
(grid-rationale.md): a sample is representative when the ЦОФ
distributions on the sample and on the general population coincide —
not just min/max ranges.

The sample side is built by joining objects to the grid via
``add_h3_index`` and keeping duplicate cells — a cell with ten objects
weighs ten times, which is exactly the sampling distribution. Raw grid
values are compared (not the object's overlap-weighted columns), so
both sides are the same physical quantity.

Output layout:

``{base_path}/region={REGION}/resolution={R}/data.parquet``
``{base_path}/region={REGION}/resolution={R}/report.md``

Long-format parquet: one row per (feature_set, feature, segment),
segments are ``overall`` plus one per asset class.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from kadastra.adapters.parquet_feature_store import ParquetFeatureStore
from kadastra.domain.asset_class import AssetClass
from kadastra.etl.h3_coverage import add_h3_index
from kadastra.etl.representativeness import compare_distributions
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort
from kadastra.usecases.get_cell_tsorf import CELL_TSORF_FEATURE_SETS

_BOOKKEEPING = {"h3_index", "resolution"}


class BuildRepresentativenessReport:
    def __init__(
        self,
        feature_store: ParquetFeatureStore,
        object_reader: ValuationObjectReaderPort,
        output_base_path: Path,
        resolution: int,
        feature_sets: tuple[str, ...] = CELL_TSORF_FEATURE_SETS,
    ) -> None:
        self._feature_store = feature_store
        self._object_reader = object_reader
        self._output_base_path = output_base_path
        self._resolution = resolution
        self._feature_sets = feature_sets

    def execute(self, region_code: str, asset_classes: list[AssetClass]) -> pl.DataFrame:
        grid, feature_sets = self._load_grid(region_code)

        segments: dict[str, pl.DataFrame] = {}
        per_class_frames: list[pl.DataFrame] = []
        for ac in asset_classes:
            objects = self._object_reader.load(region_code, ac)
            if objects.is_empty():
                continue
            cells = add_h3_index(objects.select(["lat", "lon"]), resolution=self._resolution)
            joined = cells.join(grid, on="h3_index", how="inner")
            segments[ac.value] = joined
            per_class_frames.append(joined)

        frames = [compare_distributions(grid, sample, feature_sets, segment) for segment, sample in segments.items()]
        if len(per_class_frames) > 1:
            frames.append(
                compare_distributions(
                    grid, pl.concat(per_class_frames, how="vertical_relaxed"), feature_sets, "overall"
                )
            )
        report = pl.concat(frames) if frames else pl.DataFrame()

        out_dir = self._output_base_path / f"region={region_code}" / f"resolution={self._resolution}"
        out_dir.mkdir(parents=True, exist_ok=True)
        report.write_parquet(out_dir / "data.parquet")
        (out_dir / "report.md").write_text(self._render_markdown(report, region_code), encoding="utf-8")
        return report

    def _load_grid(self, region_code: str) -> tuple[pl.DataFrame, dict[str, str]]:
        """Join all Слой 1 feature sets on ``h3_index``; map column → set."""
        grid: pl.DataFrame | None = None
        feature_sets: dict[str, str] = {}
        for feature_set in self._feature_sets:
            try:
                df = self._feature_store.load(region_code, self._resolution, feature_set)
            except FileNotFoundError:
                print(f"  ! feature_set={feature_set} missing, skipped")
                continue
            columns = [c for c in df.columns if c not in _BOOKKEEPING]
            feature_sets.update(dict.fromkeys(columns, feature_set))
            slim = df.select(["h3_index", *columns])
            grid = slim if grid is None else grid.join(slim, on="h3_index", how="inner")
        if grid is None:
            raise FileNotFoundError(f"no Слой 1 feature sets found for region={region_code}")
        return grid, feature_sets

    @staticmethod
    def _render_markdown(report: pl.DataFrame, region_code: str) -> str:
        if report.is_empty():
            return "# Репрезентативность выборки\n\nНет данных.\n"
        lines = [
            "# Репрезентативность выборки (эпик 001, этап 5)",
            "",
            f"Регион: `{region_code}`. PSI-пороги: `<0.1` ok, `0.1–0.25` moderate, `>=0.25` shift.",
            "KS p-value при N~1e5 номинален — смотреть на `ks_stat` и `psi`.",
            "",
            "## Покрытие сетки выборкой",
            "",
            "| Сегмент | Объектов | Ячеек покрыто | Coverage |",
            "| --- | ---: | ---: | ---: |",
        ]
        coverage = report.select(["segment", "n_sample", "n_sample_cells", "coverage"]).unique(
            subset=["segment"], keep="first"
        )
        for row in coverage.sort("segment").iter_rows(named=True):
            lines.append(f"| {row['segment']} | {row['n_sample']} | {row['n_sample_cells']} | {row['coverage']:.1%} |")
        lines += [
            "",
            "## Вердикты PSI по сегментам",
            "",
            "| Сегмент | ok | moderate | shift |",
            "| --- | ---: | ---: | ---: |",
        ]
        for segment in sorted(report["segment"].unique().to_list()):
            sub = report.filter(pl.col("segment") == segment)
            counts = {v: sub.filter(pl.col("verdict") == v).height for v in ("ok", "moderate", "shift")}
            lines.append(f"| {segment} | {counts['ok']} | {counts['moderate']} | {counts['shift']} |")
        lines += [
            "",
            "## Топ-20 сдвинутых ЦОФ (overall, по PSI)",
            "",
            "| Feature set | Feature | PSI | KS-stat | Вердикт |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        top = report.filter(pl.col("segment") == "overall").sort("psi", descending=True, nulls_last=True).head(20)
        for row in top.iter_rows(named=True):
            psi = f"{row['psi']:.3f}" if row["psi"] is not None else "—"
            ks = f"{row['ks_stat']:.3f}" if row["ks_stat"] is not None else "—"
            lines.append(f"| {row['feature_set']} | {row['feature']} | {psi} | {ks} | {row['verdict']} |")
        lines.append("")
        return "\n".join(lines)

"""Use case: silver NSPD frames → valuation_object_store partitions.

Per ADR-0009, this is the bridge between the new NSPD-driven silver
layer and the existing per-class object pipeline (features → train →
infer). The output schema is the same as ``ParquetValuationObjectStore``
already accepts (``object_id, asset_class, lat, lon, levels, flats``)
plus a few NSPD-only extras (``area_m2``, ``year_built``, ``materials``,
``cost_value_rub``) and the populated target column
``synthetic_target_rub_per_m2``. The legacy column name is preserved so
that ``BuildObjectFeatures`` and ``TrainObjectValuationModel`` keep
working without changes; the value is the real ``cost_index`` from
EГРН, not a synthetic proxy.

Rows with null ``asset_class`` (e.g. NSPD purpose «Гараж») or null
``cost_index_rub_per_m2`` are dropped — they can't be used for either
training or inference.
"""

from __future__ import annotations

import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.ports.nspd_silver_store import NspdSilverStorePort
from kadastra.ports.valuation_object_store import ValuationObjectStorePort

_OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "object_id": pl.Utf8,
    "asset_class": pl.Utf8,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "levels": pl.Int64,
    "flats": pl.Int64,
    "area_m2": pl.Float64,
    "year_built": pl.Int64,
    "materials": pl.Utf8,
    "cost_value_rub": pl.Float64,
    "synthetic_target_rub_per_m2": pl.Float64,
}


class AssembleNspdValuationObjects:
    def __init__(
        self,
        silver_store: NspdSilverStorePort,
        valuation_object_store: ValuationObjectStorePort,
    ) -> None:
        self._silver_store = silver_store
        self._valuation_object_store = valuation_object_store

    def execute(
        self, region_code: str, *, asset_classes: list[AssetClass]
    ) -> None:
        raise NotImplementedError

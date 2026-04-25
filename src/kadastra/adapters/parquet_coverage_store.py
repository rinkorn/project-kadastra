from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import polars as pl


class ParquetCoverageStore:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def save(self, region_code: str, cells: Iterable[tuple[str, int]]) -> None:
        groups: dict[int, list[str]] = defaultdict(list)
        for h3_index, resolution in cells:
            groups[resolution].append(h3_index)

        for resolution, indices in groups.items():
            partition = self._base_path / f"region={region_code}" / f"resolution={resolution}"
            partition.mkdir(parents=True, exist_ok=True)
            df = pl.DataFrame({"h3_index": sorted(indices)})
            df.write_parquet(partition / "data.parquet")

    def load(self, region_code: str, resolution: int) -> pl.DataFrame:
        path = self._base_path / f"region={region_code}" / f"resolution={resolution}" / "data.parquet"
        return pl.read_parquet(path).with_columns(pl.lit(resolution, dtype=pl.Int32).alias("resolution"))

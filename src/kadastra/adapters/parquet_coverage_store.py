from collections.abc import Iterable
from pathlib import Path


class ParquetCoverageStore:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def save(self, region_code: str, cells: Iterable[tuple[str, int]]) -> None:
        raise NotImplementedError

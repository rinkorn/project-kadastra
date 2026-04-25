from collections.abc import Iterable
from typing import Protocol


class CoverageStorePort(Protocol):
    def save(self, region_code: str, cells: Iterable[tuple[str, int]]) -> None: ...

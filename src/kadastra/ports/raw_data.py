from typing import Protocol


class RawDataPort(Protocol):
    def read_bytes(self, key: str) -> bytes: ...

    def list_keys(self, prefix: str) -> list[str]: ...

from dataclasses import dataclass

from kadastra.domain.asset_class import AssetClass


@dataclass(frozen=True, slots=True)
class ValuationObject:
    object_id: str
    asset_class: AssetClass
    lat: float
    lon: float
    levels: int | None = None
    flats: int | None = None

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must be a non-empty string")
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError(f"lat out of range [-90, 90]: {self.lat}")
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError(f"lon out of range [-180, 180]: {self.lon}")

from kadastra.domain.asset_class import AssetClass

_APARTMENT_TAGS = frozenset({"apartments"})
_HOUSE_TAGS = frozenset({"house", "detached", "semidetached_house", "terrace", "bungalow"})
_COMMERCIAL_TAGS = frozenset(
    {
        "retail",
        "commercial",
        "supermarket",
        "kiosk",
        "shop",
        "office",
        "industrial",
        "warehouse",
    }
)


def classify_asset_class(tag: str | None) -> AssetClass | None:
    if not tag:
        return None
    normalized = tag.strip().lower()
    if normalized in _APARTMENT_TAGS:
        return AssetClass.APARTMENT
    if normalized in _HOUSE_TAGS:
        return AssetClass.HOUSE
    if normalized in _COMMERCIAL_TAGS:
        return AssetClass.COMMERCIAL
    return None

from kadastra.domain.asset_class import AssetClass

_APARTMENT_PURPOSES = frozenset({"многоквартирный дом"})
_HOUSE_PURPOSES = frozenset({"жилой дом", "жилое", "садовый дом"})
_COMMERCIAL_PURPOSES = frozenset({"нежилое"})


def classify_nspd_building_purpose(purpose: str | None) -> AssetClass | None:
    if not purpose:
        return None
    normalized = purpose.strip().lower()
    if normalized in _APARTMENT_PURPOSES:
        return AssetClass.APARTMENT
    if normalized in _HOUSE_PURPOSES:
        return AssetClass.HOUSE
    if normalized in _COMMERCIAL_PURPOSES:
        return AssetClass.COMMERCIAL
    return None

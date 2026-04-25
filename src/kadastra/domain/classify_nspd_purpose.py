from kadastra.domain.asset_class import AssetClass

_APARTMENT_PURPOSES = frozenset({"многоквартирный дом"})
_HOUSE_PURPOSES = frozenset({"жилой дом", "жилое", "садовый дом"})
_COMMERCIAL_PURPOSES = frozenset({"нежилое"})


def classify_nspd_building_purpose(purpose: str | None) -> AssetClass | None:
    raise NotImplementedError

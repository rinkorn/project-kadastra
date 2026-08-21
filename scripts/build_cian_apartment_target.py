"""Build the apartment market-target from CIAN listings (ADR-0031, вариант б).

Читает ``data/silver/listings-mvp/all.parquet`` (source=cian, city=Казань),
чистит (квантили ₽/м², sanity-правила, хвост страниц >54) и джойнит к
НСПД-объектам apartment из ``data/gold/valuation_objects``. Пишет

``data/silver/listings_target/region=…/asset_class=apartment/{matched,unmatched}.parquet``

Переобучение модели на этом target — отдельный этап, здесь только ETL.

Запуск:
    uv run python scripts/build_cian_apartment_target.py
"""

from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass
from kadastra.usecases.build_listings_target import BuildListingsTarget


def main() -> int:
    settings = Settings()
    usecase = BuildListingsTarget(
        listings_path=settings.listings_mvp_parquet_path,
        valuation_objects_path=settings.valuation_object_store_path,
        output_base_path=settings.listings_target_store_path,
    )

    asset_class = AssetClass.APARTMENT
    print(f"Building listings target: region={settings.region_code} asset_class={asset_class.value}", flush=True)
    stats = usecase.execute(settings.region_code, asset_class)
    print(
        f"=> input={stats.n_input} clean={stats.n_clean} "
        f"(₽/м² bounds: {stats.price_per_m2_lower_bound:,.0f}–{stats.price_per_m2_upper_bound:,.0f})",
        flush=True,
    )
    print(
        f"=> matched={stats.n_matched} unmatched={stats.n_unmatched} reasons={stats.unmatched_reasons}",
        flush=True,
    )
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

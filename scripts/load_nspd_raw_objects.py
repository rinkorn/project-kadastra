"""Parse raw NSPD page-NNNN.json into the silver store.

Reads ``data/raw/nspd/{buildings,landplots}-kazan/page-*.json``,
spatially filters by the region polygon and writes to
``data/silver/nspd/region={code}/source={buildings|landplots}/data.parquet``.
"""

from kadastra.composition_root import Container
from kadastra.config import Settings


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_load_nspd_raw_objects()

    sources: list[tuple[str, "object"]] = [
        ("buildings", settings.nspd_buildings_raw_dir),
        ("landplots", settings.nspd_landplots_raw_dir),
    ]
    for source, raw_dir in sources:
        if not raw_dir.is_dir():
            print(f"skip {source}: raw dir does not exist ({raw_dir})")
            continue
        print(
            f"Loading NSPD {source}: region={settings.region_code} "
            f"raw_dir={raw_dir}"
        )
        n = usecase.execute(
            region_code=settings.region_code, source=source, raw_dir=raw_dir
        )
        print(f"  wrote {n:,} rows to silver")
    print("done")


if __name__ == "__main__":
    main()

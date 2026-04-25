"""Download geoBoundaries Russia ADM1 GeoJSON to data/raw/regions/.

Output is gitignored; re-run after a fresh clone to populate it.
"""

from pathlib import Path

import httpx

URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/"
    "releaseData/gbOpen/RUS/ADM1/geoBoundaries-RUS-ADM1.geojson"
)
DEST = Path("data/raw/regions/geoBoundaries-RUS-ADM1.geojson")


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", URL, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        with DEST.open("wb") as fh:
            for chunk in response.iter_bytes():
                fh.write(chunk)
    size_mb = DEST.stat().st_size / 1_000_000
    print(f"Saved {size_mb:.1f} MB to {DEST}")


if __name__ == "__main__":
    main()

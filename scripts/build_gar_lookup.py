"""Build the cadnum→objectid and objectid→municipality lookups from ГАР.

Reads downloaded XMLs in ``data/raw/gar/16/`` and writes three parquet
files in ``data/silver/gar_lookup/``:

- ``cadnum_index.parquet`` — ``(cad_num, objectid, source)`` derived
  from ``AS_HOUSES_PARAMS`` + ``AS_STEADS_PARAMS`` projected to
  ``TYPEID=8`` (CADNUM).
- ``mun_lookup.parquet`` — ``(objectid, mun_okrug_name,
  mun_okrug_oktmo, settlement_name)`` from ``AS_MUN_HIERARCHY`` joined
  with ``AS_ADDR_OBJ`` via the dot-separated PATH.
- ``object_params.parquet`` — ``(objectid, oktmo_full, okato,
  postal_index)`` from ``AS_HOUSES_PARAMS`` + ``AS_STEADS_PARAMS``
  pivoted on TYPEIDs {5, 6, 7}. Settlement-level OKTMO + legacy
  classifiers + ZIP — features that are not in MUN_HIERARCHY.

The PARAMS XMLs are parsed once with the widened TYPEID whitelist
``{5, 6, 7, 8}`` and the resulting frames are passed to both
builders, so we don't re-stream the 4.6 GB twice.

Idempotent: skips a file if it exists and is non-empty unless
``--force`` is passed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from kadastra.etl.gar_cadnum_index import build_cadnum_index
from kadastra.etl.gar_mun_lookup import build_mun_lookup
from kadastra.etl.gar_object_params_lookup import build_object_params_lookup
from kadastra.etl.gar_xml_addr_obj import parse_addr_obj_xml
from kadastra.etl.gar_xml_mun_hierarchy import parse_mun_hierarchy_xml
from kadastra.etl.gar_xml_object_params import parse_object_params_xml

_RAW_ROOT = Path("data/raw/gar/16")
_OUT_ROOT = Path("data/silver/gar_lookup")
_PARAMS_TYPEIDS = {5, 6, 7, 8}


def _glob_one(pattern: str) -> Path:
    matches = sorted(_RAW_ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern} under {_RAW_ROOT}")
    return matches[-1]


def _maybe_skip(path: Path, force: bool) -> bool:
    return path.is_file() and path.stat().st_size > 0 and not force


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    args = p.parse_args()

    _OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cadnum_out = _OUT_ROOT / "cadnum_index.parquet"
    mun_out = _OUT_ROOT / "mun_lookup.parquet"
    object_params_out = _OUT_ROOT / "object_params.parquet"

    cadnum_skip = _maybe_skip(cadnum_out, args.force)
    object_params_skip = _maybe_skip(object_params_out, args.force)

    if cadnum_skip and object_params_skip:
        print(f"SKIP {cadnum_out}, {object_params_out} (exist; use --force to rebuild)")
    else:
        t0 = time.perf_counter()
        # Parse PARAMS XMLs once with the widened TYPEID whitelist —
        # both downstream builders project the rows they need.
        houses = parse_object_params_xml(
            _glob_one("AS_HOUSES_PARAMS_*.XML"), typeids=_PARAMS_TYPEIDS
        )
        steads = parse_object_params_xml(
            _glob_one("AS_STEADS_PARAMS_*.XML"), typeids=_PARAMS_TYPEIDS
        )
        print(
            f"PARSED PARAMS  houses={houses.height:,}  steads={steads.height:,}  "
            f"({time.perf_counter() - t0:.1f}s)"
        )

        if cadnum_skip:
            print(f"SKIP {cadnum_out} (exists; use --force to rebuild)")
        else:
            t1 = time.perf_counter()
            cadnum_ix = build_cadnum_index(houses=houses, steads=steads)
            cadnum_ix.write_parquet(cadnum_out)
            print(
                f"WROTE {cadnum_out}  rows={cadnum_ix.height:,}  "
                f"({time.perf_counter() - t1:.1f}s)"
            )

        if object_params_skip:
            print(f"SKIP {object_params_out} (exists; use --force to rebuild)")
        else:
            t2 = time.perf_counter()
            object_params = build_object_params_lookup(
                houses=houses, steads=steads
            )
            object_params.write_parquet(object_params_out)
            print(
                f"WROTE {object_params_out}  rows={object_params.height:,}  "
                f"({time.perf_counter() - t2:.1f}s)"
            )

    if _maybe_skip(mun_out, args.force):
        print(f"SKIP {mun_out} (exists; use --force to rebuild)")
    else:
        t0 = time.perf_counter()
        addr = parse_addr_obj_xml(_glob_one("AS_ADDR_OBJ_*.XML"))
        mh = parse_mun_hierarchy_xml(_glob_one("AS_MUN_HIERARCHY_*.XML"))
        mun = build_mun_lookup(addr, mh)
        mun.write_parquet(mun_out)
        print(
            f"WROTE {mun_out}  rows={mun.height:,}  "
            f"({time.perf_counter() - t0:.1f}s)"
        )


if __name__ == "__main__":
    main()

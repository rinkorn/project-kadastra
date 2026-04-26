"""One-off profiler for train_quartet on a single asset class.

Runs the same use case the production script invokes, but wraps the
single-class call in cProfile and dumps cumulative-time stats to
stdout. Use to find which phase (load/preprocess/per-model fit/per-
fold) dominates wall time on a class that has slowed down.

Usage: uv run python scripts/profile_train_quartet.py <asset_class>
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time

from kadastra.composition_root import Container
from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: profile_train_quartet.py <asset_class>", file=sys.stderr)
        sys.exit(2)
    asset_class = AssetClass(sys.argv[1])

    settings = Settings()
    container = Container(settings)
    usecase = container.build_train_quartet()

    print(f"[profile] class={asset_class.value} starting…", flush=True)
    t0 = time.perf_counter()
    profiler = cProfile.Profile()
    profiler.enable()
    run_id = usecase.execute(settings.region_code, asset_class)
    profiler.disable()
    elapsed = time.perf_counter() - t0
    print(f"[profile] class={asset_class.value} run_id={run_id} elapsed={elapsed:.1f}s",
          flush=True)

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(40)
    print("\n[profile] cumulative top 40:")
    print(buf.getvalue())

    buf2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=buf2).sort_stats("tottime")
    stats2.print_stats(40)
    print("[profile] tottime top 40:")
    print(buf2.getvalue())


if __name__ == "__main__":
    main()

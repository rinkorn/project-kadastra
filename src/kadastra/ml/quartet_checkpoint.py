"""Crash-recovery checkpoints for the TrainQuartet pipeline (ADR-0016).

A full quartet run on landplot takes hours; without checkpoints a kill
near the end loses everything. This module persists per-fold pass1/pass2
results and the final-fit serialized models into a per-class directory
so a restarted run with the SAME fingerprint skips finished stages.

The fingerprint captures everything that changes the result: region,
class, sample count, target bytes hash, feature column lists, fold
layout params and model hyperparams. On mismatch every checkpoint is
ignored (a stale checkpoint must never leak into a different dataset's
run) and the stage files are rewritten from scratch.

Checkpoints are deleted after a successful ``log_run`` — they exist for
crash recovery, not as a cache of finished runs.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any


class QuartetCheckpointer:
    """Filesystem checkpoint store for one quartet run.

    ``dir=None`` disables checkpointing entirely (all loads miss, all
    saves no-op) — unit tests and ad-hoc experiments stay clean.
    """

    def __init__(self, dir: Path | None, fingerprint: dict[str, Any]) -> None:
        self._dir = dir
        self._fingerprint = fingerprint
        self._enabled = dir is not None
        self._valid = False
        if not self._enabled:
            return
        assert self._dir is not None
        self._dir.mkdir(parents=True, exist_ok=True)
        fp_path = self._dir / "fingerprint.json"
        if fp_path.exists():
            try:
                stored = json.loads(fp_path.read_text(encoding="utf-8"))
                self._valid = stored == fingerprint
            except (json.JSONDecodeError, OSError):
                self._valid = False
        if not self._valid:
            # Stale or missing fingerprint: drop any leftover stage files
            # so a resumed run never mixes datasets/params.
            for child in self._dir.iterdir():
                if child.name != "fingerprint.json":
                    child.unlink()
            fp_path.write_text(
                json.dumps(fingerprint, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._valid = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def load_stage(self, name: str) -> Any | None:
        """Return the pickled payload for a stage, or None on miss/disabled."""
        if not (self._enabled and self._valid):
            return None
        assert self._dir is not None
        path = self._dir / f"{name}.pkl"
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except (OSError, pickle.UnpicklingError, EOFError):
            return None

    def save_stage(self, name: str, payload: Any) -> None:
        if not (self._enabled and self._valid):
            return
        assert self._dir is not None
        path = self._dir / f"{name}.pkl"
        tmp = self._dir / f"{name}.pkl.tmp"
        with tmp.open("wb") as fh:
            pickle.dump(payload, fh)
        tmp.replace(path)

    def discard(self) -> None:
        """Remove the whole checkpoint dir after a successful run."""
        if not self._enabled:
            return
        assert self._dir is not None
        shutil.rmtree(self._dir, ignore_errors=True)


def quartet_fingerprint(
    *,
    region_code: str,
    asset_class: str,
    n_samples: int,
    y_bytes: bytes,
    full_feature_cols: list[str],
    naive_feature_cols: list[str],
    n_splits: int,
    parent_resolution: int,
    catboost_params: dict[str, Any],
    ebm_max_bins: int,
    ebm_interactions: int,
    ebm_interactions_exclude: list[str],
    grey_tree_max_depth: int,
) -> dict[str, Any]:
    """Everything that must match for a checkpoint to be reusable.

    ``y_bytes`` (the raw target array bytes) proxies the dataset: any
    gold rebuild that changes targets invalidates the checkpoints.
    """
    return {
        "region_code": region_code,
        "asset_class": asset_class,
        "n_samples": n_samples,
        "y_sha256": hashlib.sha256(y_bytes).hexdigest(),
        "full_feature_cols": full_feature_cols,
        "naive_feature_cols": naive_feature_cols,
        "n_splits": n_splits,
        "parent_resolution": parent_resolution,
        "catboost_params": catboost_params,
        "ebm_max_bins": ebm_max_bins,
        "ebm_interactions": ebm_interactions,
        "ebm_interactions_exclude": ebm_interactions_exclude,
        "grey_tree_max_depth": grey_tree_max_depth,
    }

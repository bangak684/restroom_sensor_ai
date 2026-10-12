"""Write the unified training data to a folder: sensor rows + all 16 labels of every device, per split.

    python -m restroom_unified.export_dataset --out data/unified_world_v1

Exactly the worlds train.py uses (same seeds, chunking and profiles), stored compactly:
    <split>.npz        temperature/humidity uint8, mq135_adc/mq7_adc uint16 (mq7_adc 0 = not readable), labels uint16
                       bit mask (bit j = LABELS[j]), start datetime64[s]
    devices_<split>.csv one row of simulated physical parameters per device
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import world as W
from .pipeline import CHUNK

SPLITS = {"train": (1, "train", 120), "calibration": (2, "train", 40), "test": (3, "train", 40), "shift": (4, "shift", 40)}


def _chunk(seed: int, n: int, days: int, profile: str) -> dict:
    w = W.generate(seed, n, days, profile)
    bits = np.zeros((n, w.steps), np.uint16)
    for j, name in enumerate(W.LABELS):
        bits |= (w.labels[name].astype(np.uint16) << j)
    mq7 = np.nan_to_num(w.mq7_adc, nan=0.0)
    return {"temperature": w.temperature.astype(np.uint8), "humidity": w.humidity.astype(np.uint8),
            "mq135_adc": w.mq135_adc.astype(np.uint16), "mq7_adc": mq7.astype(np.uint16), "labels": bits,
            "start": w.start, "meta": w.meta}


def export_split(name: str, devices: int, days: int, out: Path) -> dict:
    seed, profile, _ = SPLITS[name]
    parts = Parallel(n_jobs=2)(
        delayed(_chunk)(seed * 1000 + k, min(CHUNK, devices - k * CHUNK), days, profile)
        for k in range((devices + CHUNK - 1) // CHUNK))
    arrays = {key: np.concatenate([p[key] for p in parts]) for key in
              ("temperature", "humidity", "mq135_adc", "mq7_adc", "labels", "start")}
    np.savez_compressed(out / f"{name}.npz", **arrays, label_names=np.array(W.LABELS), step_seconds=np.array(W.DT))
    meta = pd.DataFrame({key: np.concatenate([p["meta"][key] for p in parts]) for key in parts[0]["meta"]})
    meta.insert(0, "device_id", [f"{name}-{i:03d}" for i in range(len(meta))])
    meta.to_csv(out / f"devices_{name}.csv", index=False, float_format="%.5g")
    prevalence = {label: round(float(((arrays["labels"] >> j) & 1).mean()), 5) for j, label in enumerate(W.LABELS)}
    return {"seed": seed, "profile": profile, "devices": devices, "days": days, "rows": int(arrays["labels"].size),
            "label_prevalence": prevalence}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"world": "unified_restroom_v1", "step_seconds": W.DT, "labels": list(W.LABELS), "splits": {}}
    for name, (_, _, devices) in SPLITS.items():
        manifest["splits"][name] = export_split(name, devices, args.days, out)
        print(name, manifest["splits"][name]["rows"], "rows", flush=True)
    manifest["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(out.glob("*"))
                         if p.suffix in (".npz", ".csv")}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Re-pick the thresholds and smoking alarm rules of a trained bundle without retraining any model.

    python -m restroom_unified.retune                                   # limits from train.py
    python -m restroom_unified.retune --alarm .03 --suspect .2 --watch-precision .6

Scores the calibration devices with every label model, picks per-label thresholds (alarm = best F1,
watch = most sensitive threshold with precision >= --watch-precision) and the smoking persistence rules
under the false-alarm limits, saves them into the bundle and policy.json, and re-evaluates the test and
shift devices in UNIFIED_EVALUATION.json (the climate-only ablation keeps its own thresholds; its "full"
column is refreshed). About 15 minutes on 4 cores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from . import MODEL_DIR
from . import pipeline as P
from .train import FALSE_LIMITS, SEEDS, WATCH_PRECISION, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alarm", type=float, default=FALSE_LIMITS["alarm"], help="흡연 경보 false alarms per device-day")
    ap.add_argument("--suspect", type=float, default=FALSE_LIMITS["suspect"], help="흡연 의심 false alarms per device-day")
    ap.add_argument("--watch-precision", type=float, default=WATCH_PRECISION)
    ap.add_argument("--dir", default=str(MODEL_DIR))
    args = ap.parse_args()
    out = Path(args.dir)
    bundle = joblib.load(out / "unified_model.joblib")
    report = json.loads((out / "UNIFIED_EVALUATION.json").read_text(encoding="utf-8"))
    devices, days = report["data"]["eval_devices"], report["data"]["days"]
    models = bundle["models"]

    probs, truth = P.score_world(SEEDS["calibration"], devices, days, "train", models)
    thresholds = P.pick_thresholds(probs, truth, args.watch_precision)
    limits = {"alarm": args.alarm, "suspect": args.suspect}
    rules = {tier: P.tune_smoking_tier(probs[..., P.SMOKING], truth[..., P.SMOKING], limit) for tier, limit in limits.items()}
    for tier, rule in rules.items():
        log(f"smoking {tier} rule: {rule}")
    del probs, truth
    bundle["thresholds"], bundle["smoking_rules"] = thresholds, rules
    joblib.dump(bundle, out / "unified_model.joblib", compress=3)

    report.update(thresholds=thresholds, smoking_rules=rules, smoking_false_limits=limits,
                  watch_precision=args.watch_precision)
    for split in ("test", "shift"):
        profile = "shift" if split == "shift" else "train"
        probs, truth = P.score_world(SEEDS[split], devices, days, profile, models)
        result = P.evaluate_split(probs, truth, thresholds, rules)
        entry = report["splits"][split]
        entry.update(result)
        for name, row in entry.get("climate_only_ablation", {}).items():
            row["full"] = result["labels"][name]["alarm"]
        log(f"{split}: smoking alarm {result['smoking_events']['alarm']}  suspect {result['smoking_events']['suspect']}")
        del probs, truth
    (out / "UNIFIED_EVALUATION.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    policy = json.loads((out / "policy.json").read_text(encoding="utf-8"))
    policy.update(thresholds=thresholds, smoking_rules=rules, smoking_false_limits=limits, watch_precision=args.watch_precision)
    (out / "policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=1), encoding="utf-8")
    log("retune done")


if __name__ == "__main__":
    main()

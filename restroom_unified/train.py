"""Train the unified model: data -> hyper-parameter selection -> one bundle -> thresholds -> held-out evaluation.

    python -m restroom_unified.train                 # full run (about 30-60 minutes on 4 cores)
    python -m restroom_unified.train --quick         # small run to test the pipeline (saved to models/unified_quick)

Splits (all devices are different restrooms, never shared between splits):
    train        seed 1, rows subsampled every `stride` rows
    calibration  seed 2, used to pick the number of boosting rounds, the thresholds and the smoking alarm rules
    test         seed 3, same world as training, evaluated once with the frozen bundle
    shift        seed 4, "shift" profile (other climates, sensor sensitivities, door / hallway differences,
                 weaker smoke plume, busier restrooms) - tests generalisation to new buildings
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from . import MODEL_DIR, RESULT
from . import features as F
from . import pipeline as P

SEEDS = {"train": 1, "calibration": 2, "test": 3, "shift": 4}
FALSE_LIMITS = {"alarm": .05, "suspect": .25}      # smoking false alarms per device-day (흡연 경보 / 흡연 의심)
WATCH_PRECISION = .5                                # 주의 tier: most sensitive threshold with at least this precision
BASE_PARAMS = dict(learning_rate=.08, min_samples_leaf=100, l2_regularization=1.0, class_weight="balanced",
                   early_stopping=False, random_state=0)
LEAF_GRID = (15, 31, 63)
LEAF_PROBE_LABELS = ("smoking", "door_open", "window_open", "leak_active")
ABLATION_LABELS = ("door_open", "window_open", "ventilation_change", "handwashing_humidity", "wet_floor",
                   "leak_active", "faucet_on", "freeze_risk_1h", "freeze_risk_3h")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def truncate(model: HistGradientBoostingClassifier, rounds: int) -> HistGradientBoostingClassifier:
    """Keep the first `rounds` boosting rounds (the number that scored best on the calibration devices)."""
    model._predictors = model._predictors[:rounds]      # n_iter_ is len(_predictors)
    return model


def staged_best(model, x_cal: np.ndarray, y_cal: np.ndarray, every: int = 20) -> tuple[int, float]:
    best_round, best_ap = 1, -1.0
    for k, proba in enumerate(model.staged_predict_proba(x_cal), start=1):
        if k % every == 0 or k == model.n_iter_:
            ap = average_precision_score(y_cal, proba[:, 1])
            if ap > best_ap:
                best_round, best_ap = k, ap
    return best_round, best_ap


def fit_label(x, y, x_cal, y_cal, leaves: int, max_iter: int, columns=None):
    cols = slice(None) if columns is None else columns
    model = HistGradientBoostingClassifier(max_iter=max_iter, max_leaf_nodes=leaves, **BASE_PARAMS)
    model.fit(x[:, cols], y)
    rounds, ap = staged_best(model, x_cal[:, cols], y_cal)
    return truncate(model, rounds), rounds, ap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--train-devices", type=int, default=120)
    ap.add_argument("--eval-devices", type=int, default=40)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--out", default=str(MODEL_DIR))
    args = ap.parse_args()
    if args.quick:
        args.train_devices, args.eval_devices, args.days, args.stride, args.max_iter = 20, 20, 4, 8, 60
        if args.out == str(MODEL_DIR):                  # never overwrite the real model with a pipeline test
            args.out = str(MODEL_DIR.parent / "unified_quick")
    started = time.time()
    labels = P.LABELS

    log(f"train set: {args.train_devices} devices x {args.days} days, every {args.stride}th row")
    x, y = P.build_training_set(SEEDS["train"], args.train_devices, args.days, "train", args.stride)
    log(f"  rows {len(x):,}, positives per label: " + ", ".join(f"{k}={int(y[:, j].sum())}" for j, k in enumerate(labels)))
    log(f"calibration set: {args.eval_devices} devices")
    x_cal, y_cal = P.build_training_set(SEEDS["calibration"], args.eval_devices, args.days, "train", 4)

    # ---- hyper-parameter: leaf count, picked on four probe labels --------------------------------------------
    rng = np.random.default_rng(0)
    probe = rng.choice(len(x), min(300_000, len(x)), replace=False)
    grid = {}
    for leaves in LEAF_GRID:
        scores = []
        for name in LEAF_PROBE_LABELS:
            j = labels.index(name)
            _, rounds, score = fit_label(x[probe], y[probe, j], x_cal, y_cal[:, j], leaves, min(args.max_iter, 300))
            scores.append(score)
        grid[leaves] = float(np.mean(scores))
        log(f"  leaf grid: max_leaf_nodes={leaves} mean AP on calibration devices {grid[leaves]:.4f}")
    leaves = max(grid, key=grid.get)

    # ---- one model per label with its best number of rounds ---------------------------------------------------
    models, rounds_used, cal_ap = {}, {}, {}
    for j, name in enumerate(labels):
        models[name], rounds_used[name], cal_ap[name] = fit_label(x, y[:, j], x_cal, y_cal[:, j], leaves, args.max_iter)
        log(f"  {name:22s} rounds {rounds_used[name]:3d}  calibration AP {cal_ap[name]:.4f}")

    # ---- climate-only ablation models (evaluation only, not part of the bundle) ------------------------------
    climate_cols = [F.FEATURE_NAMES.index(n) for n in F.CLIMATE_FEATURES]
    ablation = {}
    for name in ABLATION_LABELS:
        j = labels.index(name)
        ablation[name], _, _ = fit_label(x, y[:, j], x_cal, y_cal[:, j], leaves, args.max_iter, climate_cols)
    log("ablation models trained")
    del x, y, x_cal, y_cal

    # ---- thresholds and smoking alarm rules on the calibration devices (all rows) -----------------------------
    p_cal, y_cal_full = P.score_world(SEEDS["calibration"], args.eval_devices, args.days, "train", models)
    thresholds = P.pick_thresholds(p_cal, y_cal_full, WATCH_PRECISION)
    smoking_rules = {
        tier: P.tune_smoking_tier(p_cal[..., P.SMOKING], y_cal_full[..., P.SMOKING], limit)
        for tier, limit in FALSE_LIMITS.items()}
    log(f"smoking alarm rule {smoking_rules['alarm']}")
    log(f"smoking suspect rule {smoking_rules['suspect']}")
    del p_cal, y_cal_full

    bundle = {"world": "unified_restroom_v1", "labels": labels, "feature_names": F.FEATURE_NAMES, "models": models,
              "thresholds": thresholds, "smoking_rules": smoking_rules, "sklearn": sklearn.__version__}
    out = __import__("pathlib").Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "unified_model.joblib", compress=3)

    # ---- held-out evaluation ----------------------------------------------------------------------------------
    report = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "world": "unified_restroom_v1",
              "note": "All numbers come from the synthetic unified world. Real sensor rows: 0.",
              "data": {"train_devices": args.train_devices, "eval_devices": args.eval_devices, "days": args.days,
                       "stride": args.stride, "features": len(F.FEATURE_NAMES)},
              "hyper_parameters": {**{k: v for k, v in BASE_PARAMS.items() if k != "early_stopping"},
                                   "max_leaf_nodes": leaves, "max_rounds": args.max_iter, "rounds_used": rounds_used,
                                   "leaf_grid_mean_AP": {str(k): round(v, 4) for k, v in grid.items()}},
              "thresholds": thresholds, "smoking_rules": smoking_rules, "smoking_false_limits": FALSE_LIMITS,
              "watch_precision": WATCH_PRECISION, "splits": {}}
    ablation_columns = {name: climate_cols for name in ablation}
    for split in ("test", "shift"):
        profile = "shift" if split == "shift" else "train"
        probs, truth = P.score_world(SEEDS[split], args.eval_devices, args.days, profile, models)
        result = P.evaluate_split(probs, truth, thresholds, smoking_rules)
        rows, events = result["labels"], result["smoking_events"]
        p_ab, _ = P.score_world(SEEDS[split], args.eval_devices, args.days, profile, ablation, ablation_columns)
        abl = {}
        for name in ablation:
            j = labels.index(name)
            yy = truth[..., j].ravel()
            threshold = P.best_f1_threshold(yy[::7], p_ab[..., j].ravel()[::7])
            abl[name] = {"climate_only": P.binary_metrics(yy, p_ab[..., j].ravel(), threshold),
                         "full": rows[name]["alarm"]}
        report["splits"][split] = {"devices": args.eval_devices, "rows": int(truth.shape[0] * truth.shape[1]),
                                   "labels": rows, "smoking_events": events, "climate_only_ablation": abl}
        log(f"{split}: smoking alarm {events['alarm']}  suspect {events['suspect']}")
        del probs, truth, p_ab

    report["seconds"] = round(time.time() - started)
    (out / "UNIFIED_EVALUATION.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    policy = {k: report[k] for k in ("world", "thresholds", "smoking_rules")}
    policy["labels"] = list(labels)
    policy["buffer_rows"], policy["warmup_rows"] = F.BUFFER_ROWS, F.WARMUP_ROWS
    (out / "policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"done in {report['seconds']} s -> {out}")


if __name__ == "__main__":
    main()

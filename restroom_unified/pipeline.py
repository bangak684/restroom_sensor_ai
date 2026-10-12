"""Data building, scoring and metrics shared by training, evaluation and the checks."""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from . import features as F
from . import world as W

LABELS = W.LABELS
HORIZON_TAIL = W.FREEZE_3H          # rows at the end whose freeze labels are not known
CHUNK = 20                          # devices per worker task
SMOKING = LABELS.index("smoking")


def _valid_rows(steps: int) -> np.ndarray:
    return np.arange(F.WARMUP_ROWS, steps - HORIZON_TAIL)


def _train_chunk(seed: int, n: int, days: int, profile: str, stride: int):
    w = W.generate(seed, n, days, profile)
    rng = np.random.default_rng(seed + 7)
    xs, ys = [], []
    for i in range(n):
        table = F.compute_features(w.timestamps(i), w.temperature[i], w.humidity[i], w.mq135_adc[i], w.mq7_adc[i])
        rows = np.arange(F.WARMUP_ROWS + int(rng.integers(stride)), w.steps - HORIZON_TAIL, stride)
        xs.append(table.to_numpy()[rows])
        ys.append(np.stack([w.labels[k][i, rows] for k in LABELS], axis=1))
    return np.concatenate(xs), np.concatenate(ys)


def build_training_set(seed: int, devices: int, days: int, profile: str, stride: int, jobs: int = 4):
    """Row-subsampled (every `stride`-th row) features and labels of `devices` devices."""
    parts = Parallel(n_jobs=jobs)(
        delayed(_train_chunk)(seed * 1000 + k, min(CHUNK, devices - k * CHUNK), days, profile, stride)
        for k in range((devices + CHUNK - 1) // CHUNK))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def _score_chunk(seed: int, n: int, days: int, profile: str, models: dict, columns: dict | None):
    """Probabilities of every label for every valid row of each device in the chunk: [n, rows, labels]."""
    w = W.generate(seed, n, days, profile)
    rows = _valid_rows(w.steps)
    probs = np.zeros((n, len(rows), len(LABELS)), np.float32)
    labels = np.stack([np.stack([w.labels[k][i, rows] for k in LABELS], axis=1) for i in range(n)])
    for i in range(n):
        table = F.compute_features(w.timestamps(i), w.temperature[i], w.humidity[i], w.mq135_adc[i], w.mq7_adc[i])
        x = table.to_numpy()[rows]
        for j, name in enumerate(LABELS):
            if name in models:
                cols = None if columns is None else columns.get(name)
                probs[i, :, j] = models[name].predict_proba(x if cols is None else x[:, cols])[:, 1]
    return probs, labels


def score_world(seed: int, devices: int, days: int, profile: str, models: dict, columns: dict | None = None,
                jobs: int = 4):
    parts = Parallel(n_jobs=jobs)(
        delayed(_score_chunk)(seed * 1000 + k, min(CHUNK, devices - k * CHUNK), days, profile, models, columns)
        for k in range((devices + CHUNK - 1) // CHUNK))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


# ---- thresholds and metrics ---------------------------------------------------------------------------------

def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))]) if len(thresholds) else 0.5


def recall_threshold(y: np.ndarray, p: np.ndarray, recall_target: float = .9) -> float:
    """Highest threshold whose recall is at least `recall_target`."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    ok = np.where(recall[:-1] >= recall_target)[0]
    return float(thresholds[ok[-1]]) if len(ok) else float(thresholds[0])


def watch_threshold(y: np.ndarray, p: np.ndarray, min_precision: float = .5) -> float:
    """Most sensitive threshold whose precision is still at least `min_precision` (the 주의 tier).

    A fixed recall target fails when some positives cannot be seen at all (a freeze risk that starts with a
    sudden heating outage): reaching 90 % recall then needs a threshold near 0, which flags every row."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    ok = np.where(precision[:-1] >= min_precision)[0]
    return float(thresholds[ok[0]]) if len(ok) else float(thresholds[-1])


def pick_thresholds(probs: np.ndarray, truth: np.ndarray, watch_precision: float = .5) -> dict:
    """Per label: alarm = best-F1 threshold, watch = most sensitive threshold with precision >= watch_precision
    (never stricter than alarm). probs / truth: [devices, rows, labels]."""
    out = {}
    for j, name in enumerate(LABELS):
        yy, pp = truth[..., j].ravel(), probs[..., j].ravel()
        alarm = best_f1_threshold(yy, pp)
        out[name] = {"alarm": alarm, "watch": min(watch_threshold(yy, pp, watch_precision), alarm)}
    return out


def evaluate_split(probs: np.ndarray, truth: np.ndarray, thresholds: dict, smoking_rules: dict) -> dict:
    """Row metrics of every label at its alarm and watch thresholds, and smoking event metrics per alarm tier."""
    rows = {}
    for j, name in enumerate(LABELS):
        yy, pp = truth[..., j].ravel(), probs[..., j].ravel()
        rows[name] = {"alarm": binary_metrics(yy, pp, thresholds[name]["alarm"]),
                      "watch": binary_metrics(yy, pp, thresholds[name]["watch"])}
    events = {}
    for tier, rule in smoking_rules.items():
        state = alarm_states(probs[..., SMOKING], rule["threshold"], rule["m"], rule["window"], rule["clear"])
        events[tier] = smoking_events(state, truth[..., SMOKING])
    return {"labels": rows, "smoking_events": events}


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y = y.astype(bool)
    pred = p >= threshold
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    out = {"prevalence": round(float(y.mean()), 5), "threshold": round(float(threshold), 4),
           "precision": round(precision, 4), "recall": round(recall, 4),
           "f1": round(2 * precision * recall / max(precision + recall, 1e-12), 4)}
    if 0 < y.sum() < len(y):
        out["auc"] = round(float(roc_auc_score(y, p)), 4)
        out["average_precision"] = round(float(average_precision_score(y, p)), 4)
    return out


def alarm_states(score: np.ndarray, threshold: float, m: int, window: int, clear: int) -> np.ndarray:
    """Persistence rule: condition = at least m of the last `window` scores >= threshold; the alarm stays on
    while the condition held in any of the last `clear` rows. score: [devices, rows]."""
    hit = (score >= threshold).astype(np.int16)
    csum = np.cumsum(np.pad(hit, ((0, 0), (window, 0))), axis=1)
    cond = (csum[:, window:] - csum[:, :-window]) >= m
    csum2 = np.cumsum(np.pad(cond.astype(np.int16), ((0, 0), (clear, 0))), axis=1)
    return (csum2[:, clear:] - csum2[:, :-clear]) > 0


def episodes(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(first, last)] of consecutive True runs of a 1-D bool array."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]


def smoking_events(alarm: np.ndarray, truth: np.ndarray, grace: int = 20, lookback: int = 40) -> dict:
    """Event-level smoking metrics. A smoking event is caught when the alarm is on at some row between its start
    and `grace` rows (5 min) after its end. An alarm onset is false when no smoking happened from `lookback` rows
    (10 min) before it to `grace` rows after it. Rates are per device-day."""
    caught = total = false_alarms = 0
    delays = []
    for a, y in zip(alarm, truth):
        for first, last in episodes(y):
            total += 1
            hit = np.flatnonzero(a[first:last + grace + 1])
            if len(hit):
                caught += 1
                delays.append(hit[0] * W.DT / 60)
        yc = np.concatenate([[0], np.cumsum(y)])
        for onset in np.flatnonzero(a & ~np.concatenate([[False], a[:-1]])):
            lo, hi = max(onset - lookback, 0), min(onset + grace + 1, len(y))
            false_alarms += int(yc[hi] - yc[lo] == 0)
    device_days = alarm.shape[0] * alarm.shape[1] * W.DT / 86400
    return {"events": total, "caught": round(caught / max(total, 1), 4), "false_alarms": false_alarms,
            "false_per_device_day": round(false_alarms / device_days, 4),
            "delay_min_median": round(float(np.median(delays)), 2) if delays else None,
            "delay_min_p90": round(float(np.percentile(delays, 90)), 2) if delays else None}


SMOKING_THRESHOLDS = (.5, .6, .7, .8, .85, .9, .93, .95, .97, .98, .99, .995)
SMOKING_PERSISTENCE = ((2, 4), (3, 4), (4, 6), (4, 8), (6, 8), (8, 12), (10, 12), (12, 16), (14, 16))


def tune_smoking_tier(score: np.ndarray, truth: np.ndarray, false_limit: float, clear: int = 8) -> dict:
    """Persistence rule with the highest caught fraction whose false alarms per device-day stay under the limit
    (ties: shorter median delay). If no rule meets the limit, the rule with the fewest false alarms is returned
    and marked. Class-weighted training pushes smoking probabilities up, so the grid reaches 0.995."""
    best, best_key, fewest = None, None, None
    for threshold in SMOKING_THRESHOLDS:
        for m, window in SMOKING_PERSISTENCE:
            result = {"threshold": threshold, "m": m, "window": window, "clear": clear,
                      **smoking_events(alarm_states(score, threshold, m, window, clear), truth)}
            if fewest is None or (result["false_per_device_day"], -result["caught"]) < (fewest["false_per_device_day"], -fewest["caught"]):
                fewest = result
            if result["false_per_device_day"] > false_limit:
                continue
            key = (result["caught"], -(result["delay_min_median"] or 99))
            if best_key is None or key > best_key:
                best_key, best = key, result
    return best or {**fewest, "note": f"no rule met the false-alarm limit {false_limit}; fewest false alarms kept"}

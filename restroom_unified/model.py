"""UnifiedModel (the trained bundle) and UnifiedJudge (one restroom, streaming, one 15-second decision at a time).

    judge = UnifiedJudge("restroom-101")
    result = judge.add(at, temperature, humidity, mq135_adc, mq7_adc)      # every 15 s; mq7_adc may be None
    print(result["level"], result["headline"], result["labels"])

Without MQ-7 readings the model still answers every label, but smoking detection collapses (see docs/UNIFIED.md).
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from . import MODEL_DIR
from . import features as F
from . import rules

LAST_5S = 3                          # 5-second rows per 15-second decision


class UnifiedModel:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.labels: tuple[str, ...] = tuple(bundle["labels"])
        self.models = bundle["models"]
        self.thresholds = bundle["thresholds"]
        self.smoking_rules = bundle["smoking_rules"]
        if tuple(bundle["feature_names"]) != tuple(F.FEATURE_NAMES):
            raise ValueError("Feature definition changed since this bundle was trained")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "UnifiedModel":
        return cls(joblib.load(Path(path) if path else MODEL_DIR / "unified_model.joblib"))

    def predict(self, features: np.ndarray) -> np.ndarray:
        """[rows, features] -> [rows, labels] probabilities."""
        features = np.atleast_2d(features)
        return np.stack([self.models[name].predict_proba(features)[:, 1] for name in self.labels], axis=1)


class _Persistence:
    """The smoking alarm rule: at least m of the last `window` scores above the threshold, held for `clear` rows."""

    def __init__(self, rule: dict):
        self.rule = rule
        self.hits: deque = deque(maxlen=rule["window"])
        self.conds: deque = deque(maxlen=rule["clear"])

    def update(self, score: float) -> bool:
        self.hits.append(score >= self.rule["threshold"])
        self.conds.append(sum(self.hits) >= self.rule["m"])
        return any(self.conds)


class UnifiedJudge:
    def __init__(self, device_id: str, model: UnifiedModel | None = None):
        self.device_id = device_id
        self.model = model or UnifiedModel.load()
        self.stream = F.FeatureStream()
        self.alarm = _Persistence(self.model.smoking_rules["alarm"])
        self.suspect = _Persistence(self.model.smoking_rules["suspect"])
        self.leak_timer, self.door_timer = rules.Timer(), rules.Timer()
        self._pending: list[tuple[float, float, float, float | None]] = []

    def add(self, at: datetime, temperature: float, humidity: float, mq135_adc: float,
            mq7_adc: float | None = None) -> dict:
        """One 15-second reading. mq7_adc: the MQ-7 ADC value if this row is in the readable heater phase, else None."""
        vector = self.stream.add(np.datetime64(at.replace(tzinfo=None)), temperature, humidity, mq135_adc, mq7_adc)
        if not self.stream.ready:
            return {"device_id": self.device_id, "at": at.isoformat(), "ready": False, "level": "준비 중",
                    "headline": "센서 예열·이력 수집 중 (처음 2시간)", "items": [], "labels": [], "probabilities": {},
                    "smoking": {"alarm": False, "suspect": False, "score": None},
                    "freeze_levels": {"freeze_risk_1h": "없음", "freeze_risk_3h": "없음"}, "checks": []}
        probs = self.model.predict(vector)[0]
        return self._decide(at, dict(zip(self.model.labels, map(float, probs))), temperature)

    def add_5s(self, at: datetime, temperature: float, humidity: float, mq135_adc: float,
               mq7_adc: float | None = None) -> dict | None:
        """5-second ESP32 rows; every third row closes one 15-second decision (T / H last, MQ-135 median,
        MQ-7 the last readable value). Returns None for the other two rows."""
        self._pending.append((temperature, humidity, mq135_adc, mq7_adc))
        if len(self._pending) < LAST_5S:
            return None
        rows, self._pending = self._pending, []
        qualified = [r[3] for r in rows if r[3] is not None]
        return self.add(at, rows[-1][0], rows[-1][1], float(np.median([r[2] for r in rows])),
                        qualified[-1] if qualified else None)

    def _decide(self, at: datetime, probs: dict[str, float], temperature: float) -> dict:
        thresholds = self.model.thresholds
        active = {name for name, p in probs.items()
                  if p >= thresholds[name]["alarm"] and not name.startswith("freeze_risk")}
        freeze = {}
        for name in ("freeze_risk_1h", "freeze_risk_3h"):
            freeze[name] = ("경보" if probs[name] >= thresholds[name]["alarm"]
                            else "주의" if probs[name] >= thresholds[name]["watch"] else "없음")
        score = probs["smoking"]
        smoking = {"alarm": self.alarm.update(score), "suspect": self.suspect.update(score), "score": round(score, 4)}
        smoking["suspect"] = smoking["suspect"] or smoking["alarm"]
        if smoking["alarm"]:
            active.add("smoking")
        elif smoking["suspect"]:
            active.add("smoking")
        else:
            active.discard("smoking")
        leak_minutes = self.leak_timer.minutes("leak_active" in active, at)
        door_minutes = self.door_timer.minutes("door_open" in active, at)
        level, headline, items, checks = rules.grade(at, active, smoking, freeze, temperature, leak_minutes, door_minutes)
        return {"device_id": self.device_id, "at": at.isoformat(), "ready": True, "level": level, "headline": headline,
                "items": [{"level": lv, "code": code, "message": msg} for lv, code, msg in items],
                "labels": sorted(active), "probabilities": {k: round(v, 4) for k, v in probs.items()},
                "smoking": smoking, "freeze_levels": freeze, "checks": checks}

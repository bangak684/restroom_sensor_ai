"""Self-check of the unified model (about 2 minutes):

    python -m restroom_unified.check

1. the bundle loads and matches the feature definition
2. streaming features == batch features (same numbers once the buffer is full)
3. streaming judge probabilities == batch probabilities
4. one fresh device: the held-out evaluation numbers are reproduced in order of magnitude (sanity floor)
5. graded rule scenes behave as designed
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta

import numpy as np

from . import MODEL_DIR
from . import features as F
from . import rules
from . import world as W
from .model import UnifiedJudge, UnifiedModel

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)


def scene(month: int, active: set[str], smoking=None, freeze=None, temperature=21.0, leak_min=0.0, door_min=0.0):
    smoking = smoking or {"alarm": False, "suspect": False}
    freeze = freeze or {"freeze_risk_1h": "없음", "freeze_risk_3h": "없음"}
    return rules.grade(datetime(2026, month, 15, 12), active, smoking, freeze, temperature, leak_min, door_min)[0]


def main() -> int:
    model = UnifiedModel.load()
    record("bundle loads", True, f"{len(model.labels)} labels, {len(F.FEATURE_NAMES)} features")

    world = W.generate(77, 3, 2, "train")
    i = 1
    stamps = world.timestamps(i)
    n = F.BUFFER_ROWS + 400
    table = F.compute_features(stamps, world.temperature[i], world.humidity[i], world.mq135_adc[i], world.mq7_adc[i])
    batch = model.predict(table.to_numpy()[:n])
    stream, judge_probs = F.FeatureStream(), []
    judge = UnifiedJudge("check", model)
    rows = []
    for k in range(n):
        mq7 = None if np.isnan(world.mq7_adc[i, k]) else float(world.mq7_adc[i, k])
        rows.append(stream.add(stamps[k], world.temperature[i, k], world.humidity[i, k], world.mq135_adc[i, k], mq7))
        out = judge.add(stamps[k].astype("datetime64[s]").astype(datetime), float(world.temperature[i, k]),
                        float(world.humidity[i, k]), float(world.mq135_adc[i, k]), mq7)
        judge_probs.append([out["probabilities"].get(name, np.nan) for name in model.labels])
    diff = np.nanmax(np.abs(np.array(rows[-300:]) - table.to_numpy()[n - 300:n]))
    record("streaming features equal batch features", diff < 1e-4, f"max abs diff {diff:.2e}")
    pdiff = np.nanmax(np.abs(np.array(judge_probs[-300:]) - batch[-300:]))
    record("streaming judge probabilities equal batch probabilities", pdiff < 5e-4, f"max abs diff {pdiff:.2e} (judge rounds to 4 decimals)")

    ev = json.loads((MODEL_DIR / "UNIFIED_EVALUATION.json").read_text(encoding="utf-8"))
    test = ev["splits"]["test"]["labels"]
    weak = [k for k, v in test.items() if v["alarm"].get("auc", 1) < .6]
    record("held-out AUC above 0.6 for every label", not weak, f"weak: {weak}" if weak else "")
    record("smoking event detection on held-out devices",
           ev["splits"]["test"]["smoking_events"]["suspect"]["caught"] > .5,
           json.dumps(ev["splits"]["test"]["smoking_events"]["suspect"], ensure_ascii=False))

    record("scene: smoking alarm is 긴급", scene(6, {"smoking"}, {"alarm": True, "suspect": True}) == "긴급")
    record("scene: leak without context is 긴급", scene(6, {"leak_active"}) == "긴급")
    record("scene: leak while handwashing is 주의", scene(6, {"leak_active", "handwashing_humidity"}) == "주의")
    record("scene: leak with context for 30 minutes is 긴급",
           scene(6, {"leak_active", "handwashing_humidity"}, leak_min=31) == "긴급")
    record("scene: winter window is 주의, summer window is 정보",
           scene(1, {"window_open"}) == "주의" and scene(7, {"window_open"}) == "정보")
    record("scene: cold room + open window + 3h freeze alarm is 긴급",
           scene(1, {"window_open"}, freeze={"freeze_risk_1h": "없음", "freeze_risk_3h": "경보"}, temperature=2.0) == "긴급")
    record("scene: propped door in winter is 주의, short door pulse is 정보, nothing is 평상시",
           scene(1, {"door_open"}, door_min=45) == "주의" and scene(6, {"door_open"}) == "정보" and scene(6, set()) == "평상시")

    failed = [name for name, ok, _ in RESULTS if not ok]
    (MODEL_DIR / "CHECK_RESULT.json").write_text(
        json.dumps([{"check": n, "passed": ok, "detail": d} for n, ok, d in RESULTS], ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

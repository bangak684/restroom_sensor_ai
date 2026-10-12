"""Replay one synthetic restroom through the streaming UnifiedJudge and print the graded decisions next to the truth.

    python -m restroom_unified.demo --seed 5 --device 0 --start-hour 8 --hours 6 [--profile shift] [--csv out.csv]

The judge needs 2 hours of history, so rows are fed from two hours before --start-hour.
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np

from . import features as F
from . import world as W
from .model import UnifiedJudge, UnifiedModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--start-hour", type=float, default=8, help="first judged hour after the device's start (day 1 = 0-24)")
    ap.add_argument("--hours", type=float, default=6)
    ap.add_argument("--profile", default="train")
    ap.add_argument("--every", type=int, default=20, help="print every n-th decision")
    ap.add_argument("--csv", default=None, help="also write the replayed sensor rows and labels to this CSV")
    args = ap.parse_args()
    first = max(int(args.start_hour * 3600 / W.DT) - F.WARMUP_ROWS, 0)
    last = int((args.start_hour + args.hours) * 3600 / W.DT)
    world = W.generate(args.seed, args.device + 1, max(1, int(np.ceil(last * W.DT / 86400))), args.profile)
    i = args.device
    stamps = world.timestamps(i)
    last = min(last, world.steps)
    judge = UnifiedJudge(f"demo-{args.seed}-{i}", UnifiedModel.load())
    correct = total = 0
    for k in range(first, last):
        mq7 = None if np.isnan(world.mq7_adc[i, k]) else float(world.mq7_adc[i, k])
        at = stamps[k].astype("datetime64[s]").astype(datetime)
        result = judge.add(at, float(world.temperature[i, k]), float(world.humidity[i, k]),
                           float(world.mq135_adc[i, k]), mq7)
        if not result["ready"]:
            continue
        truth = {name for name in W.LABELS if world.labels[name][i, k]}
        predicted = set(result["labels"]) | ({"smoking"} if result["smoking"]["suspect"] else set())
        total += 1
        correct += predicted == truth
        if k % args.every == 0 or ((truth or predicted) and k % 4 == 0):      # quiet rows sparsely, busy rows every minute
            print(f"{at:%m-%d %H:%M:%S}  {result['level']:4s} {result['headline']:<26s} "
                  f"예측={sorted(predicted)} 정답={sorted(truth)}")
    print(f"\n정확히 일치한 판정 {correct}/{total} ({correct / max(total, 1):.1%}); 라벨 집합이 모두 같아야 정답으로 센 값입니다.")
    if args.csv:
        import pandas as pd
        rows = slice(first, last)
        frame = pd.DataFrame({"timestamp": stamps[rows], "temperature": world.temperature[i, rows],
                              "humidity": world.humidity[i, rows], "mq135_adc": world.mq135_adc[i, rows],
                              "mq7_adc": world.mq7_adc[i, rows],
                              **{k: world.labels[k][i, rows].astype(int) for k in W.LABELS}})
        frame.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()

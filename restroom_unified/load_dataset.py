"""Read a split written by export_dataset.py.

    from restroom_unified.load_dataset import load_split, device_frame
    split = load_split("data/unified_world_v1/test.npz")
    frame = device_frame(split, 0)           # one device as a DataFrame with a timestamp, 4 sensor columns, 16 labels
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_split(path: str) -> dict:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def device_frame(split: dict, device: int) -> pd.DataFrame:
    """One device at 15-second resolution. mq7_adc is NaN where the MQ-7 is not readable (stored as 0)."""
    step = int(split["step_seconds"])
    rows = split["labels"].shape[1]
    stamps = split["start"][device] + (np.arange(rows) * step).astype("timedelta64[s]")
    mq7 = split["mq7_adc"][device].astype("float32")
    frame = pd.DataFrame({"timestamp": stamps, "temperature": split["temperature"][device].astype("float32"),
                          "humidity": split["humidity"][device].astype("float32"),
                          "mq135_adc": split["mq135_adc"][device].astype("float32"),
                          "mq7_adc": np.where(mq7 == 0, np.nan, mq7)})
    for j, name in enumerate(split["label_names"]):
        frame[str(name)] = ((split["labels"][device] >> j) & 1).astype("int8")
    return frame

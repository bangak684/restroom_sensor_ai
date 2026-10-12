"""Causal features for the unified model, from one 15-second row per decision.

Input per row: DHT11 temperature (degC) and humidity (%RH), MQ-135 ADC (12 bit), MQ-7 ADC (12 bit, only in the
readable part of the 150 s heater cycle, NaN otherwise) and the time stamp. Every feature uses finite windows
of past rows only, so the streaming class gives exactly the same numbers as the batch function once its
buffer (BUFFER_ROWS) is full - the longest nested window is 6 h of a ratio built on a 2 h baseline.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

DT = 15
BUFFER_ROWS = 2400                 # 10 h: 1440 (6 h median) + 480 (2 h baseline) + margin
WARMUP_ROWS = 480                  # 2 h before the first ready decision
CO_PER_ODOUR = 0.1                 # MQ-7 rise explained by an MQ-135 rise in ordinary odours (global prior)
DEFAULT_SHARE = 0.074              # device odour share used until a device has enough gas events

W1M, W2M, W3M, W5M, W10M, W15M, W30M, W1H, W2H, W3H, W6H = 4, 8, 12, 20, 40, 60, 120, 240, 480, 720, 1440
CYCLE = 10


def _div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.where(den > 0)


def compute_features(timestamps: np.ndarray, temperature, humidity, mq135, mq7) -> pd.DataFrame:
    """Feature table (one row per input row). NaN readings of temperature / humidity / MQ-135 carry forward."""
    t = pd.Series(np.asarray(temperature, float)).ffill()
    h = pd.Series(np.asarray(humidity, float)).ffill()
    a135 = pd.Series(np.asarray(mq135, float)).ffill()
    a7 = pd.Series(np.asarray(mq7, float))
    f: dict[str, pd.Series] = {}

    ts = t.rolling(W1M, min_periods=1).mean()
    hs = h.rolling(W1M, min_periods=1).mean()
    f["t"], f["h"] = ts, hs
    for name, lag in (("1m", W1M), ("5m", W5M), ("15m", W15M), ("1h", W1H)):
        f[f"dt_{name}"] = ts - ts.shift(lag)
        f[f"dh_{name}"] = hs - hs.shift(lag)
    f["dt_step2"] = t - t.shift(2)
    f["dh_step2"] = h - h.shift(2)
    f["dh_jump_2m"] = h.diff().abs().rolling(W2M, min_periods=1).max()
    f["dt_jump_2m"] = t.diff().abs().rolling(W2M, min_periods=1).max()
    f["t_mean_1h"] = ts.rolling(W1H, min_periods=1).mean()
    f["t_min_3h"] = ts.rolling(W3H, min_periods=1).min()
    f["t_max_1h"] = ts.rolling(W1H, min_periods=1).max()
    f["t_dev_6h"] = ts - ts.rolling(W6H, min_periods=1).mean()
    f["t_range_30m"] = ts.rolling(W30M, min_periods=1).max() - ts.rolling(W30M, min_periods=1).min()
    f["t_slope_15m"] = (ts - ts.shift(W15M)) / (W15M * DT / 60)
    f["t_below_3_dur"] = (t <= 3).astype(float).rolling(W3H, min_periods=1).sum() * DT / 60
    f["h_mean_1h"] = hs.rolling(W1H, min_periods=1).mean()
    f["h_max_1h"] = hs.rolling(W1H, min_periods=1).max()
    f["h_min_1h"] = hs.rolling(W1H, min_periods=1).min()
    f["h_rise_30m"] = hs - hs.rolling(W30M, min_periods=1).mean()
    f["h_dev_6h"] = hs - hs.rolling(W6H, min_periods=1).mean()
    f["h_high_dur"] = (h >= 85).astype(float).rolling(W3H, min_periods=1).sum() * DT / 60
    f["h_peak_gap"] = hs.rolling(W30M, min_periods=1).max() - hs

    g135 = a135 / (4095 - a135).clip(lower=1)
    gs = g135.rolling(W1M, min_periods=1).mean()
    base135 = g135.rolling(W2M, min_periods=1).mean().rolling(W2H, min_periods=20).min()
    r1 = _div(gs, base135) - 1
    f["r135_1m"] = r1
    f["r135_3m"] = _div(g135.rolling(W3M, min_periods=1).mean(), base135) - 1
    f["r135_max10m"] = r1.rolling(W10M, min_periods=1).max()
    f["r135_max30m"] = r1.rolling(W30M, min_periods=1).max()
    f["r135_mean30m"] = r1.rolling(W30M, min_periods=1).mean()
    f["r135_slope"] = r1 - r1.shift(W3M)
    f["r135_decay"] = np.log(gs / gs.shift(W2M))
    f["r135_peak_gap"] = r1.rolling(W30M, min_periods=1).max() - r1
    win = r1.rolling(CYCLE, min_periods=1).mean()
    f["r135_win"] = win
    f["base135_drift"] = _div(base135, base135.shift(W6H)) - 1

    g7 = a7 / (4095 - a7).clip(lower=1)
    base7 = g7.rolling(W2H, min_periods=3).min()
    r7 = _div(g7, base7) - 1
    hold = r7.ffill(limit=3 * CYCLE)
    f["r7"] = hold
    f["r7_prev"] = hold.shift(CYCLE)
    f["r7_max3"] = hold.rolling(3 * CYCLE, min_periods=1).max()
    f["r7_mean10"] = r7.rolling(CYCLE, min_periods=1).mean()
    valid = r7.notna().to_numpy()
    last = np.maximum.accumulate(np.where(valid, np.arange(len(valid)), -1))
    age = np.where(last >= 0, np.arange(len(valid)) - last, np.nan)
    f["r7_age"] = pd.Series(age)
    share = _div(hold, win.clip(lower=0.03))
    f["co_share"] = share
    f["co_excess"] = hold - CO_PER_ODOUR * win
    ratio = share.where((win > .2) & valid)
    device_share = ratio.rolling(W6H, min_periods=4).median()
    f["device_share"] = device_share
    ref = device_share.fillna(DEFAULT_SHARE)
    f["share_rel"] = share / ref
    excess = hold - ref * win
    f["excess_dev"] = excess
    f["excess_dev_max3"] = excess.rolling(3 * CYCLE, min_periods=1).max()
    f["excess_dev_sum3"] = excess.clip(lower=0).rolling(3 * CYCLE, min_periods=1).sum()
    f["r7_over_win_max"] = f["r7_max3"] / win.clip(lower=0.03)

    stamps = pd.DatetimeIndex(np.asarray(timestamps, "datetime64[ns]"))
    hour = (stamps.hour + stamps.minute / 60).to_numpy()
    doy = stamps.dayofyear.to_numpy()
    f["hour_sin"] = pd.Series(np.sin(2 * np.pi * hour / 24))
    f["hour_cos"] = pd.Series(np.cos(2 * np.pi * hour / 24))
    f["season_sin"] = pd.Series(np.sin(2 * np.pi * doy / 365.25))
    f["season_cos"] = pd.Series(np.cos(2 * np.pi * doy / 365.25))
    f["weekend"] = pd.Series((stamps.dayofweek >= 5).astype(float))

    return pd.DataFrame(f).astype("float32")


FEATURE_NAMES = tuple(compute_features(np.arange(5000) * np.timedelta64(DT, "s") + np.datetime64("2025-01-01"),
                                       np.full(5000, 20.0), np.full(5000, 50.0), np.full(5000, 1500.0),
                                       np.full(5000, np.nan)).columns)
# MQ-135 / MQ-7 free subset: what a temperature-and-humidity-only box can use
CLIMATE_FEATURES = tuple(n for n in FEATURE_NAMES if not n.startswith(("r135", "r7", "co_", "excess", "device_share",
                                                                        "share_rel", "base135")))


class FeatureStream:
    """Streaming wrapper: add one row, get that row's feature vector. Same numbers as compute_features."""

    def __init__(self):
        self.stamp: deque = deque(maxlen=BUFFER_ROWS)
        self.cols = [deque(maxlen=BUFFER_ROWS) for _ in range(4)]
        self.rows = 0

    def add(self, at: np.datetime64, temperature: float, humidity: float, mq135: float, mq7: float | None) -> np.ndarray:
        self.stamp.append(np.datetime64(at, "ns"))
        for col, value in zip(self.cols, (temperature, humidity, mq135, np.nan if mq7 is None else mq7)):
            col.append(float(value))
        self.rows += 1
        table = compute_features(np.array(self.stamp), *(np.array(c) for c in self.cols))
        return table.iloc[-1].to_numpy(np.float32)

    @property
    def ready(self) -> bool:
        return self.rows >= WARMUP_ROWS

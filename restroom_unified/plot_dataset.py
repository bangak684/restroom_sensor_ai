"""Draw the unified training data as PNG charts.

    python -m restroom_unified.plot_dataset --data data/unified_world_v1 --out data/unified_world_v1/graphs

Reads the exported npz splits only. Light surface, fixed categorical order (validated palette: blue, orange,
aqua, yellow, magenta, green), one y-scale per panel. Large type (13 pt body, 21 pt title) so the charts stay
readable when GitHub shrinks them to the page width.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from .load_dataset import load_split
from .pipeline import episodes

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#3d3c3a", "#6b6a66", "#e0dfdb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
RAMP = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
KO = {"smoking": "흡연", "smoke_residual": "흡연 후 잔류", "occupancy_odor": "이용·냄새", "handwashing_humidity": "손 씻기·습기",
      "wet_floor": "젖은 바닥", "cleaning_products": "청소 제품", "cleaning_wet_floor": "청소+젖은 바닥", "sanitizer": "소독제",
      "fragrance": "방향제", "door_open": "문 열림", "window_open": "창문 열림", "ventilation_change": "환기 변화",
      "faucet_on": "수도 사용", "leak_active": "누수", "freeze_risk_1h": "동파 1시간", "freeze_risk_3h": "동파 3시간"}
plt.rcParams.update({
    "font.family": ["WenQuanYi Zen Hei", "DejaVu Sans"], "axes.unicode_minus": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 1.0, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 12.5,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 140})


def head(fig, text: str, sub: str, height: float, band: float = 1.35) -> None:
    """Title and one-line subtitle in a fixed band (inches) at the top; sets the plot's top margin."""
    fig.text(.012, 1 - .3 / height, text, fontsize=21, fontweight="bold", va="center", ha="left", color=INK)
    fig.text(.012, 1 - .78 / height, sub, fontsize=13.5, va="center", ha="left", color=INK2)
    fig.subplots_adjust(top=1 - band / height)


def bit(split: dict, name: str) -> np.ndarray:
    j = [str(n) for n in split["label_names"]].index(name)
    return ((split["labels"] >> j) & 1).astype(bool)


def conduct(adc: np.ndarray) -> np.ndarray:
    adc = adc.astype("float32")
    adc[adc == 0] = np.nan
    return adc / (4095 - adc)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name)
    plt.close(fig)
    print("saved", name, flush=True)


# 1 ---------------------------------------------------------------------------------------------------------------
def prevalence(splits: dict, names: list[str], out: Path) -> None:
    rates = {s: np.array([bit(splits[s], n).mean() * 100 for n in names]) for s in ("train", "shift")}
    order = np.argsort(rates["train"])[::-1]
    height = 11.5
    fig, ax = plt.subplots(figsize=(13, height))
    fig.subplots_adjust(left=.14, right=.97, bottom=.07)
    head(fig, "항목별 발생 비율", "전체 15초 행 중 해당 항목이 켜져 있는 비율 · 한 행에 여러 항목이 동시에 켜질 수 있음", height)
    y = np.arange(len(names))
    ax.barh(y - .21, rates["train"][order], height=.4, color=CAT[0], edgecolor=SURFACE, linewidth=2, label="학습 데이터 (장치 120대)")
    ax.barh(y + .21, rates["shift"][order], height=.4, color=CAT[1], edgecolor=SURFACE, linewidth=2, label="분포 이동 시험 (장치 40대)")
    for yi, v, w in zip(y, rates["train"][order], rates["shift"][order]):
        ax.text(v + .1, yi - .21, f"{v:.2f}%", va="center", fontsize=11.5, color=INK2)
        ax.text(w + .1, yi + .21, f"{w:.2f}%", va="center", fontsize=11.5, color=INK2)
    ax.set_yticks(y, [KO[names[i]] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("비율 (%)")
    ax.set_xlim(0, 12.3)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=13.5)
    save(fig, out, "01_label_prevalence.png")


# 2 ---------------------------------------------------------------------------------------------------------------
def one_day(split: dict, names: list[str], out: Path) -> None:
    labels = np.stack([bit(split, n) for n in names])                     # [L, dev, rows]
    rows_day = 5760
    days = labels.shape[2] // rows_day
    want = [names.index(n) for n in ("smoking", "door_open", "occupancy_odor", "handwashing_humidity", "cleaning_products")]
    best, best_score = (0, 0), -1
    for d in range(labels.shape[1]):
        for k in range(days):
            seg = labels[:, d, k * rows_day:(k + 1) * rows_day]
            score = sum(len(episodes(seg[j])) > 0 for j in want) * 100 + len(episodes(seg[names.index("smoking")]))
            if score > best_score:
                best, best_score = (d, k), score
    d, k = best
    sl = slice(k * rows_day, (k + 1) * rows_day)
    hours = np.arange(rows_day) * 15 / 3600
    g135, g7 = conduct(split["mq135_adc"][d, sl]), conduct(split["mq7_adc"][d, sl])
    smoking = [(hours[a], max((b - a + 1) * 15 / 3600, .05)) for a, b in episodes(bit(split, "smoking")[d, sl])]
    strips = ["smoking", "occupancy_odor", "handwashing_humidity", "cleaning_products", "door_open", "window_open"]
    height = 15
    fig, axes = plt.subplots(5, 1, figsize=(14, height), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1, 1.45]})
    fig.subplots_adjust(left=.1, right=.97, bottom=.05, hspace=.16)
    head(fig, f"장치 하루 기록 예시 (시험 장치 {d}번, {k + 1}일째)",
         "위 네 칸은 센서값, 맨 아래는 같은 시각의 정답 · 주황색 띠는 흡연 시간 (MQ-135와 MQ-7이 함께 오름)", height, 1.5)
    temp_1m = pd.Series(split["temperature"][d, sl].astype("float32")).rolling(4, min_periods=1).mean().to_numpy()
    panels = [(temp_1m, "온도 (℃)", "온도 (1분 평균)"), (split["humidity"][d, sl], "습도 (%RH)", "습도"),
              (g135, "전도도", "MQ-135 (냄새 가스)"), (g7, "전도도", "MQ-7 (일산화탄소, 히터 주기 중 읽는 구간만)")]
    for ax, (values, ylabel, name) in zip(axes, panels):
        for start, width in smoking:
            ax.axvspan(start, start + width, color=CAT[1], alpha=.16, linewidth=0, zorder=0)
        if name.startswith("MQ-7"):
            ax.scatter(hours, values, s=9, color=CAT[0], edgecolor="none", zorder=2)
        else:
            ax.plot(hours, values, color=CAT[0], lw=1.9, zorder=2)
        lo, hi = np.nanpercentile(values, [.5, 99.8])
        pad = (hi - lo) * .12
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(ylabel)
        ax.text(.006, .92, name, transform=ax.transAxes, fontsize=14, va="top", color=INK, fontweight="bold")
    ax = axes[4]
    for start, width in smoking:
        ax.axvspan(start, start + width, color=CAT[1], alpha=.16, linewidth=0, zorder=0)
    for i, name in enumerate(strips):
        spans = [(hours[a], max((b - a + 1) * 15 / 3600, .06)) for a, b in episodes(bit(split, name)[d, sl])]
        ax.broken_barh(spans, (i - .36, .72), facecolors=CAT[i % 6], edgecolor=CAT[i % 6], linewidth=1.2)
    ax.set_yticks(range(len(strips)), [KO[n] for n in strips], fontsize=13.5)
    ax.set_ylim(len(strips) - .4, -.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("시각 (시)")
    save(fig, out, "02_one_day_example.png")


# 3 ---------------------------------------------------------------------------------------------------------------
def distributions(split: dict, out: Path) -> None:
    sub = slice(None, None, 10)
    height = 9.5
    fig, axes = plt.subplots(2, 2, figsize=(14, height))
    fig.subplots_adjust(left=.04, right=.98, bottom=.08, hspace=.42, wspace=.12)
    head(fig, "센서값 분포", "학습 장치 120대, 10행마다 표본 · 온습도는 DHT11이라 정수 · MQ 센서는 장치마다 기준선이 달라 넓게 퍼짐", height, 1.3)
    specs = [("temperature", "온도 (℃)", np.arange(-.5, 40.6, 1)), ("humidity", "습도 (%RH)", np.arange(19.5, 90.6, 1)),
             ("mq135_adc", "MQ-135 ADC (0~4095)", np.linspace(500, 3600, 65)), ("mq7_adc", "MQ-7 ADC (읽을 수 있는 행만)", np.linspace(300, 2200, 65))]
    for ax, (key, label, bins) in zip(axes.ravel(), specs):
        values = split[key][:, sub].astype("float32").ravel()
        if key == "mq7_adc":
            values = values[values > 0]
        ax.hist(values, bins=bins, color=CAT[0], edgecolor=SURFACE, linewidth=.8)
        ax.set_xlabel(label, fontsize=14)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        q = np.percentile(values, [5, 50, 95])
        ax.text(.98, .94, f"중앙 {q[1]:.0f}\n(5%~95%: {q[0]:.0f}~{q[2]:.0f})", transform=ax.transAxes, ha="right", va="top",
                fontsize=13, color=INK)
    save(fig, out, "03_sensor_distributions.png")


# 4 ---------------------------------------------------------------------------------------------------------------
def event_response(split: dict, out: Path) -> None:
    kinds = ["smoking", "occupancy_odor", "cleaning_products", "handwashing_humidity", "door_open", "window_open"]
    offsets = np.arange(-40, 61)
    rng = np.random.default_rng(0)
    g135, g7 = conduct(split["mq135_adc"]), conduct(split["mq7_adc"])
    temp, hum = split["temperature"].astype("float32"), split["humidity"].astype("float32")
    series = {k: {m: [] for m in ("t", "h", "g135", "g7")} for k in kinds}
    for kind in kinds:
        mask = bit(split, kind)
        found = []
        for d in range(mask.shape[0]):
            for first, _ in episodes(mask[d]):
                if first >= 60 and first + 61 < mask.shape[1] and not mask[d, first - 40:first].any():
                    found.append((d, first))
        pick = rng.choice(len(found), min(500, len(found)), replace=False) if found else []
        for p in pick:
            d, first = found[p]
            idx = first + offsets
            base = slice(first - 40, first - 5)
            series[kind]["t"].append(temp[d, idx] - np.nanmedian(temp[d, base]))
            series[kind]["h"].append(hum[d, idx] - np.nanmedian(hum[d, base]))
            series[kind]["g135"].append((g135[d, idx] / np.nanmedian(g135[d, base]) - 1) * 100)
            with np.errstate(all="ignore"):
                b7 = np.nanmedian(g7[d, base]) if np.isfinite(g7[d, base]).any() else np.nan
            series[kind]["g7"].append((g7[d, idx] / b7 - 1) * 100)
    panels = [("g135", "MQ-135 (냄새 가스) 변화 (%)"), ("g7", "MQ-7 (일산화탄소) 변화 (%)"),
              ("h", "습도 변화 (%p)"), ("t", "온도 변화 (℃)")]
    height = 11.5
    fig, axes = plt.subplots(2, 2, figsize=(14.5, height), sharex=True)
    fig.subplots_adjust(left=.075, right=.98, bottom=.075, hspace=.16, wspace=.2)
    head(fig, "사건이 일어날 때 센서값은 어떻게 변하나", "항목이 처음 켜진 시점(점선)을 0분으로 맞춘 평균 변화 · 사건 종류마다 최대 500건", height, 2.05)
    minutes = offsets * 15 / 60
    for ax, (key, label) in zip(axes.ravel(), panels):
        for i, kind in enumerate(kinds):
            data = np.array(series[kind][key])
            with np.errstate(all="ignore"):
                line = np.nanmean(data, axis=0) if key in ("t", "h") else np.nanmedian(data, axis=0)
            ax.plot(minutes, line, color=CAT[i], lw=3.0, solid_capstyle="round")
        ax.axvline(0, color=MUTED, lw=1.4, ls=(0, (4, 3)))
        ax.set_title(label, fontsize=15, loc="left", fontweight="bold", color=INK, pad=8)
    for ax in axes[1]:
        ax.set_xlabel("항목이 켜진 시점 기준 (분)")
    handles = [plt.Line2D([0], [0], color=CAT[i], lw=4) for i in range(len(kinds))]
    fig.legend(handles, [KO[k] for k in kinds], loc="upper center", bbox_to_anchor=(.5, 1 - 1.18 / height), ncol=6,
               fontsize=14.5, handlelength=1.6, columnspacing=1.6)
    save(fig, out, "04_event_response.png")


# 5 ---------------------------------------------------------------------------------------------------------------
def cooccurrence(split: dict, names: list[str], out: Path) -> None:
    m = np.stack([bit(split, n).ravel()[::4] for n in names]).astype(np.float32)        # [L, rows]
    both = m @ m.T
    cond = both / np.maximum(m.sum(1)[:, None], 1)                                      # P(column | row)
    height = 13
    fig, ax = plt.subplots(figsize=(13.5, height))
    fig.subplots_adjust(left=.15, right=.93, bottom=.17)
    head(fig, "항목끼리 얼마나 겹치나 (%)", "행 항목이 켜져 있을 때 열 항목도 켜져 있는 비율 · 20% 이상만 숫자 표시", height, 1.3)
    ax.grid(False)
    im = ax.imshow(cond, cmap=RAMP, vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and cond[i, j] >= .2:
                ax.text(j, i, f"{cond[i, j] * 100:.0f}", ha="center", va="center", fontsize=13, fontweight="bold",
                        color="#fcfcfb" if cond[i, j] > .5 else INK)
    ax.set_xticks(range(len(names)), [KO[n] for n in names], rotation=55, ha="right", fontsize=13)
    ax.set_yticks(range(len(names)), [KO[n] for n in names], fontsize=13)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
    cb.set_label("비율", fontsize=13)
    save(fig, out, "05_label_cooccurrence.png")


# 6 ---------------------------------------------------------------------------------------------------------------
def hour_of_day(split: dict, names: list[str], out: Path) -> None:
    rows = split["labels"].shape[1]
    hours = (np.arange(rows) * 15 // 3600) % 24
    table = np.zeros((len(names), 24))
    for i, n in enumerate(names):
        mask = bit(split, n)
        for h in range(24):
            table[i, h] = mask[:, hours == h].mean() * 100
    norm = table / np.maximum(table.max(1, keepdims=True), 1e-9)
    height = 10.5
    fig, ax = plt.subplots(figsize=(15, height))
    fig.subplots_adjust(left=.12, right=.9, bottom=.085)
    head(fig, "시간대별 발생 패턴", "진할수록 그 항목이 많이 켜지는 시간대 (항목마다 자기 최댓값을 가장 진하게) · 오른쪽 숫자는 그 항목의 최대 비율", height, 1.3)
    ax.grid(False)
    ax.imshow(norm, cmap=RAMP, vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(names)), [KO[n] for n in names], fontsize=13.5)
    ax.set_xticks(range(24), [str(h) for h in range(24)], fontsize=13)
    ax.set_xlabel("시각 (시)")
    for i in range(len(names)):
        ax.text(23.8, i, f"최대 {table[i].max():.1f}%", va="center", fontsize=13, color=INK2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, out, "06_hour_of_day.png")


# 7 ---------------------------------------------------------------------------------------------------------------
def climate(split: dict, devices: pd.DataFrame, out: Path) -> None:
    group = np.where(devices["tmean"] < 7.5, 0, np.where(devices["tmean"] < 16.5, 1, 2))
    names = ["추운 곳 (연평균 약 3℃)", "보통 (약 12℃)", "더운 곳 (약 21℃)"]
    kinds = ["freeze_risk_1h", "freeze_risk_3h", "window_open", "leak_active"]
    height = 8.5
    fig, ax = plt.subplots(figsize=(13.5, height))
    fig.subplots_adjust(left=.08, right=.98, bottom=.12)
    head(fig, "기후별 동파·창문·누수 비율", "동파 위험은 추운 곳 장치에 몰려 있음 · 겨울에 시작하는 장치 40%와 난방 고장·난방 없는 방으로 비율을 확보", height, 1.9)
    width = .2
    for i, kind in enumerate(kinds):
        mask = bit(split, kind)
        rate = [mask[group == g].mean() * 100 if (group == g).any() else 0 for g in range(3)]
        xs = np.arange(3) + (i - 1.5) * (width + .02)
        ax.bar(xs, rate, width=width, color=CAT[i], edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, rate):
            ax.text(x, v + .06, f"{v:.1f}", ha="center", fontsize=13.5, color=INK)
    counts = [int((group == g).sum()) for g in range(3)]
    ax.set_xticks(range(3), [f"{n}\n장치 {c}대" for n, c in zip(names, counts)], fontsize=14)
    ax.set_ylabel("전체 행 중 비율 (%)")
    ax.set_ylim(0, 3.4)
    ax.grid(axis="x", visible=False)
    handles = [Patch(facecolor=CAT[i], label=KO[k]) for i, k in enumerate(kinds)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1 - 1.05 / height), ncol=4, fontsize=14.5)
    save(fig, out, "07_climate_breakdown.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, test, shift = (load_split(str(data / f"{n}.npz")) for n in ("train", "test", "shift"))
    names = list(json.loads((data / "MANIFEST.json").read_text(encoding="utf-8"))["labels"])
    prevalence({"train": train, "shift": shift}, names, out)
    one_day(test, names, out)
    distributions(train, out)
    event_response(train, out)
    cooccurrence(train, names, out)
    hour_of_day(train, names, out)
    climate(train, pd.read_csv(data / "devices_train.csv"), out)


if __name__ == "__main__":
    main()

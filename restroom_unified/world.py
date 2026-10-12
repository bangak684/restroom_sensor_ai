"""Unified synthetic restroom world: every label of the restroom lives in the same 15-second record.

The two earlier synthetic worlds disagreed (integer vs decimal humidity, no window events vs no gas
channel, door / window not modelled at all), so no single model could be trained on both. This world
simulates one restroom per device with one set of physical states and writes every judgement the
product needs as a label of the same time series:

    states      indoor air temperature, humidity (base + short-lived excess + wet floor + leak),
                two gas pools (c1 odour / VOC for MQ-135, c2 CO-like for MQ-7) and a smoke tracer
    drivers     outdoor weather and season, heating (setpoint, outages, unheated rooms), door pulses and
                propped-open doors (exchange with a hallway whose temperature / humidity differ),
                window, exhaust-fan boost, people (use, handwashing, sanitizer, fragrance, smoking),
                cleaning (with or without a wet floor), spills, taps, leaks
    sensors     DHT11 (integer, 0-50 degC, 20-90 %RH), MQ-135 and MQ-7 as 12-bit dividers
                (adc / (4095 - adc) = conductance), MQ-7 readable only in the last 60 s of each 150 s heater cycle

Every number is an engineering assumption in normalised units, not a measured value. Real restroom
data is still 0 rows; the point of this world is that all labels are consistent with each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DT = 15                              # seconds per decision row
STEPS_DAY = 86400 // DT              # 5760
HEATER_STEPS = 10                    # 150 s heater cycle
QUALIFIED_FROM = 6                   # cycle phase 90-150 s is readable
FREEZE_C = 2.0
FREEZE_1H, FREEZE_3H = 240, 720
LABELS = ("smoking", "smoke_residual", "occupancy_odor", "handwashing_humidity", "wet_floor", "cleaning_products",
          "cleaning_wet_floor", "sanitizer", "fragrance", "door_open", "window_open", "ventilation_change",
          "faucet_on", "leak_active", "freeze_risk_1h", "freeze_risk_3h")
HOURLY = np.array([.1, .05, .05, .05, .1, .3, .7, 1.4, 1.8, 1.4, 1.1, 1.2, 1.6, 1.4, 1.1, 1.0, 1.1, 1.4, 1.5, 1.0,
                   .7, .5, .3, .2])
EPOCH = np.datetime64("2025-01-01T00:00:00")


@dataclass
class World:
    n: int
    steps: int
    start: np.ndarray                                    # datetime64[s] per device
    temperature: np.ndarray                              # DHT11 reading, degC (NaN = invalid)
    humidity: np.ndarray                                 # DHT11 reading, %RH
    mq135_adc: np.ndarray
    mq7_adc: np.ndarray                                  # NaN outside the readable heater phase
    labels: dict[str, np.ndarray]                        # bool [n, steps]
    temperature_true: np.ndarray
    humidity_true: np.ndarray
    meta: dict[str, np.ndarray] = field(default_factory=dict)

    def timestamps(self, device: int) -> np.ndarray:
        return self.start[device] + (np.arange(self.steps) * DT).astype("timedelta64[s]")


def _paint(arr: np.ndarray, a: float, b: float, value: float = 1.0) -> None:
    """Add `value` x covered fraction to every step overlapped by [a, b) seconds."""
    lo, hi = max(a, 0.0), min(b, arr.shape[0] * DT)
    if hi <= lo:
        return
    i0, i1 = int(lo // DT), int((hi - 1e-9) // DT)
    if i0 == i1:
        arr[i0] += value * (hi - lo) / DT
        return
    arr[i0] += value * ((i0 + 1) * DT - lo) / DT
    if i1 > i0 + 1:
        arr[i0 + 1:i1] += value
    arr[i1] += value * (hi - i1 * DT) / DT


def _mass(arr: np.ndarray, a: float, b: float, mass: float) -> None:
    """Spread `mass` (concentration units x steps of injection) evenly over [a, b)."""
    steps = max((b - a) / DT, 1.0)
    _paint(arr, a, max(b, a + DT * 0.5), mass / steps)


def _mark(arr: np.ndarray, a: float, b: float) -> None:
    lo, hi = max(a, 0.0), min(b, arr.shape[0] * DT)
    if hi > lo:
        arr[int(lo // DT):int((hi - 1e-9) // DT) + 1] = True


def sample_devices(rng: np.random.Generator, n: int, profile: str) -> dict[str, np.ndarray]:
    """Per-device physical parameters. profile 'shift' moves climate, sensors, door and placement
    ranges away from the training ones to test generalisation to new buildings."""
    shift = profile == "shift"
    u = rng.uniform
    p: dict[str, np.ndarray] = {}
    climate = rng.choice(3, n, p=[.5, .3, .2] if shift else [.3, .5, .2])
    p["tmean"] = np.array([3., 12., 21.])[climate] + rng.normal(0, 1.5, n)
    p["aseason"] = np.array([14., 11., 6.])[climate] * u(.85, 1.15, n)
    p["adaily"] = u(3, 6, n)
    winter = rng.random(n) < .4                      # cold-season devices keep freeze-risk labels from being too rare
    p["start_doy"] = np.where(winter, (335 + rng.integers(0, 75, n)) % 365, rng.integers(0, 365, n))
    p["tset"] = u(20, 24, n) + (1.5 if shift else 0.0)
    p["setback"] = u(0, 3, n)
    p["unheated"] = (climate == 0) & (rng.random(n) < (.2 if shift else .12))
    p["outage_rate"] = u(0, .12, n) * (climate <= 1)
    p["a_th"] = DT / (60 * u(15, 50, n))
    p["hset"] = u(35, 60, n) + (6 if shift else 0)
    p["a_h"] = DT / (60 * u(10, 35, n))
    p["has_window"] = rng.random(n) < .65
    p["k_win"] = DT / (60 * u(6, 25, n))
    p["win_rate"] = u(.5, 3.0, n)
    p["k_door"] = u(.04, .15, n)
    sign = -1 if shift else 1
    p["door_dt"] = sign * u(-5, 2, n)               # hallway minus restroom temperature
    p["door_dh"] = sign * u(-20, 10, n)             # hallway minus restroom humidity
    p["has_fan"] = rng.random(n) < .7
    p["k_fan"] = u(.03, .08, n)
    p["k0"] = DT / (60 * u(5, 15, n))
    p["k_gdoor"] = u(.03, .12, n)
    p["k_gwin"] = u(.01, .05, n)
    p["kh"] = u(.015, .04, n)
    p["k_hdoor"] = u(.04, .15, n)
    p["plume"] = np.exp(rng.normal(np.log(.5 if shift else 1.0), .5, n))
    gs = np.exp(rng.normal(0, .3 if shift else .15, n))
    p["g0_135"] = u(.4, 1.2, n)
    p["a135"] = u(.7, 1.3, n) * gs
    p["g0_7"] = u(.15, .6, n)
    p["bco"] = u(.5, 1.0, n) * gs
    p["bodor"] = u(.02, .08, n)
    p["hc135"] = u(.002, .005, n)
    p["hc7"] = u(.001, .004, n)
    p["offset_t"] = rng.normal(0, .8, n)
    p["offset_h"] = rng.normal(0, 2.5, n)
    p["heater_phase"] = rng.integers(0, HEATER_STEPS, n)
    p["smoke_rate"] = np.where(rng.random(n) < .25, 0.0, np.clip(np.exp(rng.normal(np.log(.8), .9, n)), .1, 6))
    p["use_rate"] = u(15, 60, n) * (1.5 if shift else 1.0)
    p["leak_rate"] = .05 * u(.3, 2.5, n)
    p["has_fragrance"] = rng.random(n) < .4
    p["frag_rate"] = u(1, 4, n)
    p["drift"] = rng.normal(0, .003, n)
    return p


def _seasonal_temp(p: dict, i: int, doy: np.ndarray | float) -> np.ndarray | float:
    return p["tmean"][i] + p["aseason"][i] * np.cos(2 * np.pi * (doy - 200) / 365)


def _schedule(rng: np.random.Generator, p: dict, i: int, days: int, doy0: int, dow0: int, s: dict, lab: dict) -> None:
    """Draw every event of one device and paint it into the driver arrays `s` and label arrays `lab`."""
    u = rng.uniform
    plume = float(p["plume"][i])

    def secs(n: int, weights: np.ndarray = HOURLY) -> np.ndarray:
        hours = rng.choice(24, n, p=weights / weights.sum())
        return hours * 3600 + u(0, 3600, n)

    def door(a: float, length: float) -> None:
        _paint(s["door"], a, a + length)
        _mark(lab["door_open"], a, a + length)

    def odour(a: float, b: float, m1: float, m2: float) -> None:
        _mass(s["c1"], a, b, m1)
        _mass(s["c2"], a, b, m2)

    def water(a: float, b: float, excess: float) -> None:
        _mass(s["hx"], a, b, excess)
        _mark(lab["faucet_on"], a, b)

    def handwash(after: float, p_sanitizer: float) -> None:
        a = after + u(5, 40)
        b = a + u(20, 60)
        water(a, b, u(3, 8))
        _mark(lab["handwashing_humidity"], a, b + 120)
        if rng.random() < p_sanitizer:
            c = b + u(3, 15)
            m1 = u(.1, .35)
            odour(c, c + u(15, 40), m1, m1 * u(.05, .3))
            _mark(lab["sanitizer"], c, c + 120)

    for day in range(days):
        base = day * 86400.0
        weekend = (dow0 + day) % 7 >= 5
        doy = (doy0 + day) % 365
        tseason = float(_seasonal_temp(p, i, doy))

        for t0 in base + secs(rng.poisson(p["use_rate"][i] * (.6 if weekend else 1.0))):
            d = u(60, 300)
            e = t0 + d
            door(t0 - u(3, 8), u(8, 30))
            door(e + u(0, 5), u(8, 30))
            _mass(s["hx"], t0, e, u(.5, 1.5))
            if rng.random() < .7:
                m1 = float(np.clip(np.exp(rng.normal(np.log(.5), .5)), .08, 2.5)) * plume ** .5
                ratio = u(.2, .8) if rng.random() < .02 else u(.02, .15)
                odour(t0, e, m1, m1 * ratio)
                _mark(lab["occupancy_odor"], t0, e + 90)
            if rng.random() < .7:
                handwash(e, .2)

        for t0 in base + secs(rng.poisson(p["smoke_rate"][i] * (.7 if weekend else 1.0)), HOURLY + .3):
            d = u(120, 600)
            e = t0 + d
            m1 = plume * u(.8, 2.0)
            door(t0 - u(3, 8), u(8, 30))
            door(e + u(0, 5), u(8, 30))
            odour(t0, e, m1, m1 * u(.2, 1.0))
            _mass(s["smoke"], t0, e, m1)
            _mass(s["hx"], t0, e, u(.3, 1.0))
            _mark(lab["smoking"], t0, e)
            if rng.random() < .5:
                handwash(e, .05)

        for t0 in base + np.array([u(5, 9), u(12.5, 15), u(19.5, 22)])[:rng.integers(1, 4)] * 3600:
            d = u(300, 900)
            e = t0 + d
            wet = rng.random() < .6
            m1 = u(.8, 2.0)
            odour(t0, e, m1, m1 * u(.01, .06))
            _mark(lab["cleaning_products"], t0, e + 120)
            if rng.random() < .5:
                door(t0, d)
            if wet:
                tail = u(600, 1500)
                _paint(s["wet"], t0, e + tail, u(8, 18))
                _mark(lab["cleaning_wet_floor"], t0, e + 600)
                _mark(lab["wet_floor"], t0, e + tail)
                water(t0 + u(0, 60), t0 + u(60, 120), u(2, 5))

        for t0 in base + secs(rng.poisson(1.0)):
            d = u(900, 2700)
            _paint(s["wet"], t0, t0 + d, u(6, 14))
            _mark(lab["wet_floor"], t0, t0 + d + 300)

        if p["has_fragrance"][i]:
            for t0 in base + secs(rng.poisson(p["frag_rate"][i])):
                m1 = u(.1, .4)
                odour(t0, t0 + u(15, 60), m1, m1 * u(.02, .2))
                _mark(lab["fragrance"], t0, t0 + 180)

        for t0 in base + secs(rng.poisson(3.0)):
            water(t0, t0 + u(10, 40), u(1, 3))

        for t0 in base + secs(rng.poisson(4.0)):
            door(t0, u(8, 25))

        for t0 in base + secs(rng.poisson(.3)):
            door(t0, u(600, 3600))

        if p["has_window"][i]:
            factor = float(np.clip(1 + .08 * (tseason - 12), .3, 2.5))
            for t0 in base + secs(rng.poisson(p["win_rate"][i] * factor / 2)):
                if rng.random() < .08:
                    d = u(3 * 3600, 9 * 3600)
                else:
                    d = float(np.clip(np.exp(rng.normal(np.log(2400), .8)), 300, 6 * 3600))
                _paint(s["win"], t0, t0 + d)
                _mark(lab["window_open"], t0, t0 + d)

        if p["has_fan"][i]:
            for t0 in base + secs(rng.poisson(2.5)):
                d = u(600, 2400)
                _paint(s["fan"], t0, t0 + d)
                _mark(lab["ventilation_change"], t0, t0 + d)

        if rng.random() < p["leak_rate"][i]:
            t0 = base + u(0, 86400)
            d = u(1200, 18000)
            _paint(s["leak"], t0, t0 + d, u(15, 40))
            _mark(lab["leak_active"], t0, t0 + d)

        if tseason < 14 and rng.random() < p["outage_rate"][i]:
            t0 = base + u(0, 86400)
            _paint(s["outage"], t0, t0 + u(3 * 3600, 12 * 3600))


def _outdoor(rng: np.random.Generator, p: dict, n: int, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outdoor temperature / humidity [steps, n] and the seasonal mean temperature [steps, n]."""
    t = np.arange(steps) * DT / 86400.0
    hours = (t * 24) % 24
    doy = (p["start_doy"][None, :] + t[:, None]) % 365
    tseason = p["tmean"][None, :] + p["aseason"][None, :] * np.cos(2 * np.pi * (doy - 200) / 365)
    nh = steps * DT // 3600 + 2
    anomaly = np.zeros((nh, n))
    noise = rng.normal(0, 3.0 * np.sqrt(1 - .98 ** 2), (nh, n))
    anomaly[0] = rng.normal(0, 3.0, n)
    for h in range(1, nh):
        anomaly[h] = .98 * anomaly[h - 1] + noise[h]
    hour_axis = np.arange(nh)
    step_hours = t * 24
    anom = np.stack([np.interp(step_hours, hour_axis, anomaly[:, k]) for k in range(n)], axis=1)
    to = tseason + p["adaily"][None, :] * np.cos(2 * np.pi * (hours[:, None] - 15) / 24) + anom
    hn = np.zeros((nh, n))
    nn = rng.normal(0, 8.0 * np.sqrt(1 - .97 ** 2), (nh, n))
    hn[0] = rng.normal(0, 8.0, n)
    for h in range(1, nh):
        hn[h] = .97 * hn[h - 1] + nn[h]
    hnoise = np.stack([np.interp(step_hours, hour_axis, hn[:, k]) for k in range(n)], axis=1)
    ho = np.clip(60 + 12 * np.cos(2 * np.pi * (hours[:, None] - 5) / 24) - .3 * (to - tseason) + hnoise, 20, 98)
    return to, ho, tseason


def generate(seed: int, n_devices: int, days: int, profile: str = "train") -> World:
    """Simulate `n_devices` restrooms for `days` days. Deterministic for (seed, n_devices, days, profile)."""
    rng = np.random.default_rng(seed)
    n, steps = n_devices, days * STEPS_DAY
    p = sample_devices(rng, n, profile)
    start = EPOCH + (p["start_doy"].astype("timedelta64[D]")).astype("timedelta64[s]")
    d0 = start.astype("datetime64[D]").astype(np.int64)
    dow0 = (d0 + 3) % 7

    keys = ("door", "c1", "c2", "smoke", "hx", "wet", "leak", "win", "fan", "outage")
    sched = {k: np.zeros((n, steps), np.float32) for k in keys}
    labs = {k: np.zeros((n, steps), bool) for k in LABELS}
    for i in range(n):
        _schedule(rng, p, i, days, int(p["start_doy"][i]), int(dow0[i]),
                  {k: sched[k][i] for k in keys}, {k: labs[k][i] for k in LABELS})
    for k in ("door", "win", "fan", "outage"):
        np.minimum(sched[k], 1.0, out=sched[k])
    x = {k: np.ascontiguousarray(v.T) for k, v in sched.items()}               # [steps, n]

    to, ho, tseason = _outdoor(rng, p, n, steps)
    hour = ((np.arange(steps) * DT / 3600.0) % 24)[:, None]
    night = ((hour < 6) | (hour >= 22)).astype(float)
    tset = np.clip(p["tset"][None, :] + .15 * (tseason - 15), 19, 27) - p["setback"][None, :] * night
    tset_eff = np.where(p["unheated"][None, :], np.minimum(np.maximum(to + 6, 5), tset), tset)
    tset_eff = np.where(x["outage"] > .5, np.minimum(to + 3.5, tset_eff), tset_eff)
    hset = np.clip(p["hset"][None, :] + .2 * (ho - 60) + .3 * (tseason - 12), 20, 78)

    a_th, a_h, k_win, k_door = (p[k][None, :] for k in ("a_th", "a_h", "k_win", "k_door"))
    k0, kf, kgd, kgw = p["k0"], np.where(p["has_fan"], p["k_fan"], 0.0), p["k_gdoor"], p["k_gwin"]
    kh, khd, khf = p["kh"], p["k_hdoor"], .02
    ddt, ddh = p["door_dt"], p["door_dh"]
    has_win = p["has_window"].astype(float)

    tb, hb = tset_eff[0].copy(), hset[0].copy()
    c1 = np.zeros(n); c2 = np.zeros(n); sm = np.zeros(n); hx = np.zeros(n); hw = np.zeros(n); hl = np.zeros(n)
    out = {k: np.empty((steps, n), np.float32) for k in ("t", "h", "c1", "c2", "sm")}
    for t in range(steps):
        dr, wi, fa = x["door"][t], x["win"][t] * has_win, x["fan"][t]
        tb = tb + a_th[0] * (tset_eff[t] - tb) + k_win[0] * wi * (to[t] - tb) + k_door[0] * dr * (tset_eff[t] + ddt - tb)
        hb = np.clip(hb + a_h[0] * (hset[t] - hb) + k_win[0] * wi * (ho[t] - hb)
                     + k_door[0] * dr * (hset[t] + ddh - hb), 5, 98)
        decay = k0 + kf * fa + kgd * dr + kgw * wi
        c1 = c1 + x["c1"][t] - c1 * decay
        c2 = c2 + x["c2"][t] - c2 * decay
        sm = sm + x["smoke"][t] - sm * decay
        hx = hx + x["hx"][t] - hx * (kh + khd * dr + khf * fa)
        hw = hw + .006 * (x["wet"][t] - hw)
        hl = hl + .01 * (x["leak"][t] - hl)
        out["t"][t], out["c1"][t], out["c2"][t], out["sm"][t] = tb, c1, c2, sm
        out["h"][t] = np.clip(hb + hx + hw + hl, 5, 99)

    t_true, h_true, c1, c2, sm = (np.ascontiguousarray(out[k].T) for k in ("t", "h", "c1", "c2", "sm"))
    labs["smoke_residual"] = (sm > .15) & ~labs["smoking"]
    cold = np.cumsum(t_true <= FREEZE_C, axis=1, dtype=np.int32)
    before = np.concatenate([np.zeros((n, 1), np.int32), cold[:, :-1]], axis=1)
    for name, horizon in (("freeze_risk_1h", FREEZE_1H), ("freeze_risk_3h", FREEZE_3H)):
        upper = cold[:, np.minimum(np.arange(steps) + horizon, steps - 1)]
        labs[name] = (upper - before) > 0

    # ---- sensors -------------------------------------------------------------------------------
    tt = np.arange(steps)
    hours = tt * DT / 3600.0
    drift = np.exp(np.cumsum(rng.normal(0, 1, (n, steps)), axis=1) * (.0015 * np.sqrt(DT / 3600)) * 1.0
                   + p["drift"][:, None] * hours[None, :] / 24)
    col = lambda k: p[k][:, None]
    g135 = (col("g0_135") * drift * (1 + col("a135") * c1 + .1 * c2) * (1 + col("hc135") * (h_true - 50))
            * (1 - .004 * (t_true - 20)) * (1 + rng.normal(0, .015, (n, steps))))
    g7 = (col("g0_7") * drift * (1 + col("bco") * c2 + col("bodor") * c1) * (1 + col("hc7") * (h_true - 50))
          * (1 + rng.normal(0, .025, (n, steps))))
    adc135 = np.clip(np.round(4095 * g135 / (1 + g135) + rng.normal(0, 2, (n, steps))), 1, 4094)
    adc7 = np.clip(np.round(4095 * g7 / (1 + g7) + rng.normal(0, 2, (n, steps))), 1, 4094)
    phase = (tt[None, :] + p["heater_phase"][:, None]) % HEATER_STEPS
    adc7 = np.where(phase >= QUALIFIED_FROM, adc7, np.nan)
    temp = np.clip(np.round(t_true + col("offset_t") + rng.normal(0, .3, (n, steps))), 0, 50)
    hum = np.clip(np.round(h_true + col("offset_h") + rng.normal(0, 1.0, (n, steps))), 20, 90)

    meta = {k: np.asarray(v) for k, v in p.items()}
    return World(n=n, steps=steps, start=start, temperature=temp.astype(np.float32), humidity=hum.astype(np.float32),
                 mq135_adc=adc135.astype(np.float32), mq7_adc=adc7.astype(np.float32), labels=labs,
                 temperature_true=t_true, humidity_true=h_true, meta=meta)

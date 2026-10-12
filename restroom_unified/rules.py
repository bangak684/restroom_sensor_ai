"""Grades the unified model's label decisions: 긴급 / 주의 / 정보 / 평상시 (or 준비 중 while warming up).

The model answers "is this label on right now"; the rules decide how much a person should care, using what the
labels say about each other (cleaning explains water, a propped door explains a draught, ...). The numbers
(30 minutes, 3 degC, winter months) are drafts to be set from real data.
"""
from __future__ import annotations

from datetime import datetime

LEVELS = ("평상시", "정보", "주의", "긴급")
LABEL_KO = {
    "smoking": "흡연", "smoke_residual": "흡연 후 잔류", "occupancy_odor": "이용·냄새", "handwashing_humidity": "손 씻기·습기",
    "wet_floor": "젖은 바닥", "cleaning_products": "청소 제품", "cleaning_wet_floor": "청소+젖은 바닥", "sanitizer": "소독제",
    "fragrance": "방향제", "door_open": "문 열림", "window_open": "창문 열림", "ventilation_change": "환기 변화",
    "faucet_on": "수도 사용", "leak_active": "누수", "freeze_risk_1h": "동파 1시간", "freeze_risk_3h": "동파 3시간"}
WINTER_MONTHS = {12, 1, 2}
LEAK_CONFIRM_MINUTES = 30
DOOR_PROPPED_MINUTES = 30
FREEZE_COLD_C = 3.0
LEAK_CONTEXT = ("cleaning_products", "cleaning_wet_floor", "handwashing_humidity", "faucet_on")
INFO = ("occupancy_odor", "handwashing_humidity", "cleaning_products", "sanitizer", "fragrance", "ventilation_change",
        "faucet_on", "door_open")


class Timer:
    """Minutes a flag has been on without interruption."""

    def __init__(self):
        self.since: datetime | None = None

    def minutes(self, on: bool, at: datetime) -> float:
        if not on:
            self.since = None
            return 0.0
        self.since = self.since or at
        return (at - self.since).total_seconds() / 60


def grade(at: datetime, active: set[str], smoking: dict, freeze: dict, temperature: float | None,
          leak_minutes: float, door_minutes: float) -> tuple[str, str, list[tuple[str, str, str]], list[str]]:
    """-> (level, headline, [(level, code, message)], cross-checks)."""
    items: list[tuple[str, str, str]] = []
    checks: list[str] = []
    winter = at.month in WINTER_MONTHS
    cold = temperature is not None and temperature <= FREEZE_COLD_C

    if smoking["alarm"]:
        items.append(("긴급", "smoking_alarm", "흡연 경보"))
    elif smoking["suspect"]:
        items.append(("주의", "smoking_suspect", "흡연 의심"))
    if "smoke_residual" in active and not smoking["alarm"]:
        items.append(("주의", "smoke_residual", "흡연 후 잔류"))

    if "leak_active" in active:
        context = [LABEL_KO[name] for name in LEAK_CONTEXT if name in active]
        if "wet_floor" in active or "cleaning_wet_floor" in active:
            checks.append("leak: 젖은 바닥 동반")
        if leak_minutes >= LEAK_CONFIRM_MINUTES or not context:
            items.append(("긴급", "leak", f"누수 의심 ({leak_minutes:.0f}분 지속)"))
            checks.append("leak: 청소·손 씻기·수도 사용 없음" if not context else "leak: 30분 이상 지속")
        else:
            items.append(("주의", "leak_check", f"누수 확인 필요 ({', '.join(context)} 중)"))

    opening = "window_open" in active or door_minutes >= DOOR_PROPPED_MINUTES
    if freeze["freeze_risk_1h"] == "경보":
        items.append(("긴급", "freeze_1h", "1시간 내 동파 위험"))
    elif opening and cold and freeze["freeze_risk_3h"] == "경보":
        items.append(("긴급", "freeze_opening", "창문·문 열림 + 실내 3℃ 이하: 동파 위험"))
    elif freeze["freeze_risk_3h"] == "경보":
        items.append(("주의", "freeze_3h", "3시간 내 동파 위험"))
    elif "주의" in freeze.values():
        items.append(("주의", "freeze_watch", "동파 주의"))

    if "window_open" in active:
        items.append(("주의" if winter else "정보", "window_open", "창문 열림" + (" (겨울철)" if winter else "")))
    if door_minutes >= DOOR_PROPPED_MINUTES:
        items.append(("주의" if cold or winter else "정보", "door_propped", f"문이 {door_minutes:.0f}분째 열려 있음"))
    elif "door_open" in active and "window_open" in active:
        checks.append("draught: 문과 창문이 함께 열림")

    if "cleaning_wet_floor" in active:
        items.append(("주의", "cleaning_wet_floor", "청소+젖은 바닥"))
    elif "wet_floor" in active:
        items.append(("주의", "wet_floor", "젖은 바닥"))
    for name in INFO:
        if name in active and not (name == "door_open" and door_minutes >= DOOR_PROPPED_MINUTES):
            items.append(("정보", name, LABEL_KO[name]))

    items.sort(key=lambda item: -LEVELS.index(item[0]))
    if items:
        return items[0][0], items[0][2], items, checks
    return "평상시", "정상", items, checks

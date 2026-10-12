"""restroom_unified: one model for every restroom judgement (smoking, odour, cleaning, water, leak, door,
window, ventilation, freeze risk) from one sensor box (DHT11 + MQ-135 + MQ-7).

One synthetic world (world.py) holds every label in the same record; one feature function
(features.py) feeds one gradient-boosting bundle (model.py) whose decisions are graded by rules.py.
Everything it needs is in this repository: models/unified/ and data/unified_world_v1/.
"""
from pathlib import Path

RESULT = Path(__file__).resolve().parents[1]
MODEL_DIR = RESULT / "models" / "unified"

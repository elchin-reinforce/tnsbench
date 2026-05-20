"""Calibration helpers — placeholder for future human-judge calibration."""
from __future__ import annotations

from typing import Dict


def adjust_thresholds(metrics: Dict[str, float]) -> Dict[str, float]:
    """In v0, identity. In future, will fit thresholds against human-labeled set."""
    return metrics

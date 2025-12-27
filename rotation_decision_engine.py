"""Decision engine for the Cross-Asset Rotation dashboard."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

AlignmentArrow = str


@dataclass
class SignalBundle:
    bucket: str
    st: float
    mt: float
    lt: float
    align: float
    align_arrow: AlignmentArrow
    crowd_z: float | None
    crowded: bool
    effective_score: float
    current_weight: float


NORMALIZE_DIVISOR = 25.0  # keeps momentum scaling deterministic into [-1, 1]


def _normalize_momentum(value: float | None) -> float:
    """Compress raw momentum into [-1, 1] via tanh scaling.

    The divisor keeps the mapping smooth for double-digit percent returns while
    remaining deterministic for small magnitudes.
    """

    if value is None or pd.isna(value):
        return float("nan")
    return float(math.tanh(value / NORMALIZE_DIVISOR))


def compute_signals(momentums: dict[str, float | None]) -> tuple[float, float, float]:
    st_raw = 0.6 * (momentums.get("4W") or 0.0) + 0.4 * (momentums.get("12W") or 0.0)
    mt_raw = 0.5 * (momentums.get("6M") or 0.0) + 0.5 * (momentums.get("12M") or 0.0)
    lt_raw = 0.5 * (momentums.get("2Y") or 0.0) + 0.5 * (momentums.get("3Y") or 0.0)
    return (
        _normalize_momentum(st_raw),
        _normalize_momentum(mt_raw),
        _normalize_momentum(lt_raw),
    )


def align_score(st: float, mt: float, lt: float) -> float:
    scores = [v for v in (st, mt, lt) if not pd.isna(v)]
    if not scores:
        return float("nan")
    return float(0.5 * st + 0.3 * mt + 0.2 * lt)


def arrow_for(score: float) -> AlignmentArrow:
    if pd.isna(score):
        return "→"
    if score > 0.15:
        return "↑"
    if score < -0.15:
        return "↓"
    return "→"


def crowding_zscore(current_weight: float, history: Iterable[float]) -> float | None:
    data = [v for v in history if not pd.isna(v)]
    if len(data) < 3:
        return None
    mean = float(np.mean(data))
    std = float(np.std(data))
    if std == 0:
        return None
    return float((current_weight - mean) / std)


def apply_crowding_penalty(align: float, crowd_z: float | None) -> tuple[float, bool]:
    crowded = crowd_z is not None and crowd_z >= 1.0
    penalty = 0.3 if crowded else 0.0
    return float(align - penalty * max(align, 0.0)), crowded


def recommendation_plan(bundles: List[SignalBundle]) -> dict[str, list[tuple[str, float]]]:
    ordered = sorted(bundles, key=lambda b: b.effective_score, reverse=True)
    increases = ordered[:3]
    reduce_candidates = list(reversed(ordered[-3:]))
    reduces = [b for b in reduce_candidates if b not in increases]
    holds = [b for b in bundles if b not in increases and b not in reduces]

    pos_scores = [max(b.effective_score, 0.0) for b in increases]
    neg_scores = [abs(min(b.effective_score, 0.0)) for b in reduces]
    max_pos = max(pos_scores) if pos_scores else 0.0
    max_neg = max(neg_scores) if neg_scores else 0.0

    deltas: dict[str, float] = {}
    for bundle in increases:
        scaled = 0.0 if max_pos == 0 else (bundle.effective_score / max_pos) * 1.5
        deltas[bundle.bucket] = round(min(1.5, max(0.0, scaled)), 2)
    for bundle in reduces:
        scaled = 0.0 if max_neg == 0 else (bundle.effective_score / max_neg) * -1.5
        deltas[bundle.bucket] = round(max(-1.5, min(0.0, scaled)), 2)
    for bundle in holds:
        deltas[bundle.bucket] = 0.0

    imbalance = sum(deltas.values())
    if abs(imbalance) > 1e-6:
        adjust_targets = holds or increases or reduces
        if adjust_targets:
            per = round(-imbalance / len(adjust_targets), 2)
            for bundle in adjust_targets:
                new_val = deltas[bundle.bucket] + per
                deltas[bundle.bucket] = round(max(-1.5, min(1.5, new_val)), 2)

    # final tidy to reduce rounding drift
    residual = round(sum(deltas.values()), 2)
    if abs(residual) >= 0.05:
        targets = holds or increases or reduces
        if targets:
            first = targets[0]
            deltas[first.bucket] = round(max(-1.5, min(1.5, deltas[first.bucket] - residual)), 2)

    return {
        "INCREASE": [(b.bucket, deltas[b.bucket]) for b in increases],
        "HOLD": [(b.bucket, deltas[b.bucket]) for b in holds],
        "REDUCE": [(b.bucket, deltas[b.bucket]) for b in reduces],
    }


def regime_call(bundles: List[SignalBundle], cyclical: set[str], defensive: set[str]) -> tuple[str, float]:
    cyclical_score = sum(b.effective_score for b in bundles if b.bucket in cyclical)
    defensive_score = sum(b.effective_score for b in bundles if b.bucket in defensive)
    diff = cyclical_score - defensive_score
    regime = "Cyclical tilt" if diff > 0 else "Defensive tilt"
    confidence = (1 / (1 + math.exp(-abs(diff))))
    confidence = round((confidence - 0.5) * 20, 1)
    return regime, confidence


def what_breaks_view(regime: str) -> list[str]:
    if "Cyclical" in regime:
        return [
            "Credit spreads widening with USD squeeze",
            "Equities breadth rollover + rising rates volatility",
        ]
    return [
        "Equities breadth breakout with easing credit spreads",
        "Sustained drop in rates volatility and USD weakness",
    ]


CONVEXITY_MAP = {
    "Equities (global stocks)": ("No", "Low", "beta"),
    "Credit / Carry (corp & EM bonds)": ("No", "Low", "carry"),
    "Rates / Duration (govt bonds)": ("Partial", "Low", "hedge"),
    "Commodities (energy & metals)": ("No", "Medium", "beta"),
    "Gold (defensive hedge)": ("Yes", "Medium", "hedge"),
    "Crypto / Spec (BTC, ETH)": ("No", "High", "optionality"),
    "USD & FX (US dollar & majors)": ("Partial", "Low", "hedge"),
    "Cash/Sidelines (synthetic)": ("Yes", "Low", "liquidity"),
}

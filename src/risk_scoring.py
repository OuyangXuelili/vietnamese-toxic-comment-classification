
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


BASE_LABEL_RISK = {
    "CLEAN": 0.10,
    "OFFENSIVE": 0.55,
    "HATE": 0.85,
}

POLICY_ADJUSTMENT = {
    "lenient": -0.05,
    "balanced": 0.00,
    "strict": 0.05,
}


def risk_level_from_score(score: float) -> str:
    score = float(score)
    if score < 0.30:
        return "low"
    if score < 0.60:
        return "medium"
    if score < 0.80:
        return "high"
    return "critical"


def compute_risk_score(
    predicted_label: str,
    confidence: float,
    prob_offensive: float = 0.0,
    prob_hate: float = 0.0,
    toxic_spans: Sequence[Dict[str, Any]] | None = None,
    text_changed_by_normalization: bool = False,
    policy_mode: str = "balanced",
) -> Dict[str, Any]:
    label = str(predicted_label).upper()
    conf = float(confidence or 0.0)
    prob_offensive = float(prob_offensive or 0.0)
    prob_hate = float(prob_hate or 0.0)
    spans = list(toxic_spans or [])

    base = BASE_LABEL_RISK.get(label, 0.25)
    confidence_bonus = max(0.0, conf - 0.50) * 0.20
    hate_probability_bonus = prob_hate * 0.08
    offensive_probability_bonus = prob_offensive * 0.04
    span_count_bonus = min(0.18, len(spans) * 0.05)
    span_length_bonus = min(0.07, sum(len(str(s.get("text", ""))) for s in spans) / 200.0)
    normalization_bonus = 0.03 if bool(text_changed_by_normalization) and label != "CLEAN" else 0.0
    policy_adjustment = POLICY_ADJUSTMENT.get(policy_mode, 0.0)

    score = float(np.clip(
        base + confidence_bonus + hate_probability_bonus + offensive_probability_bonus
        + span_count_bonus + span_length_bonus + normalization_bonus + policy_adjustment,
        0.0,
        1.0,
    ))

    components = {
        "base_label_risk": round(base, 4),
        "confidence_bonus": round(confidence_bonus, 4),
        "hate_probability_bonus": round(hate_probability_bonus, 4),
        "offensive_probability_bonus": round(offensive_probability_bonus, 4),
        "span_count_bonus": round(span_count_bonus, 4),
        "span_length_bonus": round(span_length_bonus, 4),
        "normalization_bonus": round(normalization_bonus, 4),
        "policy_adjustment": round(policy_adjustment, 4),
        "final_risk": round(score, 4),
    }
    return {
        "risk_score": score,
        "risk_level": risk_level_from_score(score),
        "risk_components": components,
    }

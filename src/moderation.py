
from __future__ import annotations

from typing import Dict

ACTION_TABLE: Dict[str, Dict[str, str]] = {
    "lenient": {
        "low": "allow",
        "medium": "allow_with_note",
        "high": "warn_user",
        "critical": "send_to_review",
    },
    "balanced": {
        "low": "allow",
        "medium": "warn_user",
        "high": "send_to_review",
        "critical": "block_or_escalate",
    },
    "strict": {
        "low": "allow",
        "medium": "send_to_review",
        "high": "hide_comment",
        "critical": "block_or_escalate",
    },
}


def moderation_decision(risk_level: str, policy_mode: str = "balanced") -> str:
    return ACTION_TABLE.get(policy_mode, ACTION_TABLE["balanced"]).get(str(risk_level), "send_to_review")


def build_reason(
    predicted_label: str,
    confidence: float,
    toxic_span_texts: list[str],
    risk_level: str,
    action: str,
    text_changed_by_normalization: bool = False,
) -> str:
    label = str(predicted_label).upper()
    conf = float(confidence or 0.0)
    if label == "CLEAN" and not toxic_span_texts:
        return "Predicted as CLEAN and no toxic span was found."

    parts = [f"Predicted as {label} with confidence {conf:.2f}."]
    if toxic_span_texts:
        parts.append("Detected toxic span(s): " + ", ".join(toxic_span_texts[:5]) + ".")
    if text_changed_by_normalization:
        parts.append("Normalization changed the text, suggesting noisy or obfuscated input.")
    parts.append(f"Risk level is {risk_level}; suggested action is {action}.")
    return " ".join(parts)

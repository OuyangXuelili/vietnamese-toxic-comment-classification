
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd


def _collect_strings(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_collect_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_collect_strings(v))
    return out


def load_toxic_phrases(project_root: str | Path, max_phrases: int = 2000) -> List[str]:
    root = Path(project_root)
    phrases: List[str] = []

    json_candidates = [
        root / "outputs" / "resources" / "toxic_span_highlighter_rules.json",
        root / "outputs" / "resources" / "span_explainer_config.json",
    ]
    for path in json_candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                phrases.extend(_collect_strings(data))
            except Exception:
                pass

    csv_candidates = [
        root / "outputs" / "resources" / "toxic_phrases_candidates_train.csv",
        root / "outputs" / "resources" / "toxic_phrases_candidates.csv",
    ]
    for path in csv_candidates:
        if path.exists():
            try:
                df = pd.read_csv(path)
                text_cols = [c for c in df.columns if c.lower() in {"phrase", "text", "span", "toxic_phrase"}]
                if not text_cols:
                    text_cols = [df.columns[0]]
                phrases.extend(df[text_cols[0]].dropna().astype(str).tolist())
            except Exception:
                pass

    fallback = [
        "ngu", "đồ ngu", "óc chó", "đồ chó", "cút", "đm", "đmm", "vãi", "khốn",
        "đồ điên", "chó chết", "mất dạy", "rác rưởi", "súc vật", "đồ khùng",
    ]
    phrases.extend(fallback)

    cleaned = []
    for p in phrases:
        p = str(p).strip().lower()
        if 2 <= len(p) <= 40 and not p.endswith(".json") and "/" not in p and "\\" not in p:
            cleaned.append(p)
    return sorted(set(cleaned), key=len, reverse=True)[:max_phrases]


def extract_toxic_spans(text: str, phrases: Sequence[str]) -> List[Dict[str, Any]]:
    text = str(text)
    lower = text.lower()
    spans: List[Dict[str, Any]] = []
    occupied: List[Tuple[int, int]] = []

    for phrase in phrases:
        phrase = str(phrase).strip().lower()
        if not phrase:
            continue
        for m in re.finditer(re.escape(phrase), lower):
            start, end = m.start(), m.end()
            if any(not (end <= s or start >= e) for s, e in occupied):
                continue
            spans.append({
                "start": start,
                "end": end,
                "text": text[start:end],
                "source": "rule",
                "score": 0.85,
            })
            occupied.append((start, end))
    spans.sort(key=lambda s: int(s.get("start", 0)))
    return spans


def highlight_text(text: str, spans: Sequence[Dict[str, Any]]) -> str:
    text = str(text)
    if not spans:
        return html.escape(text)

    parts: List[str] = []
    last = 0
    for sp in sorted(spans, key=lambda x: int(x.get("start", 0))):
        start = max(0, int(sp.get("start", 0)))
        end = min(len(text), int(sp.get("end", start)))
        if start < last or end <= start:
            continue
        parts.append(html.escape(text[last:start]))
        parts.append(
            "<mark style='background-color:#ffdddd;border:1px solid #ff9999;"
            "padding:2px 4px;border-radius:4px;'>"
            + html.escape(text[start:end])
            + "</mark>"
        )
        last = end
    parts.append(html.escape(text[last:]))
    return "".join(parts)

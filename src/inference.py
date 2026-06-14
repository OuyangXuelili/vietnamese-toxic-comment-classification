
from __future__ import annotations

import importlib.util
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception as exc:  # pragma: no cover
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    TRANSFORMERS_IMPORT_ERROR = exc
else:
    TRANSFORMERS_IMPORT_ERROR = None

LABEL_FALLBACK = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
MAX_LENGTH = 128


def _has_transformer_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weight = any((path / name).exists() for name in [
        "pytorch_model.bin", "model.safetensors", "tf_model.h5", "flax_model.msgpack"
    ])
    return has_config and has_weight


def find_phobert_model_dir(project_root: str | Path) -> Path | None:
    root = Path(project_root)
    candidates = [
        root / "outputs" / "models" / "note08b" / "phobert_augmented_mixed",
        root / "outputs" / "models" / "note08b" / "phobert_augmented",
        root / "outputs" / "models" / "note08b" / "phobert_augmented_best",
        root / "outputs" / "models" / "note08b_v2" / "phobert_augmented_mixed",
        root / "outputs" / "models" / "note08b_v2" / "phobert_augmented",
        root / "outputs" / "models" / "phobert_augmented",
        root / "outputs" / "models" / "phobert",
        root / "outputs" / "models" / "phobert_model",
        root / "outputs" / "models" / "best_phobert",
        root / "models" / "phobert",
    ]
    for c in candidates:
        if _has_transformer_weights(c):
            return c

    found: List[Path] = []
    for search_root in [root / "outputs" / "models", root / "models"]:
        if search_root.exists():
            for config_path in search_root.rglob("config.json"):
                d = config_path.parent
                if _has_transformer_weights(d) and "phobert" in str(d).lower():
                    found.append(d)
    if not found:
        return None
    return sorted(found, key=lambda p: ("note08b" not in str(p).lower(), len(str(p))))[0]


def _load_project_normalize_text(project_root: Path):
    norm_path = project_root / "src" / "normalization.py"
    if not norm_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("project_normalization", str(norm_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "normalize_text", None)
        if callable(fn):
            return fn
    except Exception:
        return None
    return None


def fallback_normalize_text(text: str) -> str:
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"(?<=\w)[\.\*_\-]+(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    replacements = {
        "ko": "không", "k": "không", "khong": "không", "hok": "không", "hong": "không",
        "dc": "được", "đc": "được", "j": "gì", "gj": "gì",
        "vl": "vãi", "vcl": "vãi", "vc": "vãi", "dm": "đm", "dmm": "đm",
    }
    return " ".join(replacements.get(tok.lower(), tok) for tok in text.split())


def normalize_text(text: str, project_root: str | Path | None = None) -> str:
    if project_root is not None:
        fn = _load_project_normalize_text(Path(project_root))
        if fn is not None:
            try:
                return str(fn(text))
            except Exception:
                pass
    return fallback_normalize_text(text)


def get_segmenter_name() -> str:
    try:
        import underthesea  # noqa: F401
        return "underthesea"
    except Exception:
        pass
    try:
        import pyvi  # noqa: F401
        return "pyvi"
    except Exception:
        pass
    return "none"


def segment_vietnamese(text: str) -> str:
    text = str(text)
    name = get_segmenter_name()
    if name == "underthesea":
        try:
            from underthesea import word_tokenize
            return str(word_tokenize(text, format="text"))
        except Exception:
            return text
    if name == "pyvi":
        try:
            from pyvi import ViTokenizer
            return str(ViTokenizer.tokenize(text))
        except Exception:
            return text
    return text


def normalize_label(label: str) -> str:
    s = str(label).upper()
    if s in {"0", "LABEL_0"}:
        return "CLEAN"
    if s in {"1", "LABEL_1"}:
        return "OFFENSIVE"
    if s in {"2", "LABEL_2"}:
        return "HATE"
    if "CLEAN" in s:
        return "CLEAN"
    if "OFF" in s:
        return "OFFENSIVE"
    if "HATE" in s:
        return "HATE"
    return s


def load_full_phobert(model_dir: str | Path):
    if TRANSFORMERS_IMPORT_ERROR is not None:
        raise RuntimeError(f"Cannot import torch/transformers: {TRANSFORMERS_IMPORT_ERROR}")
    model_dir = Path(model_dir)
    if not _has_transformer_weights(model_dir):
        raise FileNotFoundError(f"Invalid PhoBERT model folder: {model_dir}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    raw = getattr(model.config, "id2label", None) or LABEL_FALLBACK
    id2label = {int(k): normalize_label(v) for k, v in dict(raw).items()}
    for i, label in LABEL_FALLBACK.items():
        id2label.setdefault(i, label)
    return tokenizer, model, device, id2label


def predict_texts(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    device: str,
    id2label: Dict[int, str],
    project_root: str | Path | None = None,
    normalize_input: bool = True,
    batch_size: int = 16,
    max_length: int = MAX_LENGTH,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    label_order = [id2label.get(i, LABEL_FALLBACK.get(i, f"LABEL_{i}")) for i in range(3)]

    for start in range(0, len(texts), batch_size):
        raw_batch = [str(x) for x in texts[start:start + batch_size]]
        norm_batch = [normalize_text(x, project_root=project_root) if normalize_input else x for x in raw_batch]
        seg_batch = [segment_vietnamese(x) for x in norm_batch]
        enc = tokenizer(seg_batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        for raw_text, norm_text, prob in zip(raw_batch, norm_batch, probs):
            pred_id = int(np.argmax(prob))
            row = {
                "original_text": raw_text,
                "normalized_text": norm_text,
                "text_changed_by_normalization": raw_text.strip() != norm_text.strip(),
                "predicted_label": id2label.get(pred_id, f"LABEL_{pred_id}"),
                "confidence": float(prob[pred_id]),
            }
            for i, label in enumerate(label_order):
                row[f"prob_{str(label).lower()}"] = float(prob[i]) if i < len(prob) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)

from __future__ import annotations

import html
import importlib.util
import json
import os
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib
except Exception as exc:
    joblib = None
    JOBLIB_IMPORT_ERROR = exc
else:
    JOBLIB_IMPORT_ERROR = None

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception as exc:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    TRANSFORMERS_IMPORT_ERROR = exc
else:
    TRANSFORMERS_IMPORT_ERROR = None

APP_TITLE = "Kiểm duyệt bình luận độc hại tiếng Việt"
DEFAULT_PROJECT_ROOT = Path("/content/drive/MyDrive/Deep/vietnamese-toxic-comment-classification")
LABEL_FALLBACK = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
MAX_LENGTH = 128


# =============================================================================
# Paths
# =============================================================================

def find_project_root() -> Path:
    candidates: List[Path] = []
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([Path.cwd(), DEFAULT_PROJECT_ROOT, Path(__file__).resolve().parent])

    for c in candidates:
        if (c / "data").exists() and (c / "outputs").exists():
            return c.resolve()
    return DEFAULT_PROJECT_ROOT


PROJECT_ROOT = find_project_root()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESOURCES_DIR = OUTPUTS_DIR / "resources"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"


# =============================================================================
# Normalization
# =============================================================================

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
        return fn if callable(fn) else None
    except Exception:
        return None


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


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


PROJECT_NORMALIZE_TEXT = _load_project_normalize_text(PROJECT_ROOT)


def normalize_text(text: str) -> str:
    if PROJECT_NORMALIZE_TEXT is not None:
        try:
            return str(PROJECT_NORMALIZE_TEXT(text))
        except Exception:
            return fallback_normalize_text(text)
    return fallback_normalize_text(text)


# =============================================================================
# Word segmentation for PhoBERT
# =============================================================================

@st.cache_resource(show_spinner=False)
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


# =============================================================================
# Model loading
# =============================================================================

def _has_transformer_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weight = any((path / name).exists() for name in [
        "model.safetensors", "pytorch_model.bin", "tf_model.h5", "flax_model.msgpack"
    ])
    return has_config and has_weight


def find_phobert_model_dir(project_root: Path) -> Optional[Path]:
    env_model = os.environ.get("PHOBERT_MODEL_DIR")
    candidates: List[Path] = []
    if env_model:
        candidates.append(Path(env_model))

    candidates.extend([
        project_root / "outputs" / "models" / "note08b" / "phobert_augmented_mixed",
        project_root / "outputs" / "models" / "note08b" / "phobert_augmented",
        project_root / "outputs" / "models" / "note08b" / "phobert_augmented_best",
        project_root / "outputs" / "models" / "note08b_v2" / "phobert_augmented_mixed",
        project_root / "outputs" / "models" / "note08b_v2" / "phobert_augmented",
        project_root / "outputs" / "models" / "phobert_augmented",
        project_root / "outputs" / "models" / "phobert",
        project_root / "outputs" / "models" / "phobert_model",
        project_root / "outputs" / "models" / "best_phobert",
    ])

    for c in candidates:
        if _has_transformer_weights(c):
            return c

    found: List[Path] = []
    for root in [project_root / "outputs" / "models", project_root / "models"]:
        if root.exists():
            for config_path in root.rglob("config.json"):
                d = config_path.parent
                if _has_transformer_weights(d) and "phobert" in str(d).lower():
                    found.append(d)
    if found:
        found = sorted(found, key=lambda p: ("note08b" not in str(p).lower(), len(str(p))))
        return found[0]
    return None


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


@st.cache_resource(show_spinner="Loading full PhoBERT model...")
def load_full_phobert(model_dir_str: str):
    if TRANSFORMERS_IMPORT_ERROR is not None:
        raise RuntimeError(f"Cannot import torch/transformers: {TRANSFORMERS_IMPORT_ERROR}")

    model_dir = Path(model_dir_str)
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

    id2label_raw = getattr(model.config, "id2label", None) or LABEL_FALLBACK
    id2label = {int(k): normalize_label(v) for k, v in dict(id2label_raw).items()}
    for i, fallback in LABEL_FALLBACK.items():
        id2label.setdefault(i, fallback)
    return tokenizer, model, device, id2label


def predict_phobert(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    device: str,
    id2label: Dict[int, str],
    normalize_input: bool = True,
    batch_size: int = 16,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    label_order = [id2label.get(i, LABEL_FALLBACK.get(i, f"LABEL_{i}")) for i in range(3)]

    for start in range(0, len(texts), batch_size):
        raw_batch = [str(x) for x in texts[start:start + batch_size]]
        norm_batch = [normalize_text(x) if normalize_input else x for x in raw_batch]
        seg_batch = [segment_vietnamese(x) for x in norm_batch]
        enc = tokenizer(seg_batch, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        for raw_text, norm_text, prob in zip(raw_batch, norm_batch, probs):
            pred_id = int(np.argmax(prob))
            row: Dict[str, Any] = {
                "original_text": raw_text,
                "normalized_text": norm_text,
                "text_changed_by_normalization": raw_text.strip() != norm_text.strip(),
                "predicted_label": id2label.get(pred_id, f"LABEL_{pred_id}"),
                "confidence": float(prob[pred_id]),
            }
            for i, label in enumerate(label_order):
                row[f"prob_{label.lower()}"] = float(prob[i]) if i < len(prob) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def find_xlmr_model_dir(project_root: Path) -> Optional[Path]:
    candidates = [
        project_root / "outputs" / "models" / "note08" / "xlmr_augmented_balanced_sample",
        project_root / "outputs" / "models" / "note08" / "xlmr_original_balanced_sample",
    ]
    for path in candidates:
        if _has_transformer_weights(path):
            return path
    return None


@st.cache_resource(show_spinner=False)
def load_xlmr_model(project_root_str: str, device: str) -> Dict[str, Any]:
    if TRANSFORMERS_IMPORT_ERROR is not None:
        return {"ok": False, "error": f"Không import được transformers: {TRANSFORMERS_IMPORT_ERROR}"}
    model_dir = find_xlmr_model_dir(Path(project_root_str))
    if model_dir is None:
        return {"ok": False, "error": "Không tìm thấy artifact XLM-R Note 8."}
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()
        model.to(device)
        raw = getattr(model.config, "id2label", None) or LABEL_FALLBACK
        id2label_xlmr = {int(k): normalize_label(v) for k, v in dict(raw).items()}
        for i, label in LABEL_FALLBACK.items():
            id2label_xlmr.setdefault(i, label)
    except Exception as exc:
        detail = str(exc)
        if "paging file" in detail.lower() or "os error 1455" in detail.lower():
            detail = "Máy hiện tại thiếu RAM/pagefile để load model XLM-R đầy đủ."
        return {"ok": False, "error": f"Không load được XLM-R: {detail}"}
    return {
        "ok": True,
        "name": "XLM-R",
        "tokenizer": tokenizer,
        "model": model,
        "id2label": id2label_xlmr,
        "path": str(model_dir),
        "device": device,
    }


def predict_xlmr(
    texts: Sequence[str],
    bundle: Dict[str, Any],
    normalize_input: bool = True,
    batch_size: int = 16,
) -> pd.DataFrame:
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle.get("device", "cpu")
    id2label_xlmr = bundle.get("id2label", LABEL_FALLBACK)
    rows: List[Dict[str, Any]] = []

    for start in range(0, len(texts), batch_size):
        raw_batch = [str(x) for x in texts[start:start + batch_size]]
        norm_batch = [normalize_text(x) if normalize_input else x for x in raw_batch]
        enc = tokenizer(norm_batch, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        for original, normalized, prob in zip(raw_batch, norm_batch, probs):
            rows.append(_prediction_row_from_probs(
                "XLM-R",
                original,
                normalized,
                prob,
                id2label_xlmr,
                "Baseline nghiên cứu Note 8; macro-F1 clean khoảng 0.497, không dùng làm quyết định chính.",
            ))
    return pd.DataFrame(rows)


# =============================================================================
# Lightweight model comparison
# =============================================================================

def _softmax_np(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = arr - np.nanmax(arr, axis=1, keepdims=True)
    exp = np.exp(arr)
    denom = np.sum(exp, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return exp / denom


def _base_prediction_row(model_name: str, original: str, normalized: str, note: str = "") -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "original_text": original,
        "normalized_text": normalized,
        "text_changed_by_normalization": str(original).strip() != str(normalized).strip(),
        "predicted_label": "CLEAN",
        "confidence": np.nan,
        "prob_clean": np.nan,
        "prob_offensive": np.nan,
        "prob_hate": np.nan,
        "comparison_note": note,
    }


def _unavailable_prediction_row(model_name: str, text: str, note: str) -> Dict[str, Any]:
    row = _base_prediction_row(model_name, str(text), str(text), note)
    row["predicted_label"] = "UNAVAILABLE"
    return row


def _prediction_row_from_probs(
    model_name: str,
    original: str,
    normalized: str,
    probs: Sequence[float],
    id2label: Dict[int, str],
    note: str = "",
) -> Dict[str, Any]:
    prob_arr = np.asarray(probs, dtype=np.float64)
    pred_id = int(np.nanargmax(prob_arr))
    label = normalize_label(id2label.get(pred_id, LABEL_FALLBACK.get(pred_id, f"LABEL_{pred_id}")))
    row = _base_prediction_row(model_name, original, normalized, note)
    row.update({
        "predicted_label": label,
        "confidence": float(prob_arr[pred_id]),
    })
    for i in range(3):
        label_name = normalize_label(id2label.get(i, LABEL_FALLBACK.get(i, f"LABEL_{i}")))
        row[f"prob_{label_name.lower()}"] = float(prob_arr[i]) if i < len(prob_arr) else np.nan
    return row


def find_svm_artifacts(project_root: Path) -> Optional[Tuple[Path, Path]]:
    candidates = [
        (
            project_root / "outputs" / "models" / "tfidf_vectorizer.joblib",
            project_root / "outputs" / "models" / "svm_model.joblib",
        ),
        (
            project_root / "outputs" / "models" / "note08" / "tfidf_augmented_mixed_note08.joblib",
            project_root / "outputs" / "models" / "note08" / "svm_augmented_mixed_note08.joblib",
        ),
        (
            project_root / "outputs" / "models" / "note08" / "tfidf_original_note08.joblib",
            project_root / "outputs" / "models" / "note08" / "svm_original_note08.joblib",
        ),
    ]
    for vectorizer_path, model_path in candidates:
        if vectorizer_path.exists() and model_path.exists():
            return vectorizer_path, model_path
    return None


@st.cache_resource(show_spinner=False)
def load_svm_artifacts(project_root_str: str) -> Dict[str, Any]:
    if JOBLIB_IMPORT_ERROR is not None or joblib is None:
        return {"ok": False, "error": f"Không import được joblib: {JOBLIB_IMPORT_ERROR}"}
    artifacts = find_svm_artifacts(Path(project_root_str))
    if artifacts is None:
        return {"ok": False, "error": "Không tìm thấy artifact TF-IDF + SVM."}
    vectorizer_path, model_path = artifacts
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vectorizer = joblib.load(vectorizer_path)
            model = joblib.load(model_path)
    except Exception as exc:
        return {"ok": False, "error": f"Không load được SVM: {exc}"}
    return {
        "ok": True,
        "name": "TF-IDF + SVM",
        "vectorizer": vectorizer,
        "model": model,
        "vectorizer_path": str(vectorizer_path),
        "model_path": str(model_path),
    }


def predict_svm(
    texts: Sequence[str],
    bundle: Dict[str, Any],
    normalize_input: bool = True,
) -> pd.DataFrame:
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]
    raw_batch = [str(x) for x in texts]
    norm_batch = [normalize_text(x) if normalize_input else x for x in raw_batch]
    X = vectorizer.transform(norm_batch)
    pred = model.predict(X)
    classes = [int(c) if str(c).isdigit() else c for c in getattr(model, "classes_", [0, 1, 2])]

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=np.float64)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        probs_by_class = _softmax_np(scores)
    elif hasattr(model, "predict_proba"):
        probs_by_class = np.asarray(model.predict_proba(X), dtype=np.float64)
    else:
        probs_by_class = np.full((len(raw_batch), len(classes)), np.nan)

    rows: List[Dict[str, Any]] = []
    for i, (original, normalized, pred_id_raw) in enumerate(zip(raw_batch, norm_batch, pred)):
        probs = np.full(3, np.nan, dtype=np.float64)
        for j, cls in enumerate(classes):
            try:
                cls_id = int(cls)
            except Exception:
                cls_id = int(pred_id_raw) if str(cls) == str(pred_id_raw) else j
            if 0 <= cls_id < 3 and j < probs_by_class.shape[1]:
                probs[cls_id] = probs_by_class[i, j]
        if np.all(np.isnan(probs)):
            try:
                pred_id = int(pred_id_raw)
            except Exception:
                pred_id = 0
            probs = np.zeros(3, dtype=np.float64)
            probs[pred_id] = 1.0
        rows.append(_prediction_row_from_probs(
            "TF-IDF + SVM",
            original,
            normalized,
            probs,
            LABEL_FALLBACK,
            "Độ tin cậy xấp xỉ từ LinearSVC margin.",
        ))
    return pd.DataFrame(rows)


def find_bilstm_artifact(project_root: Path) -> Optional[Path]:
    candidates = [
        project_root / "outputs" / "models" / "note08b" / "bilstm_augmented_mixed" / "bilstm_best.pt",
        project_root / "outputs" / "models" / "note08b" / "bilstm_original_recheck" / "bilstm_best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


class BiLSTMClassifier(torch.nn.Module if torch is not None else object):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_labels: int,
        dropout: float,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = torch.nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids: Any, lengths: Any) -> Any:
        emb = self.embedding(input_ids)
        lengths_cpu = lengths.detach().cpu().clamp(min=1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(self.dropout(h))


@st.cache_resource(show_spinner=False)
def load_bilstm_artifact(project_root_str: str, device: str) -> Dict[str, Any]:
    if torch is None:
        return {"ok": False, "error": "Không import được torch nên không load được BiLSTM."}
    artifact_path = find_bilstm_artifact(Path(project_root_str))
    if artifact_path is None:
        return {"ok": False, "error": "Không tìm thấy checkpoint BiLSTM Note08b."}
    try:
        checkpoint = torch.load(artifact_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        vocab = checkpoint.get("vocab")
        if vocab is None:
            vocab_path = artifact_path.parent / "vocab.json"
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        config = checkpoint.get("config", {})
        id2label_raw = checkpoint.get("id_to_label", LABEL_FALLBACK)
        id2label_bilstm = {int(k): normalize_label(v) for k, v in dict(id2label_raw).items()}
        for i, label in LABEL_FALLBACK.items():
            id2label_bilstm.setdefault(i, label)

        embedding_weight = state_dict["embedding.weight"]
        hidden_dim = int(config.get("hidden_dim", state_dict["lstm.weight_hh_l0"].shape[1]))
        model = BiLSTMClassifier(
            vocab_size=int(embedding_weight.shape[0]),
            embedding_dim=int(config.get("embedding_dim", embedding_weight.shape[1])),
            hidden_dim=hidden_dim,
            num_layers=int(config.get("num_layers", 2)),
            num_labels=int(state_dict["classifier.bias"].shape[0]),
            dropout=float(config.get("dropout", 0.4)),
            pad_idx=int(vocab.get("<PAD>", vocab.get("[PAD]", 0))),
        )
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
    except Exception as exc:
        return {"ok": False, "error": f"Không load được BiLSTM: {exc}"}
    return {
        "ok": True,
        "name": "BiLSTM",
        "model": model,
        "vocab": vocab,
        "config": config,
        "id2label": id2label_bilstm,
        "path": str(artifact_path),
        "device": device,
    }


def _encode_bilstm_text(text: str, vocab: Dict[str, int], max_len: int) -> Tuple[List[int], int]:
    pad_idx = int(vocab.get("<PAD>", vocab.get("[PAD]", 0)))
    unk_idx = int(vocab.get("<UNK>", vocab.get("[UNK]", 1)))
    tokens = re.findall(r"\w+|[^\w\s]", str(text).lower(), flags=re.UNICODE)
    ids = [int(vocab.get(tok, unk_idx)) for tok in tokens[:max_len]]
    if not ids:
        ids = [unk_idx]
    length = len(ids)
    if length < max_len:
        ids.extend([pad_idx] * (max_len - length))
    return ids, length


def predict_bilstm(
    texts: Sequence[str],
    bundle: Dict[str, Any],
    normalize_input: bool = True,
    batch_size: int = 16,
) -> pd.DataFrame:
    model = bundle["model"]
    vocab = bundle["vocab"]
    config = bundle.get("config", {})
    device = bundle.get("device", "cpu")
    id2label_bilstm = bundle.get("id2label", LABEL_FALLBACK)
    max_len = int(config.get("max_len", 128))
    rows: List[Dict[str, Any]] = []
    raw_texts = [str(x) for x in texts]
    norm_texts = [normalize_text(x) if normalize_input else x for x in raw_texts]

    for start in range(0, len(raw_texts), batch_size):
        raw_batch = raw_texts[start:start + batch_size]
        norm_batch = norm_texts[start:start + batch_size]
        encoded = [_encode_bilstm_text(text, vocab, max_len) for text in norm_batch]
        input_ids = torch.tensor([item[0] for item in encoded], dtype=torch.long, device=device)
        lengths = torch.tensor([item[1] for item in encoded], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids, lengths)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        for original, normalized, prob in zip(raw_batch, norm_batch, probs):
            rows.append(_prediction_row_from_probs("BiLSTM", original, normalized, prob, id2label_bilstm))
    return pd.DataFrame(rows)


def build_model_comparison(
    texts: Sequence[str],
    phobert_df: pd.DataFrame,
    normalize_input: bool,
    batch_size: int,
) -> Tuple[pd.DataFrame, List[str]]:
    frames: List[pd.DataFrame] = []
    warnings_out: List[str] = []

    phobert_view = phobert_df.copy()
    phobert_view.insert(0, "model_name", "PhoBERT")
    phobert_view["comparison_note"] = "Mô hình triển khai chính; macro-F1 clean khoảng 0.666."
    frames.append(phobert_view)

    svm_bundle = load_svm_artifacts(str(PROJECT_ROOT))
    if svm_bundle.get("ok"):
        try:
            frames.append(predict_svm(texts, svm_bundle, normalize_input))
        except Exception as exc:
            warnings_out.append(f"SVM không chạy được: {exc}")
    else:
        warnings_out.append(str(svm_bundle.get("error", "SVM chưa sẵn sàng.")))

    bilstm_bundle = load_bilstm_artifact(str(PROJECT_ROOT), str(device))
    if bilstm_bundle.get("ok"):
        try:
            frames.append(predict_bilstm(texts, bilstm_bundle, normalize_input, batch_size))
        except Exception as exc:
            warnings_out.append(f"BiLSTM không chạy được: {exc}")
    else:
        msg = str(bilstm_bundle.get("error", "BiLSTM chưa sẵn sàng."))
        warnings_out.append(msg)
        frames.append(pd.DataFrame([_unavailable_prediction_row("BiLSTM", texts[0], msg)]))

    xlmr_bundle = load_xlmr_model(str(PROJECT_ROOT), str(device))
    if xlmr_bundle.get("ok"):
        try:
            frames.append(predict_xlmr(texts, xlmr_bundle, normalize_input, batch_size))
        except Exception as exc:
            msg = f"XLM-R không chạy được: {exc}"
            warnings_out.append(msg)
            frames.append(pd.DataFrame([_unavailable_prediction_row("XLM-R", texts[0], msg)]))
    else:
        msg = str(xlmr_bundle.get("error", "XLM-R chưa sẵn sàng."))
        warnings_out.append(msg)
        frames.append(pd.DataFrame([_unavailable_prediction_row("XLM-R", texts[0], msg)]))

    if not frames:
        return pd.DataFrame(), warnings_out
    return pd.concat(frames, ignore_index=True, sort=False), warnings_out


# =============================================================================
# Toxic span explanation
# =============================================================================

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


@st.cache_data(show_spinner=False)
def load_toxic_phrases(project_root_str: str) -> List[str]:
    project_root = Path(project_root_str)
    phrases: List[str] = []
    priority_phrases = [
        "ngu", "vl", "vcl", "vc", "dm", "dmm", "đm", "đmm", "cút", "vãi", "khốn",
        "con chó", "óc chó", "đồ chó", "chó chết", "chó", "mất dạy", "rác rưởi", "rác", "súc vật",
        "đồ ngu", "đồ khùng", "đồ điên", "thằng ngu", "con ngu",
    ]

    for path in [
        project_root / "outputs" / "resources" / "toxic_span_highlighter_rules.json",
        project_root / "outputs" / "resources" / "span_explainer_config.json",
    ]:
        if path.exists():
            try:
                phrases.extend(_collect_strings(json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                pass

    for path in [
        project_root / "outputs" / "resources" / "toxic_phrases_candidates_train.csv",
        project_root / "outputs" / "resources" / "toxic_phrases_candidates.csv",
    ]:
        if path.exists():
            try:
                df = pd.read_csv(path)
                cols = [c for c in df.columns if c.lower() in {"phrase", "text", "span", "toxic_phrase"}]
                col = cols[0] if cols else df.columns[0]
                phrases.extend(df[col].dropna().astype(str).tolist())
            except Exception:
                pass

    phrases.extend(priority_phrases)
    cleaned = []
    for p in phrases:
        p = str(p).strip().lower()
        if 2 <= len(p) <= 40 and "/" not in p and not p.endswith(".json"):
            cleaned.append(p)

    priority_cleaned = []
    for p in priority_phrases:
        p = p.strip().lower()
        if p in cleaned and p not in priority_cleaned:
            priority_cleaned.append(p)

    ranked = sorted(set(cleaned) - set(priority_cleaned), key=len, reverse=True)
    return (priority_cleaned + ranked)[:2000]


def _phrase_pattern(phrase: str) -> str:
    parts: List[str] = []
    for ch in str(phrase):
        if ch.isspace():
            parts.append(r"\s+")
        elif ch.isalnum():
            parts.append(re.escape(ch) + r"+")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


def extract_toxic_spans(text: str, phrases: Sequence[str]) -> List[Dict[str, Any]]:
    text = str(text)
    lower = text.lower()
    spans: List[Dict[str, Any]] = []
    occupied: List[Tuple[int, int]] = []
    for phrase in phrases:
        phrase = str(phrase).strip().lower()
        if not phrase:
            continue
        pattern = rf"(?<!\w){_phrase_pattern(phrase)}(?!\w)"
        for m in re.finditer(pattern, lower, flags=re.UNICODE):
            s, e = m.start(), m.end()
            if any(not (e <= os_ or s >= oe) for os_, oe in occupied):
                continue
            spans.append({"start": s, "end": e, "text": text[s:e], "source": "rule", "score": 0.85})
            occupied.append((s, e))
    spans.sort(key=lambda x: int(x["start"]))
    return spans


def _collapse_repeats(text: str) -> str:
    return re.sub(r"(.)\1{1,}", r"\1", str(text).lower())


def extract_signal_spans(original: str, normalized: str, row: Dict[str, Any]) -> List[Dict[str, Any]]:
    label = str(row.get("predicted_label", "CLEAN")).upper()
    confidence = float(row.get("confidence", 0.0) or 0.0)
    if label == "CLEAN" or confidence < 0.40:
        return []

    text = str(original)
    risky_stems = {
        "ngu", "cho", "rac", "cut", "khon", "suc", "vat", "dien", "dm", "vl", "vcl",
        "chui", "bien", "matday", "occho",
    }
    stopwords = {
        "toi", "ban", "minh", "may", "tao", "nay", "kia", "thi", "la", "ma", "va", "co",
        "khong", "duoc", "cho", "cai", "nguoi", "noi", "that", "qua", "roi", "nhe",
    }
    spans: List[Dict[str, Any]] = []
    occupied: List[Tuple[int, int]] = []

    def add_span(start: int, end: int, score: float = 0.35) -> None:
        if end <= start or any(not (end <= os_ or start >= oe) for os_, oe in occupied):
            return
        spans.append({"start": start, "end": end, "text": text[start:end], "source": "model_signal", "score": score})
        occupied.append((start, end))

    for match in re.finditer(r"\w+", text, flags=re.UNICODE):
        token = match.group(0)
        folded = strip_accents(token).lower()
        compact = _collapse_repeats(folded)
        compact_no_space = re.sub(r"\W+", "", compact)
        repeated = bool(re.search(r"(.)\1{2,}", token.lower()))
        normalized_token = normalize_text(token)
        changed = token != normalized_token and len(token) >= 3
        stem_hit = any(stem in compact_no_space for stem in risky_stems)
        if repeated or changed or stem_hit:
            add_span(match.start(), match.end(), 0.45 if stem_hit else 0.32)

    if spans:
        return spans[:4]

    if confidence >= 0.50:
        content_tokens = []
        for match in re.finditer(r"\w+", text, flags=re.UNICODE):
            token = match.group(0)
            folded = strip_accents(token).lower()
            if len(folded) >= 4 and folded not in stopwords:
                content_tokens.append(match)
        for match in content_tokens[:2]:
            add_span(match.start(), match.end(), 0.25)

    return spans[:3]


def highlight_text(text: str, spans: Sequence[Dict[str, Any]]) -> str:
    text = str(text)
    if not spans:
        return html.escape(text)
    parts: List[str] = []
    last = 0
    for sp in sorted(spans, key=lambda x: int(x.get("start", 0))):
        s = max(0, int(sp.get("start", 0)))
        e = min(len(text), int(sp.get("end", s)))
        if s < last or e <= s:
            continue
        parts.append(html.escape(text[last:s]))
        parts.append(
            "<mark style='background-color:#ffdddd;border:1px solid #ff9999;"
            "padding:2px 4px;border-radius:4px;'>" + html.escape(text[s:e]) + "</mark>"
        )
        last = e
    parts.append(html.escape(text[last:]))
    return "".join(parts)


# =============================================================================
# Risk scoring
# =============================================================================

def compute_risk(row: Dict[str, Any], spans: Sequence[Dict[str, Any]], policy_mode: str = "balanced") -> Dict[str, Any]:
    label = str(row.get("predicted_label", "CLEAN")).upper()
    conf = float(row.get("confidence", 0.0) or 0.0)
    prob_hate = float(row.get("prob_hate", 0.0) or 0.0)
    prob_off = float(row.get("prob_offensive", 0.0) or 0.0)
    changed = bool(row.get("text_changed_by_normalization", False))

    base = {"CLEAN": 0.10, "OFFENSIVE": 0.55, "HATE": 0.85}.get(label, 0.25)
    confidence_bonus = max(0.0, conf - 0.50) * 0.20
    hate_bonus = prob_hate * 0.08
    offensive_bonus = prob_off * 0.04
    span_bonus = min(0.18, len(spans) * 0.05)
    span_length_bonus = min(0.07, sum(len(str(sp.get("text", ""))) for sp in spans) / 200.0)
    normalization_bonus = 0.03 if changed and label != "CLEAN" else 0.0
    policy_adjustment = 0.05 if policy_mode == "strict" else (-0.05 if policy_mode == "lenient" else 0.0)

    risk = float(np.clip(base + confidence_bonus + hate_bonus + offensive_bonus + span_bonus + span_length_bonus + normalization_bonus + policy_adjustment, 0, 1))
    level = "low" if risk < 0.30 else "medium" if risk < 0.60 else "high" if risk < 0.80 else "critical"
    components = {
        "base_label_risk": round(base, 4),
        "confidence_bonus": round(confidence_bonus, 4),
        "hate_probability_bonus": round(hate_bonus, 4),
        "offensive_probability_bonus": round(offensive_bonus, 4),
        "span_count_bonus": round(span_bonus, 4),
        "span_length_bonus": round(span_length_bonus, 4),
        "normalization_bonus": round(normalization_bonus, 4),
        "policy_adjustment": round(policy_adjustment, 4),
        "final_risk": round(risk, 4),
    }
    return {"risk_score": risk, "risk_level": level, "risk_components": components}


def moderation_action(risk_level: str, policy_mode: str = "balanced") -> str:
    table = {
        "lenient": {"low": "allow", "medium": "allow_with_note", "high": "warn_user", "critical": "send_to_review"},
        "balanced": {"low": "allow", "medium": "warn_user", "high": "send_to_review", "critical": "block_or_escalate"},
        "strict": {"low": "allow", "medium": "send_to_review", "high": "hide_comment", "critical": "block_or_escalate"},
    }
    return table.get(policy_mode, table["balanced"]).get(risk_level, "send_to_review")


def reason_text(row: Dict[str, Any], spans: Sequence[Dict[str, Any]], risk: Dict[str, Any], action: str) -> str:
    label = row.get("predicted_label", "CLEAN")
    conf = float(row.get("confidence", 0.0) or 0.0)
    label_display = str(label).upper()
    risk_vi = {"low": "thấp", "medium": "trung bình", "high": "cao", "critical": "nghiêm trọng"}.get(str(risk.get("risk_level", "")).lower(), str(risk.get("risk_level", "")))
    action_vi = {
        "allow": "cho phép",
        "allow_with_note": "cho phép kèm lưu ý",
        "warn_user": "cảnh báo người dùng",
        "send_to_review": "chuyển duyệt thủ công",
        "hide_comment": "ẩn bình luận",
        "block_or_escalate": "chặn hoặc chuyển duyệt",
    }.get(str(action), str(action))
    if label == "CLEAN" and not spans:
        return "Mô hình dự đoán nhãn CLEAN và không phát hiện cụm độc hại rõ ràng."
    bits = [f"Mô hình dự đoán nhãn {label_display} với độ tin cậy {conf:.2f}."]
    if spans:
        span_text = ", ".join(str(s.get("text", "")) for s in spans[:5])
        if all(str(s.get("source", "")) == "model_signal" for s in spans):
            bits.append(f"Chưa có toxic span rule chắc chắn; đánh dấu {len(spans)} tín hiệu gợi ý: {span_text}.")
        else:
            bits.append(f"Phát hiện {len(spans)} cụm độc hại: {span_text}.")
    if row.get("text_changed_by_normalization", False):
        bits.append("Chuẩn hóa đã thay đổi đầu vào, cho thấy bình luận có thể chứa nhiễu, teencode hoặc cách viết né lọc.")
    bits.append(f"Mức rủi ro {risk_vi}; hành động đề xuất: {action_vi}.")
    return " ".join(bits)


# =============================================================================
# Robustness playground noise
# =============================================================================

def add_repeated_char_noise(text: str) -> str:
    out = []
    vowels = "aeiouyăâêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịđ"
    for i, ch in enumerate(str(text)):
        out.append(ch)
        if ch.lower() in vowels and i % 5 == 0:
            out.append(ch * 2)
    return "".join(out)


def add_special_mask_noise(text: str) -> str:
    return re.sub(r"(?<=\w)([aeiouăâêôơư])(?=\w)", ".", str(text), flags=re.IGNORECASE)


def add_teencode_noise(text: str) -> str:
    repl = {"không": "ko", "được": "dc", "gì": "j", "vãi": "vl", "biết": "biet", "mày": "may", "tao": "t", "người": "nguoi"}
    return " ".join(repl.get(tok.lower(), tok) for tok in str(text).split())


def apply_noise(text: str, noise_type: str) -> str:
    if noise_type == "no_accent":
        return strip_accents(text)
    if noise_type == "repeated_char":
        return add_repeated_char_noise(text)
    if noise_type == "special_mask":
        return add_special_mask_noise(text)
    if noise_type == "teencode":
        return add_teencode_noise(text)
    if noise_type == "mixed_noise":
        return add_teencode_noise(add_special_mask_noise(add_repeated_char_noise(strip_accents(text))))
    return text


# =============================================================================
# End-to-end moderation
# =============================================================================

def moderate_texts(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    device: str,
    id2label: Dict[int, str],
    phrases: Sequence[str],
    normalize_input: bool,
    policy_mode: str,
    batch_size: int,
) -> pd.DataFrame:
    pred_df = predict_phobert(texts, tokenizer, model, device, id2label, normalize_input, batch_size)
    rows = []
    for _, row in pred_df.iterrows():
        row_dict = row.to_dict()
        original = str(row_dict["original_text"])
        normalized = str(row_dict.get("normalized_text", ""))
        original_spans = extract_toxic_spans(original, phrases)
        normalized_spans = []
        if normalized and normalized != original:
            normalized_spans = extract_toxic_spans(normalized, phrases)
        signal_spans = []
        if not original_spans and not normalized_spans:
            signal_spans = extract_signal_spans(original, normalized, row_dict)

        spans = original_spans if original_spans else normalized_spans if normalized_spans else signal_spans
        span_source = "original" if original_spans else "normalized" if normalized_spans else "model_signal" if signal_spans else "none"
        risk_spans = original_spans if original_spans else normalized_spans
        risk = compute_risk(row_dict, risk_spans, policy_mode)
        action = moderation_action(risk["risk_level"], policy_mode)
        row_dict.update({
            "toxic_spans_json": json.dumps(spans, ensure_ascii=False),
            "toxic_spans_original_json": json.dumps(original_spans, ensure_ascii=False),
            "toxic_spans_normalized_json": json.dumps(normalized_spans, ensure_ascii=False),
            "signal_spans_json": json.dumps(signal_spans, ensure_ascii=False),
            "toxic_span_count_original": len(original_spans),
            "toxic_span_count_normalized": len(normalized_spans),
            "toxic_span_count": len(spans),
            "toxic_span_texts": ", ".join(str(s.get("text", "")) for s in spans),
            "toxic_span_source": span_source,
            "highlighted_text": highlight_text(original, original_spans if original_spans else signal_spans),
            "highlighted_normalized_text": highlight_text(normalized, normalized_spans),
            "risk_score": risk["risk_score"],
            "risk_level": risk["risk_level"],
            "risk_components_json": json.dumps(risk["risk_components"], ensure_ascii=False),
            "moderation_action": action,
            "explanation": reason_text(row_dict, spans, risk, action),
        })
        rows.append(row_dict)
    return pd.DataFrame(rows)


# =============================================================================
# UI
# =============================================================================

st.set_page_config(page_title=APP_TITLE, page_icon="shield", layout="wide")


def inject_console_theme() -> None:
    st.markdown(
        """
<style>
:root {
  --ink: #101828;
  --muted: #667085;
  --line: #e4e7ec;
  --panel: #ffffff;
  --surface: #f6f8fb;
  --navy: #101828;
  --accent: #b42318;
  --blue: #175cd3;
}
.stApp {
  background: linear-gradient(180deg, #fbfcfe 0%, #f6f8fb 100%);
}
[data-testid="stHeader"] {
  background: rgba(251, 252, 254, 0.9);
  border-bottom: 1px solid rgba(228, 231, 236, 0.72);
  backdrop-filter: blur(10px);
}
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu {
  visibility: hidden;
  height: 0;
}
[data-testid="stSidebar"] {
  background: #0f172a;
}
[data-testid="stSidebar"] * {
  color: #eef2ff;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] p {
  color: #dbe3f4;
}
[data-testid="stSidebar"] code {
  color: #0f172a !important;
  white-space: nowrap;
}
[data-testid="stSidebar"] pre,
[data-testid="stSidebar"] code,
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] input {
  color: #0f172a !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: #ffffff !important;
  border-color: rgba(203, 213, 225, 0.72) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
  color: #0f172a !important;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  color: #f8fafc !important;
  font-weight: 760 !important;
}
[data-testid="stSidebar"] [data-testid="stCheckbox"] p,
[data-testid="stSidebar"] [data-testid="stToggle"] p {
  color: #eef2ff !important;
}
.sidebar-card {
  border: 1px solid rgba(148, 163, 184, 0.24);
  background: rgba(15, 23, 42, 0.82);
  border-radius: 8px;
  padding: 0.85rem;
  margin: 0.75rem 0 0.9rem;
}
.sidebar-kicker {
  color: #93c5fd !important;
  font-size: 0.72rem;
  font-weight: 850;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 0.55rem;
}
.sidebar-row {
  display: flex;
  justify-content: space-between;
  gap: 0.65rem;
  border-top: 1px solid rgba(148, 163, 184, 0.16);
  padding: 0.58rem 0;
}
.sidebar-row:first-of-type {
  border-top: 0;
  padding-top: 0;
}
.sidebar-row:last-child {
  padding-bottom: 0;
}
.sidebar-row .name {
  color: #cbd5e1 !important;
  font-size: 0.78rem;
  line-height: 1.35;
}
.sidebar-row .value {
  color: #ffffff !important;
  font-size: 0.78rem;
  font-weight: 800;
  line-height: 1.35;
  text-align: right;
}
.sidebar-chipline {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin-top: 0.55rem;
}
.sidebar-chip {
  border: 1px solid rgba(191, 219, 254, 0.25);
  background: rgba(30, 41, 59, 0.86);
  border-radius: 999px;
  color: #e0f2fe !important;
  font-size: 0.72rem;
  font-weight: 760;
  padding: 0.22rem 0.48rem;
}
.sidebar-note {
  color: #94a3b8 !important;
  font-size: 0.78rem;
  line-height: 1.45;
  margin-top: 0.55rem;
}
.block-container {
  max-width: 1280px;
  padding-top: 3.2rem;
  padding-bottom: 4rem;
}
h1, h2, h3 {
  letter-spacing: 0;
}
.hero {
  border-bottom: 1px solid var(--line);
  margin-bottom: 1rem;
  padding-bottom: 1.25rem;
}
.eyebrow {
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 850;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0.55rem;
}
.hero-title {
  color: var(--ink);
  font-size: clamp(2rem, 4vw, 3.25rem);
  font-weight: 850;
  line-height: 1.06;
  margin: 0;
  max-width: 940px;
}
.hero-copy {
  color: var(--muted);
  font-size: 1.02rem;
  line-height: 1.55;
  margin-top: 0.9rem;
  max-width: 800px;
}
.status-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem;
  margin: 1rem 0 1.15rem;
}
.status-item {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  min-height: 76px;
  padding: 0.78rem 0.9rem;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
.status-label {
  color: var(--muted);
  font-size: 0.76rem;
  font-weight: 760;
  text-transform: uppercase;
  margin-bottom: 0.32rem;
}
.status-value {
  color: var(--ink);
  font-size: 1.05rem;
  font-weight: 820;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 1.05rem;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
.section-title {
  color: var(--ink);
  font-size: 1.18rem;
  font-weight: 840;
  margin: 0 0 0.22rem;
}
.section-copy {
  color: var(--muted);
  font-size: 0.92rem;
  line-height: 1.48;
  margin: 0 0 0.85rem;
}
.workflow-card {
  background: #ffffff;
  border: 1px solid var(--line);
  border-left: 4px solid #ef4444;
  border-radius: 8px;
  padding: 0.85rem 1rem;
  margin: 0.2rem 0 1rem;
}
.workflow-title {
  color: var(--ink);
  font-size: 0.92rem;
  font-weight: 850;
  margin-bottom: 0.24rem;
}
.workflow-copy {
  color: var(--muted);
  font-size: 0.88rem;
  line-height: 1.48;
  margin: 0;
}
.workflow-output {
  color: #344054;
  font-size: 0.82rem;
  font-weight: 760;
  margin-top: 0.55rem;
}
.decision {
  background: #111827;
  border: 1px solid #1f2937;
  border-radius: 8px;
  color: #ffffff;
  min-height: 276px;
  padding: 1rem;
}
.decision .muted {
  color: #cbd5e1;
  line-height: 1.5;
}
.decision-label {
  display: inline-flex;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 850;
  padding: 0.28rem 0.62rem;
  text-transform: uppercase;
  margin-bottom: 0.8rem;
}
.decision-action {
  color: #ffffff;
  font-size: 2rem;
  font-weight: 850;
  line-height: 1.1;
  margin-bottom: 0.72rem;
  overflow-wrap: anywhere;
}
.badge-clean { background: #ecfdf3; color: #067647; border: 1px solid #abefc6; }
.badge-offensive { background: #fffaeb; color: #b54708; border: 1px solid #fedf89; }
.badge-hate { background: #fff1f0; color: #b42318; border: 1px solid #fecdca; }
.badge-unknown { background: #f2f4f7; color: #344054; border: 1px solid #d0d5dd; }
.risk-low { background: #ecfdf3; color: #067647; border: 1px solid #abefc6; }
.risk-medium { background: #fffaeb; color: #b54708; border: 1px solid #fedf89; }
.risk-high { background: #fff4ed; color: #c4320a; border: 1px solid #f9dbaf; }
.risk-critical { background: #fff1f0; color: #b42318; border: 1px solid #fecdca; }
.score-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.6rem;
  margin-top: 0.8rem;
}
.score-box {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 8px;
  padding: 0.65rem;
}
.score-box .label {
  color: #cbd5e1;
  font-size: 0.74rem;
  font-weight: 760;
  text-transform: uppercase;
}
.score-box .value {
  color: #ffffff;
  font-size: 1.28rem;
  font-weight: 850;
  margin-top: 0.2rem;
  overflow-wrap: anywhere;
}
.highlight-box {
  background: #fcfcfd;
  border: 1px solid var(--line);
  border-radius: 8px;
  font-size: 1.08rem;
  line-height: 1.7;
  min-height: 58px;
  padding: 0.9rem 1rem;
}
.explain-box {
  background: #f8fafc;
  border: 1px solid var(--line);
  border-left: 4px solid var(--blue);
  border-radius: 8px;
  color: #344054;
  line-height: 1.5;
  padding: 0.85rem 1rem;
}
.prob-row {
  margin: 0.62rem 0;
}
.prob-head {
  color: var(--ink);
  display: flex;
  font-size: 0.88rem;
  font-weight: 760;
  justify-content: space-between;
  margin-bottom: 0.28rem;
}
.prob-track {
  background: #eef2f6;
  border-radius: 999px;
  height: 10px;
  overflow: hidden;
}
.prob-fill {
  background: linear-gradient(90deg, #175cd3, #b42318);
  height: 100%;
}
div[data-testid="stButton"] > button {
  border-radius: 8px;
  font-weight: 850;
  min-height: 2.8rem;
}
div[data-testid="stTextArea"] textarea {
  background: #ffffff;
  border: 1px solid #d0d5dd;
  border-radius: 8px;
  font-size: 1rem;
}
div[data-testid="stTabs"] button {
  font-weight: 760;
}
@media (max-width: 900px) {
  .block-container { padding-top: 2.8rem; }
  .status-strip, .score-grid { grid-template-columns: 1fr 1fr; }
  .hero-title { font-size: 2.05rem; line-height: 1.12; }
}
@media (max-width: 560px) {
  .status-strip {
    grid-template-columns: 1fr 1fr;
    gap: 0.55rem;
  }
  .score-grid { grid-template-columns: 1fr; }
  .status-item {
    min-height: 68px;
    padding: 0.65rem 0.7rem;
  }
  .status-label {
    font-size: 0.68rem;
  }
  .status-value {
    font-size: 0.9rem;
  }
  .status-value {
    white-space: normal;
    overflow-wrap: anywhere;
  }
  .hero-title { font-size: 1.58rem; }
  .hero-copy { font-size: 0.94rem; }
  .decision-action { font-size: 1.45rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _label_class(label: str) -> str:
    label = str(label).lower()
    if "clean" in label:
        return "badge-clean"
    if "offensive" in label:
        return "badge-offensive"
    if "hate" in label:
        return "badge-hate"
    return "badge-unknown"


def _risk_class(level: str) -> str:
    level = str(level).lower()
    return f"risk-{level}" if level in {"low", "medium", "high", "critical"} else "badge-unknown"


def _display_action(action: str) -> str:
    actions = {
        "allow": "Cho phép",
        "allow_with_note": "Cho phép kèm lưu ý",
        "warn_user": "Cảnh báo người dùng",
        "send_to_review": "Chuyển duyệt thủ công",
        "hide_comment": "Ẩn bình luận",
        "block_or_escalate": "Chặn hoặc chuyển duyệt",
    }
    return actions.get(str(action), str(action).replace("_", " ").strip().title())


def _display_label(label: str) -> str:
    label = str(label).upper()
    if label == "UNAVAILABLE":
        return "--"
    return label if label in {"CLEAN", "OFFENSIVE", "HATE"} else str(label)


def _display_risk(level: str) -> str:
    levels = {
        "low": "Thấp",
        "medium": "Trung bình",
        "high": "Cao",
        "critical": "Nghiêm trọng",
    }
    return levels.get(str(level).lower(), str(level))


def _display_policy(policy: str) -> str:
    policies = {
        "balanced": "Cân bằng",
        "strict": "Nghiêm ngặt",
        "lenient": "Nới lỏng",
    }
    return policies.get(str(policy), str(policy))


def _display_noise(noise: str) -> str:
    labels = {
        "original": "Gốc",
        "no_accent": "Bỏ dấu",
        "no_accent_normalized": "Bỏ dấu + chuẩn hóa",
        "repeated_char": "Lặp ký tự",
        "repeated_char_normalized": "Lặp ký tự + chuẩn hóa",
        "special_mask": "Che ký tự",
        "special_mask_normalized": "Che ký tự + chuẩn hóa",
        "teencode": "Teencode",
        "teencode_normalized": "Teencode + chuẩn hóa",
        "mixed_noise": "Nhiễu tổng hợp",
        "mixed_noise_normalized": "Nhiễu tổng hợp + chuẩn hóa",
    }
    return labels.get(str(noise), str(noise))


def _runtime_label(device: str) -> str:
    value = str(device).lower()
    if value == "cuda":
        return "GPU/CUDA"
    if value == "mps":
        return "Apple GPU"
    return "Local CPU"


DEMO_CASES = {
    "Trung tính": "Mình không đồng ý với quan điểm này nhưng vẫn tôn trọng bạn.",
    "Công kích": "bạn thật ngu ngốc",
    "Né lọc": "đồ raaac này",
    "Thù ghét": "đồ súc vật biến khỏi đây",
}


def _render_workflow_card(title: str, copy: str, output: str) -> None:
    st.markdown(
        f"""
<div class="workflow-card">
  <div class="workflow-title">{html.escape(title)}</div>
  <p class="workflow-copy">{html.escape(copy)}</p>
  <div class="workflow-output">Đầu ra: {html.escape(output)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _review_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame()
    if "original_text" in df.columns:
        out["Bình luận"] = df["original_text"]
    if "predicted_label" in df.columns:
        out["Nhãn"] = df["predicted_label"].map(_display_label)
    if "confidence" in df.columns:
        out["Độ tin cậy"] = df["confidence"].map(lambda v: f"{float(v):.3f}")
    if "risk_score" in df.columns:
        out["Điểm rủi ro"] = df["risk_score"].map(lambda v: f"{float(v):.3f}")
    if "risk_level" in df.columns:
        out["Mức rủi ro"] = df["risk_level"].map(_display_risk)
    if "moderation_action" in df.columns:
        out["Hành động"] = df["moderation_action"].map(_display_action)
    if "toxic_span_texts" in df.columns:
        out["Bằng chứng"] = df["toxic_span_texts"].fillna("").replace("", "Không phát hiện")
    return out


def _model_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame()
    model_names = df.get("model_name", pd.Series([""] * len(df))).astype(str)
    out["Mô hình"] = model_names
    status = df.get("predicted_label", pd.Series([""] * len(df))).astype(str).str.upper().map(lambda v: "Không chạy" if v == "UNAVAILABLE" else "Đã chạy")
    out["Trạng thái"] = status
    role_map = {
        "PhoBERT": "Quyết định chính",
        "TF-IDF + SVM": "Đối chiếu phụ",
        "BiLSTM": "Đối chiếu phụ",
        "XLM-R": "Baseline nghiên cứu",
    }
    out["Vai trò"] = model_names.map(lambda name: role_map.get(name, "Đối chiếu"))
    out["Nhãn"] = df.get("predicted_label", pd.Series([""] * len(df))).map(_display_label)
    if "confidence" in df.columns:
        out["Độ tin cậy"] = df["confidence"].map(lambda v: "--" if pd.isna(v) else f"{float(v):.3f}")
    for source, target in [("prob_clean", "CLEAN"), ("prob_offensive", "OFFENSIVE"), ("prob_hate", "HATE")]:
        if source in df.columns:
            out[target] = df[source].map(lambda v: "--" if pd.isna(v) else f"{float(v):.1%}")
    if "comparison_note" in df.columns:
        out["Ghi chú"] = df["comparison_note"].fillna("")
    return out


@st.cache_data(show_spinner=False)
def _load_research_context(project_root_str: str) -> Dict[str, pd.DataFrame]:
    project_root = Path(project_root_str)
    out: Dict[str, pd.DataFrame] = {}

    multilingual_path = project_root / "outputs" / "results" / "note08" / "multilingual_vs_phobert_comparison.csv"
    if multilingual_path.exists():
        df = pd.read_csv(multilingual_path)
        df = df.rename(columns={
            "model_name": "Mô hình",
            "source": "Nguồn",
            "accuracy": "Accuracy",
            "macro_f1": "Macro F1",
            "macro_precision": "Macro Precision",
            "macro_recall": "Macro Recall",
        })
        name_map = {
            "svm_baseline_note02": "TF-IDF + SVM",
            "bilstm_baseline_note03": "BiLSTM",
            "phobert_baseline_note04": "PhoBERT",
            "xlmr_note08_original": "XLM-R đa ngôn ngữ",
            "xlmr_note08_augmented_mixed": "XLM-R đa ngôn ngữ + augmentation",
        }
        if "Mô hình" in df.columns:
            df["Mô hình"] = df["Mô hình"].map(lambda v: name_map.get(str(v), str(v)))
        for col in ["Accuracy", "Macro F1", "Macro Precision", "Macro Recall"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
        out["multilingual"] = df

    augmentation_path = project_root / "outputs" / "results" / "note08" / "augmentation_vs_original_comparison.csv"
    if augmentation_path.exists():
        df = pd.read_csv(augmentation_path)
        focus = df[
            (df.get("noise_type", "") == "mixed_noise")
            & (df.get("eval_text_variant", "") == "before_norm")
        ].copy()
        if focus.empty:
            focus = df.head(12).copy()
        focus = focus.rename(columns={
            "model_name": "Mô hình",
            "noise_type": "Nhiễu",
            "eval_text_variant": "Biến thể đánh giá",
            "original_macro_f1": "F1 gốc",
            "augmented_macro_f1": "F1 augmentation",
            "augmentation_gain_macro_f1": "Tăng F1",
            "original_accuracy": "Accuracy gốc",
            "augmented_accuracy": "Accuracy augmentation",
            "augmentation_gain_accuracy": "Tăng accuracy",
        })
        if "Mô hình" in focus.columns:
            focus["Mô hình"] = focus["Mô hình"].map({
                "svm_note08": "TF-IDF + SVM",
                "xlmr_note08": "XLM-R đa ngôn ngữ",
            }).fillna(focus["Mô hình"])
        for col in ["F1 gốc", "F1 augmentation", "Tăng F1", "Accuracy gốc", "Accuracy augmentation", "Tăng accuracy"]:
            if col in focus.columns:
                focus[col] = pd.to_numeric(focus[col], errors="coerce").round(4)
        keep = [c for c in ["Mô hình", "Nhiễu", "Biến thể đánh giá", "F1 gốc", "F1 augmentation", "Tăng F1", "Accuracy gốc", "Accuracy augmentation", "Tăng accuracy"] if c in focus.columns]
        out["augmentation"] = focus[keep]

    vihos_metrics_path = project_root / "outputs" / "results" / "vihos_span_baseline_metrics.csv"
    if vihos_metrics_path.exists():
        df = pd.read_csv(vihos_metrics_path)
        df = df.rename(columns={
            "method": "Phương pháp",
            "split": "Split",
            "comment_precision": "Comment precision",
            "comment_recall": "Comment recall",
            "comment_f1": "Comment F1",
            "char_precision": "Char precision",
            "char_recall": "Char recall",
            "char_f1": "Char F1",
            "rows_per_sec": "Dòng/giây",
        })
        keep = [c for c in ["Phương pháp", "Split", "Comment precision", "Comment recall", "Comment F1", "Char precision", "Char recall", "Char F1", "Dòng/giây"] if c in df.columns]
        for col in keep:
            if col not in {"Phương pháp", "Split"}:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
        out["vihos_metrics"] = df[keep]

    coverage_path = project_root / "outputs" / "results" / "vihos_phrase_coverage_on_vihsd_test.csv"
    if coverage_path.exists():
        df = pd.read_csv(coverage_path)
        df = df.rename(columns={
            "group": "Nhóm",
            "n_rows": "Số dòng",
            "n_hit": "Có bằng chứng",
            "coverage_rate": "Tỷ lệ phủ",
        })
        if "Tỷ lệ phủ" in df.columns:
            df["Tỷ lệ phủ"] = pd.to_numeric(df["Tỷ lệ phủ"], errors="coerce").round(4)
        out["vihos_coverage"] = df

    return out


@st.cache_data(show_spinner=False)
def _load_audit_context(project_root_str: str) -> Dict[str, Any]:
    project_root = Path(project_root_str)
    out: Dict[str, Any] = {}

    csv_specs = {
        "artifact_status": project_root / "outputs" / "results" / "extension_artifact_status.csv",
        "leakage": project_root / "outputs" / "results" / "leakage_overlap_report.csv",
        "vihsd_audit": project_root / "outputs" / "results" / "data_audit_vihsd.csv",
        "vihos_audit": project_root / "outputs" / "results" / "data_audit_vihos_span.csv",
        "failure_summary": project_root / "outputs" / "results" / "note08b" / "bilstm_failure_summary.csv",
        "case_analysis": project_root / "outputs" / "results" / "note09" / "moderation_case_analysis.csv",
    }
    for key, path in csv_specs.items():
        if path.exists():
            try:
                df = pd.read_csv(path)
                for col in df.columns:
                    if col.lower() in {"exists"}:
                        df[col] = df[col].map(lambda v: "OK" if bool(v) else "Thiếu")
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
                out[key] = df
            except Exception:
                pass

    summary_path = project_root / "outputs" / "results" / "note09" / "model_card_summary.md"
    if summary_path.exists():
        try:
            out["model_card_summary"] = summary_path.read_text(encoding="utf-8")
        except Exception:
            pass

    figure_specs = [
        ("So sánh F1 mô hình", project_root / "outputs" / "figures" / "model_comparison_f1.png"),
        ("PhoBERT confusion matrix", project_root / "outputs" / "figures" / "phobert_confusion_matrix_recheck.png"),
        ("SVM confusion matrix", project_root / "outputs" / "figures" / "svm_confusion_matrix.png"),
        ("Robustness theo nhiễu", project_root / "outputs" / "figures" / "note08" / "robustness_f1_by_noise_type.png"),
        ("Gain augmentation", project_root / "outputs" / "figures" / "note08b" / "augmentation_mixed_noise_macro_f1_gain_by_model.png"),
    ]
    out["figures"] = [(label, path) for label, path in figure_specs if path.exists()]
    return out


@st.cache_data(show_spinner=False)
def _load_model_comparison(project_root_str: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    project_root = Path(project_root_str)
    rows: List[Dict[str, Any]] = []
    metric_specs = [
        ("TF-IDF + SVM", project_root / "outputs" / "results" / "svm_metrics.csv"),
        ("BiLSTM", project_root / "outputs" / "results" / "bilstm_metrics.csv"),
        ("PhoBERT", project_root / "outputs" / "results" / "phobert_metrics.csv"),
    ]
    for model_name, path in metric_specs:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            rows.append({
                "Mô hình": model_name,
                "Accuracy": row.get("accuracy", np.nan),
                "Macro Precision": row.get("precision_macro", np.nan),
                "Macro Recall": row.get("recall_macro", np.nan),
                "Macro F1": row.get("f1_macro", np.nan),
            })
        except Exception:
            continue

    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        for col in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce").round(4)

    robust_path = project_root / "outputs" / "results" / "note08b" / "augmentation_all_models_comparison.csv"
    robust_df = pd.DataFrame()
    if robust_path.exists():
        try:
            raw = pd.read_csv(robust_path)
            cols = [
                "model", "train_setting", "clean_macro_f1", "no_accent_macro_f1",
                "repeated_char_macro_f1", "mixed_noise_macro_f1_before_norm",
                "mixed_noise_macro_f1_after_norm", "normalization_gain_on_mixed",
            ]
            raw = raw[[c for c in cols if c in raw.columns]].copy()
            robust_df = raw.rename(columns={
                "model": "Mô hình",
                "train_setting": "Thiết lập huấn luyện",
                "clean_macro_f1": "Clean F1",
                "no_accent_macro_f1": "Bỏ dấu F1",
                "repeated_char_macro_f1": "Lặp ký tự F1",
                "mixed_noise_macro_f1_before_norm": "Nhiễu tổng hợp F1",
                "mixed_noise_macro_f1_after_norm": "Sau chuẩn hóa F1",
                "normalization_gain_on_mixed": "Lợi ích chuẩn hóa",
            })
            for col in robust_df.columns:
                if col not in {"Mô hình", "Thiết lập huấn luyện"}:
                    robust_df[col] = pd.to_numeric(robust_df[col], errors="coerce").round(4)
        except Exception:
            robust_df = pd.DataFrame()

    return metrics_df, robust_df


def _render_probability_bars(row: Dict[str, Any]) -> None:
    labels = [
        ("CLEAN", float(row.get("prob_clean", 0.0) or 0.0)),
        ("OFFENSIVE", float(row.get("prob_offensive", 0.0) or 0.0)),
        ("HATE", float(row.get("prob_hate", 0.0) or 0.0)),
    ]
    for name, value in sorted(labels, key=lambda item: item[1], reverse=True):
        width = max(0.0, min(100.0, value * 100.0))
        st.markdown(
            f"""
<div class="prob-row">
  <div class="prob-head"><span>{html.escape(name)}</span><span>{width:.1f}%</span></div>
  <div class="prob-track"><div class="prob-fill" style="width:{width:.1f}%"></div></div>
</div>
            """,
            unsafe_allow_html=True,
        )


def _render_hero(device: str, phrase_count: int, model_dir: Path) -> None:
    model_name = "PhoBERT + so sánh"
    st.markdown(
        f"""
<section class="hero">
  <div class="eyebrow">Hệ thống kiểm duyệt tiếng Việt</div>
  <h1 class="hero-title">Kiểm duyệt bình luận độc hại tiếng Việt</h1>
  <p class="hero-copy">
    Bảng điều khiển demo sử dụng PhoBERT cho quyết định chính, đối chiếu SVM/BiLSTM/XLM-R khi có artifact,
    chuẩn hóa văn bản, bằng chứng cụm độc hại và điểm rủi ro minh bạch.
  </p>
</section>
<div class="status-strip">
  <div class="status-item"><div class="status-label">Quyết định chính</div><div class="status-value">PhoBERT</div></div>
  <div class="status-item"><div class="status-label">So sánh mô hình</div><div class="status-value">SVM / BiLSTM / XLM-R</div></div>
  <div class="status-item"><div class="status-label">Bằng chứng</div><div class="status-value">VIHOS + rules</div></div>
  <div class="status-item"><div class="status-label">Xử lý</div><div class="status-value">Risk score + policy</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )


inject_console_theme()

with st.sidebar:
    st.markdown("## Vận hành")
    st.caption("Thiết lập cho phiên demo kiểm duyệt.")

    model_dir = find_phobert_model_dir(PROJECT_ROOT)
    if model_dir is None:
        st.error("Không tìm thấy artifact mô hình PhoBERT trong outputs/models.")
        st.code("Expected: outputs/models/note08b/phobert_augmented_mixed")
        st.stop()

    policy_mode = st.selectbox("Chính sách duyệt", ["balanced", "strict", "lenient"], index=0, format_func=_display_policy)
    normalize_input = st.toggle("Chuẩn hóa văn bản nhiễu", value=True)
    compare_models = st.toggle("So sánh mô hình", value=True)
    batch_size = 16

try:
    tokenizer, phobert_model, device, id2label = load_full_phobert(str(model_dir))
except Exception as exc:
    st.error("Không load được mô hình PhoBERT đầy đủ.")
    st.exception(exc)
    st.stop()

phrases = load_toxic_phrases(str(PROJECT_ROOT))
with st.sidebar:
    st.divider()
    st.markdown(
        f"""
<div class="sidebar-card">
  <div class="sidebar-kicker">Tổng quan hệ thống</div>
  <div class="sidebar-row"><span class="name">Quyết định chính</span><span class="value">PhoBERT</span></div>
  <div class="sidebar-row"><span class="name">Đối chiếu</span><span class="value">SVM / BiLSTM / XLM-R</span></div>
  <div class="sidebar-row"><span class="name">Bằng chứng</span><span class="value">VIHOS/rules</span></div>
  <div class="sidebar-row"><span class="name">Môi trường chạy</span><span class="value">{html.escape(_runtime_label(str(device)))}</span></div>
  <div class="sidebar-chipline">
    <span class="sidebar-chip">CLEAN</span>
    <span class="sidebar-chip">OFFENSIVE</span>
    <span class="sidebar-chip">HATE</span>
  </div>
  <div class="sidebar-note">XLM-R sẽ chạy khi máy/Colab đủ RAM; nếu thiếu tài nguyên, bảng so sánh sẽ ghi rõ trạng thái.</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Chi tiết artifact"):
        st.write("Thư mục project")
        st.code(str(PROJECT_ROOT))
        st.write("Mô hình PhoBERT")
        try:
            st.code(str(model_dir.relative_to(PROJECT_ROOT)))
        except Exception:
            st.code(str(model_dir))
        st.write("Bộ tách từ")
        st.code(get_segmenter_name())

_render_hero(device, len(phrases), model_dir)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Kiểm duyệt", "Hàng loạt", "Độ bền nhiễu", "Dashboard", "Model card"])

with tab1:
    _render_workflow_card(
        "Kiểm duyệt một bình luận",
        "Không gian thao tác chính để xem PhoBERT ra quyết định, bằng chứng độc hại nằm ở đâu và các model khác có đồng ý không.",
        "nhãn CLEAN/OFFENSIVE/HATE, risk score, hành động đề xuất, toxic span và bảng so sánh mô hình.",
    )
    if "single_text" not in st.session_state:
        st.session_state["single_text"] = DEMO_CASES["Công kích"]
    elif st.session_state.get("single_text") == "mày ngu vl":
        st.session_state["single_text"] = DEMO_CASES["Công kích"]
    input_col, result_col = st.columns([1.05, 0.95], gap="large")
    with input_col:
        st.markdown(
            """
<div class="panel">
  <p class="section-title">Kiểm duyệt bình luận</p>
  <p class="section-copy">Nhập bình luận để hệ thống phân loại, trích bằng chứng và đề xuất hướng xử lý.</p>
</div>
            """,
            unsafe_allow_html=True,
        )
        sample_cols = st.columns(4)
        for idx, (sample_name, sample_text) in enumerate(DEMO_CASES.items()):
            if sample_cols[idx].button(sample_name, width="stretch"):
                st.session_state["single_text"] = sample_text
        text = st.text_area(
            "Nội dung bình luận",
            height=170,
            key="single_text",
            label_visibility="collapsed",
            placeholder="Dán hoặc nhập bình luận tiếng Việt...",
        )
        run_single = st.button("Phân tích kiểm duyệt", type="primary", width="stretch")

    if run_single and text.strip():
        with st.spinner("Đang chạy PhoBERT và tính điểm rủi ro..."):
            single_result = moderate_texts([text], tokenizer, phobert_model, device, id2label, phrases, normalize_input, policy_mode, batch_size=1)
        comparison_result = pd.DataFrame()
        comparison_warnings: List[str] = []
        if compare_models:
            with st.spinner("Đang đối chiếu SVM, BiLSTM, XLM-R và PhoBERT..."):
                comparison_result, comparison_warnings = build_model_comparison([text], single_result, normalize_input, batch_size=1)
        st.session_state["latest_single_result"] = single_result
        st.session_state["latest_model_comparison"] = comparison_result
        st.session_state["latest_model_comparison_warnings"] = comparison_warnings
    elif run_single:
        st.warning("Vui lòng nhập bình luận trước khi phân tích.")

    single_result = st.session_state.get("latest_single_result")
    with result_col:
        if single_result is None:
            st.markdown(
                """
<div class="decision">
  <div class="decision-label badge-unknown">Chờ phân tích</div>
  <div class="decision-action">Chưa có kết luận</div>
  <p class="muted">Chạy một bình luận qua mô hình để xem nhãn, độ tin cậy, mức rủi ro và hành động đề xuất.</p>
  <div class="score-grid">
    <div class="score-box"><div class="label">Nhãn</div><div class="value">--</div></div>
    <div class="score-box"><div class="label">Rủi ro</div><div class="value">--</div></div>
    <div class="score-box"><div class="label">Độ tin cậy</div><div class="value">--</div></div>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            r = single_result.iloc[0].to_dict()
            st.markdown(
                f"""
<div class="decision">
  <div class="decision-label {_risk_class(r.get('risk_level', ''))}">{html.escape(_display_risk(str(r.get('risk_level', ''))).upper())}</div>
  <div class="decision-action">{html.escape(_display_action(str(r.get('moderation_action', ''))))}</div>
  <p class="muted">{html.escape(str(r.get('explanation', '')))}</p>
  <div class="score-grid">
    <div class="score-box"><div class="label">Nhãn</div><div class="value">{html.escape(_display_label(str(r.get('predicted_label', ''))))}</div></div>
    <div class="score-box"><div class="label">Rủi ro</div><div class="value">{float(r.get('risk_score', 0.0) or 0.0):.3f}</div></div>
    <div class="score-box"><div class="label">Độ tin cậy</div><div class="value">{float(r.get('confidence', 0.0) or 0.0):.3f}</div></div>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )

    if single_result is not None:
        r = single_result.iloc[0].to_dict()
        st.markdown("")
        evidence_col, prob_col = st.columns([1.05, 0.95], gap="large")
        with evidence_col:
            evidence_source = str(r.get("toxic_span_source", "none"))
            evidence_html = str(r.get("highlighted_text", ""))
            evidence_copy = "Các cụm nghi vấn được tô sáng bằng lớp bằng chứng xây từ VIHOS và luật toxic span."
            if evidence_source == "normalized":
                evidence_html = str(r.get("highlighted_normalized_text", ""))
                evidence_copy = "Không phát hiện rõ trên văn bản gốc; lớp bằng chứng VIHOS/rules tìm thấy cụm độc hại sau bước chuẩn hóa."
            elif evidence_source == "model_signal":
                evidence_copy = "VIHOS/rules chưa khớp toxic span chắc chắn; các đoạn tô sáng là tín hiệu gợi ý từ chuẩn hóa, nhiễu ký tự hoặc dự đoán mô hình."
            st.markdown(
                f"""
<div class="panel">
  <p class="section-title">Bằng chứng và chuẩn hóa</p>
  <p class="section-copy">{html.escape(evidence_copy)}</p>
  <div class="highlight-box">{evidence_html}</div>
</div>
                """,
                unsafe_allow_html=True,
            )
            norm_changed = "Đã thay đổi" if bool(r.get("text_changed_by_normalization", False)) else "Không đổi"
            evidence_metric_label = "Tín hiệu" if evidence_source == "model_signal" else "Cụm độc hại"
            st.markdown(
                f"""
<div class="status-strip" style="grid-template-columns: repeat(2, minmax(0, 1fr)); margin-bottom:0.75rem;">
  <div class="status-item"><div class="status-label">Chuẩn hóa</div><div class="status-value">{norm_changed}</div></div>
  <div class="status-item"><div class="status-label">{evidence_metric_label}</div><div class="status-value">{int(r.get('toxic_span_count', 0) or 0)}</div></div>
</div>
<div class="explain-box"><strong>Văn bản gốc</strong><br>{html.escape(str(r.get('original_text', '')))}<br><br><strong>Sau chuẩn hóa</strong><br>{html.escape(str(r.get('normalized_text', '')))}</div>
                """,
                unsafe_allow_html=True,
            )

        with prob_col:
            st.markdown(
                """
<div class="panel">
  <p class="section-title">Xác suất mô hình</p>
  <p class="section-copy">Phân bố xác suất từ bộ phân loại PhoBERT (sequence classifier).</p>
</div>
                """,
                unsafe_allow_html=True,
            )
            _render_probability_bars(r)

        comparison_result = st.session_state.get("latest_model_comparison", pd.DataFrame())
        comparison_warnings = st.session_state.get("latest_model_comparison_warnings", [])
        if compare_models and isinstance(comparison_result, pd.DataFrame) and not comparison_result.empty:
            st.markdown("#### Đối chiếu dự đoán mô hình")
            st.caption("PhoBERT là mô hình triển khai chính. SVM/BiLSTM dùng để đối chiếu phụ; XLM-R là baseline nghiên cứu từ Note 8 nên điểm thấp hơn và không dùng làm quyết định kiểm duyệt.")
            st.dataframe(_model_prediction_table(comparison_result), width="stretch", hide_index=True)
            if comparison_warnings:
                st.caption("Một số artifact phụ chưa chạy được: " + " | ".join(comparison_warnings))
        elif compare_models and comparison_warnings:
            st.info("Chưa chạy được đối chiếu mô hình phụ: " + " | ".join(comparison_warnings))

        with st.expander("Chi tiết tính điểm rủi ro"):
            comp = json.loads(r["risk_components_json"])
            st.dataframe(pd.DataFrame([comp]).T.rename(columns={0: "value"}), width="stretch")

with tab2:
    _render_workflow_card(
        "Kiểm duyệt hàng loạt bằng CSV",
        "Dùng cho kịch bản thật khi cần rà nhiều bình luận cùng lúc thay vì nhập từng câu thủ công.",
        "bảng reviewer ưu tiên rủi ro và file CSV đầy đủ để lưu/chuyển cho bước kiểm duyệt tiếp theo.",
    )
    st.markdown(
        """
<div class="panel">
  <p class="section-title">Nguồn dữ liệu CSV</p>
  <p class="section-copy">Nhận file có cột bình luận, chạy cùng pipeline PhoBERT + risk score và xuất kết quả có thể audit.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    sample_csv = pd.DataFrame({
        "comment": list(DEMO_CASES.values()),
        "case_type": list(DEMO_CASES.keys()),
    }).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Tải CSV mẫu",
        sample_csv,
        "demo_comments_sample.csv",
        "text/csv",
        width="stretch",
    )
    up = st.file_uploader("Tải CSV có cột text/comment/content/original_text", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        st.dataframe(df.head(), width="stretch")
        default_col = next((c for c in ["text", "comment", "content", "original_text"] if c in df.columns), df.columns[0])
        text_col = st.selectbox("Cột văn bản", list(df.columns), index=list(df.columns).index(default_col))
        max_rows = st.number_input("Số dòng tối đa", min_value=1, max_value=max(1, len(df)), value=min(200, len(df)))
        if st.button("Chạy kiểm duyệt hàng loạt", type="primary", width="stretch"):
            sub = df.head(int(max_rows)).copy()
            with st.spinner("Đang chạy PhoBERT cho dữ liệu hàng loạt..."):
                res = moderate_texts(sub[text_col].astype(str).tolist(), tokenizer, phobert_model, device, id2label, phrases, normalize_input, policy_mode, batch_size)
            merged = pd.concat([sub.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
            st.session_state["latest_batch_result"] = merged
            st.markdown("### Bảng quyết định")
            st.dataframe(_review_table(res), width="stretch", hide_index=True)
            st.download_button("Tải CSV đầy đủ", merged.to_csv(index=False).encode("utf-8-sig"), "moderated_comments_full_phobert.csv", "text/csv", width="stretch")

with tab3:
    _render_workflow_card(
        "Kiểm thử độ bền trước văn bản né lọc",
        "Dùng để kiểm tra mô hình có giữ quyết định khi bình luận bị bỏ dấu, lặp ký tự, che ký tự hoặc viết teencode hay không.",
        "bảng so sánh từng biến thể nhiễu, nhãn dự đoán, rủi ro và bằng chứng sau chuẩn hóa.",
    )
    st.markdown(
        """
<div class="panel">
  <p class="section-title">Bộ tạo nhiễu kiểm thử</p>
  <p class="section-copy">Tạo các phiên bản né lọc của cùng một bình luận để kiểm tra độ ổn định của pipeline.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    base_text = st.text_area("Bình luận kiểm thử", value=DEMO_CASES["Công kích"], height=120, key="robust_text_new")
    selected_noise = st.multiselect(
        "Loại nhiễu",
        ["no_accent", "repeated_char", "special_mask", "teencode", "mixed_noise"],
        default=["no_accent", "repeated_char", "mixed_noise"],
        format_func=_display_noise,
    )
    if st.button("So sánh độ bền", type="primary", width="stretch"):
        versions = [("original", base_text)]
        for nt in selected_noise:
            noisy = apply_noise(base_text, nt)
            versions.append((nt, noisy))
            versions.append((nt + "_normalized", normalize_text(noisy)))
        names, texts = zip(*versions)
        with st.spinner("Đang chạy PhoBERT cho các biến thể nhiễu..."):
            res = moderate_texts(list(texts), tokenizer, phobert_model, device, id2label, phrases, normalize_input=False, policy_mode=policy_mode, batch_size=batch_size)
        res.insert(0, "version", names)
        robust_view = _review_table(res)
        robust_view.insert(0, "Biến thể", res["version"].map(_display_noise))
        st.dataframe(robust_view, width="stretch", hide_index=True)

with tab4:
    _render_workflow_card(
        "Dashboard kết quả kiểm duyệt",
        "Dùng để đọc nhanh phân bố nhãn, mức rủi ro, hành động và các bình luận cần ưu tiên xem lại.",
        "biểu đồ phân bố, chỉ số trung bình và danh sách bình luận rủi ro cao nhất.",
    )
    st.markdown('<div class="panel"><p class="section-title">Tổng hợp phiên kiểm duyệt</p></div>', unsafe_allow_html=True)
    data_source = None
    demo_path = PROJECT_ROOT / "outputs" / "results" / "note09" / "demo_cases.csv"
    if "latest_batch_result" in st.session_state:
        data_source = st.session_state["latest_batch_result"]
        st.caption("Đang dùng kết quả hàng loạt mới nhất trong phiên này.")
    elif demo_path.exists():
        data_source = pd.read_csv(demo_path)
        st.caption(f"Đang dùng {demo_path.relative_to(PROJECT_ROOT)}")
    if data_source is None:
        st.info("Hãy chạy kiểm duyệt hàng loạt trước, hoặc cung cấp outputs/results/note09/demo_cases.csv.")
    else:
        d = data_source.copy()
        c1, c2, c3 = st.columns(3)
        c1.metric("Số dòng", len(d))
        if "risk_score" in d.columns:
            c2.metric("Rủi ro TB", f"{d['risk_score'].mean():.3f}")
        if "confidence" in d.columns:
            c3.metric("Độ tin cậy TB", f"{d['confidence'].mean():.3f}")
        if "predicted_label" in d.columns:
            st.markdown("### Phân bố nhãn")
            st.bar_chart(d["predicted_label"].map(_display_label).value_counts())
        if "risk_level" in d.columns:
            st.markdown("### Phân bố mức rủi ro")
            st.bar_chart(d["risk_level"].map(_display_risk).value_counts())
        if "moderation_action" in d.columns:
            st.markdown("### Phân bố hành động")
            st.bar_chart(d["moderation_action"].map(_display_action).value_counts())
        if "risk_score" in d.columns:
            st.markdown("### Bình luận rủi ro cao nhất")
            cols = [c for c in ["original_text", "predicted_label", "confidence", "risk_score", "risk_level", "moderation_action"] if c in d.columns]
            st.dataframe(_review_table(d.sort_values("risk_score", ascending=False)[cols].head(20)), width="stretch", hide_index=True)

with tab5:
    _render_workflow_card(
        "Model card và bằng chứng thực nghiệm",
        "Dùng để chứng minh vì sao chọn PhoBERT làm mô hình chính, XLM-R/VIHOS có vai trò gì và giới hạn hệ thống nằm ở đâu.",
        "metric SVM/BiLSTM/PhoBERT/XLM-R, bảng robustness, vai trò VIHOS và danh sách artifact chạy demo.",
    )
    st.markdown('<div class="panel"><p class="section-title">Hồ sơ mô hình</p></div>', unsafe_allow_html=True)
    st.markdown(
        """
#### Mục đích hệ thống
Ứng dụng demo này hỗ trợ kiểm duyệt bình luận tiếng Việt trên mạng xã hội bằng PhoBERT làm mô hình quyết định chính, đồng thời đối chiếu SVM/BiLSTM/XLM-R/PhoBERT trên cùng đầu vào khi artifact có sẵn.

#### Mô hình mặc định
- **Bộ phân loại (classifier):** PhoBERT sequence classification artifact từ `outputs/models/note08b/phobert_augmented_mixed`.
- **Đối chiếu trực tiếp:** SVM dùng TF-IDF + LinearSVC artifact; BiLSTM dùng checkpoint Note08b; XLM-R dùng artifact đa ngôn ngữ Note 8; PhoBERT vẫn là nguồn quyết định kiểm duyệt chính.
- **Giải thích:** Tô sáng cụm độc hại bằng luật từ tài nguyên Note 7, kèm danh sách cụm dự phòng.
- **Điểm rủi ro (risk score):** Kết hợp nhãn, độ tin cậy, xác suất lớp, cụm độc hại và tín hiệu chuẩn hóa.

#### Cách diễn giải khuyến nghị
Hệ thống nên đóng vai trò hỗ trợ kiểm duyệt, không thay thế hoàn toàn người duyệt. Các ca nhạy cảm, ranh giới hoặc có ảnh hưởng cao cần được con người xem lại.

#### Hạn chế đã biết
- Tiếng lóng, teencode và cách viết né lọc trong tiếng Việt thay đổi nhanh.
- Ranh giới giữa `OFFENSIVE` và `HATE` có thể phụ thuộc ngữ cảnh.
- Tô sáng cụm độc hại chỉ mang tính xấp xỉ và có thể bỏ sót độc hại theo ngữ cảnh.
- Nếu artifact PhoBERT triển khai khác cấu hình tokenizer/tiền xử lý khi huấn luyện, dự đoán có thể lệch so với metric trong notebook.
        """
    )

    metrics_df, robust_df = _load_model_comparison(str(PROJECT_ROOT))
    st.markdown("#### So sánh mô hình")
    if metrics_df.empty:
        st.info("Chưa tìm thấy file metrics tổng hợp để so sánh SVM, BiLSTM và PhoBERT.")
    else:
        st.dataframe(metrics_df, width="stretch", hide_index=True)
        chart_cols = [c for c in ["Macro F1", "Accuracy"] if c in metrics_df.columns]
        if chart_cols:
            st.bar_chart(metrics_df.set_index("Mô hình")[chart_cols])

    if not robust_df.empty:
        st.markdown("#### So sánh độ bền và augmentation")
        st.caption("Macro F1 trên dữ liệu sạch và các biến thể nhiễu. Bảng này đọc từ outputs/results/note08b/augmentation_all_models_comparison.csv.")
        st.dataframe(robust_df, width="stretch", hide_index=True)

    research_context = _load_research_context(str(PROJECT_ROOT))
    st.markdown("#### Vai trò của Note 8 và VIHOS")
    st.markdown(
        """
- **Note 8 / XLM-R đa ngôn ngữ:** dùng làm baseline nghiên cứu để so với PhoBERT và đo tác động của augmentation trên dữ liệu nhiễu. Kết quả này giúp chứng minh vì sao demo chọn PhoBERT làm mô hình chính.
- **VIHOS:** dùng cho lớp giải thích toxic span. Bộ dữ liệu này không thay classifier chính, mà cung cấp nhãn span/cụm độc hại để tạo luật, kiểm tra độ phủ và tô sáng bằng chứng trong giao diện.
        """
    )

    if "multilingual" in research_context:
        st.markdown("##### Note 8: mô hình đa ngôn ngữ so với PhoBERT")
        st.dataframe(research_context["multilingual"], width="stretch", hide_index=True)
    if "augmentation" in research_context:
        st.markdown("##### Note 8: augmentation trên nhiễu tổng hợp")
        st.caption("Bảng tập trung vào mixed noise trước chuẩn hóa, nơi augmentation thể hiện rõ giá trị nhất.")
        st.dataframe(research_context["augmentation"], width="stretch", hide_index=True)
    vihos_cols = st.columns(2)
    with vihos_cols[0]:
        if "vihos_metrics" in research_context:
            st.markdown("##### VIHOS: chất lượng span")
            st.dataframe(research_context["vihos_metrics"], width="stretch", hide_index=True)
    with vihos_cols[1]:
        if "vihos_coverage" in research_context:
            st.markdown("##### VIHOS/rules: độ phủ bằng chứng")
            st.dataframe(research_context["vihos_coverage"], width="stretch", hide_index=True)

    audit_context = _load_audit_context(str(PROJECT_ROOT))
    st.markdown("#### Độ phủ output notebook trong demo")
    st.caption("Phần này kiểm tra các artifact sinh từ notebook đã được dùng làm model, dữ liệu giải thích, metric hoặc bằng chứng audit cho demo.")
    audit_cols = st.columns(2)
    with audit_cols[0]:
        if "artifact_status" in audit_context:
            st.markdown("##### Artifact sẵn sàng")
            st.dataframe(audit_context["artifact_status"], width="stretch", hide_index=True)
        if "leakage" in audit_context:
            st.markdown("##### Kiểm tra leakage dữ liệu")
            st.dataframe(audit_context["leakage"], width="stretch", hide_index=True)
    with audit_cols[1]:
        if "vihsd_audit" in audit_context:
            st.markdown("##### Audit tập VIHSD")
            st.dataframe(audit_context["vihsd_audit"], width="stretch", hide_index=True)
        if "failure_summary" in audit_context:
            st.markdown("##### Phân tích lỗi SVM/BiLSTM")
            st.dataframe(audit_context["failure_summary"], width="stretch", hide_index=True)

    if "case_analysis" in audit_context:
        st.markdown("##### Note09: ca kiểm duyệt bên ngoài")
        case_df = audit_context["case_analysis"]
        show_cols = [c for c in ["sample_id", "case_type", "expected_behavior", "predicted_label", "confidence", "risk_score", "risk_level", "moderation_action", "analysis_tags"] if c in case_df.columns]
        st.dataframe(case_df[show_cols].head(20), width="stretch", hide_index=True)

    figures = audit_context.get("figures", [])
    if figures:
        st.markdown("##### Hình sinh từ notebook")
        for row_start in range(0, len(figures), 2):
            fig_cols = st.columns(2)
            for col, (label, path) in zip(fig_cols, figures[row_start:row_start + 2]):
                with col:
                    st.caption(label)
                    st.image(str(path), width="stretch")

    if "model_card_summary" in audit_context:
        with st.expander("Model card summary từ Note09"):
            st.markdown(str(audit_context["model_card_summary"]))

    c1, c2 = st.columns(2)
    c1.write("Nhãn phát hiện")
    c1.json(id2label)
    c2.write("Artifact chạy trực tiếp")
    artifact_lines = [f"PhoBERT: {model_dir}"]
    svm_bundle = load_svm_artifacts(str(PROJECT_ROOT))
    if svm_bundle.get("ok"):
        artifact_lines.append(f"SVM: {svm_bundle.get('model_path')}")
        artifact_lines.append(f"TF-IDF: {svm_bundle.get('vectorizer_path')}")
    bilstm_bundle = load_bilstm_artifact(str(PROJECT_ROOT), str(device))
    if bilstm_bundle.get("ok"):
        artifact_lines.append(f"BiLSTM: {bilstm_bundle.get('path')}")
    xlmr_bundle = load_xlmr_model(str(PROJECT_ROOT), str(device))
    if xlmr_bundle.get("ok"):
        artifact_lines.append(f"XLM-R: {xlmr_bundle.get('path')}")
    c2.code("\n".join(artifact_lines))

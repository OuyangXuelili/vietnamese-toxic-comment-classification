from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from pyvi import ViTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="DLNLP Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
RESULT_DIR = PROJECT_ROOT / "outputs" / "results"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
PHOBERT_DIR = MODEL_DIR / "phobert_base"

LABEL_MAP = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
LABEL_VI = {0: "Bình thường", 1: "Công kích", 2: "Thù ghét"}
LABEL_COLOR = {0: "#16a34a", 1: "#f59e0b", 2: "#dc2626"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLES = {
    "Câu tích cực": "Bài viết này rất hữu ích, cảm ơn bạn đã chia sẻ.",
    "Câu công kích nhẹ": "Đọc mà chán thật, nói năng kiểu này thì ai chịu nổi.",
    "Câu thù ghét": "Mấy đứa đó đúng là phải biến khỏi đây ngay.",
    "Câu trung tính": "Hôm nay trời khá mát, mình vừa đi học về.",
}


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, input_ids, lengths):
        emb = self.emb_dropout(self.embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=input_ids.size(1),
        )

        mask = (input_ids != 0).unsqueeze(-1)
        output_masked = output.masked_fill(~mask, -1e9)
        max_pool = output_masked.max(dim=1).values

        sum_pool = (output * mask).sum(dim=1)
        valid_len = mask.sum(dim=1).clamp(min=1)
        mean_pool = sum_pool / valid_len

        feat = torch.cat([max_pool, mean_pool], dim=1)
        logits = self.fc(self.dropout(feat))
        return logits


class TextDataset(Dataset):
    def __init__(self, texts, vocab, max_len):
        self.texts = list(texts)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids, length = encode_text(self.texts[idx], self.vocab, self.max_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
        }


@st.cache_resource(show_spinner=False)
def load_svm_artifacts() -> Tuple[Any, Any]:
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(MODEL_DIR / "svm_model.joblib")
    return vectorizer, model


@st.cache_resource(show_spinner=False)
def load_bilstm_artifacts() -> Optional[Dict[str, Any]]:
    vocab_path = MODEL_DIR / "bilstm_vocab.json"
    model_path = MODEL_DIR / "bilstm_best.pt"
    metrics_path = RESULT_DIR / "bilstm_metrics.csv"

    if not vocab_path.exists() or not model_path.exists() or not metrics_path.exists():
        return None

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    metrics_df = pd.read_csv(metrics_path)
    if metrics_df.empty:
        return None

    row = metrics_df.iloc[0]
    max_len = int(row.get("max_len", 64))
    embed_dim = int(row.get("embed_dim", 300))
    hidden_dim = int(row.get("hidden_dim", 128))
    num_layers = int(row.get("num_layers", 2))

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=3,
        num_layers=num_layers,
        dropout=0.4,
    ).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "model": model,
        "vocab": vocab,
        "max_len": max_len,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }


@st.cache_resource(show_spinner=False)
def find_phobert_model_dir() -> Optional[Path]:
    if not PHOBERT_DIR.exists():
        return None

    candidates = []
    for pattern in ("**/config.json", "**/pytorch_model.bin", "**/model.safetensors"):
        for file_path in PHOBERT_DIR.glob(pattern):
            candidates.append(file_path.parent)

    if not candidates:
        return None

    unique_dirs = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_dirs.append(candidate)

    return max(unique_dirs, key=lambda item: item.stat().st_mtime)


@st.cache_resource(show_spinner=False)
def load_phobert_artifacts() -> Optional[Dict[str, Any]]:
    model_dir = find_phobert_model_dir()
    if model_dir is None:
        return None

    try:
        from transformers import AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
        model.eval()
        return {"tokenizer": tokenizer, "model": model, "model_dir": model_dir}
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_data(show_spinner=False)
def segment_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text.strip().lower())
    return ViTokenizer.tokenize(text)


@st.cache_data(show_spinner=False)
def tokenize_text(text: str):
    return str(text).split()


@st.cache_data(show_spinner=False)
def encode_text(text: str, vocab: Dict[str, int], max_len: int):
    tokens = tokenize_text(text)[:max_len]
    ids = [vocab.get(token, 1) for token in tokens]

    if len(ids) == 0:
        ids = [1]

    length = len(ids)
    if length < max_len:
        ids += [0] * (max_len - length)

    return ids, length


@torch.no_grad()
def predict_bilstm(text: str, artifacts: Dict[str, Any]):
    model = artifacts["model"]
    vocab = artifacts["vocab"]
    max_len = artifacts["max_len"]

    seg_text = segment_text(clean_text(text))
    input_ids, length = encode_text(seg_text, vocab, max_len)
    batch = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=DEVICE),
        "length": torch.tensor([length], dtype=torch.long, device=DEVICE),
    }
    logits = model(batch["input_ids"], batch["length"])
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    pred_id = int(np.argmax(probs))
    return pred_id, probs


@torch.no_grad()
def predict_phobert(text: str, artifacts: Dict[str, Any]):
    tokenizer = artifacts["tokenizer"]
    model = artifacts["model"]

    seg_text = segment_text(clean_text(text))
    encoding = tokenizer(
        [seg_text],
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    encoding = {key: value.to(DEVICE) for key, value in encoding.items()}
    outputs = model(**encoding)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    pred_id = int(np.argmax(probs))
    return pred_id, probs


@st.cache_resource(show_spinner=False)
def load_models():
    svm_vectorizer, svm_model = load_svm_artifacts()
    bilstm_artifacts = load_bilstm_artifacts()
    phobert_artifacts = load_phobert_artifacts()
    return {
        "svm": (svm_vectorizer, svm_model),
        "bilstm": bilstm_artifacts,
        "phobert": phobert_artifacts,
    }


def predict_svm(text: str, artifacts: Tuple[Any, Any]):
    vectorizer, model = artifacts
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred_id = int(model.predict(features)[0])
    return pred_id


def label_card(label_id: int, title: str, subtitle: str, prob: Optional[float] = None):
    color = LABEL_COLOR.get(label_id, "#334155")
    probability_text = f"{prob:.2%}" if prob is not None else ""
    st.markdown(
        f"""
        <div class="result-card" style="border-left: 6px solid {color};">
            <div class="result-title">{title}</div>
            <div class="result-subtitle">{subtitle}</div>
            <div class="result-badge" style="background: {color};">{LABEL_MAP.get(label_id, 'Unknown')}</div>
            <div class="result-prob">{probability_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(59, 130, 246, 0.16), transparent 32%),
            radial-gradient(circle at top right, rgba(16, 185, 129, 0.12), transparent 28%),
            linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }
    .hero {
        padding: 1.4rem 1.5rem;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.05rem;
        color: #0f172a;
    }
    .hero p {
        margin: 0.4rem 0 0 0;
        color: #475569;
        font-size: 1rem;
        line-height: 1.55;
    }
    .result-card {
        padding: 1rem 1rem 0.9rem 1rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.92);
        box-shadow: 0 12px 26px rgba(15, 23, 42, 0.07);
        min-height: 140px;
    }
    .result-title {
        font-size: 0.86rem;
        font-weight: 700;
        color: #475569;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .result-subtitle {
        color: #64748b;
        margin-top: 0.3rem;
        margin-bottom: 0.7rem;
        font-size: 0.92rem;
    }
    .result-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        color: white;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }
    .result-prob {
        color: #0f172a;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .small-panel {
        padding: 0.9rem 1rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Phân loại bình luận độc hại tiếng Việt</h1>
        <p>
            Giao diện demo để nhập một câu bất kỳ, sau đó so sánh dự đoán từ TF-IDF + SVM, BiLSTM,
            và PhoBERT nếu mô hình đã sẵn sàng. Toàn bộ model được đọc từ các file đã lưu trong workspace.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

models = None
with st.spinner("Đang tải mô hình đã lưu..."):
    try:
        models = load_models()
    except Exception as exc:
        st.error(f"Không thể tải model: {exc}")
        st.stop()

svm_available = (MODEL_DIR / "tfidf_vectorizer.joblib").exists() and (MODEL_DIR / "svm_model.joblib").exists()
bilstm_available = models["bilstm"] is not None
phobert_available = models["phobert"] is not None

with st.sidebar:
    st.markdown("### Trạng thái demo")
    st.write(f"**Thiết bị:** {DEVICE}")
    st.write(f"**SVM:** {'Sẵn sàng' if svm_available else 'Thiếu file'}")
    st.write(f"**BiLSTM:** {'Sẵn sàng' if bilstm_available else 'Thiếu file'}")
    st.write(f"**PhoBERT:** {'Sẵn sàng' if phobert_available else 'Chưa có model hoàn chỉnh'}")

    st.markdown("### Văn bản mẫu")
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""

    for label, sample_text in SAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state["input_text"] = sample_text

    st.markdown("### Ghi chú")
    st.caption("Notebook chỉ dùng để train và lưu artifact. Ứng dụng này đọc lại file đã lưu để demo dự đoán.")

left_col, right_col = st.columns([1.35, 0.95], gap="large")

with left_col:
    st.markdown("### Nhập nội dung cần kiểm tra")
    input_text = st.text_area(
        "Nội dung bình luận",
        key="input_text",
        height=220,
        placeholder="Nhập một câu bình luận tiếng Việt ở đây...",
        label_visibility="collapsed",
    )

    action_col, helper_col = st.columns([0.25, 0.75])
    with action_col:
        predict_clicked = st.button("Dự đoán", use_container_width=True, type="primary")
    with helper_col:
        if st.button("Xóa nội dung", use_container_width=True):
            st.session_state["input_text"] = ""
            st.rerun()

    st.markdown("### Chuẩn hóa đầu vào")
    cleaned_preview = clean_text(input_text)
    segmented_preview = segment_text(cleaned_preview)
    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        st.markdown("**Text sau làm sạch**")
        st.code(cleaned_preview or "(trống)")
    with preview_col2:
        st.markdown("**Text sau tách từ**")
        st.code(segmented_preview or "(trống)")

with right_col:
    st.markdown("### Tổng quan mô hình")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("SVM", "Sẵn sàng" if svm_available else "Thiếu")
    with c2:
        st.metric("BiLSTM", "Sẵn sàng" if bilstm_available else "Thiếu")
    with c3:
        st.metric("PhoBERT", "Sẵn sàng" if phobert_available else "Thiếu")

    st.markdown("### Dự đoán nhanh")
    if not input_text.strip():
        st.info("Nhập một bình luận rồi bấm **Dự đoán** để xem kết quả.")
    elif predict_clicked:
        results = []

        if svm_available:
            svm_pred = predict_svm(input_text, models["svm"])
            results.append(("TF-IDF + SVM", svm_pred, None))
        else:
            results.append(("TF-IDF + SVM", -1, None))

        if bilstm_available:
            bilstm_pred, bilstm_probs = predict_bilstm(input_text, models["bilstm"])
            results.append(("BiLSTM", bilstm_pred, float(bilstm_probs[bilstm_pred])))
        else:
            results.append(("BiLSTM", -1, None))

        if phobert_available:
            phobert_pred, phobert_probs = predict_phobert(input_text, models["phobert"])
            results.append(("PhoBERT", phobert_pred, float(phobert_probs[phobert_pred])))
        else:
            results.append(("PhoBERT", -1, None))

        valid_votes = [pred for _, pred, _ in results if pred in LABEL_MAP]
        if valid_votes:
            vote = Counter(valid_votes).most_common(1)[0][0]
        else:
            vote = 0

        top_left, top_mid, top_right = st.columns(3)
        with top_left:
            label_card(vote, "Kết luận gợi ý", "Theo đa số mô hình", None)
        with top_mid:
            label_card(results[0][1], "SVM", "Mô hình nền tảng", results[0][2])
        with top_right:
            label_card(results[1][1], "BiLSTM", "Mô hình tuần tự", results[1][2])

        if phobert_available:
            st.markdown("### PhoBERT")
            phobert_row = results[2]
            label_card(phobert_row[1], "PhoBERT", "Mô hình ngữ cảnh", phobert_row[2])
        else:
            st.warning("PhoBERT chưa có artifact hoàn chỉnh nên app bỏ qua phần dự đoán này.")

        st.markdown("### Giải thích nhanh")
        if vote == 0:
            st.success("Hệ thống nghiêng về bình thường / không độc hại.")
        elif vote == 1:
            st.warning("Hệ thống nghiêng về công kích / tiêu cực.")
        else:
            st.error("Hệ thống nghiêng về nội dung thù ghét / độc hại nặng.")

        summary_rows = []
        for model_name, pred_id, prob in results:
            if pred_id in LABEL_MAP:
                summary_rows.append(
                    {
                        "Mô hình": model_name,
                        "Nhãn dự đoán": LABEL_VI.get(pred_id, LABEL_MAP[pred_id]),
                        "Mã nhãn": LABEL_MAP[pred_id],
                        "Xác suất / độ tin cậy": None if prob is None else round(prob, 4),
                    }
                )
            else:
                summary_rows.append(
                    {
                        "Mô hình": model_name,
                        "Nhãn dự đoán": "Không khả dụng",
                        "Mã nhãn": "-",
                        "Xác suất / độ tin cậy": "-",
                    }
                )

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Bấm **Dự đoán** để chạy các mô hình đã lưu trên văn bản hiện tại.")

st.markdown("---")
st.markdown("### Cấu trúc dữ liệu được dùng")
info_col1, info_col2, info_col3 = st.columns(3)
with info_col1:
    st.markdown(
        f"""
        <div class="small-panel">
        <strong>Input</strong><br/>
        `data/raw` → notebook 1 → `data/processed`
        </div>
        """,
        unsafe_allow_html=True,
    )
with info_col2:
    st.markdown(
        f"""
        <div class="small-panel">
        <strong>Model</strong><br/>
        `outputs/models` chứa SVM, BiLSTM và PhoBERT
        </div>
        """,
        unsafe_allow_html=True,
    )
with info_col3:
    st.markdown(
        f"""
        <div class="small-panel">
        <strong>Kết quả</strong><br/>
        `outputs/results` và `outputs/figures` lưu báo cáo
        </div>
        """,
        unsafe_allow_html=True,
    )

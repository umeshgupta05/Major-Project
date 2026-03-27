import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_curve,
    roc_auc_score,
)

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(ROOT_DIR, "Dataset.csv")
CHECKPOINT_PATH = os.path.join(ROOT_DIR, "mstct_sepsis_model.pth")
SCALER_PATH = os.path.join(ROOT_DIR, "feature_scaler.pkl")
FEATURE_CONFIG_PATH = os.path.join(ROOT_DIR, "feature_config.json")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        v = float(value)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default


def normalize_id(value: Any) -> str:
    s = str(value).strip()
    if s == "":
        return ""
    # Make matching tolerant to leading zeros and numeric formatting differences.
    if s.isdigit():
        return s.lstrip("0") or "0"
    return s


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, : x.size(2)]
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class MultiScaleTCN(nn.Module):
    def __init__(self, in_channels: int, tcn_channels: int, kernel_sizes: List[int], dilations: List[int], tcn_dropout: float = 0.2):
        super().__init__()
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            branch = nn.Sequential(
                *[
                    CausalConvBlock(
                        in_channels if i == 0 else tcn_channels,
                        tcn_channels,
                        ks,
                        d,
                        dropout=tcn_dropout,
                    )
                    for i, d in enumerate(dilations)
                ]
            )
            self.branches.append(branch)

        total_channels = tcn_channels * len(kernel_sizes)
        self.fusion = nn.Sequential(
            nn.Conv1d(total_channels, tcn_channels, 1),
            nn.BatchNorm1d(tcn_channels),
            nn.GELU(),
            nn.Dropout(tcn_dropout),
        )

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        concat = torch.cat(branch_outputs, dim=1)
        return self.fusion(concat)


class MSTCT(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        input_dim = int(config["INPUT_DIM"])
        tcn_channels = int(config["TCN_CHANNELS"])
        kernel_sizes = list(config["TCN_KERNEL_SIZES"])
        dilations = list(config["TCN_DILATIONS"])
        tf_dim = int(config["TRANSFORMER_DIM"])
        tf_heads = int(config["TRANSFORMER_HEADS"])
        tf_layers = int(config["TRANSFORMER_LAYERS"])
        tf_ff_dim = int(config["TRANSFORMER_FF_DIM"])
        dropout = float(config["DROPOUT"])
        tcn_dropout = float(config.get("TCN_DROPOUT", 0.2))

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, tcn_channels),
            nn.LayerNorm(tcn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tcn = MultiScaleTCN(tcn_channels, tcn_channels, kernel_sizes, dilations, tcn_dropout=tcn_dropout)
        self.tcn_to_transformer = nn.Sequential(
            nn.Linear(tcn_channels, tf_dim),
            nn.LayerNorm(tf_dim),
            nn.Dropout(dropout),
        )
        self.pos_encoder = PositionalEncoding(tf_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_dim,
            nhead=tf_heads,
            dim_feedforward=tf_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
        self.classifier = nn.Sequential(
            nn.Linear(tf_dim, tf_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tf_dim // 2, 1),
        )

    def forward(self, x, return_intermediates: bool = False):
        input_projected = self.input_proj(x)
        tcn_input = input_projected.permute(0, 2, 1)
        tcn_out = self.tcn(tcn_input)
        transformer_in = tcn_out.permute(0, 2, 1)
        transformer_in = self.tcn_to_transformer(transformer_in)

        padding_mask = torch.isclose(
            transformer_in.abs().sum(dim=-1),
            torch.zeros(1, device=x.device),
            atol=1e-12,
        )

        transformer_encoded = self.pos_encoder(transformer_in)
        transformer_encoded = self.transformer(transformer_encoded, src_key_padding_mask=padding_mask)

        valid_mask = (~padding_mask).float().unsqueeze(-1)
        denom = valid_mask.sum(dim=1).clamp(min=1.0)
        pooled = (transformer_encoded * valid_mask).sum(dim=1) / denom

        logits = self.classifier(pooled)
        logits = logits.squeeze(-1)

        if return_intermediates:
            intermediates = {
                "input_projected": input_projected,
                "tcn_out": tcn_out.permute(0, 2, 1),
                "transformer_in": transformer_in,
                "transformer_out": transformer_encoded,
                "pooled": pooled,
            }
            return logits, intermediates

        return logits


@dataclass
class Artifacts:
    feature_config: Dict[str, Any]
    scaler: Any
    checkpoint: Dict[str, Any]
    model: Any
    ready: bool
    status: str


def load_artifacts() -> Artifacts:
    feature_config = {}
    scaler = None
    checkpoint = {}
    model = None

    if not os.path.exists(FEATURE_CONFIG_PATH):
        return Artifacts({}, None, {}, None, False, "feature_config.json missing")
    if not os.path.exists(SCALER_PATH):
        return Artifacts({}, None, {}, None, False, "feature_scaler.pkl missing")
    if not os.path.exists(CHECKPOINT_PATH):
        return Artifacts({}, None, {}, None, False, "mstct_sepsis_model.pth missing")

    with open(FEATURE_CONFIG_PATH, "r", encoding="utf-8") as f:
        feature_config = json.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    if torch is None:
        return Artifacts(feature_config, scaler, {}, None, False, "torch not installed in current Python environment")

    # PyTorch 2.6 changed torch.load default to weights_only=True.
    # This checkpoint stores extra Python objects (config/metrics), so force full load.
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        # Backward compatibility for torch versions that do not expose weights_only.
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    except Exception as ex:
        return Artifacts(
            feature_config,
            scaler,
            {},
            None,
            False,
            f"checkpoint load failed: {ex}",
        )

    config = dict(checkpoint.get("config", {}))
    if "INPUT_DIM" not in config or config.get("INPUT_DIM") in (None, 0):
        config["INPUT_DIM"] = len(feature_config.get("FINAL_FEATURES", []))

    model = MSTCT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return Artifacts(feature_config, scaler, checkpoint, model, True, "ok")


ARTIFACTS = load_artifacts()

app = Flask(__name__, template_folder="templates", static_folder="static")


def compute_dataset_overview(feature_config: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "rows": 0,
        "columns": 0,
        "patients": 0,
        "label_col": feature_config.get("LABEL_COL"),
        "patient_col": feature_config.get("PATIENT_ID_COL"),
        "time_col": feature_config.get("TIME_COL"),
        "sepsis_prevalence": None,
        "top_missing": [],
    }

    if not os.path.exists(DATASET_PATH):
        return summary

    df = pd.read_csv(DATASET_PATH)
    summary["rows"] = int(df.shape[0])
    summary["columns"] = int(df.shape[1])

    patient_col = feature_config.get("PATIENT_ID_COL")
    label_col = feature_config.get("LABEL_COL")

    if patient_col in df.columns:
        summary["patients"] = int(df[patient_col].nunique(dropna=True))
    if label_col in df.columns:
        y = pd.to_numeric(df[label_col], errors="coerce").fillna(0)
        summary["sepsis_prevalence"] = round(float((y > 0.5).mean() * 100.0), 3)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        miss = (df[numeric_cols].isna().mean() * 100.0).sort_values(ascending=False).head(12)
        summary["top_missing"] = [{"feature": str(k), "missing_pct": round(float(v), 2)} for k, v in miss.items()]

    return summary


def build_overview() -> Dict[str, Any]:
    fc = ARTIFACTS.feature_config
    ckpt = ARTIFACTS.checkpoint
    metrics = ckpt.get("test_metrics", {}) if isinstance(ckpt, dict) else {}

    return {
        "artifact_status": ARTIFACTS.status,
        "model_ready": ARTIFACTS.ready,
        "sequence_length": int(ckpt.get("config", {}).get("SEQUENCE_LENGTH", 48)) if isinstance(ckpt, dict) else 48,
        "stride": int(ckpt.get("config", {}).get("STRIDE", 12)) if isinstance(ckpt, dict) else 12,
        "threshold": to_float(ckpt.get("optimal_threshold", 0.5), 0.5) if isinstance(ckpt, dict) else 0.5,
        "input_dim": len(fc.get("FINAL_FEATURES", [])),
        "metrics": {
            "accuracy": to_float(metrics.get("accuracy", 0.0)),
            "precision": to_float(metrics.get("precision", 0.0)),
            "recall": to_float(metrics.get("recall", 0.0)),
            "f1": to_float(metrics.get("f1", 0.0)),
            "auroc": to_float(metrics.get("auroc", 0.0)),
            "auprc": to_float(metrics.get("auprc", 0.0)),
        },
        "feature_groups": {
            "temporal": len(fc.get("ALL_TEMPORAL_FEATURES", [])),
            "vitals": len(fc.get("VITAL_COLS", [])),
            "labs": len(fc.get("LAB_COLS", [])),
            "static": len(fc.get("STATIC_DEMO", [])),
            "missing_flags": len(fc.get("MASK_COLS", [])),
            "time_since_obs": len(fc.get("DELTA_COLS", [])),
            "rolling": len(fc.get("ROLLING_COLS", [])),
        },
        "dataset": compute_dataset_overview(fc),
    }


def _compute_window_ranges(n_rows: int, seq_len: int, stride: int) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    if n_rows <= seq_len:
        ranges.append((0, n_rows))
        return ranges

    for start in range(0, n_rows - seq_len + 1, stride):
        ranges.append((start, start + seq_len))

    last_range = (n_rows - seq_len, n_rows)
    if not ranges or ranges[-1] != last_range:
        ranges.append(last_range)
    return ranges


def summarize_tensor(tensor: Any) -> Dict[str, Any]:
    arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "shape": list(arr.shape),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
    }


def attention_proxy(transformer_out: Any, max_tokens: int = 32) -> List[List[float]]:
    arr = transformer_out.detach().cpu().numpy() if hasattr(transformer_out, "detach") else np.asarray(transformer_out)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return []

    last_window = arr[-1]
    token_count = min(last_window.shape[0], max_tokens)
    x = last_window[:token_count]
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    x = x / norm
    sim = np.matmul(x, x.T)
    sim = (sim + 1.0) / 2.0
    sim = np.clip(sim, 0.0, 1.0)
    return sim.astype(float).tolist()


def run_preprocessing_trace(df_input: pd.DataFrame) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]], Dict[str, Any]]:
    fc = ARTIFACTS.feature_config
    seq_len = int(ARTIFACTS.checkpoint.get("config", {}).get("SEQUENCE_LENGTH", 48)) if ARTIFACTS.checkpoint else 48
    stride = int(ARTIFACTS.checkpoint.get("config", {}).get("STRIDE", 12)) if ARTIFACTS.checkpoint else 12

    temporal_features = list(fc.get("ALL_TEMPORAL_FEATURES", []))
    static_demo = list(fc.get("STATIC_DEMO", []))
    vital_cols = list(fc.get("VITAL_COLS", []))
    lab_cols = list(fc.get("LAB_COLS", []))
    final_features = list(fc.get("FINAL_FEATURES", []))
    medians = fc.get("global_medians", {})
    clip_lower = fc.get("clip_lower", {})
    clip_upper = fc.get("clip_upper", {})

    steps: List[Dict[str, Any]] = []

    df_raw = df_input.copy()
    df_proc = df_input.copy()

    raw_missing_series = df_input.isna().mean().sort_values(ascending=False)

    for col in temporal_features + static_demo:
        if col not in df_proc.columns:
            df_proc[col] = medians.get(col, 0.0)
            df_raw[col] = np.nan

    before_missing = int(df_proc[temporal_features].isna().sum().sum()) if temporal_features else 0
    steps.append({"step": "Input validation", "details": "Missing required columns added with training medians", "missing_before": before_missing})

    for col in temporal_features:
        df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce")

    if vital_cols:
        df_proc[vital_cols] = df_proc[vital_cols].interpolate(method="linear", limit_direction="both")

    for col in temporal_features:
        df_proc[col] = df_proc[col].ffill().bfill()
        if df_proc[col].isna().any():
            df_proc[col] = df_proc[col].fillna(medians.get(col, 0.0))

    after_temporal_missing = int(df_proc[temporal_features].isna().sum().sum()) if temporal_features else 0
    steps.append({"step": "Temporal imputation", "details": "Interpolation plus forward/backward fill", "missing_after": after_temporal_missing})

    for col in static_demo:
        df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce").fillna(medians.get(col, 0.0))

    for col in ["Unit1", "Unit2"]:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].fillna(0)

    steps.append({"step": "Static fill", "details": "Static demographics filled using medians", "missing_after": int(df_proc.isna().sum().sum())})

    for col in lab_cols:
        mask_col = f"{col}_missing"
        delta_col = f"{col}_tsince_obs"
        raw_mask = df_raw[col].isna().astype(np.float32).to_numpy() if col in df_raw.columns else np.ones(len(df_proc), dtype=np.float32)
        df_proc[mask_col] = raw_mask

        tsince = np.zeros(len(raw_mask), dtype=np.float32)
        counter = 0.0
        for i, m in enumerate(raw_mask):
            if m < 0.5:
                counter = 0.0
            else:
                counter += 1.0
            tsince[i] = counter
        df_proc[delta_col] = tsince

    for col in vital_cols:
        if col in df_proc.columns:
            df_proc[f"{col}_roll_mean"] = df_proc[col].rolling(window=6, min_periods=1).mean()
            df_proc[f"{col}_roll_std"] = df_proc[col].rolling(window=6, min_periods=1).std().fillna(0)

    missing_engineered = [c for c in final_features if c not in df_proc.columns]
    for c in missing_engineered:
        df_proc[c] = 0.0

    steps.append({"step": "Feature engineering", "details": "Missing flags, time-since-observation, and rolling stats created", "features_created": len(final_features)})

    feat = df_proc[final_features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

    if clip_lower and clip_upper:
        lower = np.array([to_float(clip_lower.get(c, -np.inf), -np.inf) for c in final_features], dtype=np.float32)
        upper = np.array([to_float(clip_upper.get(c, np.inf), np.inf) for c in final_features], dtype=np.float32)
        feat = np.minimum(np.maximum(feat, lower), upper)

    feat = ARTIFACTS.scaler.transform(feat)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    steps.append({"step": "Scaling and clipping", "details": "Train-time quantile clipping and StandardScaler transform applied", "rows": int(feat.shape[0]), "cols": int(feat.shape[1])})

    windows: List[np.ndarray] = []
    n_rows = len(feat)
    ranges = _compute_window_ranges(n_rows, seq_len, stride)

    if n_rows <= seq_len:
        pad_len = seq_len - n_rows
        padded = np.vstack([np.zeros((pad_len, feat.shape[1]), dtype=np.float32), feat])
        windows.append(padded)
    else:
        for start, end in ranges:
            windows.append(feat[start:end])

    steps.append({"step": "Windowing", "details": "Fixed-length windows generated for sequence model", "window_count": len(windows), "window_length": seq_len, "stride": stride})

    proc_missing_series = df_proc[final_features].isna().mean().sort_values(ascending=False)
    raw_sample_cols = [c for c in df_input.columns[: min(10, len(df_input.columns))]]
    clean_sample_cols = [c for c in final_features[: min(10, len(final_features))]]

    preprocessing_viz = {
        "raw_sample": df_input[raw_sample_cols].head(8).replace({np.nan: None}).to_dict(orient="records") if raw_sample_cols else [],
        "clean_sample": df_proc[clean_sample_cols].head(8).replace({np.nan: None}).to_dict(orient="records") if clean_sample_cols else [],
        "missing_before": [
            {"feature": str(k), "missing_pct": round(float(v * 100.0), 2)}
            for k, v in raw_missing_series.head(20).items()
        ],
        "missing_after": [
            {"feature": str(k), "missing_pct": round(float(v * 100.0), 2)}
            for k, v in proc_missing_series.head(20).items()
        ],
    }

    return np.array(windows, dtype=np.float32), steps, ranges, preprocessing_viz


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/overview", methods=["GET"])
def api_overview():
    return jsonify(build_overview())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not ARTIFACTS.ready:
        return jsonify({"ok": False, "error": f"Model is not ready: {ARTIFACTS.status}"}), 500

    incoming_json = request.json if request.is_json else {}
    patient_id = None
    uploaded_filename = None
    records: List[Dict[str, Any]] = []

    upload_file = None
    if request.files:
        upload_file = request.files.get("file") or request.files.get("csvFile")

    if upload_file is not None:
        file = upload_file
        uploaded_filename = file.filename
        if not file.filename:
            return jsonify({"ok": False, "error": "Uploaded file has no name."}), 400
        # Accept upload even when extension is missing; validate by parse instead.
        if not file.filename.lower().endswith(".csv"):
            if "csv" not in (file.content_type or "").lower():
                return jsonify({"ok": False, "error": "Please upload a CSV file."}), 400
        try:
            df_all = pd.read_csv(file)
        except Exception as ex:
            return jsonify({"ok": False, "error": f"Could not read uploaded CSV: {ex}"}), 400

        patient_col = ARTIFACTS.feature_config.get("PATIENT_ID_COL")
        patient_id_raw = request.form.get("patient_id", "").strip()
        if patient_id_raw and patient_col and patient_col in df_all.columns:
            patient_id = patient_id_raw
            target_id = normalize_id(patient_id_raw)
            df_ids = df_all[patient_col].astype(str).map(normalize_id)
            df_all = df_all[df_ids == target_id]
            if df_all.empty:
                sample_ids = (
                    pd.Series(df_ids.unique())
                    .replace("", np.nan)
                    .dropna()
                    .head(5)
                    .tolist()
                )
                return jsonify(
                    {
                        "ok": False,
                        "error": (
                            f"No rows found for patient_id '{patient_id_raw}' in column '{patient_col}'. "
                            f"Sample available IDs: {sample_ids}"
                        ),
                    }
                ), 400

        records = df_all.to_dict(orient="records")
    else:
        records = incoming_json.get("records", []) if isinstance(incoming_json, dict) else []
        patient_id = incoming_json.get("patient_id") if isinstance(incoming_json, dict) else None

    if not records:
        return jsonify({"ok": False, "error": "No input records provided."}), 400

    df_input = pd.DataFrame(records)
    time_col = ARTIFACTS.feature_config.get("TIME_COL")
    if time_col in df_input.columns:
        df_input = df_input.sort_values(time_col).reset_index(drop=True)

    try:
        windows, steps, ranges, preprocessing_viz = run_preprocessing_trace(df_input)
    except Exception as ex:
        return jsonify({"ok": False, "error": f"Preprocessing failed: {ex}"}), 400

    x = torch.FloatTensor(windows)
    with torch.no_grad():
        logits, intermediates = ARTIFACTS.model(x, return_intermediates=True)
        probs = torch.sigmoid(logits).cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)

    stage_viz = {
        "input_window_shape": list(x.shape),
        "tcn": summarize_tensor(intermediates.get("tcn_out")),
        "transformer_in": summarize_tensor(intermediates.get("transformer_in")),
        "transformer_out": summarize_tensor(intermediates.get("transformer_out")),
        "attention_proxy": attention_proxy(intermediates.get("transformer_out")),
    }

    threshold = to_float(ARTIFACTS.checkpoint.get("optimal_threshold", 0.5), 0.5)
    risk = float(probs[-1])
    max_risk = float(probs.max())
    pred = int(risk >= threshold)

    if pred == 1:
        confidence = (risk - threshold) / max(1.0 - threshold, 1e-6)
    else:
        confidence = (threshold - risk) / max(threshold, 1e-6)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    uploaded_metrics = None
    uploaded_metrics_note = None
    metrics_scope = "window_level_any_positive"
    confusion = None
    roc_points = []
    pr_points = []
    label_col = ARTIFACTS.feature_config.get("LABEL_COL")
    if label_col in df_input.columns:
        y_rows = pd.to_numeric(df_input[label_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y_rows = (y_rows > 0.5).astype(np.int32)
        y_windows = []
        for start, end in ranges:
            y_windows.append(int(y_rows[start:end].max() > 0))
        y_windows = np.asarray(y_windows, dtype=np.int32)
        pred_windows = (probs >= threshold).astype(np.int32)

        uploaded_metrics = {
            "accuracy": float(accuracy_score(y_windows, pred_windows)),
            "precision": float(precision_score(y_windows, pred_windows, zero_division=0)),
            "recall": float(recall_score(y_windows, pred_windows, zero_division=0)),
            "f1": float(f1_score(y_windows, pred_windows, zero_division=0)),
            "auroc": None,
            "auprc": None,
            "sample_count": int(len(y_windows)),
            "positive_rate": float(y_windows.mean()) if len(y_windows) > 0 else 0.0,
            "tp": int(((pred_windows == 1) & (y_windows == 1)).sum()),
            "tn": int(((pred_windows == 0) & (y_windows == 0)).sum()),
            "fp": int(((pred_windows == 1) & (y_windows == 0)).sum()),
            "fn": int(((pred_windows == 0) & (y_windows == 1)).sum()),
        }
        if len(np.unique(y_windows)) >= 2:
            uploaded_metrics["auroc"] = float(roc_auc_score(y_windows, probs))
            uploaded_metrics["auprc"] = float(average_precision_score(y_windows, probs))
            fpr, tpr, _ = roc_curve(y_windows, probs)
            prec, rec, _ = precision_recall_curve(y_windows, probs)
            roc_points = [{"x": float(xv), "y": float(yv)} for xv, yv in zip(fpr.tolist(), tpr.tolist())]
            pr_points = [{"x": float(xv), "y": float(yv)} for xv, yv in zip(rec.tolist(), prec.tolist())]
        else:
            uploaded_metrics_note = "Uploaded labels have a single class; AUROC/AUPRC need both classes."

        confusion = {
            "labels": ["Non-Sepsis", "Sepsis"],
            "matrix": [
                [uploaded_metrics["tn"], uploaded_metrics["fp"]],
                [uploaded_metrics["fn"], uploaded_metrics["tp"]],
            ],
        }
    else:
        uploaded_metrics_note = f"Uploaded file does not contain label column '{label_col}'."

    return jsonify(
        {
            "ok": True,
            "patient_id": patient_id,
            "prediction": pred,
            "label": "SEPSIS RISK" if pred == 1 else "LOW RISK",
            "risk_score": round(risk, 6),
            "max_risk_score": round(max_risk, 6),
            "threshold": threshold,
            "confidence": round(confidence, 6),
            "window_count": int(len(probs)),
            "window_probabilities": [float(x) for x in probs.tolist()],
            "preprocessing_trace": steps,
            "preprocessing_viz": preprocessing_viz,
            "stage_viz": stage_viz,
            "uploaded_metrics": uploaded_metrics,
            "uploaded_metrics_note": uploaded_metrics_note,
            "metrics_scope": metrics_scope,
            "confusion_matrix": confusion,
            "roc_curve": roc_points,
            "pr_curve": pr_points,
            "upload_ack": {
                "filename": uploaded_filename,
                "rows_received": int(len(df_input)),
                "columns_received": int(df_input.shape[1]),
                "patient_filter": patient_id,
                "label_column_present": bool(label_col in df_input.columns),
            },
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

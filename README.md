# Faculty Dashboard - MSTCT Sepsis Prediction

This project provides an interactive Flask dashboard for sepsis prediction using a Multi-Scale Temporal Convolution + Transformer model (MSTCT), with stage-wise visualization of the full inference pipeline.

## 1) Project Purpose

The dashboard allows faculty/users to:

- Upload a patient time-series CSV.
- Run the same preprocessing and model inference logic used by the Python backend.
- Visualize each stage of the pipeline:
  - Data preprocessing
  - Feature transformation and windowing
  - TCN stage
  - Transformer stage
  - Final output and evaluation
- View prediction outputs and evaluation metrics dynamically from uploaded data (when labels are present).

## 2) Current Project Structure

- `app.py`: Flask backend + model loading + preprocessing + inference + API payload generation.
- `templates/index.html`: Main dashboard UI layout (stage-based workflow screens).
- `static/app.js`: Frontend behavior, animations, chart rendering, and API interaction.
- `static/styles.css`: Dashboard styling.
- `Dataset.csv`: Main dataset file.
- `mstct_sepsis_model.pth`: Saved PyTorch model checkpoint.
- `feature_scaler.pkl`: Saved scaler used during training.
- `feature_config.json`: Feature definitions, medians, clipping values, and schema metadata.
- `test_patient_017072.csv`: Single-patient test CSV.
- `requirements.txt`: Python dependencies.

## 3) Backend Implementation Details (`app.py`)

### 3.1 Artifact and Path Setup

The backend uses fixed local paths:

- `DATASET_PATH = Dataset.csv`
- `CHECKPOINT_PATH = mstct_sepsis_model.pth`
- `SCALER_PATH = feature_scaler.pkl`
- `FEATURE_CONFIG_PATH = feature_config.json`

### 3.2 Core Utility Functions

- `to_float(value, default)`: Safe numeric conversion with finite checks.
- `normalize_id(value)`: Normalizes patient IDs for robust matching (handles leading zeros).

### 3.3 Model Architecture Classes

The following classes are implemented directly in `app.py`:

- `PositionalEncoding`
- `CausalConvBlock`
- `MultiScaleTCN`
- `MSTCT`

`MSTCT.forward(..., return_intermediates=True)` returns:

- Final logits
- Intermediate tensors for visualization:
  - `input_projected`
  - `tcn_out`
  - `transformer_in`
  - `transformer_out`
  - `pooled`

This is important because the frontend visualizations are derived from these real intermediate outputs.

### 3.4 Artifact Loading

`load_artifacts()` does:

1. Existence check for config/scaler/checkpoint files.
2. Loads:
   - feature config JSON
   - scaler pickle
   - checkpoint (`torch.load(..., weights_only=False)` with backward-compatible fallback)
3. Reconstructs model from checkpoint config.
4. Loads `model_state_dict` and switches model to eval mode.

### 3.5 Overview API Logic

`build_overview()` compiles:

- Artifact readiness and status
- Model params (`sequence_length`, `stride`, threshold, input dim)
- Saved test metrics from checkpoint (`accuracy`, `precision`, `recall`, `f1`, `auroc`, `auprc`)
- Feature group counts
- Dataset profile (`rows`, `columns`, `patients`, prevalence, top missing columns)

### 3.6 Preprocessing Pipeline (`run_preprocessing_trace`)

The inference preprocessing logic follows training-aligned behavior:

1. Validate and add missing expected columns using medians.
2. Convert temporal columns to numeric.
3. Interpolate vital columns (`linear`, both directions).
4. Apply forward/backward fill on temporal features.
5. Fill remaining NaNs with training medians.
6. Fill static columns and unit columns.
7. Generate missingness indicators (`*_missing`) for lab-like features.
8. Generate time-since-observed features (`*_tsince_obs`).
9. Create rolling stats for vitals (`*_roll_mean`, `*_roll_std`).
10. Ensure final engineered feature set exists.
11. Apply clipping using train-time lower/upper bounds.
12. Apply scaler transform.
13. Create fixed-length windows using sequence length + stride.

It returns:

- `windows`: model input tensor array
- `steps`: textual stage trace
- `ranges`: window row ranges
- `preprocessing_viz`: raw/clean sample tables + missingness summaries

### 3.7 Model Stage Visualization Data

Additional helper outputs for UI:

- `summarize_tensor(...)`: shape/mean/std/min/max
- `attention_proxy(...)`: token similarity matrix derived from transformer outputs

Returned under `stage_viz`:

- input window shape
- TCN summary
- Transformer input summary
- Transformer output summary
- attention proxy matrix

### 3.8 Prediction + Metrics Computation

In `/api/predict`:

- File upload is accepted from `file` or `csvFile` field.
- Optional `patient_id` filter is supported.
- Prediction uses sigmoid probabilities from model logits.
- Final class uses checkpoint threshold.

Confidence score:

- If positive prediction: `(risk - threshold) / (1 - threshold)`
- If negative prediction: `(threshold - risk) / threshold`
- Clipped to `[0, 1]`

If label column exists in uploaded input:

- Window-level labels are built with "any-positive-in-window" strategy.
- Computes dynamic metrics on uploaded input:
  - accuracy, precision, recall, f1
  - auroc, auprc (only if both classes present)
  - tp, tn, fp, fn
- Also computes:
  - confusion matrix payload
  - ROC curve points
  - PR curve points

## 4) API Contracts

### 4.1 `GET /api/overview`

Returns:

- artifact status/readiness
- model config summary
- saved checkpoint metrics
- dataset-level profile
- feature-group counts

### 4.2 `POST /api/predict`

Input:

- Multipart form-data (recommended):
  - `file` or `csvFile`: CSV file
  - `patient_id` (optional)
- JSON fallback:
  - `records`
  - `patient_id` (optional)

Output (main fields):

- `prediction`, `label`, `risk_score`, `confidence`, `threshold`
- `window_probabilities`
- `preprocessing_trace`, `preprocessing_viz`
- `stage_viz`
- `uploaded_metrics`, `uploaded_metrics_note`, `metrics_scope`
- `confusion_matrix`, `roc_curve`, `pr_curve`
- `upload_ack`

## 5) Frontend Implementation Details

### 5.1 Main UI Layout (`templates/index.html`)

The UI is structured into workflow stages:

1. Upload
2. Preprocess
3. Features
4. TCN
5. Transformer
6. Output

It includes:

- Upload zone with drag-and-drop and file picker
- Model architecture panel
- Pipeline step tracker
- Navigation controls for stage progression
- Dedicated canvases for stage-wise charts/animations

### 5.2 Frontend Logic (`static/app.js`)

Key behavior:

- Loads overview metadata on startup.
- Handles upload and calls `/api/predict`.
- Updates tracker and transitions stage-by-stage.
- Renders dynamic visuals from backend payload, including:
  - missingness before/after charts
  - raw vs clean tables
  - feature/window stats
  - TCN/Transformer stage visuals
  - attention heatmap
  - probability chart and output metrics

Important: visuals are payload-driven, not fixed constants.

## 6) How to Run

1. Open terminal in project root.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start server:

   ```bash
   python app.py
   ```

4. Open browser:

   - `http://127.0.0.1:5000`

## 7) How to Test Quickly

Use:

- `test_patient_017072.csv`

Recommended test flow:

1. Start app.
2. Upload test file.
3. Leave patient ID empty first.
4. Inspect stage-by-stage outputs.
5. If labels are present, verify dynamic metrics and curve visualizations.

## 8) Dynamic Behavior Guarantee

The pipeline visualization is tied to real backend execution outputs:

- Preprocessing visuals use `preprocessing_viz` from actual transformed data.
- Stage tensors come from real model forward pass (`return_intermediates=True`).
- Output metrics and curve points are computed from uploaded input labels (if available).

No dummy hardcoded prediction results are used in backend API responses.

## 9) Common Issues and Fixes

### Upload fails

- Ensure CSV is valid and readable.
- If filtering by patient ID, verify matching value (leading zeros are normalized).

### AUROC/AUPRC missing

- Uploaded labels may contain only one class; both classes are required.

### Artifact load errors

- Ensure all files exist:
  - `mstct_sepsis_model.pth`
  - `feature_scaler.pkl`
  - `feature_config.json`

### Dependency import errors

- Reinstall packages using `requirements.txt` in the same Python environment used to run Flask.

## 10) Notes for Further Extension

Possible next additions:

- Per-layer feature attribution (SHAP/Integrated Gradients)
- Multi-patient batch upload with cohort comparison
- Exportable pipeline run report (PDF/JSON)
- Real-time streaming input mode for bedside monitoring

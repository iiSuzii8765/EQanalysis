# EQ Philosophy Backend

This repository contains a runnable backend for video-based emotion regulation analytics with OpenFace extraction, Stage 2/3 learned models, Stage 4 philosophy-derived scoring, and Stage 5 Bayesian fusion.

## Abstract

Emotional quotient (EQ) is pertinent in every aspect of life. Philosophical principles combined with EQ can help balance intelligence, support mental health, and strengthen human resources in fast-moving environments. This project operationalizes that idea as a backend system that extracts facial emotion dynamics from unconstrained real-world video and processes them through four analysis layers: spatial perception, temporal modeling, interpretable philosophy-derived feature scoring, and Bayesian score fusion. The philosophy frameworks used are Appraisal (F1), Somatic Marker (F2), Phenomenology (F3), and Cognitive Load (F4). Using AffectNet-based training assets, the system applies temporal sequence modeling with a bidirectional LSTM and uncertainty-aware fusion with Monte Carlo dropout. In the current prototype, Stage 2 reaches validation macro-F1 of about 0.666 on a stratified AffectNet split, and Stage 5 reaches strong correlation and confidence-interval behavior on bootstrapped proxy targets. These results demonstrate architectural novelty, engineering feasibility, and a practical path toward future supervised clinical and executive-coaching validation.

## System architecture

The service is implemented as an asynchronous backend:

- Client uploads video through FastAPI endpoint.
- API stores session metadata in PostgreSQL and enqueues processing via Celery.
- Worker extracts face/action-unit features using OpenFace.
- Stage 2 predicts frame-level affective representations.
- Stage 3 aggregates frame dynamics into temporal sequence signals.
- Stage 4 computes interpretable philosophy-derived scores (F1-F4).
- Stage 5 fuses Stage 4 signals into ERS with uncertainty using Bayesian inference.
- API returns window-level and session-level JSON outputs.

## Five-stage analytics pipeline

### Stage 1: Video preprocessing and facial signal extraction
- Input is an unconstrained uploaded video (`.mp4`, `.mov`, `.webm`).
- Frames are decoded, resized, and aligned with OpenFace outputs.
- OpenFace provides facial landmarks, action units, gaze/head-pose-related signals used downstream.

### Stage 2: Spatial perception (ResNet-based)
- Per-frame model predicts emotion class, valence/arousal, and AU-related targets.
- Trained on AffectNet-derived assets with stratified splitting and class-imbalance mitigation.
- Current best reported validation macro-F1 is around `0.666` in this project setup.

### Stage 3: Temporal modeling (Bi-LSTM)
- Stage 2 frame embeddings/features are transformed into temporal sequences.
- Bi-directional LSTM models sequence context and emotion dynamics.
- Produces sequence-level temporal signals that feed philosophy scoring.

### Stage 4: Interpretable philosophy-derived scoring
- Deterministic rule-based module computes:
  - `F1_appraisal`
  - `F2_somatic`
  - `F3_coherence` (phenomenology coherence proxy)
  - `F4_cognitive` (cognitive load)
- Stage 4 is not trained; it is explicit and interpretable by design.

### Stage 5: Bayesian score fusion (ERS + uncertainty)
- Bayesian MLP with Monte Carlo dropout combines Stage 4 signals into ERS.
- Returns uncertainty-aware outputs (`ERS_uncertainty`, confidence interval bounds).
- Current Stage 5 quality is reported on proxy labels and should be upgraded to supervised targets in future work.

## What is implemented in Week 3

- FastAPI service with async analysis submission endpoint
- PostgreSQL persistence for session lifecycle and result payloads
- Celery + Redis worker queue for background processing
- OpenFace extraction in worker container (`FeatureExtraction`)
- Stage 2 ResNet baseline modules and inference wrapper
- Stage 3 Bi-LSTM baseline modules and inference wrapper
- Refactored Stage 4 rule-based scoring
- Stage 5 Bayesian combiner with MC Dropout inference path
- Docker Compose local environment (API, OpenFace worker, Redis, Postgres)

## Project structure (current)

```text
app/
  main.py            # FastAPI app and API endpoints
  config.py          # Environment settings
  db.py              # SQLAlchemy engine/session/base
  models.py          # AnalysisSession model
  schemas.py         # API response models
  services.py        # Session lifecycle helpers
  celery_app.py      # Celery app config
  tasks.py           # Background task entrypoint
  openface_extractor.py  # OpenFace FeatureExtraction runner
  scoring.py             # Stage-4 score orchestration helpers
  pipeline.py            # Week 3 orchestration pipeline
models/
  stage2_spatial/        # ResNet baseline, loss, training, inference
  stage3_temporal/       # Bi-LSTM baseline, training, inference
  philosophy_module/
    stage4_rule_based/   # F1-F4 functions
    stage5_bayesian/     # BayesPhiloNet training/inference
data/preprocessing/
  extract_frames.py      # Frame decoding/resizing
  segment_windows.py     # Window segmentation helpers
evaluation/
  metrics.py             # Result summarization helpers
  smoke_test.py          # End-to-end API smoke test
```

## Local setup

1. Create env file (if needed):
   - copy `.env.example` to `.env`
2. Build and run (first build is long because worker compiles OpenFace):
   - `docker compose up --build`
3. API docs:
   - [http://localhost:8001/docs](http://localhost:8001/docs)

### Notes for current build

- Worker image compiles OpenFace from source and downloads models.
- Initial worker build can take several minutes.
- If the worker fails during OpenFace compilation, rerun build with:
  - `docker compose build worker --no-cache`
- Week 3 adds PyTorch and torchvision to the images, so rebuild times are longer than Week 2.

## API endpoints

- `GET /health`
- `POST /api/v1/sessions/analyse`
  - form fields:
    - `context` (string)
    - `video` (file: `.mp4`, `.mov`, `.webm`)
- `GET /api/v1/sessions/{session_id}/status`
- `GET /api/v1/sessions/{session_id}/result`

## Quick test flow

1. Submit video in Swagger UI (`/docs`) using `POST /api/v1/sessions/analyse`.
2. Copy returned `session_id`.
3. Poll `GET /status` until `completed`.
4. Fetch `GET /result` for JSON output.

## Output semantics

- `pipeline_version = week3-stage2-stage3-stage5` means the Week 3 orchestration path is active.
- If checkpoints are missing, Stage 2/3/5 fall back to checkpoint-free baseline behavior while preserving the same interfaces.
- Once checkpoints are trained and placed under `artifacts/checkpoints/`, the worker will load them automatically.
- Response now includes `model_status` so you can verify whether Stage 2/3/5 checkpoints were actually loaded.

## Full output JSON example

The following is a full result payload example from a completed async run:

```json
{
  "session_id": "7a82f3da-de18-4732-9d9e-128bf44c2fb9",
  "status": "completed",
  "result": {
    "context": "string",
    "windows": [
      {
        "ERS": 0.816,
        "context": "string",
        "ERS_ci_low": 0.726,
        "ERS_ci_high": 0.887,
        "timestamp_end": 2.96,
        "ERS_uncertainty": 0.053,
        "confidence_flag": true,
        "deployment_mode": "async",
        "microexpression": true,
        "timestamp_start": 0,
        "dominant_emotion": "happiness",
        "philosophy_scores": {
          "F2_somatic": 0.625,
          "F1_appraisal": 0.536,
          "F3_coherence": 0.583,
          "F4_cognitive": 0.583
        },
        "shap_attributions": [
          0.536,
          0.625,
          0.417,
          0.583
        ],
        "regulation_strategy": "expression"
      },
      {
        "ERS": 0.803,
        "context": "string",
        "ERS_ci_low": 0.676,
        "ERS_ci_high": 0.874,
        "timestamp_end": 3.96,
        "ERS_uncertainty": 0.061,
        "confidence_flag": true,
        "deployment_mode": "async",
        "microexpression": true,
        "timestamp_start": 1,
        "dominant_emotion": "happiness",
        "philosophy_scores": {
          "F2_somatic": 0.602,
          "F1_appraisal": 0.559,
          "F3_coherence": 0.601,
          "F4_cognitive": 0.534
        },
        "shap_attributions": [
          0.559,
          0.602,
          0.399,
          0.534
        ],
        "regulation_strategy": "expression"
      },
      {
        "ERS": 0.81,
        "context": "string",
        "ERS_ci_low": 0.74,
        "ERS_ci_high": 0.873,
        "timestamp_end": 4.96,
        "ERS_uncertainty": 0.062,
        "confidence_flag": true,
        "deployment_mode": "async",
        "microexpression": true,
        "timestamp_start": 2,
        "dominant_emotion": "happiness",
        "philosophy_scores": {
          "F2_somatic": 0.553,
          "F1_appraisal": 0.581,
          "F3_coherence": 0.639,
          "F4_cognitive": 0.407
        },
        "shap_attributions": [
          0.581,
          0.553,
          0.361,
          0.407
        ],
        "regulation_strategy": "expression"
      },
      {
        "ERS": 0.731,
        "context": "string",
        "ERS_ci_low": 0.634,
        "ERS_ci_high": 0.807,
        "timestamp_end": 5.96,
        "ERS_uncertainty": 0.064,
        "confidence_flag": true,
        "deployment_mode": "async",
        "microexpression": true,
        "timestamp_start": 3,
        "dominant_emotion": "neutral",
        "philosophy_scores": {
          "F2_somatic": 0.581,
          "F1_appraisal": 0.62,
          "F3_coherence": 0.555,
          "F4_cognitive": 0.374
        },
        "shap_attributions": [
          0.62,
          0.581,
          0.445,
          0.374
        ],
        "regulation_strategy": "expression"
      },
      {
        "ERS": 0.767,
        "context": "string",
        "ERS_ci_low": 0.668,
        "ERS_ci_high": 0.824,
        "timestamp_end": 6.96,
        "ERS_uncertainty": 0.053,
        "confidence_flag": true,
        "deployment_mode": "async",
        "microexpression": true,
        "timestamp_start": 4,
        "dominant_emotion": "neutral",
        "philosophy_scores": {
          "F2_somatic": 0.601,
          "F1_appraisal": 0.62,
          "F3_coherence": 0.559,
          "F4_cognitive": 0.544
        },
        "shap_attributions": [
          0.62,
          0.601,
          0.441,
          0.544
        ],
        "regulation_strategy": "expression"
      },
      {
        "ERS": 0.763,
        "context": "string",
        "ERS_ci_low": 0.653,
        "ERS_ci_high": 0.829,
        "timestamp_end": 7.96,
        "ERS_uncertainty": 0.054,
        "confidence_flag": true,
        "deployment_mode": "async",
        "microexpression": true,
        "timestamp_start": 5,
        "dominant_emotion": "neutral",
        "philosophy_scores": {
          "F2_somatic": 0.638,
          "F1_appraisal": 0.579,
          "F3_coherence": 0.555,
          "F4_cognitive": 0.514
        },
        "shap_attributions": [
          0.579,
          0.638,
          0.445,
          0.514
        ],
        "regulation_strategy": "expression"
      },
      {
        "ERS": 0.765,
        "context": "string",
        "ERS_ci_low": 0.663,
        "ERS_ci_high": 0.851,
        "timestamp_end": 8.96,
        "ERS_uncertainty": 0.062,
        "confidence_flag": true,
        "deployment_mode": "async",
        "microexpression": true,
        "timestamp_start": 6,
        "dominant_emotion": "neutral",
        "philosophy_scores": {
          "F2_somatic": 0.597,
          "F1_appraisal": 0.606,
          "F3_coherence": 0.654,
          "F4_cognitive": 0.445
        },
        "shap_attributions": [
          0.606,
          0.597,
          0.346,
          0.445
        ],
        "regulation_strategy": "expression"
      },
      {
        "ERS": 0.777,
        "context": "string",
        "ERS_ci_low": 0.656,
        "ERS_ci_high": 0.853,
        "timestamp_end": 9.96,
        "ERS_uncertainty": 0.069,
        "confidence_flag": true,
        "deployment_mode": "async",
        "microexpression": true,
        "timestamp_start": 7,
        "dominant_emotion": "neutral",
        "philosophy_scores": {
          "F2_somatic": 0.577,
          "F1_appraisal": 0.65,
          "F3_coherence": 0.648,
          "F4_cognitive": 0.58
        },
        "shap_attributions": [
          0.65,
          0.577,
          0.352,
          0.58
        ],
        "regulation_strategy": "expression"
      },
      {
        "ERS": 0.735,
        "context": "string",
        "ERS_ci_low": 0.631,
        "ERS_ci_high": 0.809,
        "timestamp_end": 10.96,
        "ERS_uncertainty": 0.054,
        "confidence_flag": true,
        "deployment_mode": "async",
        "microexpression": true,
        "timestamp_start": 8,
        "dominant_emotion": "neutral",
        "philosophy_scores": {
          "F2_somatic": 0.532,
          "F1_appraisal": 0.683,
          "F3_coherence": 0.537,
          "F4_cognitive": 0.622
        },
        "shap_attributions": [
          0.683,
          0.532,
          0.463,
          0.622
        ],
        "regulation_strategy": "expression"
      },
      {
        "ERS": 0.775,
        "context": "string",
        "ERS_ci_low": 0.662,
        "ERS_ci_high": 0.85,
        "timestamp_end": 11.96,
        "ERS_uncertainty": 0.06,
        "confidence_flag": true,
        "deployment_mode": "async",
        "microexpression": true,
        "timestamp_start": 9,
        "dominant_emotion": "neutral",
        "philosophy_scores": {
          "F2_somatic": 0.541,
          "F1_appraisal": 0.623,
          "F3_coherence": 0.477,
          "F4_cognitive": 0.597
        },
        "shap_attributions": [
          0.623,
          0.541,
          0.523,
          0.597
        ],
        "regulation_strategy": "expression"
      },
      {
        "ERS": 0.85,
        "context": "string",
        "ERS_ci_low": 0.752,
        "ERS_ci_high": 0.922,
        "timestamp_end": 12.96,
        "ERS_uncertainty": 0.057,
        "confidence_flag": true,
        "deployment_mode": "async",
        "microexpression": true,
        "timestamp_start": 10,
        "dominant_emotion": "neutral",
        "philosophy_scores": {
          "F2_somatic": 0.541,
          "F1_appraisal": 0.56,
          "F3_coherence": 0.659,
          "F4_cognitive": 0.57
        },
        "shap_attributions": [
          0.56,
          0.541,
          0.341,
          0.57
        ],
        "regulation_strategy": "expression"
      },
      {
        "ERS": 0.845,
        "context": "string",
        "ERS_ci_low": 0.768,
        "ERS_ci_high": 0.915,
        "timestamp_end": 13.96,
        "ERS_uncertainty": 0.048,
        "confidence_flag": true,
        "deployment_mode": "async",
        "microexpression": true,
        "timestamp_start": 11,
        "dominant_emotion": "happiness",
        "philosophy_scores": {
          "F2_somatic": 0.542,
          "F1_appraisal": 0.517,
          "F3_coherence": 0.594,
          "F4_cognitive": 0.564
        },
        "shap_attributions": [
          0.517,
          0.542,
          0.406,
          0.564
        ],
        "regulation_strategy": "expression"
      },
      {
        "ERS": 0.817,
        "context": "string",
        "ERS_ci_low": 0.687,
        "ERS_ci_high": 0.895,
        "timestamp_end": 14.96,
        "ERS_uncertainty": 0.063,
        "confidence_flag": true,
        "deployment_mode": "async",
        "microexpression": true,
        "timestamp_start": 12,
        "dominant_emotion": "happiness",
        "philosophy_scores": {
          "F2_somatic": 0.561,
          "F1_appraisal": 0.548,
          "F3_coherence": 0.568,
          "F4_cognitive": 0.542
        },
        "shap_attributions": [
          0.548,
          0.561,
          0.432,
          0.542
        ],
        "regulation_strategy": "expression"
      },
      {
        "ERS": 0.856,
        "context": "string",
        "ERS_ci_low": 0.733,
        "ERS_ci_high": 0.931,
        "timestamp_end": 15.96,
        "ERS_uncertainty": 0.064,
        "confidence_flag": true,
        "deployment_mode": "async",
        "microexpression": true,
        "timestamp_start": 13,
        "dominant_emotion": "happiness",
        "philosophy_scores": {
          "F2_somatic": 0.519,
          "F1_appraisal": 0.535,
          "F3_coherence": 0.573,
          "F4_cognitive": 0.608
        },
        "shap_attributions": [
          0.535,
          0.519,
          0.427,
          0.608
        ],
        "regulation_strategy": "expression"
      },
      {
        "ERS": 0.859,
        "context": "string",
        "ERS_ci_low": 0.765,
        "ERS_ci_high": 0.928,
        "timestamp_end": 16.96,
        "ERS_uncertainty": 0.061,
        "confidence_flag": true,
        "deployment_mode": "async",
        "microexpression": true,
        "timestamp_start": 14,
        "dominant_emotion": "neutral",
        "philosophy_scores": {
          "F2_somatic": 0.565,
          "F1_appraisal": 0.498,
          "F3_coherence": 0.549,
          "F4_cognitive": 0.621
        },
        "shap_attributions": [
          0.498,
          0.565,
          0.451,
          0.621
        ],
        "regulation_strategy": "expression"
      }
    ],
    "session_id": "7a82f3da-de18-4732-9d9e-128bf44c2fb9",
    "video_path": "storage/uploads/03d2bb66-0834-40f1-a5f9-122d9ea348dc.mp4",
    "model_status": {
      "stage2_checkpoint_loaded": true,
      "stage3_checkpoint_loaded": true,
      "stage5_checkpoint_loaded": true
    },
    "windows_count": 15,
    "session_summary": {
      "ERS_mean": 0.798,
      "ERS_peak": 0.859,
      "ERS_variability": 0.041,
      "mean_uncertainty": 0.059,
      "reappraisal_ratio": 0,
      "suppression_ratio": 0,
      "dysregulation_ratio": 0
    },
    "pipeline_version": "week3-stage2-stage3-stage5",
    "openface_csv_path": "artifacts/openface/7a82f3da-de18-4732-9d9e-128bf44c2fb9/03d2bb66-0834-40f1-a5f9-122d9ea348dc.csv",
    "session_processed_at": "2026-04-24T08:28:41.172542+00:00"
  },
  "error_message": null
}
```

## Model artifact paths

- `artifacts/checkpoints/stage2_resnet.pt`
- `artifacts/checkpoints/stage3_bilstm.pt`
- `artifacts/checkpoints/stage5_bayes.pt`

## Training entrypoints

- `python models/stage2_spatial/train_stage2.py ...`
- `python data/preprocessing/build_stage3_tensors.py ...`
- `python models/stage3_temporal/train_stage3.py --tensors-dir <stage3_tensors_dir> ...`
- `python models/stage3_temporal/run_stage3_pipeline.py ...`
- `python models/philosophy_module/stage5_bayesian/train_combiner.py ...`

## Stage 2 checkpoint generation pipeline

Generate a real Stage 2 checkpoint in three steps:

0) (Recommended) Build clean labels from folder names if your dataset is arranged as `images/<emotion>/*`:
- `python data/datasets/affectnet/build_labels_from_folders.py --images-dir data/datasets/affectnet/images --output data/datasets/affectnet/labels_stage2.csv`

1) Build manifest from labeled images:
- `python data/preprocessing/build_stage2_manifest.py --images-dir <images_dir> --labels-csv <labels_csv> --output artifacts/stage2_training/manifest.csv`

2) Create split CSVs + class weights:
- `python data/preprocessing/split_stage2_manifest.py --manifest artifacts/stage2_training/manifest.csv --output-dir artifacts/stage2_training/splits`

3) Train and save checkpoint:
- `python models/stage2_spatial/train_stage2.py --splits-dir artifacts/stage2_training/splits --output artifacts/checkpoints/stage2_resnet.pt --epochs 30 --device cpu`

One-command wrapper:
- `python models/stage2_spatial/run_stage2_pipeline.py --images-dir <images_dir> --labels-csv <labels_csv> --output-root artifacts/stage2_training --epochs 30 --device cpu`

Useful Stage 2 tuning flags:
- `--class-balanced-loss` to reduce majority-class bias
- `--label-smoothing 0.05` to improve generalization
- `--weighted-sampler` for balanced training batches
- `--lambda-au 0.2` to reduce AU-loss dominance when AU labels are weak
- Stage 2 training now logs `val_acc` and `val_f1_macro` each epoch and stores best confusion matrix in metrics JSON

## Stage 3 checkpoint generation pipeline

Generate Stage 3 tensors from Stage 2 split CSVs + a trained Stage 2 checkpoint, then train Bi-LSTM:

1) Build Stage 3 tensors:
- `python data/preprocessing/build_stage3_tensors.py --splits-dir artifacts/stage2_training/splits --stage2-checkpoint artifacts/checkpoints/stage2_resnet.pt --output-dir artifacts/stage3_training/tensors --window-size 75 --stride 25 --device cpu`

2) Train Stage 3 checkpoint:
- `python models/stage3_temporal/train_stage3.py --tensors-dir artifacts/stage3_training/tensors --output artifacts/checkpoints/stage3_bilstm.pt --epochs 20 --device cpu`

One-command wrapper:
- `python models/stage3_temporal/run_stage3_pipeline.py --splits-dir artifacts/stage2_training/splits --stage2-checkpoint artifacts/checkpoints/stage2_resnet.pt --output-root artifacts/stage3_training --epochs 20 --device cpu`

Stage 3 evaluation (test split accuracy/F1/confusion matrix):
- `python models/stage3_temporal/evaluate_stage3.py --tensors-dir artifacts/stage3_training/tensors --checkpoint artifacts/stage3_training/stage3_bilstm.pt --output artifacts/stage3_training/stage3_eval.json --device cpu`

Useful tuning flags:
- `--class-balanced-loss` to mitigate class imbalance
- `--label-smoothing 0.05` to reduce overconfidence
- `--sequence-mode grouped_by_emotion` (recommended for frame datasets like AffectNet)

## Stage 5 checkpoint generation pipeline

Build Stage 5 proxy training data from Stage 3 tensors, train Bayesian combiner, and evaluate:

1) Build Stage 5 dataset:
- `python data/preprocessing/build_stage5_dataset.py --stage3-tensors-dir artifacts/stage3_training/tensors --output-dir artifacts/stage5_training/dataset`

2) Train Stage 5 checkpoint:
- `python models/philosophy_module/stage5_bayesian/train_combiner.py --features artifacts/stage5_training/dataset/train_features.npy --labels artifacts/stage5_training/dataset/train_labels.npy --val-features artifacts/stage5_training/dataset/val_features.npy --val-labels artifacts/stage5_training/dataset/val_labels.npy --output artifacts/checkpoints/stage5_bayes.pt --epochs 120 --device cpu`

3) Evaluate Stage 5 checkpoint:
- `python models/philosophy_module/stage5_bayesian/evaluate_stage5.py --features artifacts/stage5_training/dataset/test_features.npy --labels artifacts/stage5_training/dataset/test_labels.npy --checkpoint artifacts/checkpoints/stage5_bayes.pt --output artifacts/stage5_training/stage5_eval.json --device cpu`

One-command wrapper:
- `python models/philosophy_module/stage5_bayesian/run_stage5_pipeline.py --stage3-tensors-dir artifacts/stage3_training/tensors --output-root artifacts/stage5_training --epochs 120 --device cpu`

## Future changes

- Stage 5 labels are currently proxy ERS targets derived from Stage 4 features.
- Replace proxy labels with real supervised targets (for example EQ-i aligned labels) when available.
- Add leakage-safe temporal evaluation splits (identity/session/source separated).
- Introduce external real-video benchmark validation beyond internal tensor splits.
- Add expert-annotated labels for coaching/clinical scenarios and recalibrate uncertainty thresholds.

Label CSV schema (`data/datasets/stage2_labels_template.csv`):
- `image`: relative filename under `images-dir` (for example `subdir/frame_001.jpg`)
- `emotion`: integer class id in `[0..7]`
- `valence`: float in `[-1,1]`
- `arousal`: float in `[0,1]`

Emotion id mapping used by Stage 2:
- `0=anger, 1=disgust, 2=fear, 3=happiness, 4=sadness, 5=surprise, 6=contempt, 7=neutral`

## Validation

- End-to-end smoke test:
  - `python evaluation/smoke_test.py --video path/to/test.mp4 --context pitch`

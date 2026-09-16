# Validity-Informed Attenuation in Open-Ended Teaching Feedback

An auditable aspect-based sentiment analysis (ABSA) framework that examines rating sensitivity to explicit relevance criteria in open-ended teaching feedback. The method attenuates emotion features from clauses assigned outside a stated pedagogical taxonomy in proportion to their share of review text, then records the resulting prediction change in a fixed rating model. Original reviews and the classification and adjustment trail remain available for inspection.

## Pipeline Overview

The framework processes open-ended teaching reviews through five core stages, with optional validation and robustness analyses:

1. **Preprocessing** — cleans raw reviews (encoding repair, lowercase normalization, abbreviation expansion, punctuation standardization) and applies structural filters.
2. **Aspect-Term Categorization (ATC)** — splits reviews into clauses using spaCy, embeds each clause with `all-mpnet-base-v2`, and assigns it to one of three pedagogical categories (Instructional Effectiveness, Fairness & Grading, Workload & Difficulty) or Miscellaneous via cosine similarity against topic descriptors.
3. **Emotion Extraction** — runs each clause through a quantized RoBERTa-GoEmotions model to produce a 27-dimensional emotion probability vector (Neutral excluded).
4. **Rating Regression** — trains a CatBoost model on topic-level emotion features and density metrics (112 features) with instructor-level grouped cross-validation, evaluated on held-out instructors.
5. **Validity-Informed Attenuation** — down-weights Miscellaneous emotion features by ζ = 1 − D_misc, predicts baseline and attenuated ratings using the fixed model, and records the signed adjustment Δ per review.
6. **ATC Validation** (optional) — inter-rater LLM agreement (Cohen's κ) and expert accuracy scoring against human-labelled clauses.
7. **Attenuation Validation** (optional) — three-expert paired comparison with Fleiss' κ, Wilson intervals, density-only baseline, instructor recurrence audit, and discordant-pair identification.
8. **Robustness** (opt-in) — permutation control, attenuation-exponent sensitivity, and instructor-level bootstrap.

## Project Structure

```
├── pipeline.py              # Main entry point
├── constants.py             # All configuration, paths, and hyperparameters
├── setup.py                 # One-command environment setup (auto-detects GPU)
├── requirements.txt         # Python dependencies (excluding PyTorch)
│
├── src/
│   ├── preprocessing.py     # Stage 1: Data cleaning and text normalization
│   ├── atc.py               # Stage 2: Clause extraction and topic categorization
│   ├── sentiment.py         # Stage 3: RoBERTa-GoEmotions emotion extraction
│   ├── regression.py        # Stage 4: Feature engineering and CatBoost training
│   ├── attenuation.py       # Stage 5: Validity-informed rating adjustment
│   ├── atc_validation.py    # Stage 6: ATC inter-rater agreement and expert accuracy
│   ├── validation.py        # Stage 7: Expert attenuation validation
│   ├── robustness.py        # Stage 8: Permutation control and sensitivity (opt-in)
│   ├── splits.py            # Instructor-level train/test split
│   └── visualizations/
│       ├── descriptive_plots.py    # Topic frequencies, emotion heatmaps
│       ├── attenuation_plots.py    # SHAP, Δ distributions
│       └── correlation_plots.py    # Correlation changes before/after attenuation
│
├── data/
│   ├── raw/                 # Input files (see Data Requirements)
│   └── processed/           # Intermediate and final outputs (generated)
│
└── models/                  # Trained CatBoost model and feature importance
```

## Setup

Requires **Python 3.10**.

```bash
conda create -n absa_env python=3.10
conda activate absa_env
python setup.py
```

`setup.py` detects GPU availability, installs the appropriate PyTorch build, remaining dependencies from `requirements.txt`, and the spaCy language model.

Two pretrained models are downloaded automatically on first run and cached locally:
- `sentence-transformers/all-mpnet-base-v2` (~420 MB) — clause-topic embeddings
- `SamLowe/roberta-base-go_emotions-onnx` (~80 MB) — clause-level emotion extraction

## Usage

Run from the project root with the environment activated.

```bash
# Full pipeline (stages 1-7)
python pipeline.py

# Full pipeline with robustness analyses
python pipeline.py --robustness

# Full pipeline with visualization generation
python pipeline.py --visualize

# Skip validation stages (no expert label files needed)
python pipeline.py --skip-validation --skip-atc-validation

# Generate plots from existing outputs only
python pipeline.py --visualize-only
```

| Flag | Description |
|------|-------------|
| `--skip-validation` | Skip expert attenuation validation |
| `--skip-atc-validation` | Skip ATC validation |
| `--robustness` | Run permutation control and sensitivity analyses |
| `--visualize` | Generate all plots after pipeline completes |
| `--visualize-only` | Skip pipeline stages; generate plots from existing outputs |

Individual stages can be run independently given their required inputs exist:

```bash
python -m src.preprocessing
python -m src.atc
python -m src.sentiment
python -m src.regression
python -m src.attenuation
python -m src.atc_validation
python -m src.validation
python -m src.robustness
```

## Data Requirements

### Required — `data/raw/`

| File | Description |
|------|-------------|
| `RateMyProfessor_Sample data.csv` | RateMyProfessors reviews dataset ([source](https://doi.org/10.17632/fvtfjyvw7d.2)) |

The original corpus is not redistributed. The dataset must contain at minimum: an instructor identifier, institution, a numerical rating (1–5), and review text.

### Optional validation files — `data/raw/`

Required only when running validation stages. Use `--skip-validation` and `--skip-atc-validation` to bypass.

| File | Required by | Description |
|------|-------------|-------------|
| `expert_labels.csv`, `expert2_labels.csv`, `expert3_labels.csv` | `validation.py` | Three experts' paired review judgments |
| `atc_expert_labels.csv` | `atc_validation.py` | Expert-labelled clause categories |
| `atc_predictions.csv` | `atc_validation.py` | ATC predictions for the labelled clause set |
| `tau_selection.csv` | `atc_validation.py` | Tau threshold development set (overlap check) |
| `grok.csv` | `atc_validation.py` | LLM clause categorizations for inter-rater agreement |
| `gpt.csv` | `atc_validation.py` | LLM clause categorizations for inter-rater agreement |

### Generated directories

`data/processed/` and `models/` are populated by the pipeline. They can be empty at the start.

## Output Files

### `data/processed/`

| File | Stage | Description |
|------|-------|-------------|
| `professors_cleaned.csv` | 1 | One row per instructor with average rating and review count |
| `reviews_cleaned.csv` | 1 | Filtered reviews with assigned instructor IDs |
| `reviews_text_cleaned.csv` | 1 | Reviews after text normalization |
| `clause_dataset.csv` | 2 | Reviews with extracted clause lists |
| `exploded_clauses.csv` | 2 | One row per clause with predicted topic and similarity scores |
| `ATExtracted_reviews.csv` | 2 | Pivoted clauses per review with density metrics |
| `final_clause_vectors.parquet` | 3 | Per-clause 28-dimensional GoEmotions probability vectors |
| `final_emotions.csv` | 4 | Per-review feature matrix (112 features) |
| `weighted_emotions.csv` | 5 | Feature matrix after Miscellaneous emotion down-weighting |
| `attuned_ratings.csv` | 5 | Reviews with D_misc > 0: baseline predictions, attenuated predictions, and Δ |
| `attuned_ratings_full.csv` | 5 | All reviews including those with no Miscellaneous content |
| `discordant_pairs.csv` | 7 | Six expert pairs where attenuation and density-only rankings diverge |

### `models/`

| File | Stage | Description |
|------|-------|-------------|
| `cat_boost_final.cbm` | 4 | Trained CatBoost regression model |
| `final_feature_importance.csv` | 4 | Per-feature importance scores |

### Console output

Validation and robustness stages print results to the terminal: accuracy with Wilson intervals, Fleiss' κ, density-only baseline comparison, instructor recurrence, permutation summaries, and sensitivity tables.

## Hardware Requirements & Estimated Runtimes

No GPU is required. ATC embedding benefits from GPU; all other stages run on CPU.

Runtimes below are measured on mid-range hardware (Intel i5-13600K, RTX 4070 Super).

| Stage | Module | Estimated Runtime |
|-------|--------|-------------------|
| Preprocessing | `preprocessing.py` | 3.6 seconds |
| ATC (clause extraction + embedding) | `atc.py` | 1.27 minutes (GPU) / 9 minutes (CPU) |
| Sentiment extraction | `sentiment.py` | 6.61 minutes |
| Regression (5-fold CV + final model) | `regression.py` | 40 seconds |
| Attenuation | `attenuation.py` | 3.6 seconds |
| ATC Validation | `atc_validation.py` | 0.2 seconds |
| Validation | `validation.py` | 0.1 seconds |
| Descriptive visualizations | `descriptive_plots.py` | 7.2 seconds |
| Attenuation visualizations | `attenuation_plots.py` | 5.3 seconds |
| Correlation visualizations | `correlation_plots.py` | 1.2 seconds |
| Robustness | `robustness.py` | 1.1 minutes |

**Total with GPU:** approximately 10 minutes.
**Total CPU only:** approximately 17.1 minutes.

### Disk Space Requirements

**Pipeline outputs** (`data/processed/` + `models/`): ~80 MB

**Python environment** — installed package sizes (approximate):

| Package | Size |
|---------|------|
| PyTorch cu128 | ~3.5 GB |
| CatBoost | ~350 MB |
| SciPy | ~150 MB |
| PyArrow | ~100 MB |
| llvmlite (Numba dependency) | ~120 MB |
| Numba | ~30 MB |
| spaCy | ~30 MB |
| ONNX Runtime | ~50 MB |
| Transformers + Tokenizers | ~45 MB |
| Matplotlib | ~30 MB |
| scikit-learn | ~30 MB |
| All other dependencies | ~150 MB |
| **Subtotal (packages)** | **~4.6 GB** |

**Downloaded model weights** (cached in `~/.cache/`):

| Model | Size |
|-------|------|
| `sentence-transformers/all-mpnet-base-v2` | ~420 MB |
| `SamLowe/roberta-base-go_emotions-onnx` (INT8) | ~80 MB |
| `en_core_web_sm` (spaCy) | ~15 MB |
| **Subtotal (model cache)** | **~515 MB** |

**Total estimated disk usage: ~5.2 GB**

> Note: PyTorch cu128 accounts for the majority of disk usage (~3.5 GB). If running CPU-only, the CPU PyTorch build is approximately 300 MB, reducing total disk usage to approximately 2.0 GB.

**Memory:** peak memory usage occurs during the ATC embedding and sentiment extraction stages. Approximately 2–3 GB RAM is sufficient for datasets up to ~20,000 reviews.

## Ethical Considerations

The study used an existing research dataset. For expert paired comparisons, instructor and institution names were replaced with placeholder tags. All review texts retained in the repository were manually checked and personal names replaced.

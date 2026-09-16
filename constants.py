"""
constants.py
------------
Single source for all configuration values, file paths,
hyperparameters, and domain definitions used across the framework.
"""

from pathlib import Path

# ==============================================================================
# PATHS
# ==============================================================================

# 1. Establish the absolute project root (where constants.py lives)
ROOT_DIR = Path(__file__).resolve().parent

# 2. Anchor all top-level directories to the project root
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# Inputs
RAW_DATASET        = DATA_DIR / "raw" / "RateMyProfessor_Sample data.csv"

# Intermediate outputs (written and read between stages)
PROFESSORS_CLEANED = DATA_DIR / "processed" / "professors_cleaned.csv"
REVIEWS_CLEANED    = DATA_DIR / "processed" / "reviews_cleaned.csv"
REVIEWS_TEXT       = DATA_DIR / "processed" / "reviews_text_cleaned.csv"
CLAUSE_DATASET     = DATA_DIR / "processed" / "clause_dataset.csv"
EXPLODED_CLAUSES   = DATA_DIR / "processed" / "exploded_clauses.csv"
ATC_EXTRACTED      = DATA_DIR / "processed" / "ATExtracted_reviews.csv"
CLAUSE_VECTORS     = DATA_DIR / "processed" / "final_clause_vectors.parquet"
FINAL_EMOTIONS     = DATA_DIR / "processed" / "final_emotions.csv"
WEIGHTED_EMOTIONS  = DATA_DIR / "processed" / "weighted_emotions.csv"

# Final outputs
ATTUNED_RATINGS      = DATA_DIR / "processed" / "attuned_ratings.csv"
ATTUNED_RATINGS_FULL = DATA_DIR / "processed" / "attuned_ratings_full.csv"

# Models
CATBOOST_MODEL       = MODELS_DIR / "cat_boost.cbm"
CATBOOST_FINAL_MODEL = MODELS_DIR / "cat_boost_final.cbm"
FEATURE_IMPORTANCE   = MODELS_DIR / "final_feature_importance.csv"


# ==============================================================================
# PREPROCESSING
# ==============================================================================

MIN_REVIEWS_PER_PROF = 8

# ==============================================================================
# ATC (Aspect Term Categorization)
# ==============================================================================

SIMILARITY_THRESHOLD = 0.25
EMBEDDING_MODEL      = "sentence-transformers/all-mpnet-base-v2"

TOPIC_DESCRIPTIONS = {
    "instructional_effectiveness": {
        "descriptions": [
            "clear and effective explanations of concepts and material",
            "makes difficult topics understandable with good examples or structure",
            "lectures are well-organized, logical, and easy to follow",
            "helps students learn and grasp the subject matter successfully",
            "I've learned a lot / learned more useful stuff than in other classes",
            "very knowledgeable about the topic and conveys it well",
            "provides helpful clarification when students don't understand",
            "answers questions, reteaches concepts, or explains in different ways",
            "always willing to help students understand the content",
            "works with different learning styles to make material accessible",
            "devoted to teaching and making the subject engaging through delivery",
            "educational, interesting, or makes the class intellectually valuable",
            "clear and organized lectures/assignments that aid studying and learning",
        ]
    },
    "fairness": {
        "descriptions": [
            "grading feels fair, earned, and based on actual performance",
            "tough but fair grader, very detailed and consistent evaluation",
            "grades by skill level or progress rather than strict letter grades",
            "adjusts grading to student's current ability or advancement",
            "drops lowest scores or has reasonable grading accommodations",
            "lenient, understanding, or kind-hearted in grading policies",
            "tough grading or strict rules but ultimately fair and preparatory",
            "grade based purely on number of tests, assignments, or clear criteria",
        ]
    },
    "workload": {
        "descriptions": [
            "heavy workload, lots of effort and time required to succeed",
            "very stressful, requires many hours per week on homework/assignments",
            "tests, exams, labs, or quizzes are hard or demanding",
            "easy class, light workload, assignments are very easy",
            "sections or tasks feel easy but pointless or minimal",
            "just show up, pay attention, take notes and you'll be fine",
            "easy exams if you study basic materials like powerpoints",
            "do the packets, board work, assigned readings, or simple daily tasks",
            "specific assignments like vlogs, research papers, major papers required",
            "no textbook, only assigned readings or light materials",
            "must attend class or do basic work to pass / low attendance penalty",
            "didn't go to class much and still passed / low effort to decent grade",
            "intense finals or content that requires knowing everything",
        ]
    },
}


TOPICS = list(TOPIC_DESCRIPTIONS.keys()) + ["misc"]

TOPIC_DISPLAY_NAMES = {
    "instructional_effectiveness": "Instructional\nEffectiveness",
    "workload":                    "Workload",
    "fairness":                    "Fairness",
    "misc":                        "Miscellaneous",
}

TOPIC_DISPLAY_NAMES_WIDE = {
    "instructional_effectiveness": "Instructional Eff.",
    "workload":                    "Workload",
    "fairness":                    "Fairness",
    "misc":                        "Miscellaneous",
}
# Short codes used where full topic names do not fit (e.g. topic-combination labels)
TOPIC_ABBREVIATIONS = {
    "instructional_effectiveness": "IE",
    "workload":                    "W",
    "fairness":                    "F",
    "misc":                        "M",
}

# ==============================================================================
# SENTIMENT (RoBERTa GoEmotions)
# ==============================================================================

ROBERTA_MODEL_REPO = "SamLowe/roberta-base-go_emotions-onnx"
ROBERTA_MODEL_FILE = "onnx/model_quantized.onnx"
ROBERTA_TOKENIZER  = "SamLowe/roberta-base-go_emotions"

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

# Neutral is excluded from downstream modelling
EMOTION_LABELS_NO_NEUTRAL = [e for e in EMOTION_LABELS if e != "neutral"]

POS_EMOTIONS = {
    "admiration", "amusement", "approval", "caring", "desire",
    "excitement", "gratitude", "joy", "love", "optimism",
    "pride", "relief", "curiosity", "realization",
}
NEG_EMOTIONS = {
    "anger", "annoyance", "disappointment", "disapproval",
    "disgust", "embarrassment", "fear", "grief",
    "nervousness", "remorse", "sadness", "confusion",
}

# ==============================================================================
# REGRESSION (CatBoost)
# ==============================================================================

METADATA_COLS = ["review_id", "prof_ID", "rating"]

# No early stopping: every fit runs the full `iterations`. Early stopping needs
# an eval_set, and any set handed to it both sizes the model and is then scored,
# which biases whatever is reported on it. See train_final_model in
# src/regression.py.
CATBOOST_PARAMS = {
    "iterations":            1000,
    "learning_rate":         0.02,
    "depth":                 7,
    "l2_leaf_reg":           6,
    "loss_function":         "MAE",
    "eval_metric":           "MAE",
    "verbose":               100,
}

CV_N_SPLITS  = 5
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ==============================================================================
# ATTENUATION
# ==============================================================================

# Down-weighting exponent, FIXED A PRIORI -- not estimated from data.
#
# At s = 1, zeta = 1 - D_misc: the misc emotion channel is retained in exact
# proportion to the review's on-topic fraction. That is the proportional
# attenuation Section 3.5 argues for on measurement-theoretic grounds, so the
# value needs no justification beyond the argument already in the paper.
#
# It is fixed rather than tuned because the grid search did not identify it. The
# objective is built from the same correlation improvements the paper reports,
# so tuning made those results partly restatements of the objective -- an
# in-criterion problem no train/test split can fix. It was also unstable:
# dropping five blank records (0.06% of the tuning rows) moved the selected
# value from 0.93 to the search floor of 0.2. Fixing s removes both problems.
# The s-sensitivity tables in stage 7 are the evidence that the choice does not
# matter downstream (delta agreement r = 0.95-1.00 across s in [0.37, 1.2]).
S_VALUE = 1.0
MISC_D_COL = "misc_d"



# ==============================================================================
# VALIDATION
# ==============================================================================

# Coarse and fine bins partition the expert pairs by how far apart the two
# reviews' |delta|s are. FINE_DELTA_MAX meets COARSE_DELTA_THRESHOLD exactly:
# the fine bin is half-open [MIN, MAX) and the coarse bin is [THRESHOLD, inf),
# so every pair falls in exactly one.
COARSE_DELTA_THRESHOLD   = 0.5
FINE_DELTA_MIN           = 0
FINE_DELTA_MAX           = COARSE_DELTA_THRESHOLD

EXPERT_LABELS_PATH = (
    DATA_DIR / "raw" / "expert_labels.csv",
    DATA_DIR / "raw" / "expert2_labels.csv",
    DATA_DIR / "raw" / "expert3_labels.csv",
)


# ATC validation
ATC_EXPERT_LABELS        = DATA_DIR / "raw" / "atc_expert_labels.csv"
ATC_PREDICTIONS          = DATA_DIR / "raw" / "atc_predictions.csv"
TAU_SELECTION            = DATA_DIR / "raw" / "tau_selection.csv"
GROK_LABELS              = DATA_DIR / "raw" / "grok.csv"
GPT_LABELS               = DATA_DIR / "raw" / "gpt.csv"


# ==============================================================================
# ROBUSTNESS (Stage 7 — opt-in, not part of a default pipeline run)
# ==============================================================================

# Permutation control: how many times D_misc is shuffled to build the null.
N_PERMUTATIONS   = 1000
PERMUTATION_SEED = 42

# Comparison points for the s-sensitivity tables. S_VALUE is added at runtime,
# so list only the alternatives.
#
# These are spread comparison points, not estimates of anything. Their job is to
# span the range over which s is weakly identified and show that the downstream
# deltas barely move across it; they do not need to be the argmins of any
# particular fit. The first three originate as the modes of a tuning-split
# resampling study (20 professor-level resamples gave SD 0.23, range 0.23–0.94)
# run against an earlier model. That study was a throwaway and its code is not
# in the repo, but the multimodality it found is a property of the loss surface,
# which no change to the model reshapes — so the values remain a fair spread.
#
# 1.2 sits deliberately above the operating point. Since zeta = 1 - D^s and D is
# in (0, 1], a larger s attenuates less, so it probes the low-attenuation
# direction; 0.37 probes the high-attenuation one. Together they bracket
# S_VALUE = 1.0 widely enough that the agreement figures are informative.
SENSITIVITY_S_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5]

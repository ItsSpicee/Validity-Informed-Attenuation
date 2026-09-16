"""
src/sentiment.py
----------------
Stage 3: Per-clause emotion extraction using SamLowe/RoBERTa-GoEmotions (ONNX).

https://huggingface.co/SamLowe/roberta-base-go_emotions-onnx

Produces:
  - final_clause_vectors.parquet  (clause rows + 28 emotion probability columns)
"""

import numpy as np
import pandas as pd
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from tqdm.auto import tqdm

from constants import (
    EXPLODED_CLAUSES,
    CLAUSE_VECTORS,
    ROBERTA_MODEL_REPO,
    ROBERTA_MODEL_FILE,
    ROBERTA_TOKENIZER,
    EMOTION_LABELS,
)


# ==============================================================================
# MODEL SETUP
# ==============================================================================

def load_model() -> tuple[ort.InferenceSession, Tokenizer, str]:
    """Download (or load from cache) the quantized ONNX model and tokenizer."""
    print("  Loading RoBERTa-GoEmotions ONNX model...")
    model_path = hf_hub_download(
        repo_id=ROBERTA_MODEL_REPO,
        filename=ROBERTA_MODEL_FILE,
    )

    tokenizer = Tokenizer.from_pretrained(ROBERTA_TOKENIZER)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.intra_op_num_threads = 2

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    output_name = session.get_outputs()[0].name

    return session, tokenizer, output_name


# ==============================================================================
# INFERENCE
# ==============================================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

"""
Inference uses a batch size of 1 as recommended on the hugging face page for this model.
"""
def run_inference(
    texts: list[str],
    session: ort.InferenceSession,
    tokenizer: Tokenizer,
    output_name: str,
) -> np.ndarray:

    """Return (n_texts, 28) float32 probability matrix."""
    logits = []
    for text in tqdm(texts, desc="  Running inference"):
        enc = tokenizer.encode(text)
        inputs = {
            "input_ids":      np.array([enc.ids],            dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        }
        logits.append(session.run([output_name], inputs)[0][0])

    return _sigmoid(np.vstack(logits).astype(np.float32))


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def run() -> None:
    print("\n=== Stage 3: Sentiment extraction ===")

    print("Loading exploded clauses...")
    df = pd.read_csv(EXPLODED_CLAUSES)
    texts = df["review_clauses"].astype(str).tolist()

    session, tokenizer, output_name = load_model()

    probs = run_inference(texts, session, tokenizer, output_name)

    emotion_df = pd.DataFrame(probs, columns=EMOTION_LABELS)
    df = pd.concat([df.reset_index(drop=True), emotion_df], axis=1)

    CLAUSE_VECTORS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLAUSE_VECTORS, compression="zstd", compression_level=3)
    print(f"  Saved: {CLAUSE_VECTORS}")


if __name__ == "__main__":
    run()

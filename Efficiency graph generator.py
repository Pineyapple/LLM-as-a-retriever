"""V8 Token-Scaling Analysis (GridProbe vs. Naive) - Strategy 1, headless script.

Measures how latency and compute (TFLOPs) scale with document length for the Naive
full-context baseline vs. the sub-quadratic GridProbe, using the BRIGHT dataset.
Writes a results CSV and PNG figures; prints an NDCG@10 comparison. No display needed.

Usage:
    export HF_TOKEN=hf_xxx        # optional; only needed for gated models
    python obliq_scaling_analysis_v8.py

Requires internet (or a warm Hugging Face cache) to fetch the model + BRIGHT dataset.
Dependencies (bitsandbytes only needed if LOAD_IN_4BIT is set True in the config):
    pip install -U transformers accelerate datasets pandas scikit-learn huggingface_hub matplotlib seaborn tqdm
"""


import os
import json
import math
import time
import random
import inspect
from collections import namedtuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")   # headless backend: render figures to files, no display
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sns.set_theme(style="whitegrid")


# Hugging Face authentication (non-interactive: reads HF_TOKEN from the environment).
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("HF_TOKEN not set; continuing unauthenticated (fine for public models).")


# ==========================================================================
# EXPERIMENT CONFIGURATION
# ==========================================================================
MODEL_ID = "Qwen/Qwen3.5-9B"          # must be a valid, accessible HF repo id

# Sampling / grid parameters.
QUERIES_TO_TEST       = 1
DOCS_PER_QUERY        = 50
GOLDEN_DOCS_PER_QUERY = 2
WORDS_PER_CHUNK       = 100            # GridProbe chunk granularity

# Precision. A100 has plenty of VRAM, so run bf16 with no quantization.
LOAD_IN_4BIT = False                  # set True only on small GPUs (e.g. a T4)
COMPUTE_DTYPE = torch.bfloat16        # A100 supports bf16 natively
ATTN_IMPLEMENTATION = "sdpa"          # memory-efficient; no custom masks are used

# OOM guard for the Naive baseline (tokens). Documents longer than this skip the
# Naive pass (recorded as NaN); GridProbe still runs on them. A runtime
# OutOfMemoryError is also caught and treated as a skip (see evaluate_prompt).
MAX_NAIVE_TOKENS = 32000              # A100-40GB-safe; raise toward 64000+ on 80GB

# Measured FLOPs off by default (recording them adds forward passes). Set True to
# also log op-level FLOPs via torch.utils.flop_counter alongside the analytical value.
MEASURE_FLOPS = False

# Output files.
RESULTS_CSV       = "gridprobe_results.csv"
FIG_PATH          = "gridprobe_scaling.png"
FIG_MEASURED_PATH = "gridprobe_scaling_measured.png"


# ==========================================================================
# LOAD MODEL
# ==========================================================================
print(f"Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

load_kwargs = dict(device_map="auto", attn_implementation=ATTN_IMPLEMENTATION)
if LOAD_IN_4BIT:
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=COMPUTE_DTYPE)
    print("  -> 4-bit quantized")
else:
    load_kwargs["torch_dtype"] = COMPUTE_DTYPE
    print(f"  -> full precision ({COMPUTE_DTYPE})")

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
model.eval()
print(f"Model loaded on {model.device} (attn_implementation='{ATTN_IMPLEMENTATION}').")


# --------------------------------------------------------------------------
# Dataset builder: BRIGHT (long_documents) - wide length spread, short -> ~100k words
# --------------------------------------------------------------------------
BRIGHT_DATASET = "xlangai/BRIGHT"
BRIGHT_SPLIT   = "biology"        # research papers; swap for longer-document splits

print(f"Downloading BRIGHT examples + long_documents ({BRIGHT_SPLIT}) ...")
examples_ds = load_dataset(BRIGHT_DATASET, "examples", split=BRIGHT_SPLIT)
corpus_ds   = load_dataset(BRIGHT_DATASET, "long_documents", split=BRIGHT_SPLIT)

corpus_dict = {str(doc["id"]): doc["content"] for doc in corpus_ds}

# Length buckets (words) spanning short documents up to ~100k words.
BRIGHT_THRESHOLDS = [
    (2000,   "xs (<2k words)"),
    (8000,   "s (2k-8k words)"),
    (20000,  "m (8k-20k words)"),
    (50000,  "l (20k-50k words)"),
    (100000, "xl (50k-100k words)"),
    (float("inf"), "xxl (>100k words)"),
]

def bright_bucket(word_count):
    for upper, label in BRIGHT_THRESHOLDS:
        if word_count < upper:
            return label
    return BRIGHT_THRESHOLDS[-1][1]

neg_buckets = {label: [] for _, label in BRIGHT_THRESHOLDS}
print("Bucketing corpus by length ...")
for doc_id, text in corpus_dict.items():
    neg_buckets[bright_bucket(len(text.split()))].append(doc_id)

print("Bucket sizes:")
for label, ids in neg_buckets.items():
    print(f"  {label}: {len(ids)} docs")

# Short -> long ordering; round-robin over it gives an even spread of lengths.
bucket_order = [label for _, label in BRIGHT_THRESHOLDS]

sampled_queries = list(examples_ds)
random.shuffle(sampled_queries)
sampled_queries = sampled_queries[:QUERIES_TO_TEST]

evaluation_dataset = []
for q in sampled_queries:
    q_id = str(q["id"])
    query_text = q["query"]

    g_ids = [str(gid) for gid in (q.get("gold_ids_long") or q.get("gold_ids") or [])]

    final_golden = []
    for gid in g_ids:
        if gid in corpus_dict:
            final_golden.append({"id": gid, "is_golden": True, "text": corpus_dict[gid]})
            if len(final_golden) >= GOLDEN_DOCS_PER_QUERY:
                break

    excluded = set(g_ids) | {str(eid) for eid in q.get("excluded_ids", [])}

    # Even round-robin over buckets, sampling WITHOUT replacement. `exhausted`
    # breaks the loop once no bucket has an unused document (prevents infinite loop).
    negatives_needed = DOCS_PER_QUERY - len(final_golden)
    final_neg = []
    exhausted = False
    while len(final_neg) < negatives_needed and not exhausted:
        exhausted = True
        for b in bucket_order:
            available = [nid for nid in neg_buckets[b] if nid not in excluded]
            if not available:
                continue
            exhausted = False
            if len(final_neg) >= negatives_needed:
                break
            chosen = random.choice(available)
            final_neg.append({"id": chosen, "is_golden": False, "text": corpus_dict[chosen]})
            excluded.add(chosen)

    doc_pool = final_golden + final_neg
    random.shuffle(doc_pool)
    evaluation_dataset.append({"query_id": q_id, "query": query_text, "documents": doc_pool})

lengths = sorted(len(d["text"].split()) for e in evaluation_dataset for d in e["documents"])
print(f"Prepared {len(evaluation_dataset)} query pool(s); "
      f"doc word-lengths span {lengths[0]:,} -> {lengths[-1]:,}.")


# ==========================================================================
# SHARED HELPERS
# ==========================================================================

# --- Analytical FLOPs model ----------------------------------------------
# Read the architecture from the loaded model so the estimate matches reality.
NUM_LAYERS = model.config.num_hidden_layers
D_MODEL    = model.config.hidden_size
PARAMS_BILLIONS = 9        # nominal parameter count of MODEL_ID

def calculate_tflops(num_tokens, params_billions=PARAMS_BILLIONS,
                     num_layers=NUM_LAYERS, d_model=D_MODEL):
    """Analytical forward-pass TFLOPs for `num_tokens` input tokens.

    Two terms, because *where they cross over* is the whole point of the analysis:
      * Linear / projection FLOPs ~ 2 * P * T       (grows with total tokens)
      * Attention FLOPs           ~ 4 * L * d * T^2  (the quadratic bottleneck)
    """
    param_flops     = 2 * (params_billions * 1e9) * num_tokens
    attention_flops = 4 * num_layers * d_model * (num_tokens ** 2)
    return (param_flops + attention_flops) / 1e12

# --- Chunking -------------------------------------------------------------
def chunk_text(text):
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[i:i + WORDS_PER_CHUNK])
            for i in range(0, len(words), WORDS_PER_CHUNK)]

# --- Prompt scaffold (built once) -----------------------------------------
_SYS_PROMPT = "You are an expert reasoning AI."
_QUESTION = (
    "\nOptions:\n"
    "A) Yes, this document is highly relevant to the query.\n"
    "B) No, this document is completely irrelevant to the query.\n\n"
    "Question: Is this document relevant to the query? You MUST choose A or B."
)
_messages = [{"role": "system", "content": _SYS_PROMPT},
             {"role": "user", "content": "PLACEHOLDER"}]
_scaffold = tokenizer.apply_chat_template(_messages, tokenize=False, add_generation_prompt=True)
_scaffold += "\nTo answer, I will select the single best option letter. The correct option is ("
_PREFIX_SCAFFOLD, _SUFFIX_SCAFFOLD = _scaffold.split("PLACEHOLDER")

# Token ids for the "A" / "B" readout (computed once).
OPTION_TOKEN_IDS = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in ("A", "B")]

# --- Forward-pass plumbing -------------------------------------------------
try:
    from torch.utils.flop_counter import FlopCounterMode
    _HAS_FLOP_COUNTER = True
except Exception:
    _HAS_FLOP_COUNTER = False

_OOM_ERROR = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)

# Compute only the final-position logits when supported (big memory saver on long
# docs). The argument was renamed across transformers versions.
_fwd_params = inspect.signature(model.forward).parameters
if "logits_to_keep" in _fwd_params:
    _LOGITS_KWARG = "logits_to_keep"
elif "num_logits_to_keep" in _fwd_params:
    _LOGITS_KWARG = "num_logits_to_keep"
else:
    _LOGITS_KWARG = None

Probe = namedtuple("Probe", ["prob", "latency", "calc_tflops", "meas_tflops"])
_MEAS_CACHE = {}   # token-bucket -> measured TFLOPs (FLOPs are shape-determined)

def _forward_kwargs(input_ids):
    # use_cache=False: we do a single scoring pass and never generate, so the KV
    # cache is pure wasted memory. logits_to_keep=1: only the last logit is needed.
    kw = {"input_ids": input_ids, "use_cache": False}
    if _LOGITS_KWARG:
        kw[_LOGITS_KWARG] = 1
    return kw

def _measured_tflops(forward_kwargs, num_tokens):
    """Op-level TFLOPs via FlopCounterMode, cached per ~256-token bucket."""
    if not _HAS_FLOP_COUNTER:
        return np.nan
    key = round(num_tokens / 256)
    if key in _MEAS_CACHE:
        return _MEAS_CACHE[key]
    try:
        counter = FlopCounterMode(display=False)
        with counter:
            model(**forward_kwargs)
        val = counter.get_total_flops() / 1e12
    except _OOM_ERROR:
        torch.cuda.empty_cache()
        return np.nan
    _MEAS_CACHE[key] = val
    return val

def _build_prompt_ids(query, text_blocks):
    """Tokenize the query + document block(s) into a single input_ids tensor."""
    was_list = not isinstance(text_blocks, str)
    blocks = [text_blocks] if isinstance(text_blocks, str) else list(text_blocks)

    prefix_ids = tokenizer.encode(_PREFIX_SCAFFOLD + f"Query: {query}\n\nDocument:\n",
                                  add_special_tokens=False)
    suffix_ids = tokenizer.encode(_QUESTION + _SUFFIX_SCAFFOLD, add_special_tokens=False)

    ids = list(prefix_ids)
    for i, block in enumerate(blocks):
        chunk_str = f"[CHUNK {i + 1}]: {block}\n\n" if was_list else f"{block}\n\n"
        ids.extend(tokenizer.encode(chunk_str, add_special_tokens=False))
    ids.extend(suffix_ids)
    return torch.tensor([ids], device=model.device)

@torch.no_grad()
def evaluate_prompt(query, text_blocks, measure_flops=False, skip_on_oom=True):
    """Score P(relevant) for a query over one text block (str) or several (list of str).

    On a CUDA OutOfMemoryError, returns an all-NaN Probe when skip_on_oom=True
    (default) instead of raising. Returns Probe(prob, latency, calc_tflops, meas_tflops).
    """
    start_time = time.time()
    input_ids = _build_prompt_ids(query, text_blocks)
    num_tokens = input_ids.shape[1]
    fkw = _forward_kwargs(input_ids)

    try:
        outputs = model(**fkw)
    except _OOM_ERROR:
        del input_ids, fkw
        torch.cuda.empty_cache()
        if skip_on_oom:
            return Probe(np.nan, np.nan, np.nan, np.nan)
        raise

    logits = outputs.logits[0, -1, :].float()
    probs = torch.softmax(logits[OPTION_TOKEN_IDS], dim=0)
    prob_relevant = float(probs[0])
    latency = time.time() - start_time
    calc = calculate_tflops(num_tokens)
    del outputs

    meas = _measured_tflops(fkw, num_tokens) if measure_flops else np.nan

    del input_ids, fkw
    torch.cuda.empty_cache()
    return Probe(prob_relevant, latency, calc, meas)


# ==========================================================================
# STRATEGY 1: Per-document GridProbe
# ==========================================================================
results_log = []

for q_idx, data in enumerate(evaluation_dataset):
    print(f"\nProcessing query {q_idx + 1}/{len(evaluation_dataset)}")
    query_naive_tflops = 0.0
    query_grid_tflops = 0.0

    for doc in tqdm(data["documents"], desc="Documents"):
        doc_tokens = len(tokenizer.encode(doc["text"]))
        doc_words = len(doc["text"].split())

        # 1) Naive baseline (skipped for docs that would OOM; still gets GridProbe).
        if MAX_NAIVE_TOKENS is not None and doc_tokens > MAX_NAIVE_TOKENS:
            naive = Probe(np.nan, np.nan, np.nan, np.nan)
            tqdm.write(f"  [skip naive] {doc['id']}: {doc_tokens} tok > MAX_NAIVE_TOKENS")
        else:
            naive = evaluate_prompt(data["query"], doc["text"], measure_flops=MEASURE_FLOPS)
            if np.isnan(naive.calc_tflops):
                tqdm.write(f"  [skip naive: OOM] {doc['id']}: {doc_tokens} tok")
        if not np.isnan(naive.calc_tflops):
            query_naive_tflops += naive.calc_tflops

        # 2) GridProbe: chunk -> grid -> probe rows and columns.
        chunks = chunk_text(doc["text"])
        N = len(chunks)
        R = math.ceil(math.sqrt(N))
        C = math.ceil(N / R)
        padded = chunks + [""] * (R * C - N)
        grid = [padded[i * C:(i + 1) * C] for i in range(R)]

        g_lat = g_calc = g_meas = 0.0
        row_probs, col_probs = [], []
        for i in range(R):
            res = evaluate_prompt(data["query"], [c for c in grid[i] if c], measure_flops=MEASURE_FLOPS)
            row_probs.append(res.prob); g_lat += res.latency
            g_calc += res.calc_tflops; g_meas += res.meas_tflops
        for j in range(C):
            res = evaluate_prompt(data["query"], [grid[i][j] for i in range(R) if grid[i][j]], measure_flops=MEASURE_FLOPS)
            col_probs.append(res.prob); g_lat += res.latency
            g_calc += res.calc_tflops; g_meas += res.meas_tflops
        query_grid_tflops += g_calc

        # Combine row/column relevance into per-cell scores; keep the top cells.
        cell_scores = [row_probs[i] * col_probs[j]
                       for i in range(R) for j in range(C) if grid[i][j]]
        cell_scores = [s for s in cell_scores if s == s]   # drop NaN from any skipped probe
        cell_scores.sort(reverse=True)
        g_prob = float(np.mean(cell_scores[:5])) if cell_scores else np.nan

        results_log.append({
            "query_id": data["query_id"], "doc_id": doc["id"], "is_golden": doc["is_golden"],
            "tokens": doc_tokens, "words": doc_words,
            "naive_score": naive.prob, "naive_lat": naive.latency,
            "naive_tflops": naive.calc_tflops, "naive_meas_tflops": naive.meas_tflops,
            "grid_score": g_prob, "grid_lat": g_lat,
            "grid_tflops": g_calc, "grid_meas_tflops": g_meas,
        })

    print(f"--- Query {q_idx + 1}: Naive {query_naive_tflops:.2f} TFLOPs | "
          f"GridProbe {query_grid_tflops:.2f} TFLOPs (analytical) ---")


# ==========================================================================
# ANALYSIS: save CSV, plot scaling, report NDCG@10
# ==========================================================================
df = pd.DataFrame(results_log)

# Save the full results first, so the CSV exists even if plotting has issues.
df.to_csv(RESULTS_CSV, index=False)
print(f"Saved {len(df)} rows -> {RESULTS_CSV}")

def scaling_panel(ax, x, y_naive, y_grid, title, xlabel, ylabel):
    sns.scatterplot(data=df, x=x, y=y_naive, ax=ax, color="tab:blue", label="Naive",     alpha=0.6)
    sns.scatterplot(data=df, x=x, y=y_grid,  ax=ax, color="tab:red",  label="GridProbe", alpha=0.6)
    if df[y_naive].notna().any():
        sns.lineplot(data=df, x=x, y=y_naive, ax=ax, color="tab:blue", estimator=np.mean, errorbar=None)
    if df[y_grid].notna().any():
        sns.lineplot(data=df, x=x, y=y_grid,  ax=ax, color="tab:red",  estimator=np.mean, errorbar=None)
    ax.set_title(title, fontsize=14); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

# Figure 1: latency + analytical TFLOPs vs tokens and words.
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
scaling_panel(axes[0, 0], "tokens", "naive_lat",    "grid_lat",    "Latency vs Tokens",        "Document Length (Tokens)", "Latency (s)")
scaling_panel(axes[0, 1], "tokens", "naive_tflops", "grid_tflops", "Analytical Cost vs Tokens", "Document Length (Tokens)", "Cost (TFLOPs)")
scaling_panel(axes[1, 0], "words",  "naive_lat",    "grid_lat",    "Latency vs Words",         "Document Length (Words)",  "Latency (s)")
scaling_panel(axes[1, 1], "words",  "naive_tflops", "grid_tflops", "Analytical Cost vs Words",  "Document Length (Words)",  "Cost (TFLOPs)")
plt.tight_layout()
fig.savefig(FIG_PATH, dpi=120, bbox_inches="tight")
print(f"Saved figure -> {FIG_PATH}")

# Figure 2: MEASURED TFLOPs (only if MEASURE_FLOPS produced data).
if df[["naive_meas_tflops", "grid_meas_tflops"]].notna().any().any():
    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 6))
    scaling_panel(axes2[0], "tokens", "naive_meas_tflops", "grid_meas_tflops",
                  "Measured Cost vs Tokens", "Document Length (Tokens)", "Measured Cost (TFLOPs)")
    scaling_panel(axes2[1], "words", "naive_meas_tflops", "grid_meas_tflops",
                  "Measured Cost vs Words", "Document Length (Words)", "Measured Cost (TFLOPs)")
    plt.tight_layout()
    fig2.savefig(FIG_MEASURED_PATH, dpi=120, bbox_inches="tight")
    print(f"Saved measured figure -> {FIG_MEASURED_PATH}")

# NDCG@10.
def ndcg_at_k(scored_docs, num_relevant, k=10):
    ranked = sorted(scored_docs, key=lambda d: d["score"], reverse=True)
    dcg = sum(1.0 / np.log2(rank + 1)
              for rank, d in enumerate(ranked, start=1)
              if d["is_golden"] and rank <= k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(k, num_relevant) + 1))
    return dcg / idcg if idcg > 0 else 0.0

def _rank_score(v):
    return v if v == v else -np.inf   # NaN (skipped doc) ranks last

naive_ndcgs, grid_ndcgs = [], []
for q_id in df["query_id"].unique():
    q_df = df[df["query_id"] == q_id]
    num_relevant = int(q_df["is_golden"].sum())
    if num_relevant == 0:
        continue
    naive_ndcgs.append(ndcg_at_k(
        [{"score": _rank_score(r.naive_score), "is_golden": r.is_golden} for r in q_df.itertuples()], num_relevant))
    grid_ndcgs.append(ndcg_at_k(
        [{"score": _rank_score(r.grid_score), "is_golden": r.is_golden} for r in q_df.itertuples()], num_relevant))

print(f"\n{'=' * 40}\nOVERALL ACCURACY (NDCG@10)\n{'=' * 40}")
print(f"Naive baseline:      {np.mean(naive_ndcgs):.4f}")
print(f"GridProbe (Strat 1): {np.mean(grid_ndcgs):.4f}")

plt.close("all")

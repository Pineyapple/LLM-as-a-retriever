# LLM-as-a-Retriever

Using a generative LLM as the relevance judge for document retrieval, and asking a
practical question: **you don't have to feed the whole document to the model to rank it.**
Instead of reading a document end-to-end (which is quadratic in length and eventually
runs out of context), the document is chunked and probed with cheaper reading strategies,
and their accuracy/cost trade-offs are compared on the [BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT)
benchmark.

## Reading strategies compared

Every strategy scores the same `Query + Document → "Is this relevant? Yes/No"` prompt and
ranks documents by the log-odds of the "Yes" token. They differ only in **how much of the
document reaches the model**:

| Strategy | What it sends |
|----------|---------------|
| **naive / FullRead** | the whole document in one prompt (the baseline; fails when it exceeds the context budget) |
| **firstp** | only the first chunk |
| **perchunk (MaxP)** | one probe per chunk, take the max — true MaxP (Dai & Callan) |
| **rows** | √N probes, each a span of consecutive chunks |
| **grid / GridProbe** | arranges chunks in a √N × √N grid and probes rows *and* columns (sub-quadratic) |

The central claim being tested: **GridProbe** approaches FullRead accuracy at a fraction of
the compute, and keeps working on documents too long for FullRead to fit at all.

---

## Repository contents

| File | Purpose |
|------|---------|
| `bright_runner.py` | Main experiment. Runs all reading modes on BRIGHT in one process (via [vLLM](https://github.com/vllm-project/vllm)) so the model and pools are held equal. Outputs per-domain metric tables. |
| `Efficiency graph generator.py` | Token-scaling study: measures how latency and compute (TFLOPs) grow with document length, GridProbe vs. Naive. Uses `transformers` directly and writes CSV + PNG figures. |
| `Weighted localization code.py` | Standalone evaluator for the 2D grid-cell *localization* task — scores predicted cell coordinates against annotated ground truth with a suite of custom rank-weighted metrics. |
| `annotations.xlsx` | Ground-truth annotations: for each query, which cell(s) hold the answer. |
| `annotations with coords of 50 chunk size qwen.xlsx` | Same annotations plus resolved `(row, col)` grid coordinates (Qwen, 50-token chunks). |

---

## Setup

Requires **Python 3.9+**. A CUDA GPU is needed for `bright_runner.py` and
`Efficiency graph generator.py` (they load an LLM); the localization evaluator runs on CPU.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# For the two model-driven scripts:
pip install -U vllm transformers accelerate datasets huggingface_hub \
               numpy pandas tqdm matplotlib seaborn scikit-learn tabulate

# For the localization evaluator only:
pip install pandas numpy openpyxl
```

The BRIGHT dataset and the model weights download automatically from Hugging Face on first
run. To control where they cache, set `HF_HOME` (or pass `--cache-dir` to `bright_runner.py`):

```bash
export HF_HOME=/path/to/hf_cache
export HF_TOKEN=hf_xxx   # only needed for gated/private models
```

---

## Running the code

### 1. Main benchmark — `bright_runner.py`

Runs every reading mode on one (or all) BRIGHT domains and prints a metrics table
(NDCG@5/10, Recall, AUC, Success%, TFLOPs, latency).

```bash
python bright_runner.py --domain biology --queries 10 --pool-size 100
```

Common options:

| Flag | Meaning |
|------|---------|
| `--domain` | `biology`, `earth_science`, `economics`, `pony`, `psychology`, `robotics`, `stackoverflow`, `sustainable_living`, or `ALL` |
| `--queries N` | queries per domain (`0` = all) |
| `--pool-size N` | `0` = full-corpus retrieval (comparable to published BRIGHT numbers). `>0` = pooled reranking with golds injected (measures the judge, **not** retrieval — not comparable to BRIGHT rows) |
| `--modes` | comma list of `naive,perchunk,rows,grid,firstp` |
| `--max-model-len` | context budget = the OOM wall to emulate (e.g. `8192` ≈ 16 GB card, `32768` ≈ 80 GB). FullRead is scored as FAILED above this. |
| `--model` | default `Qwen/Qwen3-4B-Instruct-2507` |
| `--server URL` | score through a running `vllm serve` instead of loading the model in-process (lets several domains share one GPU) |

Results are written to `runs_bright/` as `.pkl` checkpoints (resumable — rerun the same
command to continue) plus a summary `table_*.csv`.

> **Note:** `Success%` is relative to the `--max-model-len` budget. The OOM wall is
> hardware-dependent, so always report it together with the GPU it was measured on.

### 2. Efficiency / scaling figures — `Efficiency graph generator.py`

Measures latency and TFLOPs vs. document length for Naive vs. GridProbe, and reports an
NDCG@10 comparison. Configuration lives in the `EXPERIMENT CONFIGURATION` block near the
top of the file (model id, docs per query, chunk size, OOM guard). Then just run it:

```bash
python "Efficiency graph generator.py"
```

Outputs: `gridprobe_results.csv` and the figures `gridprobe_scaling.png`
(and `gridprobe_scaling_measured.png` if `MEASURE_FLOPS = True`).

### 3. Localization metrics — `Weighted localization code.py`

Scores a predictions file (top-10 retrieved grid cells per query) against the annotated
ground truth, using custom rank-weighted metrics (Top-N overlap, Top-3 precision,
rank-weighted Top-3, priority-cell reward, and a composite).

```bash
python "Weighted localization code.py" \
    --predictions top10gridcells.csv \
    --ground_truth "annotations with coords of 50 chunk size qwen.xlsx"
```

Both file paths also have defaults at the top of the script. Outputs
`custom_metrics_evaluation_results.xlsx` / `.csv` and prints dataset-wide averages.

---

## Tips

- Start small (`--queries 5 --pool-size 50`) to confirm everything loads before a full run.
- Out of GPU memory? Lower `--max-model-len`, switch to a smaller `--model`, or (in
  `Efficiency graph generator.py`) set `LOAD_IN_4BIT = True`.
- `bright_runner.py` checkpoints after every query, so an interrupted run resumes where it
  left off — pass `--overwrite` to rescore from scratch.

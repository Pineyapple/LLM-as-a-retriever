from IPython.display import clear_output
#!pip install datasets huggingface_hub torch accelerate scipy bitsandbytes
#!pip install -U transformers
import gc
import json
import math
import random
import time
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
clear_output()

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

clear_output()

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)

MODELS = [
    # gemma-4-E2B-it is MatFormer-based: "2.3B effective" params but a larger on-disk footprint
    # (Gemma 3n's E2B precedent was 5B total on disk) -- quantized for memory headroom accordingly.
    {"id": "google/gemma-4-E2B-it", "quantize": True, "n_active_params": 2.3e9, "score_temperature": 1.0},
]

# Point this at the file's location as a Kaggle input, e.g.
# "/kaggle/input/<your-dataset-slug>/candidates_congress_top300_goldinjected.jsonl"
DATA_PATH = "candidates_congress_top300_goldinjected.jsonl"

# This file has 20 queries, each with its own ~300-doc hard-candidate pool. Naive/per-chunk/grid
# reads are far pricier per document than the Level-1 skim work elsewhere in this repo, so this
# run samples a small subset per query (gold guaranteed + random negatives) rather than using the
# full ~300-doc pool -- a deliberate first-pass scope, bump these once the mechanics check out.
N_QUERIES = 10
SAMPLE_SIZE = 20
SEED = 67

EVAL_KS = [5, 10]
TARGET_CHUNK_TOKENS = 100   # drives per-document K (grid granularity) -- NOT a truncation cap
ROW_AGGREGATION = "peak"   # "peak" (max) or "avg" (mean) -- same open ablation as elsewhere

# Stride step for the strided GridProbe variants: read every STRIDE-th row/column instead of all
# k. STRIDE=2 means row0, row2, row4, ... (skip every other one) -- the knob that controls how
# much of the k x k grid factorization actually gets approximated vs read in full.
STRIDE = 2

# One model copy per visible GPU (e.g. ["cuda:0", "cuda:1"] on Kaggle's T4 x2), so the query loop
# below can hand each device its own share of the work and run them concurrently. Falls back to a
# single device (GPU or CPU) automatically if only one is visible.
if torch.cuda.is_available():
    DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
else:
    DEVICES = ["cpu"]
print(f"Detected devices: {DEVICES}")


def get_model_dims(m):
    config = m.config
    text_config = getattr(config, "text_config", config)
    return text_config.num_hidden_layers, text_config.hidden_size


def load_data(path):
    # Each line is a full query with its own pre-built candidate pool -- gold docs stage-1 missed
    # were injected (rank=None, injected=True) so is_gold is the only reliable gold signal.
    queries_data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            queries_data.append(json.loads(line))
    return queries_data


def build_query_pool(query_obj):
    candidates = query_obj["candidates"]
    corpus = {c["doc_id"]: c["text"] for c in candidates}
    gold_ids = {c["doc_id"] for c in candidates if c["is_gold"]}
    other_ids = [cid for cid in corpus if cid not in gold_ids]
    random.Random(SEED).shuffle(other_ids)
    filler = other_ids[: max(0, SAMPLE_SIZE - len(gold_ids))]
    sample_ids = sorted(gold_ids | set(filler))
    return sample_ids, corpus, gold_ids


def build_relevance_prompt(query_text, chunks_text):
    return f"""You are an expert relevance judge for OBLIQUE queries — queries where relevance is latent.
    A chunk can be relevant even if it shares no keywords or topic with the query
    — relevance may show up as an implicit signal, a structural or abstract similarity,
    a tone, or a fuzzy impressionistic resemblance rather than explicit content overlap. Do not judge
    relevance by keyword or topic overlap alone.

    Query:
    {query_text}

    Candidate chunks:
    {chunks_text}

    Question: Is ANY chunk above relevant to the query, even if only through implicit, structural, or non-obvious similarity?
    Answer with a single word: Yes or No.
    """


def format_chunks(chunks):
    return "\n\n".join(f"chunk {i+1}: {c}" for i, c in enumerate(chunks))


def recall_at_k(ranked_has_gold, k):
    total_gold = sum(ranked_has_gold)
    if total_gold == 0:
        return None
    return sum(ranked_has_gold[:k]) / total_gold


def ndcg_at_k(ranked_has_gold, k):
    total_gold = sum(ranked_has_gold)
    if total_gold == 0:
        return None
    dcg = sum(1.0 / math.log2(rank + 2) for rank, hit in enumerate(ranked_has_gold[:k]) if hit)
    ideal_hits = min(total_gold, k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else None


def auc_score(scores_and_labels):
    pos = [s for s, has_gold in scores_and_labels if has_gold]
    neg = [s for s, has_gold in scores_and_labels if not has_gold]
    if not pos or not neg:
        return None
    wins = sum((sp > sn) + 0.5 * (sp == sn) for sp in pos for sn in neg)
    return wins / (len(pos) * len(neg))


def evaluate_doc_scores(doc_scores, gold_in_sample):
    pairs = [(score, doc_id in gold_in_sample) for doc_id, score in doc_scores.items()]
    ranked = [has_gold for _, has_gold in sorted(pairs, key=lambda x: x[0], reverse=True)]
    metrics = {
        "recall": {k: recall_at_k(ranked, k) for k in EVAL_KS},
        "ndcg": {k: ndcg_at_k(ranked, k) for k in EVAL_KS},
    }
    return metrics, auc_score(pairs)


def build_rows_cols(k, chunks):
    rows = [chunks[r * k : (r + 1) * k] for r in range(k)]
    cols = [chunks[c::k] for c in range(k)]
    return rows, cols


# ---- Worker: one full model copy pinned to one device ----

class Worker:
    def __init__(self, device):
        self.device = device
        self.tokenizer = None
        self.model = None
        self.n_layers = None
        self.d_model = None
        self.n_active_params = None
        self.score_temperature = None
        self.option_ids = None

    def load(self, model_cfg):
        print(f"\n[{self.device}] Loading model: {model_cfg['id']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["id"])
        if model_cfg["quantize"]:
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_cfg["id"], dtype=torch.bfloat16, quantization_config=quant_config
            ).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_cfg["id"], dtype=torch.bfloat16).to(self.device)

        self.n_layers, self.d_model = get_model_dims(self.model)
        self.n_active_params = model_cfg["n_active_params"]
        self.score_temperature = model_cfg["score_temperature"]
        self.option_ids = None

    def unload(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def estimate_tflops(self, context_length):
        flops = 2 * self.n_active_params * context_length + 2 * self.n_layers * self.d_model * context_length**2
        return flops / 1e12

    def prompt_inputs(self, prompt):
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def yesno_token_ids(self):
        ids = {}
        for label, variants in (("Yes", ["Yes", " Yes", "yes", " yes"]), ("No", ["No", " No", "no", " no"])):
            for variant in variants:
                ids[self.tokenizer.encode(variant, add_special_tokens=False)[-1]] = label
        return ids

    def score_relevance(self, query_text, chunks_text):
        if self.option_ids is None:
            self.option_ids = self.yesno_token_ids()

        prompt = build_relevance_prompt(query_text, chunks_text)
        inputs = self.prompt_inputs(prompt)
        context_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]

        ids = list(self.option_ids)
        probs = torch.softmax(next_token_logits[ids] / self.score_temperature, dim=0)
        yes_prob = sum(p for tid, p in zip(ids, probs) if self.option_ids[tid] == "Yes")
        return float(yes_prob), self.estimate_tflops(context_length)

    def split_document(self, text):
        # Divides the FULL document into k*k pieces -- every token belongs to exactly one chunk,
        # nothing is cropped. Rows = k consecutive chunks (local/consecutive context); columns =
        # every k-th chunk (spread across the whole document -- global context).
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        total = len(token_ids)
        k = max(2, math.ceil(math.sqrt(total / TARGET_CHUNK_TOKENS)))
        k = min(k, max(1, math.isqrt(total)))
        n_chunks = k * k
        base, remainder = divmod(total, n_chunks)
        chunks, start = [], 0
        for i in range(n_chunks):
            size = base + (1 if i < remainder else 0)
            chunks.append(self.tokenizer.decode(token_ids[start : start + size], skip_special_tokens=True))
            start += size
        return k, chunks


# ---- the 6 methods, run head-to-head on the SAME pool per query (no funnel, no M_eff gating) ----

def run_naive(worker, query_text, sample_ids, corpus):
    # Congress excerpts are far shorter than BRIGHT's long_documents, so an OOM here is unlikely,
    # but the same safety net is kept: skip (never truncate) and keep going if it happens anyway.
    t0 = time.perf_counter()
    scores = {}
    total_tflops = 0.0
    failed = 0
    for doc_id in sample_ids:
        try:
            score, tflops = worker.score_relevance(query_text, format_chunks([corpus[doc_id]]))
            scores[doc_id] = score
            total_tflops += tflops
        except torch.cuda.OutOfMemoryError:
            failed += 1
            torch.cuda.empty_cache()
            print(f"    [naive] OOM on doc={doc_id} -- skipped, not truncated ({len(corpus[doc_id])} chars)")
    latency = time.perf_counter() - t0
    return {"scores": scores, "calls": len(sample_ids), "latency": latency, "tflops": total_tflops, "failed": failed}


def run_per_chunks_and_gridprobe(worker, query_text, sample_ids, corpus):
    # Full row pass is shared: "per_chunks" (renamed from row-verifier) IS this pass, and the
    # un-strided GridProbe reuses the same row scores -- only the (also full) column pass is
    # extra work. Both need the SAME full k rows, so sharing here is exact, not an approximation.
    row_t0 = time.perf_counter()
    doc_grids = {}
    row_scores_by_doc = {}
    row_calls = 0
    row_tflops = 0.0
    for doc_id in sample_ids:
        k, chunks = worker.split_document(corpus[doc_id])
        rows, cols = build_rows_cols(k, chunks)
        row_scores = []
        for row in rows:
            score, tflops = worker.score_relevance(query_text, format_chunks(row))
            row_scores.append(score)
            row_tflops += tflops
            row_calls += 1
        doc_grids[doc_id] = {"k": k, "cols": cols, "row_scores": row_scores}
        row_scores_by_doc[doc_id] = max(row_scores) if ROW_AGGREGATION == "peak" else sum(row_scores) / len(row_scores)
    row_latency = time.perf_counter() - row_t0

    col_t0 = time.perf_counter()
    grid_scores = {}
    col_calls = 0
    col_tflops = 0.0
    for doc_id in sample_ids:
        grid = doc_grids[doc_id]
        k = grid["k"]
        col_scores = []
        for col in grid["cols"]:
            score, tflops = worker.score_relevance(query_text, format_chunks(col))
            col_scores.append(score)
            col_tflops += tflops
            col_calls += 1
        grid_scores[doc_id] = max(grid["row_scores"][r] * col_scores[c] for r in range(k) for c in range(k))
    col_latency = time.perf_counter() - col_t0

    per_chunks = {
        "scores": row_scores_by_doc, "calls": row_calls, "latency": row_latency, "tflops": row_tflops, "failed": 0,
    }
    gridprobe = {
        "scores": grid_scores,
        "calls": row_calls + col_calls, "latency": row_latency + col_latency, "tflops": row_tflops + col_tflops,
        "failed": 0,
    }
    return per_chunks, gridprobe


def run_gridprobe_strided(worker, query_text, sample_ids, corpus, stride_rows, stride_cols):
    # Independent row/col passes using only every stride-th row/column -- NOT reused from the full
    # pass above, since the whole point is to measure what striding actually saves in calls, not
    # just to subset an already-paid-for full read. stride_rows=stride_cols=1 would reproduce
    # un-strided GridProbe (kept separate above so the two full-read methods can share their pass).
    t0 = time.perf_counter()
    scores = {}
    total_tflops = 0.0
    total_calls = 0
    for doc_id in sample_ids:
        k, chunks = worker.split_document(corpus[doc_id])
        rows, cols = build_rows_cols(k, chunks)
        row_indices = list(range(0, k, stride_rows))
        col_indices = list(range(0, k, stride_cols))

        row_scores = {}
        for r in row_indices:
            score, tflops = worker.score_relevance(query_text, format_chunks(rows[r]))
            row_scores[r] = score
            total_tflops += tflops
            total_calls += 1

        col_scores = {}
        for c in col_indices:
            score, tflops = worker.score_relevance(query_text, format_chunks(cols[c]))
            col_scores[c] = score
            total_tflops += tflops
            total_calls += 1

        scores[doc_id] = max(row_scores[r] * col_scores[c] for r in row_indices for c in col_indices)
    latency = time.perf_counter() - t0
    return {"scores": scores, "calls": total_calls, "latency": latency, "tflops": total_tflops, "failed": 0}


ALL_METHODS = [
    "naive", "per_chunks", "gridprobe",
    "gridprobe_stride_both", "gridprobe_stride_rows", "gridprobe_stride_cols",
]


def run_query_congress(worker, query_obj):
    sample_ids, corpus, gold_ids = build_query_pool(query_obj)
    query_text = query_obj["query"]
    gold_in_sample = gold_ids & set(sample_ids)

    naive = run_naive(worker, query_text, sample_ids, corpus)
    per_chunks, gridprobe = run_per_chunks_and_gridprobe(worker, query_text, sample_ids, corpus)
    # both dims strided; rows only (cols full); cols only (rows full) -- same STRIDE step throughout.
    grid_both = run_gridprobe_strided(worker, query_text, sample_ids, corpus, STRIDE, STRIDE)
    grid_rows = run_gridprobe_strided(worker, query_text, sample_ids, corpus, STRIDE, 1)
    grid_cols = run_gridprobe_strided(worker, query_text, sample_ids, corpus, 1, STRIDE)

    results = {}
    for name, r in (
        ("naive", naive), ("per_chunks", per_chunks), ("gridprobe", gridprobe),
        ("gridprobe_stride_both", grid_both), ("gridprobe_stride_rows", grid_rows), ("gridprobe_stride_cols", grid_cols),
    ):
        metrics, auc = evaluate_doc_scores(r["scores"], gold_in_sample)
        results[name] = {
            "calls": r["calls"], "latency": r["latency"], "tflops": r["tflops"],
            "failed": r.get("failed", 0), "metrics": metrics, "auc": auc,
        }
    return results, len(gold_in_sample)


def split_round_robin(items, n):
    return [items[i::n] for i in range(n)]


def run_worker_queries(worker, query_objs):
    if worker.device.startswith("cuda"):
        torch.cuda.set_device(worker.device)

    per_method = {name: [] for name in ALL_METHODS}
    logs = []
    for query_obj in query_objs:
        q_start = time.perf_counter()
        results, gold_in_sample = run_query_congress(worker, query_obj)
        for name in ALL_METHODS:
            per_method[name].append(results[name])
        qid = query_obj["query_id"]
        logs.append(f"[{worker.device}] {qid} done in {time.perf_counter() - q_start:.2f}s (gold_in_sample={gold_in_sample})")
    return per_method, logs


def aggregate(results_list):
    agg = {
        "calls": sum(r["calls"] for r in results_list) / len(results_list),
        "latency": sum(r["latency"] for r in results_list) / len(results_list),
        "tflops": sum(r["tflops"] for r in results_list) / len(results_list),
        "failed": sum(r["failed"] for r in results_list) / len(results_list),
        "recall": {}, "ndcg": {},
    }
    for name in ("recall", "ndcg"):
        for k in EVAL_KS:
            vals = [r["metrics"][name][k] for r in results_list if r["metrics"][name][k] is not None]
            agg[name][k] = sum(vals) / len(vals) if vals else float("nan")
    aucs = [r["auc"] for r in results_list if r["auc"] is not None]
    agg["auc"] = sum(aucs) / len(aucs) if aucs else float("nan")
    return agg


def print_grid(headers, rows):
    widths = [max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) + 2 for i in range(len(headers))]
    print("".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("".join(v.ljust(w) for v, w in zip(r, widths)))


def print_results_table(agg):
    headers = ["Method", "Calls", "Failed", "Latency(s)", "Lat/Call(s)", "TFLOPs"]
    for k in EVAL_KS:
        headers += [f"Recall@{k}", f"NDCG@{k}"]
    headers.append("AUC")
    rows = []
    for name in ALL_METHODS:
        m = agg[name]
        lat_per_call = m["latency"] / m["calls"] if m["calls"] else 0.0
        row = [
            name, f"{m['calls']:.1f}", f"{m['failed']:.1f}",
            f"{m['latency']:.2f}", f"{lat_per_call:.2f}", f"{m['tflops']:.2f}",
        ]
        for k in EVAL_KS:
            row += [f"{m['recall'][k]:.3f}", f"{m['ndcg'][k]:.3f}"]
        row.append(f"{m['auc']:.3f}")
        rows.append(row)
    print_grid(headers, rows)


def main():
    all_queries = load_data(DATA_PATH)
    queries_to_run = all_queries[:N_QUERIES]
    print(f"Loaded {len(all_queries)} queries from {DATA_PATH}, running {len(queries_to_run)} across {len(DEVICES)} device(s)")

    for model_cfg in MODELS:
        model_name = model_cfg["id"]
        safe_model_name = model_name.replace("/", "_")

        workers = [Worker(d) for d in DEVICES]
        for w in workers:
            w.load(model_cfg)

        print(f"\n{'#'*70}\n MODEL: {model_name}  |  DATASET: congress top-300 gold-injected (sampled)\n{'#'*70}")

        chunks = split_round_robin(queries_to_run, len(workers))
        with ThreadPoolExecutor(max_workers=len(workers)) as ex:
            futures = [ex.submit(run_worker_queries, w, chunk) for w, chunk in zip(workers, chunks)]
            per_worker_results = [f.result() for f in futures]

        per_method = {name: [] for name in ALL_METHODS}
        for worker_per_method, logs in per_worker_results:
            for line in logs:
                print(line)
            for name in ALL_METHODS:
                per_method[name].extend(worker_per_method[name])

        agg = {name: aggregate(per_method[name]) for name in ALL_METHODS}

        print(f"\n{'='*70}\n NAIVE vs PER-CHUNKS vs GRIDPROBE (+ strided) — {model_name} — avg over {len(queries_to_run)} queries\n{'='*70}")
        print_results_table(agg)

        results_path = f"congress_stride_{safe_model_name}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
        print(f"\nSaved results to {results_path}")

        for w in workers:
            w.unload()

    print(f"\n{'#'*70}\n ALL DONE\n{'#'*70}")


if __name__ == "__main__":
    main()

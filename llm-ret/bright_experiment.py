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
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import skew, kurtosis
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
clear_output()

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

clear_output()

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)

MODELS = [
    # score_temperature=8.0 carries over from Qwen3-4B-Instruct-2507's established value for this
    # exact prompt style elsewhere in this repo (hierarchy_experiment.py) -- a reasonable starting
    # point for the closely related base Qwen3-4B, worth re-checking once real scores come in.
    {"id": "Qwen/Qwen3-4B", "quantize": True, "n_active_params": 4e9, "score_temperature": 8.0},
]

# BRIGHT (xlangai/BRIGHT): reasoning-intensive retrieval benchmark, 12 domains. "documents" is
# BRIGHT's own pre-split short passages (ids like "<page>_<n>.txt"); "long_documents" is the
# original unsplit page per domain (id == the same page without the numeric suffix) and only
# exists for the StackExchange-sourced domains -- see LONG_DOCUMENTS_DOMAINS below (verified
# against the actual repo listing). Some of those pages run to tens of thousands of tokens, which
# OOM'd Naive's uncapped full-document read -- see run_naive for how that's now handled.
DATASET_ID = "xlangai/BRIGHT"
LONG_DOCUMENTS_DOMAINS = {
    "biology", "earth_science", "economics", "pony",
    "psychology", "robotics", "stackoverflow", "sustainable_living",
}

# First validation pass: small on purpose. GridProbe costs up to 2K model calls per document
# (vs 1 for the Level-1 skim work elsewhere in this repo), so this checks the mechanics/cost are
# sane before scaling to more domains / more queries / a bigger pool.
DOMAINS = ["biology", "psychology"]
N_QUERIES_PER_DOMAIN = 10

# Each query's candidate pool is built as: gold doc(s) guaranteed + random negatives, half drawn
# from "documents" (short) and half from "long_documents" (long). If a domain has no
# long_documents split, the pool is filled entirely from "documents" instead (build_query_pool).
POOL_SIZE = 20
# Bumped from 67 (the convention used everywhere else in this repo) specifically so this doesn't
# resample the same oversized long_documents page that OOM'd the biology pool before.
SEED = 68

EVAL_KS = [5, 10]
TARGET_CHUNK_TOKENS = 100   # drives per-document K (grid granularity) -- NOT a truncation cap
ROW_AGGREGATION = "peak"   # "peak" (max) or "avg" (mean) -- same open ablation as hierarchy_experiment.py

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


def load_bright_domain(domain):
    examples_ds = load_dataset(DATASET_ID, "examples")[domain]
    short_ds = load_dataset(DATASET_ID, "documents")[domain]
    short_corpus = {r["id"]: r["content"] for r in short_ds}

    long_corpus = {}
    if domain in LONG_DOCUMENTS_DOMAINS:
        long_ds = load_dataset(DATASET_ID, "long_documents")[domain]
        long_corpus = {r["id"]: r["content"] for r in long_ds}

    queries, gold_short, gold_long = {}, {}, {}
    for r in examples_ds:
        qid = r["id"]
        queries[qid] = r["query"]
        gold_short[qid] = set(r["gold_ids"])
        gold_long[qid] = set(r.get("gold_ids_long", []))

    return queries, short_corpus, long_corpus, gold_short, gold_long


def build_half(ids_pool, gold_ids, half_size, seed):
    gold_in_pool = gold_ids & set(ids_pool)
    other = [cid for cid in ids_pool if cid not in gold_in_pool]
    random.Random(seed).shuffle(other)
    filler = other[: max(0, half_size - len(gold_in_pool))]
    return sorted(gold_in_pool | set(filler))


def build_query_pool(qid, short_corpus, long_corpus, gold_short, gold_long):
    half = POOL_SIZE // 2
    short_ids = build_half(list(short_corpus), gold_short.get(qid, set()), half, SEED)

    if long_corpus:
        long_ids = build_half(list(long_corpus), gold_long.get(qid, set()), POOL_SIZE - half, SEED + 1)
    else:
        # Domain has no long_documents split -- fill the whole pool from short docs instead.
        extra_needed = POOL_SIZE - len(short_ids)
        remaining = [cid for cid in short_corpus if cid not in short_ids]
        random.Random(SEED + 2).shuffle(remaining)
        short_ids = short_ids + remaining[:extra_needed]
        long_ids = []

    corpus = {cid: short_corpus[cid] for cid in short_ids}
    corpus.update({cid: long_corpus[cid] for cid in long_ids})
    sample_ids = short_ids + long_ids
    gold_in_sample = (gold_short.get(qid, set()) & set(short_ids)) | (gold_long.get(qid, set()) & set(long_ids))
    return sample_ids, corpus, gold_in_sample


def build_relevance_prompt(query_text, chunks_text):
    return f"""You are an expert relevance judge for a reasoning-intensive retrieval benchmark.
    Queries require multi-step reasoning to connect to the right document -- a document can be
    relevant even if it shares no keywords with the query, and irrelevant even if it shares many.
    Judge relevance based on whether the content below is actually needed to answer the query,
    not on surface-level keyword or topic overlap.

    Query:
    {query_text}

    Candidate chunks:
    {chunks_text}

    Question: Is ANY chunk above relevant to the query, even if only through indirect or non-obvious reasoning?
    Answer with a single word: Yes or No.
    """


def format_chunks(chunks):
    return "\n\n".join(f"chunk {i+1}: {c}" for i, c in enumerate(chunks))


def recall_at_k(ranked_has_gold, k):
    total_gold = sum(ranked_has_gold)
    if total_gold == 0:
        return None
    return sum(ranked_has_gold[:k]) / total_gold


def precision_at_k(ranked_has_gold, k):
    if sum(ranked_has_gold) == 0:
        return None
    return sum(ranked_has_gold[:k]) / k


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
        # nothing is cropped (this is the mechanism, not a length cap -- k grows with doc length).
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
        return k, chunks, total

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def build_rows_cols(k, chunks):
    rows = [chunks[r * k : (r + 1) * k] for r in range(k)]
    cols = [chunks[c::k] for c in range(k)]
    return rows, cols


# ---- the 3 methods, run head-to-head on the SAME pool per query (no funnel, no M_eff gating) ----

def run_naive(worker, query_text, sample_ids, corpus):
    # Naive reads the whole document in one prompt, uncapped -- occasionally a raw long_documents
    # page is too large for one forward pass to fit in memory. That's a real property of Naive,
    # not a bug: record it as a failure for that document (never truncate it) and keep going,
    # clearing the CUDA cache so the failed allocation doesn't leave fragmented memory behind.
    # "records" is per-document (tokens, tflops, latency) for the cost-vs-tokens plots -- only for
    # documents that actually got scored, failed ones are excluded entirely (not just zeroed).
    t0 = time.perf_counter()
    scores = {}
    total_tflops = 0.0
    failed = 0
    records = []
    failed_tokens = []
    for doc_id in sample_ids:
        doc_t0 = time.perf_counter()
        try:
            score, tflops = worker.score_relevance(query_text, format_chunks([corpus[doc_id]]))
            scores[doc_id] = score
            total_tflops += tflops
            records.append({
                "tokens": worker.count_tokens(corpus[doc_id]),
                "tflops": tflops, "latency": time.perf_counter() - doc_t0,
            })
        except torch.cuda.OutOfMemoryError:
            failed += 1
            torch.cuda.empty_cache()
            # Tokenization never touches the GPU, so this is safe even right after an OOM --
            # keeps the failed document's length visible on the plot without ever scoring it.
            failed_tokens.append(worker.count_tokens(corpus[doc_id]))
            print(f"    [naive] OOM on doc={doc_id} -- skipped, not truncated ({len(corpus[doc_id])} chars)")
    latency = time.perf_counter() - t0
    return {
        "scores": scores, "calls": len(sample_ids), "latency": latency, "tflops": total_tflops,
        "failed": failed, "records": records, "failed_tokens": failed_tokens,
    }


def run_row_and_grid(worker, query_text, sample_ids, corpus):
    # Row pass is shared: Row-Verifier's score IS this pass, and GridProbe reuses its row_scores
    # instead of recomputing them -- only the column pass is extra work. Per-document tokens/tflops/
    # latency are tracked for both (row-only for Row-Verifier, row+col combined for GridProbe) for
    # the cost-vs-tokens plots.
    row_t0 = time.perf_counter()
    doc_grids = {}
    row_scores_by_doc = {}
    row_calls = 0
    row_tflops = 0.0
    row_records_by_doc = {}
    for doc_id in sample_ids:
        doc_row_t0 = time.perf_counter()
        k, chunks, total_tokens = worker.split_document(corpus[doc_id])
        rows, cols = build_rows_cols(k, chunks)
        row_scores = []
        doc_row_tflops = 0.0
        for row in rows:
            score, tflops = worker.score_relevance(query_text, format_chunks(row))
            row_scores.append(score)
            row_tflops += tflops
            doc_row_tflops += tflops
            row_calls += 1
        doc_grids[doc_id] = {"k": k, "rows": rows, "cols": cols, "row_scores": row_scores}
        row_scores_by_doc[doc_id] = max(row_scores) if ROW_AGGREGATION == "peak" else sum(row_scores) / len(row_scores)
        row_records_by_doc[doc_id] = {
            "tokens": total_tokens, "tflops": doc_row_tflops, "latency": time.perf_counter() - doc_row_t0,
        }
    row_latency = time.perf_counter() - row_t0

    col_t0 = time.perf_counter()
    grid_scores = {}
    col_calls = 0
    col_tflops = 0.0
    grid_records = []
    for doc_id in sample_ids:
        doc_col_t0 = time.perf_counter()
        grid = doc_grids[doc_id]
        k = grid["k"]
        col_scores = []
        doc_col_tflops = 0.0
        for col in grid["cols"]:
            score, tflops = worker.score_relevance(query_text, format_chunks(col))
            col_scores.append(score)
            col_tflops += tflops
            doc_col_tflops += tflops
            col_calls += 1
        grid_scores[doc_id] = max(grid["row_scores"][r] * col_scores[c] for r in range(k) for c in range(k))
        row_rec = row_records_by_doc[doc_id]
        grid_records.append({
            "tokens": row_rec["tokens"],
            "tflops": row_rec["tflops"] + doc_col_tflops,
            "latency": row_rec["latency"] + (time.perf_counter() - doc_col_t0),
        })
    col_latency = time.perf_counter() - col_t0

    row_verifier = {
        "scores": row_scores_by_doc, "calls": row_calls, "latency": row_latency, "tflops": row_tflops,
        "failed": 0, "records": list(row_records_by_doc.values()),
    }
    gridprobe = {
        "scores": grid_scores,
        "calls": row_calls + col_calls, "latency": row_latency + col_latency, "tflops": row_tflops + col_tflops,
        "failed": 0, "records": grid_records,
    }
    return row_verifier, gridprobe


ALL_METHODS = ["naive", "row_verifier", "gridprobe"]


def run_query_bright(worker, qid, query_text, short_corpus, long_corpus, gold_short, gold_long):
    sample_ids, corpus, gold_in_sample = build_query_pool(qid, short_corpus, long_corpus, gold_short, gold_long)

    naive = run_naive(worker, query_text, sample_ids, corpus)
    row_verifier, gridprobe = run_row_and_grid(worker, query_text, sample_ids, corpus)

    results = {}
    for name, r in (("naive", naive), ("row_verifier", row_verifier), ("gridprobe", gridprobe)):
        metrics, auc = evaluate_doc_scores(r["scores"], gold_in_sample)
        results[name] = {
            "calls": r["calls"], "latency": r["latency"], "tflops": r["tflops"],
            "failed": r.get("failed", 0), "records": r.get("records", []),
            "failed_tokens": r.get("failed_tokens", []), "metrics": metrics, "auc": auc,
        }
    return results, len(gold_in_sample)


def split_round_robin(items, n):
    return [items[i::n] for i in range(n)]


def run_worker_queries(worker, domain, query_ids, queries, short_corpus, long_corpus, gold_short, gold_long):
    if worker.device.startswith("cuda"):
        torch.cuda.set_device(worker.device)

    per_method = {name: [] for name in ALL_METHODS}
    logs = []
    for qid in query_ids:
        q_start = time.perf_counter()
        results, gold_in_sample = run_query_bright(worker, qid, queries[qid], short_corpus, long_corpus, gold_short, gold_long)
        for name in ALL_METHODS:
            per_method[name].append(results[name])
        logs.append(f"[{worker.device}] {domain}/{qid} done in {time.perf_counter() - q_start:.2f}s (gold_in_sample={gold_in_sample})")
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


def fit_loglog(xs, ys):
    # Least-squares power-law fit: log10(y) = a*log10(x) + b, i.e. y = 10^b * x^a. Needs at least
    # 2 points at distinct token counts; returns None otherwise (nothing to fit against).
    if len(xs) < 2 or len(set(xs)) < 2:
        return None
    a, b = np.polyfit(np.log10(xs), np.log10(ys), 1)
    return a, b


def fit_crossing(fit_naive, fit_other, token_range):
    # Intersection of two fitted log-log lines: a1*log(x)+b1 = a2*log(x)+b2. Only reported if it
    # actually falls inside the observed token range -- outside that range it's extrapolation,
    # not a real crossing this run's data can support.
    if fit_naive is None or fit_other is None:
        return None
    a1, b1 = fit_naive
    a2, b2 = fit_other
    if a1 == a2:
        return None
    x_cross = 10 ** ((b2 - b1) / (a1 - a2))
    return x_cross if token_range[0] <= x_cross <= token_range[1] else None


def plot_cost_vs_tokens(model_name, domain, per_method):
    # Log-log axes: TFLOPs/latency vs token count both span orders of magnitude (short passages
    # to 20k+ token pages), so a linear scale crushes the short end and hides exactly the
    # crossing behavior these plots exist to show. Each method's actual measured points are
    # connected directly (sorted by token count) rather than smoothed into a fit line -- the fit
    # is still computed under the hood (fit_loglog/fit_crossing) purely to locate the "overtakes"
    # annotation, it just isn't drawn. Naive's OOM'd documents (see run_naive) are marked, not
    # silently missing.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    metric_specs = [("tflops", "TFLOPs", axes[0]), ("latency", "Latency (s)", axes[1])]

    all_tokens = [rec["tokens"] for name in ALL_METHODS for r in per_method[name] for rec in r["records"]]
    token_range = (min(all_tokens), max(all_tokens)) if all_tokens else (1, 1)
    failed_tokens = sorted({t for r in per_method["naive"] for t in r["failed_tokens"]})

    for metric_key, metric_label, ax in metric_specs:
        fits = {}
        for name in ALL_METHODS:
            records = sorted(
                (rec for r in per_method[name] for rec in r["records"]),
                key=lambda rec: rec["tokens"],
            )
            xs = [rec["tokens"] for rec in records]
            ys = [rec[metric_key] for rec in records]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=2, label=name)
            fits[name] = fit_loglog(xs, ys)

        for other in ("row_verifier", "gridprobe"):
            cross = fit_crossing(fits.get("naive"), fits.get(other), token_range)
            if cross is not None:
                ax.axvline(cross, linestyle="--", color="gray", linewidth=1, alpha=0.7)
                ax.annotate(
                    f"{other} overtakes naive\n~{cross:,.0f} tokens",
                    xy=(cross, 0.95), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", fontsize=8, color="dimgray", va="top",
                )

        for ft in failed_tokens:
            ax.axvline(ft, linestyle=":", color="crimson", linewidth=1, alpha=0.4)
        if failed_tokens:
            ax.annotate(
                "naive OOM'd here\n(excluded, not plotted)", xy=(failed_tokens[0], 1.0),
                xycoords=("data", "axes fraction"), xytext=(4, -10), textcoords="offset points",
                fontsize=8, color="crimson", va="top",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of Tokens (log scale)")
        ax.set_ylabel(f"{metric_label} (log scale)")
        ax.set_title(f"{metric_label} vs Tokens")
        ax.legend(fontsize=8)

    fig.suptitle(f"{model_name} — {domain}: cost vs document length")
    fig.tight_layout()
    safe_model_name = model_name.replace("/", "_")
    fig.savefig(f"bright_{domain}_{safe_model_name}_cost_vs_tokens.png")
    plt.show()


def main():
    for model_cfg in MODELS:
        model_name = model_cfg["id"]
        safe_model_name = model_name.replace("/", "_")

        workers = [Worker(d) for d in DEVICES]
        for w in workers:
            w.load(model_cfg)

        for domain in DOMAINS:
            print(f"\n{'#'*70}\n MODEL: {model_name}  |  DOMAIN: {domain}\n{'#'*70}")
            queries, short_corpus, long_corpus, gold_short, gold_long = load_bright_domain(domain)
            has_long = domain in LONG_DOCUMENTS_DOMAINS
            print(f"Loaded {len(queries)} queries, {len(short_corpus)} short docs, "
                  f"{len(long_corpus)} long docs (has_long={has_long})")

            query_ids = list(queries)[:N_QUERIES_PER_DOMAIN]
            chunks = split_round_robin(query_ids, len(workers))
            with ThreadPoolExecutor(max_workers=len(workers)) as ex:
                futures = [
                    ex.submit(run_worker_queries, w, domain, chunk, queries, short_corpus, long_corpus, gold_short, gold_long)
                    for w, chunk in zip(workers, chunks)
                ]
                per_worker_results = [f.result() for f in futures]

            per_method = {name: [] for name in ALL_METHODS}
            for worker_per_method, logs in per_worker_results:
                for line in logs:
                    print(line)
                for name in ALL_METHODS:
                    per_method[name].extend(worker_per_method[name])

            agg = {name: aggregate(per_method[name]) for name in ALL_METHODS}

            print(f"\n{'='*70}\n NAIVE vs ROW-VERIFIER vs GRIDPROBE — {model_name} — {domain} — avg over {len(query_ids)} queries\n{'='*70}")
            print_results_table(agg)
            plot_cost_vs_tokens(model_name, domain, per_method)

            results_path = f"bright_{domain}_{safe_model_name}.json"
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(agg, f, indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
            print(f"\nSaved results to {results_path}")

        for w in workers:
            w.unload()

    print(f"\n{'#'*70}\n ALL DONE\n{'#'*70}")


if __name__ == "__main__":
    main()

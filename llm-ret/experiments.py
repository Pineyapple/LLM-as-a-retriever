from IPython.display import clear_output
#!pip install transformers datasets huggingface_hub torch accelerate bitsandbytes scipy
import json
import math
import random
import time
import torch
import numpy as np
from scipy.stats import skew, kurtosis
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
clear_output()

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
DATASET_ID = "dianetc/OBLIQ-Bench"
CONFIGS = ["math", "writing", "twitter", "wildchat", "congress"]
QRELS_PATHS = {
    "math": "analogues/math/queries+qrels/qrels.tsv",
    "writing": "analogues/writing/queries+qrels/qrels.tsv",
    "twitter": "descriptive/twitter/queries+qrels/qrels.tsv",
    "wildchat": "descriptive/wildchat/queries+qrels/qrels.tsv",
    "congress": "tip-of-tongue/congress/queries+qrels/qrels.tsv",
}

GRID_K = 10                     # grid dimension -> sample is always GRID_K x GRID_K docs
SAMPLE_SIZE = GRID_K * GRID_K
N_QUERIES = 2
EVAL_FRACTIONS = [0.05, 0.10]   # top 5% / top 10% of the candidate pool
EVAL_KS = [max(1, round(SAMPLE_SIZE * frac)) for frac in EVAL_FRACTIONS]
GAMMA_0 = 0.25                  # sigma-adaptive M_eff shape coefficient, per GridProbe Eq.
SEED = 67
MAX_CHUNK_TOKENS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPERIMENTS = ["1_plain_doc", "2_row_verifier", "3_gridprobe_plain", "4_gridprobe_global"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, quantization_config=quant_config).to(DEVICE)


def load_data(config):
    ds = load_dataset(DATASET_ID, config)
    corpus = {r["_id"]: r["text"] for r in ds["corpus"]}
    queries = {r["_id"]: r["text"] for r in ds["queries"]}
    qrels_path = hf_hub_download(DATASET_ID, QRELS_PATHS[config], repo_type="dataset")
    gold_by_query = {}
    with open(qrels_path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            qid, cid, score = line.strip().split("\t")
            if int(score) > 0:
                gold_by_query.setdefault(qid, set()).add(cid)
    return corpus, queries, gold_by_query


def sort_key(cid):
    try:
        return (0, int(cid))
    except ValueError:
        return (1, cid)


def build_sample(corpus_ids, gold_ids):
    other_ids = [cid for cid in corpus_ids if cid not in gold_ids]
    random.Random(SEED).shuffle(other_ids)
    filler = other_ids[: max(0, SAMPLE_SIZE - len(gold_ids))]
    return sorted(gold_ids | set(filler), key=sort_key)


def build_rows_cols(sample_ids):
    rows = [sample_ids[r * GRID_K : (r + 1) * GRID_K] for r in range(GRID_K)]
    cols = [sample_ids[c::GRID_K] for c in range(GRID_K)]
    return rows, cols


def prompt_inputs(prompt, enable_thinking=False):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return tokenizer(text, return_tensors="pt").to(DEVICE)


def truncate_chunk(text, max_tokens=MAX_CHUNK_TOKENS):
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def yesno_token_ids():
    ids = {}
    for label, variants in (("Yes", ["Yes", " Yes", "yes", " yes"]), ("No", ["No", " No", "no", " no"])):
        for variant in variants:
            ids[tokenizer.encode(variant, add_special_tokens=False)[-1]] = label
    return ids


OPTION_IDS = None  # filled in lazily after tokenizer loads


def score_relevance(query_text, chunks_text):
    global OPTION_IDS
    if OPTION_IDS is None:
        OPTION_IDS = yesno_token_ids()

    prompt = f"""You are an expert relevance judge for OBLIQUE queries — queries where relevance is latent.
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
    inputs = prompt_inputs(prompt)
    with torch.no_grad():
        outputs = model(**inputs)
    next_token_logits = outputs.logits[0, -1, :]

    ids = list(OPTION_IDS)
    probs = torch.softmax(next_token_logits[ids], dim=0)
    yes_prob = sum(p for tid, p in zip(ids, probs) if OPTION_IDS[tid] == "Yes")
    return float(yes_prob)


def score_chunks(query_text, doc_ids, corpus):
    chunks_text = "\n\n".join(f"chunk {i+1}: {truncate_chunk(corpus[d])}" for i, d in enumerate(doc_ids))
    return score_relevance(query_text, chunks_text)


def recall_at_k(ranked_has_gold, k):
    total_gold = sum(ranked_has_gold)
    if total_gold == 0:
        return None
    hits = sum(ranked_has_gold[:k])
    return hits / total_gold


def precision_at_k(ranked_has_gold, k):
    if sum(ranked_has_gold) == 0:
        return None
    hits = sum(ranked_has_gold[:k])
    return hits / k


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


def compute_metrics(ranked):
    return {
        "recall": {k: recall_at_k(ranked, k) for k in EVAL_KS},
        "precision": {k: precision_at_k(ranked, k) for k in EVAL_KS},
        "ndcg": {k: ndcg_at_k(ranked, k) for k in EVAL_KS},
    }


def average_metrics(metrics_list):
    avg = {"recall": {}, "precision": {}, "ndcg": {}}
    for name in ("recall", "precision", "ndcg"):
        for k in EVAL_KS:
            vals = [m[name][k] for m in metrics_list if m[name][k] is not None]
            avg[name][k] = sum(vals) / len(vals) if vals else float("nan")
    return avg


def evaluate_doc_scores(doc_scores, gold):
    pairs = [(score, doc_id in gold) for doc_id, score in doc_scores.items()]
    ranked = [has_gold for _, has_gold in sorted(pairs, key=lambda x: x[0], reverse=True)]
    return compute_metrics(ranked), auc_score(pairs)


def compute_sigma_meff(values):
    arr = np.array(values, dtype=float)
    sigma = 0.0
    if arr.var() > 1e-9:
        sigma = abs(skew(arr)) + 0.5 * max(0.0, kurtosis(arr, fisher=True))
    m_eff = max(1, min(GRID_K**2, math.ceil((GRID_K**2) / (1 + GAMMA_0 * GRID_K * sigma))))
    return sigma, m_eff


def run_query(qid, corpus, queries, gold_by_query, sample_ids, rows, cols):
    gold = gold_by_query.get(qid, set()) - {qid}
    query_text = queries[qid]

    # Phase 1: score every doc individually -> feeds Experiments 1 & 3
    t0 = time.perf_counter()
    doc_scores = {d: score_chunks(query_text, [d], corpus) for d in sample_ids}
    doc_latency = time.perf_counter() - t0

    # Phase 2: score each row as one batched call -> feeds Experiments 2 & 4
    t0 = time.perf_counter()
    row_scores = [score_chunks(query_text, row, corpus) for row in rows]
    row_latency = time.perf_counter() - t0

    # Phase 3: score each column as one batched call -> feeds Experiment 4 only
    t0 = time.perf_counter()
    col_scores = [score_chunks(query_text, col, corpus) for col in cols]
    col_latency = time.perf_counter() - t0

    results = {}

    # Experiment 1: plain per-document scoring, no grid
    metrics, auc = evaluate_doc_scores(doc_scores, gold)
    results["1_plain_doc"] = {
        "metrics": metrics, "auc": auc, "calls": len(sample_ids), "latency": doc_latency,
    }

    # Experiment 2: row-verifier (current main.py mechanism) -- every doc inherits its row's score
    row_inherited = {d: row_scores[r] for r, row in enumerate(rows) for d in row}
    metrics, auc = evaluate_doc_scores(row_inherited, gold)
    results["2_row_verifier"] = {
        "metrics": metrics, "auc": auc, "calls": len(rows), "latency": row_latency,
    }

    # Experiment 3: GridProbe grid built from individually-scored docs (max-aggregated per row/col)
    c_row = [max(doc_scores[d] for d in row) for row in rows]
    c_col = [max(doc_scores[d] for d in col) for col in cols]
    grid_scores_3 = {d: c_row[r] * c_col[j] for r, row in enumerate(rows) for j, d in enumerate(row)}
    metrics, auc = evaluate_doc_scores(grid_scores_3, gold)
    sigma_3, meff_3 = compute_sigma_meff(list(grid_scores_3.values()))
    results["3_gridprobe_plain"] = {
        "metrics": metrics, "auc": auc, "calls": len(sample_ids), "latency": doc_latency,
        "sigma": sigma_3, "m_eff": meff_3,
    }

    # Experiment 4: literal GridProbe -- batched row probes x batched col probes
    grid_scores_4 = {d: row_scores[r] * col_scores[j] for r, row in enumerate(rows) for j, d in enumerate(row)}
    metrics, auc = evaluate_doc_scores(grid_scores_4, gold)
    sigma_4, meff_4 = compute_sigma_meff(list(grid_scores_4.values()))
    results["4_gridprobe_global"] = {
        "metrics": metrics, "auc": auc, "calls": 2 * len(rows), "latency": row_latency + col_latency,
        "sigma": sigma_4, "m_eff": meff_4,
    }

    return results


NAME_COL = 26
COL = 11


def print_table(rows):
    headers = ["Experiment", "Calls", "Latency(s)"]
    for name in ("Recall", "Precision", "NDCG"):
        headers += [f"{name}@{k}" for k in EVAL_KS]
    headers += ["AUC", "M_eff"]
    print(headers[0].ljust(NAME_COL) + "".join(h.ljust(COL) for h in headers[1:]))
    for exp_name, m in rows:
        row = [f"{m['calls']:.0f}", f"{m['latency']:.2f}"]
        for name in ("recall", "precision", "ndcg"):
            row += [f"{m[name][k]:.3f}" for k in EVAL_KS]
        row += [f"{m['auc']:.3f}", f"{m['m_eff']:.1f}" if "m_eff" in m else "-"]
        print(exp_name.ljust(NAME_COL) + "".join(v.ljust(COL) for v in row))


def aggregate_results(results_list):
    avg = average_metrics([r["metrics"] for r in results_list])
    aucs = [r["auc"] for r in results_list if r["auc"] is not None]
    avg["auc"] = sum(aucs) / len(aucs) if aucs else float("nan")
    avg["calls"] = results_list[0]["calls"]
    avg["latency"] = sum(r["latency"] for r in results_list) / len(results_list)
    if "m_eff" in results_list[0]:
        avg["m_eff"] = sum(r["m_eff"] for r in results_list) / len(results_list)
    return avg


def run_config(config):
    print(f"\n{'#'*70}\n TASK BEING TESTED: {config.upper()}\n{'#'*70}")
    corpus, queries, gold_by_query = load_data(config)

    query_ids = list(queries)[:N_QUERIES]
    gold_ids = set().union(*(gold_by_query.get(q, set()) for q in query_ids))
    sample_ids = build_sample(list(corpus), gold_ids)[:SAMPLE_SIZE]
    rows, cols = build_rows_cols(sample_ids)

    print(f"Sample: {len(sample_ids)} docs -> {GRID_K}x{GRID_K} grid\n")

    per_experiment = {name: [] for name in EXPERIMENTS}
    for qid in query_ids:
        query_start = time.perf_counter()
        query_results = run_query(qid, corpus, queries, gold_by_query, sample_ids, rows, cols)
        for name in EXPERIMENTS:
            per_experiment[name].append(query_results[name])
        print(f"=== Query {qid} done in {time.perf_counter() - query_start:.2f}s ===")

    return per_experiment


def main():
    all_results = {}
    global_per_experiment = {name: [] for name in EXPERIMENTS}
    for config in CONFIGS:
        per_experiment = run_config(config)
        all_results[config] = {name: aggregate_results(r) for name, r in per_experiment.items()}
        for name in EXPERIMENTS:
            global_per_experiment[name].extend(per_experiment[name])

    global_summary = {name: aggregate_results(r) for name, r in global_per_experiment.items()}

    print(f"\n{'='*70}\n GLOBAL SUMMARY — avg across all {len(CONFIGS)} tasks x {N_QUERIES} queries\n{'='*70}")
    print_table([(name, global_summary[name]) for name in EXPERIMENTS])

    results_path = "experiment_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()

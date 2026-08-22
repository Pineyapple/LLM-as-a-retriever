from IPython.display import clear_output
#!pip install transformers datasets huggingface_hub torch accelerate scipy bitsandbytes
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

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_ID = "dianetc/OBLIQ-Bench"
CONFIGS = ["writing"]
QRELS_PATHS = {
    "math": "analogues/math/queries+qrels/qrels.tsv",
    "writing": "analogues/writing/queries+qrels/qrels.tsv",
    "twitter": "descriptive/twitter/queries+qrels/qrels.tsv",
    "wildchat": "descriptive/wildchat/queries+qrels/qrels.tsv",
    "congress": "tip-of-tongue/congress/queries+qrels/qrels.tsv",
}

N_QUERIES = 3
SAMPLE_SIZE = 50
EVAL_KS = [5, 10]
SEED = 67
TARGET_CHUNK_TOKENS = 100      # drives per-document K (grid granularity) -- NOT a truncation cap
WORDS_PER_PARAGRAPH = 12        # Level 1 skim: first N words of each paragraph
GAMMA_0 = 0.25
SCORE_TEMPERATURE = 8.0         # softens saturated yes/no logit gaps back into a continuous [0,1] score
ROW_AGGREGATION = "peak"       # "peak" (max) or "avg" (mean) -- open ablation
L1_MEFF_MULTIPLIER = 1          # scales Level 1's computed M_eff (capped at sample size) -- ablation knob
L2_MEFF_MULTIPLIER = 1          # scales Level 2's computed M_eff (capped at survivor count) -- ablation knob
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, quantization_config=quant_config).to(DEVICE)

N_LAYERS = model.config.num_hidden_layers
D_MODEL = model.config.hidden_size
N_ACTIVE_PARAMS = 4e9  # Qwen3-4B-Instruct-2507 is dense -- all ~4B params are active per forward pass


def estimate_tflops(context_length):
    flops = 2 * N_ACTIVE_PARAMS * context_length + 2 * N_LAYERS * D_MODEL * context_length**2
    return flops / 1e12


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
    return sorted(gold_ids | set(filler), key=sort_key)[:SAMPLE_SIZE]


def prompt_inputs(prompt):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt").to(DEVICE)


def yesno_token_ids():
    ids = {}
    for label, variants in (("Yes", ["Yes", " Yes", "yes", " yes"]), ("No", ["No", " No", "no", " no"])):
        for variant in variants:
            ids[tokenizer.encode(variant, add_special_tokens=False)[-1]] = label
    return ids


OPTION_IDS = None  # filled in lazily after tokenizer loads


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


def score_relevance(query_text, chunks_text):
    global OPTION_IDS
    if OPTION_IDS is None:
        OPTION_IDS = yesno_token_ids()

    prompt = build_relevance_prompt(query_text, chunks_text)
    inputs = prompt_inputs(prompt)
    context_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model(**inputs)
    next_token_logits = outputs.logits[0, -1, :]

    ids = list(OPTION_IDS)
    probs = torch.softmax(next_token_logits[ids] / SCORE_TEMPERATURE, dim=0)
    yes_prob = sum(p for tid, p in zip(ids, probs) if OPTION_IDS[tid] == "Yes")
    return float(yes_prob), estimate_tflops(context_length)


def format_chunks(chunks):
    return "\n\n".join(f"chunk {i+1}: {c}" for i, c in enumerate(chunks))


def split_document(text):
    # Divides the FULL document into k*k pieces -- every token belongs to exactly one chunk, nothing is cropped.
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    total = len(token_ids)
    k = max(2, math.ceil(math.sqrt(total / TARGET_CHUNK_TOKENS)))
    k = min(k, max(1, math.isqrt(total)))
    n_chunks = k * k
    base, remainder = divmod(total, n_chunks)
    chunks, start = [], 0
    for i in range(n_chunks):
        size = base + (1 if i < remainder else 0)
        chunks.append(tokenizer.decode(token_ids[start : start + size], skip_special_tokens=True))
        start += size
    return k, chunks


def build_rows_cols(k, chunks):
    rows = [chunks[r * k : (r + 1) * k] for r in range(k)]
    cols = [chunks[c::k] for c in range(k)]
    return rows, cols


def compute_sigma_meff(values, k):
    # k is the "grid dimension" this list of scores should be judged against -- literally K for a
    # true K x K grid (Level 3), or round(sqrt(N)) when values is just a flat list of N scores (Levels 1 & 2).
    arr = np.array(values, dtype=float)
    sigma = 0.0
    if arr.var() > 1e-9:
        sigma = abs(skew(arr)) + 0.5 * max(0.0, kurtosis(arr, fisher=True))
    n = len(values)
    m_eff = max(1, min(n, math.ceil(n / (1 + GAMMA_0 * k * sigma))))
    return sigma, m_eff


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


def build_skim_text(doc_text):
    paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [doc_text.strip()]
    snippets = []
    for p in paragraphs:
        words = p.split()[:WORDS_PER_PARAGRAPH]
        if words:
            snippets.append(" ".join(words))
    return " / ".join(snippets)


def level1_skim(query_text, sample_ids, corpus):
    t0 = time.perf_counter()
    scores = {}
    total_tflops = 0.0
    n = len(sample_ids)
    print(f"  [Level 1] skimming {n} docs...")
    for i, doc_id in enumerate(sample_ids):
        call_t0 = time.perf_counter()
        skim_text = build_skim_text(corpus[doc_id])
        score, tflops = score_relevance(query_text, format_chunks([skim_text]))
        scores[doc_id] = score
        total_tflops += tflops
        print(f"    [L1 {i+1}/{n}] doc={doc_id} score={scores[doc_id]:.3f} ({time.perf_counter()-call_t0:.2f}s)")
    latency = time.perf_counter() - t0

    n = len(sample_ids)
    sigma, m_eff = compute_sigma_meff(list(scores.values()), k=round(math.sqrt(n)))
    m_eff = min(n, m_eff * L1_MEFF_MULTIPLIER)
    survivors = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:m_eff]]
    return scores, survivors, sigma, m_eff, latency, total_tflops


def level2_rows(query_text, survivor_ids, corpus):
    t0 = time.perf_counter()
    doc_grids = {}
    total_tflops = 0.0
    m = len(survivor_ids)
    print(f"  [Level 2] row-probing {m} survivors...")
    for i, doc_id in enumerate(survivor_ids):
        k, chunks = split_document(corpus[doc_id])
        rows, cols = build_rows_cols(k, chunks)
        row_scores = []
        for r, row in enumerate(rows):
            call_t0 = time.perf_counter()
            score, tflops = score_relevance(query_text, format_chunks(row))
            row_scores.append(score)
            total_tflops += tflops
            print(
                f"    [L2 doc {i+1}/{m} row {r+1}/{k}] doc={doc_id} "
                f"score={row_scores[-1]:.3f} ({time.perf_counter()-call_t0:.2f}s)"
            )
        combined = sum(row_scores) / len(row_scores) if ROW_AGGREGATION == "avg" else max(row_scores)
        doc_grids[doc_id] = {
            "k": k, "chunks": chunks, "rows": rows, "cols": cols,
            "row_scores": row_scores, "combined_score": combined,
        }
    latency = time.perf_counter() - t0

    m = len(survivor_ids)
    sigma, m_eff = 0.0, m
    if m > 0:
        combined_scores = [doc_grids[d]["combined_score"] for d in survivor_ids]
        sigma, m_eff = compute_sigma_meff(combined_scores, k=round(math.sqrt(m)))
        m_eff = min(m, m_eff * L2_MEFF_MULTIPLIER)
    finalists = [d for d, _ in sorted(
        ((d, doc_grids[d]["combined_score"]) for d in survivor_ids), key=lambda x: x[1], reverse=True
    )[:m_eff]]
    return doc_grids, finalists, sigma, m_eff, latency, total_tflops


def level3_columns(query_text, finalist_ids, doc_grids):
    t0 = time.perf_counter()
    evidence = {}
    total_tflops = 0.0
    n = len(finalist_ids)
    print(f"  [Level 3] column-probing {n} finalists...")
    for i, doc_id in enumerate(finalist_ids):
        grid = doc_grids[doc_id]
        k = grid["k"]
        col_scores = []
        for c, col in enumerate(grid["cols"]):
            call_t0 = time.perf_counter()
            score, tflops = score_relevance(query_text, format_chunks(col))
            col_scores.append(score)
            total_tflops += tflops
            print(
                f"    [L3 doc {i+1}/{n} col {c+1}/{k}] doc={doc_id} "
                f"score={col_scores[-1]:.3f} ({time.perf_counter()-call_t0:.2f}s)"
            )
        row_scores = grid["row_scores"]

        best_r, best_c, best_score = 0, 0, -1.0
        for r in range(k):
            for c in range(k):
                cell_score = row_scores[r] * col_scores[c]
                if cell_score > best_score:
                    best_r, best_c, best_score = r, c, cell_score

        evidence[doc_id] = {
            "col_scores": col_scores,
            "peak_row": best_r,
            "peak_col": best_c,
            "peak_score": best_score,
            "peak_chunk": grid["chunks"][best_r * k + best_c],
        }
    latency = time.perf_counter() - t0
    return evidence, latency, total_tflops


def run_query(qid, corpus, queries, gold_by_query, sample_ids):
    gold = gold_by_query.get(qid, set()) - {qid}
    query_text = queries[qid]
    gold_in_sample = gold & set(sample_ids)

    skim_scores, survivors, sigma1, meff1, lat1, tflops1 = level1_skim(query_text, sample_ids, corpus)
    l1_calls = len(sample_ids)

    doc_grids, finalists, sigma2, meff2, lat2, tflops2 = level2_rows(query_text, survivors, corpus)
    evidence, lat3, tflops3 = level3_columns(query_text, finalists, doc_grids)

    metrics1, auc1 = evaluate_doc_scores(skim_scores, gold)
    gold_missed_l1 = len(gold_in_sample - set(survivors))

    l2_scores = {d: doc_grids[d]["combined_score"] for d in survivors}
    metrics2, auc2 = evaluate_doc_scores(l2_scores, gold)
    gold_missed_l2 = len((gold_in_sample & set(survivors)) - set(finalists))

    level3_calls = sum(doc_grids[d]["k"] for d in finalists)

    return {
        "gold_in_sample": len(gold_in_sample),
        "total_latency": lat1 + lat2 + lat3,
        "total_tflops": tflops1 + tflops2 + tflops3,
        "level1": {
            "metrics": metrics1, "auc": auc1, "calls": l1_calls, "latency": lat1, "tflops": tflops1,
            "sigma": sigma1, "m_eff": meff1, "survivors": len(survivors), "gold_missed": gold_missed_l1,
        },
        "level2": {
            "metrics": metrics2, "auc": auc2, "calls": sum(doc_grids[d]["k"] for d in survivors), "latency": lat2,
            "tflops": tflops2, "sigma": sigma2, "m_eff": meff2, "finalists": len(finalists),
            "gold_missed": gold_missed_l2,
        },
        "level3": {
            "calls": level3_calls, "latency": lat3, "tflops": tflops3,
            "evidence": {
                d: {**evidence[d], "is_gold": d in gold} for d in finalists
            },
        },
    }


def aggregate_level(level_results):
    avg = average_metrics([r["metrics"] for r in level_results])
    aucs = [r["auc"] for r in level_results if r["auc"] is not None]
    avg["auc"] = sum(aucs) / len(aucs) if aucs else float("nan")
    avg["calls"] = sum(r["calls"] for r in level_results) / len(level_results)
    avg["latency"] = sum(r["latency"] for r in level_results) / len(level_results)
    avg["tflops"] = sum(r["tflops"] for r in level_results) / len(level_results)
    avg["m_eff"] = sum(r["m_eff"] for r in level_results) / len(level_results)
    avg["gold_missed"] = sum(r["gold_missed"] for r in level_results) / len(level_results)
    return avg


NAME_COL = 18
COL = 14


def print_config_info():
    print(
        f"Gamma={GAMMA_0}  Temperature={SCORE_TEMPERATURE}  "
        f"Strategy={ROW_AGGREGATION}  WordsPerParagraph={WORDS_PER_PARAGRAPH}\n"
    )


def print_table(level_rows):
    headers = ["Level", "Calls", "Latency(s)", "Lat/Call(s)", "TFLOPs"]
    for name in ("Recall", "Precision", "NDCG"):
        headers += [f"{name}@{k}" for k in EVAL_KS]
    headers += ["AUC", "M_eff", "GoldMissed"]
    print(headers[0].ljust(NAME_COL) + "".join(h.ljust(COL) for h in headers[1:]))
    for label, m in level_rows:
        lat_per_call = m["latency"] / m["calls"] if m["calls"] else 0.0
        row = [f"{m['calls']:.1f}", f"{m['latency']:.2f}", f"{lat_per_call:.2f}", f"{m['tflops']:.2f}"]
        for name in ("recall", "precision", "ndcg"):
            row += [f"{m[name][k]:.3f}" for k in EVAL_KS]
        row += [f"{m['auc']:.3f}", f"{m['m_eff']:.1f}", f"{m['gold_missed']:.1f}"]
        print(label.ljust(NAME_COL) + "".join(v.ljust(COL) for v in row))


def run_config(config):
    print(f"\n{'#'*70}\n TASK BEING TESTED: {config.upper()}\n{'#'*70}")
    corpus, queries, gold_by_query = load_data(config)

    query_ids = list(queries)[:N_QUERIES]
    gold_ids = set().union(*(gold_by_query.get(q, set()) for q in query_ids))
    sample_ids = build_sample(list(corpus), gold_ids)

    print(f"Sample: {len(sample_ids)} docs, {N_QUERIES} queries\n")

    level1_results, level2_results, level3_all = [], [], []
    per_query_detail = {}
    query_totals = []
    for qid in query_ids:
        q_start = time.perf_counter()
        result = run_query(qid, corpus, queries, gold_by_query, sample_ids)
        level1_results.append(result["level1"])
        level2_results.append(result["level2"])
        level3_all.append(result["level3"])
        per_query_detail[qid] = result
        query_totals.append((result["total_latency"], result["total_tflops"]))
        print(
            f"=== Query {qid} done in {time.perf_counter() - q_start:.2f}s "
            f"(gold_in_sample={result['gold_in_sample']}, "
            f"L1 survivors={result['level1']['survivors']}, L2 finalists={result['level2']['finalists']}) ==="
        )

    return level1_results, level2_results, level3_all, per_query_detail, query_totals


def main():
    all_results = {}
    for config in CONFIGS:
        level1_results, level2_results, level3_all, per_query_detail, query_totals = run_config(config)

        agg1 = aggregate_level(level1_results)
        agg2 = aggregate_level(level2_results)

        l3_calls = sum(r["calls"] for r in level3_all) / len(level3_all)
        l3_latency = sum(r["latency"] for r in level3_all) / len(level3_all)
        l3_tflops = sum(r["tflops"] for r in level3_all) / len(level3_all)
        total_finalists = sum(len(r["evidence"]) for r in level3_all)
        gold_hits = sum(1 for r in level3_all for ev in r["evidence"].values() if ev["is_gold"])

        avg_query_latency = sum(t[0] for t in query_totals) / len(query_totals)
        avg_query_tflops = sum(t[1] for t in query_totals) / len(query_totals)

        print(f"\n{'='*70}\n SUMMARY — {config} — avg over {N_QUERIES} queries\n{'='*70}")
        print_config_info()
        print_table([("Level1_skim", agg1), ("Level2_rows", agg2)])
        l3_lat_per_call = l3_latency / l3_calls if l3_calls else 0.0
        print(
            f"\n{'Level3_columns'.ljust(NAME_COL)}calls={l3_calls:.1f}  latency={l3_latency:.2f}s  "
            f"lat/call={l3_lat_per_call:.2f}s  tflops={l3_tflops:.2f} TF  "
            f"finalists_that_are_gold={gold_hits}/{total_finalists}"
        )
        print(f"\nTotal per query (avg): latency={avg_query_latency:.2f}s  tflops={avg_query_tflops:.2f} TF")

        all_results[config] = {
            "level1": agg1,
            "level2": agg2,
            "level3": {
                "calls": l3_calls, "latency": l3_latency, "tflops": l3_tflops,
                "finalists_gold_hits": gold_hits, "finalists_total": total_finalists,
            },
            "avg_query_latency": avg_query_latency,
            "avg_query_tflops": avg_query_tflops,
            "per_query": per_query_detail,
        }

    results_path = "hierarchy_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
def run_meff_ablation(config=CONFIGS[0], seeds=(67, 76)):
    global GAMMA_0, SCORE_TEMPERATURE, WORDS_PER_PARAGRAPH, ROW_AGGREGATION
    global L1_MEFF_MULTIPLIER, L2_MEFF_MULTIPLIER, SEED
    original = (
        GAMMA_0, SCORE_TEMPERATURE, WORDS_PER_PARAGRAPH, ROW_AGGREGATION,
        L1_MEFF_MULTIPLIER, L2_MEFF_MULTIPLIER, SEED,
    )

    experiments = [
        {"name": "avg",              "row_aggregation": "avg",  "l1_mult": 1, "l2_mult": 1},
        {"name": "peak",             "row_aggregation": "peak", "l1_mult": 1, "l2_mult": 1},
        {"name": "peak_2xMeff_L2",   "row_aggregation": "peak", "l1_mult": 1, "l2_mult": 2},
        {"name": "peak_2xMeff_L1",   "row_aggregation": "peak", "l1_mult": 2, "l2_mult": 1},
        {"name": "peak_2xMeff_L1L2", "row_aggregation": "peak", "l1_mult": 2, "l2_mult": 2},
    ]

    all_runs_by_seed = {}
    for seed in seeds:
        print(f"\n{'@'*70}\n SEED = {seed}\n{'@'*70}")
        SEED = seed
        all_runs = []
        for exp in experiments:
            print(f"\n{'#'*70}\n RUNNING EXPERIMENT: {exp['name']} (seed={seed})\n{'#'*70}")
            GAMMA_0 = 0.1
            SCORE_TEMPERATURE = 5
            WORDS_PER_PARAGRAPH = 12
            ROW_AGGREGATION = exp["row_aggregation"]
            L1_MEFF_MULTIPLIER = exp["l1_mult"]
            L2_MEFF_MULTIPLIER = exp["l2_mult"]

            level1_results, level2_results, level3_all, per_query_detail, query_totals = run_config(config)
            agg1 = aggregate_level(level1_results)
            agg2 = aggregate_level(level2_results)

            l3_calls = sum(r["calls"] for r in level3_all) / len(level3_all)
            l3_latency = sum(r["latency"] for r in level3_all) / len(level3_all)
            l3_tflops = sum(r["tflops"] for r in level3_all) / len(level3_all)
            l3_lat_per_call = l3_latency / l3_calls if l3_calls else 0.0
            total_finalists = sum(len(r["evidence"]) for r in level3_all)
            gold_hits = sum(1 for r in level3_all for ev in r["evidence"].values() if ev["is_gold"])
            avg_query_latency = sum(t[0] for t in query_totals) / len(query_totals)
            avg_query_tflops = sum(t[1] for t in query_totals) / len(query_totals)

            all_runs.append({
                "name": exp["name"], "row_aggregation": exp["row_aggregation"],
                "l1_mult": exp["l1_mult"], "l2_mult": exp["l2_mult"],
                "gamma": GAMMA_0, "temperature": SCORE_TEMPERATURE, "words_per_paragraph": WORDS_PER_PARAGRAPH,
                "agg1": agg1, "agg2": agg2,
                "l3_calls": l3_calls, "l3_latency": l3_latency, "l3_tflops": l3_tflops,
                "l3_lat_per_call": l3_lat_per_call,
                "gold_hits": gold_hits, "total_finalists": total_finalists,
                "avg_query_latency": avg_query_latency, "avg_query_tflops": avg_query_tflops,
            })
        all_runs_by_seed[seed] = all_runs

    GAMMA_0, SCORE_TEMPERATURE, WORDS_PER_PARAGRAPH, ROW_AGGREGATION, \
        L1_MEFF_MULTIPLIER, L2_MEFF_MULTIPLIER, SEED = original

    for seed in seeds:
        print(f"\n{'@'*70}\n RESULTS FOR SEED = {seed}\n{'@'*70}")
        for run in all_runs_by_seed[seed]:
            print(f"\n{'='*70}\n SUMMARY — {config} — {run['name']} — avg over {N_QUERIES} queries\n{'='*70}")
            print(
                f"Gamma={run['gamma']}  Temperature={run['temperature']}  Strategy={run['row_aggregation']}  "
                f"WordsPerParagraph={run['words_per_paragraph']}  L1_MeffMult={run['l1_mult']}  "
                f"L2_MeffMult={run['l2_mult']}\n"
            )
            print_table([("Level1_skim", run["agg1"]), ("Level2_rows", run["agg2"])])
            print(
                f"\n{'Level3_columns'.ljust(NAME_COL)}calls={run['l3_calls']:.1f}  latency={run['l3_latency']:.2f}s  "
                f"lat/call={run['l3_lat_per_call']:.2f}s  tflops={run['l3_tflops']:.2f} TF  "
                f"finalists_that_are_gold={run['gold_hits']}/{run['total_finalists']}"
            )
            print(
                f"\nTotal per query (avg): latency={run['avg_query_latency']:.2f}s  "
                f"tflops={run['avg_query_tflops']:.2f} TF"
            )

    results_path = "meff_ablation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({str(seed): runs for seed, runs in all_runs_by_seed.items()}, f, indent=2)
    print(f"\nSaved results to {results_path}")

    return all_runs_by_seed


def build_relevance_prompt_aggressive(query_text, chunks_text):
    return f"""You are a strict relevance judge for OBLIQUE queries — queries where relevance is latent rather than explicit.
    A chunk can be relevant without keyword or topic overlap, through an implicit signal, a structural
    similarity, or a tone that specifically echoes the query. However, most candidate chunks are NOT relevant.
    Only answer Yes if the connection is clear and specific to this query — not merely because some loose,
    generic resemblance could be argued with enough interpretation. When in doubt, answer No.

    Query:
    {query_text}

    Candidate chunks:
    {chunks_text}

    Question: Is ANY chunk above clearly and specifically relevant to the query, through explicit or implicit similarity?
    Answer with a single word: Yes or No.
    """


def build_relevance_prompt_light(query_text, chunks_text):
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


def build_relevance_prompt_general(query_text, chunks_text):
    return f"""You are a relevance judge. Given a query and a set of candidate chunks, determine whether any
    of the chunks are relevant to the query.

    Query:
    {query_text}

    Candidate chunks:
    {chunks_text}

    Question: Is ANY chunk above relevant to the query?
    Answer with a single word: Yes or No.
    """


def run_prompt_ablation(config=CONFIGS[0], seeds=(67, 76)):
    global build_relevance_prompt, SEED
    original_prompt_fn = build_relevance_prompt
    original_seed = SEED

    variants = {
        "aggressive": build_relevance_prompt_aggressive,
        "light": build_relevance_prompt_light,
        "general": build_relevance_prompt_general,
    }

    results_by_seed = {}
    for seed in seeds:
        print(f"\n{'@'*70}\n SEED = {seed}\n{'@'*70}")
        SEED = seed
        seed_results = {}
        for name, prompt_fn in variants.items():
            print(f"\n{'#'*70}\n RUNNING PROMPT VARIANT: {name} (seed={seed})\n{'#'*70}")
            build_relevance_prompt = prompt_fn

            level1_results, level2_results, level3_all, per_query_detail, query_totals = run_config(config)
            agg1 = aggregate_level(level1_results)
            agg2 = aggregate_level(level2_results)

            l3_calls = sum(r["calls"] for r in level3_all) / len(level3_all)
            l3_latency = sum(r["latency"] for r in level3_all) / len(level3_all)
            l3_tflops = sum(r["tflops"] for r in level3_all) / len(level3_all)
            l3_lat_per_call = l3_latency / l3_calls if l3_calls else 0.0
            total_finalists = sum(len(r["evidence"]) for r in level3_all)
            gold_hits = sum(1 for r in level3_all for ev in r["evidence"].values() if ev["is_gold"])
            avg_query_latency = sum(t[0] for t in query_totals) / len(query_totals)
            avg_query_tflops = sum(t[1] for t in query_totals) / len(query_totals)

            seed_results[name] = {
                "level1": agg1,
                "level2": agg2,
                "level3": {
                    "calls": l3_calls, "latency": l3_latency, "tflops": l3_tflops,
                    "lat_per_call": l3_lat_per_call,
                    "finalists_gold_hits": gold_hits, "finalists_total": total_finalists,
                },
                "avg_query_latency": avg_query_latency,
                "avg_query_tflops": avg_query_tflops,
            }
        results_by_seed[seed] = seed_results

    build_relevance_prompt = original_prompt_fn
    SEED = original_seed

    for seed in seeds:
        print(f"\n{'@'*70}\n RESULTS FOR SEED = {seed}\n{'@'*70}")
        for name, r in results_by_seed[seed].items():
            print(f"\n{'='*70}\n SUMMARY — {config} — {name} prompt — avg over {N_QUERIES} queries\n{'='*70}")
            print_config_info()
            print_table([("Level1_skim", r["level1"]), ("Level2_rows", r["level2"])])
            l3 = r["level3"]
            print(
                f"\n{'Level3_columns'.ljust(NAME_COL)}calls={l3['calls']:.1f}  latency={l3['latency']:.2f}s  "
                f"lat/call={l3['lat_per_call']:.2f}s  tflops={l3['tflops']:.2f} TF  "
                f"finalists_that_are_gold={l3['finalists_gold_hits']}/{l3['finalists_total']}"
            )
            print(
                f"\nTotal per query (avg): latency={r['avg_query_latency']:.2f}s  "
                f"tflops={r['avg_query_tflops']:.2f} TF"
            )

    results_path = "prompt_ablation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({str(seed): r for seed, r in results_by_seed.items()}, f, indent=2)
    print(f"\nSaved results to {results_path}")

    return results_by_seed


if __name__ == "__main__":
    main()

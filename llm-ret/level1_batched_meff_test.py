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

# ---- copied as-is from hierarchy_experiment.py (4B model) ----
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
WORDS_PER_PARAGRAPH = 12
GAMMA_0 = 0.25
SCORE_TEMPERATURE = 8.0
BATCH_SIZE = 10                 # the new idea under test
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, quantization_config=quant_config).to(DEVICE)

N_LAYERS = model.config.num_hidden_layers
D_MODEL = model.config.hidden_size
N_ACTIVE_PARAMS = 4e9


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


OPTION_IDS = None


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


def yesno_letter_token_ids():
    ids = {}
    for label, variants in (("Yes", ["Y", " Y", "y", " y"]), ("No", ["N", " N", "n", " n"])):
        for variant in variants:
            ids[tokenizer.encode(variant, add_special_tokens=False)[-1]] = label
    return ids


LETTER_OPTION_IDS = None


def build_listwise_prompt(query_text, skim_texts):
    chunks_block = "\n\n".join(f"Chunk {i+1}: {text}" for i, text in enumerate(skim_texts))
    n = len(skim_texts)
    return f"""You are an expert relevance judge for OBLIQUE queries — queries where relevance is latent.
    A chunk can be relevant even if it shares no keywords or topic with the query
    — relevance may show up as an implicit signal, a structural or abstract similarity,
    a tone, or a fuzzy impressionistic resemblance rather than explicit content overlap.

    Query:
    {query_text}

    {chunks_block}

    For EACH chunk above, judge whether it is relevant to the query.
    Answer with exactly {n} single letters separated by spaces, one per chunk, in order:
    Y = relevant, N = not relevant.
    Answer with nothing else -- just the {n} letters, e.g.: Y N Y N ...
    """


def score_batch_listwise(query_text, doc_ids, corpus):
    # ONE call scores the whole batch -- the model must name a verdict per chunk (not one
    # blanket yes/no), so per-document signal survives instead of collapsing like the
    # earlier concatenated-row-verifier attempt did. Single-letter answers keep the
    # generation budget short since decode, not prefill, is the latency bottleneck here.
    global LETTER_OPTION_IDS
    if LETTER_OPTION_IDS is None:
        LETTER_OPTION_IDS = yesno_letter_token_ids()

    skim_texts = [build_skim_text(corpus[d]) for d in doc_ids]
    prompt = build_listwise_prompt(query_text, skim_texts)
    inputs = prompt_inputs(prompt)
    prompt_len = inputs["input_ids"].shape[1]
    max_new_tokens = len(doc_ids) * 2  # "Y "/"N " style single-letter answers

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )

    generated_ids = out.sequences[0][prompt_len:]
    total_context = out.sequences.shape[1]
    tflops = estimate_tflops(total_context)  # approximation -- treats the whole generation as one pass

    ids = list(LETTER_OPTION_IDS)
    scores = {}
    doc_idx = 0
    for step, token_id in enumerate(generated_ids):
        if doc_idx >= len(doc_ids):
            break
        tid = token_id.item()
        if tid in LETTER_OPTION_IDS:
            logits = out.scores[step][0]
            probs = torch.softmax(logits[ids] / SCORE_TEMPERATURE, dim=0)
            yes_prob = sum(p for t, p in zip(ids, probs) if LETTER_OPTION_IDS[t] == "Yes")
            scores[doc_ids[doc_idx]] = float(yes_prob)
            doc_idx += 1

    for d in doc_ids:
        if d not in scores:
            scores[d] = 0.5  # model never gave a clean verdict for this one -- neutral fallback

    return scores, tflops


def compute_sigma_meff(values, k):
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


# ---- batched M_eff: per-batch cut only, pooled survivors go straight to Level 2 ----

def build_batches(sample_ids, batch_size=BATCH_SIZE):
    # shuffle before slicing -- sample_ids is sorted by doc ID, and gold IDs cluster low,
    # so slicing the sorted list directly would dump almost all gold into batch 0.
    shuffled = list(sample_ids)
    random.Random(SEED).shuffle(shuffled)
    return [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]


def run_query_batched(qid, corpus, queries, gold_by_query, sample_ids, batches):
    gold = gold_by_query.get(qid, set()) - {qid}
    query_text = queries[qid]
    gold_in_sample = gold & set(sample_ids)

    t0 = time.perf_counter()
    total_tflops = 0.0
    total_calls = 0
    batch_stats = []
    survivor_scores = {}

    for b_idx, batch_ids in enumerate(batches):
        scores, tflops = score_batch_listwise(query_text, batch_ids, corpus)
        total_tflops += tflops
        total_calls += 1

        sigma, m_eff = compute_sigma_meff(list(scores.values()), k=round(math.sqrt(len(batch_ids))))
        survivors = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:m_eff]]
        gold_in_batch = len(gold_in_sample & set(batch_ids))
        gold_missed_batch = len(gold_in_sample & set(batch_ids) - set(survivors))

        batch_stats.append({
            "batch": b_idx, "size": len(batch_ids), "sigma": sigma, "m_eff": m_eff,
            "gold_in_batch": gold_in_batch, "gold_missed": gold_missed_batch,
        })
        for d in survivors:
            survivor_scores[d] = scores[d]

    n_final = len(survivor_scores)
    final_sigma, final_m_eff = (0.0, 0)
    final_survivors = []
    if n_final > 0:
        final_sigma, final_m_eff = compute_sigma_meff(list(survivor_scores.values()), k=round(math.sqrt(n_final)))
        final_survivors = [d for d, _ in sorted(survivor_scores.items(), key=lambda x: x[1], reverse=True)[:final_m_eff]]

    latency = time.perf_counter() - t0
    final_gold_missed = len(gold_in_sample - set(final_survivors))
    metrics, auc = evaluate_doc_scores(survivor_scores, gold)

    final_stats = {
        "calls": total_calls, "latency": latency, "tflops": total_tflops,
        "final_sigma": final_sigma, "final_m_eff": final_m_eff,
        "gold_in_sample": len(gold_in_sample), "gold_missed": final_gold_missed,
        "metrics": metrics, "auc": auc,
    }
    return batch_stats, final_stats


NAME_COL = 10
COL = 14


def print_batch_table(all_batch_stats, n_batches):
    print("--- Per-batch M_eff (averaged across queries) ---")
    headers = ["Batch", "Size", "Sigma", "M_eff", "GoldInBatch", "GoldMissed"]
    print(headers[0].ljust(NAME_COL) + "".join(h.ljust(COL) for h in headers[1:]))
    for b in range(n_batches):
        rows = [q[b] for q in all_batch_stats]
        size = rows[0]["size"]
        sigma = sum(r["sigma"] for r in rows) / len(rows)
        m_eff = sum(r["m_eff"] for r in rows) / len(rows)
        gold_in_batch = sum(r["gold_in_batch"] for r in rows) / len(rows)
        gold_missed = sum(r["gold_missed"] for r in rows) / len(rows)
        row = [f"{size}", f"{sigma:.3f}", f"{m_eff:.1f}", f"{gold_in_batch:.1f}", f"{gold_missed:.1f}"]
        print(str(b).ljust(NAME_COL) + "".join(v.ljust(COL) for v in row))


def print_final_table(all_final_stats):
    print("\n--- Final M_eff (after pooling batch survivors, averaged across queries) ---")
    headers = ["Calls", "Latency(s)", "TFLOPs", "FinalM_eff", "GoldInSample", "GoldMissed"]
    for name in ("Recall", "Precision", "NDCG"):
        headers += [f"{name}@{k}" for k in EVAL_KS]
    headers += ["AUC"]
    print("".join(h.ljust(COL) for h in headers))

    calls = sum(r["calls"] for r in all_final_stats) / len(all_final_stats)
    latency = sum(r["latency"] for r in all_final_stats) / len(all_final_stats)
    tflops = sum(r["tflops"] for r in all_final_stats) / len(all_final_stats)
    final_m_eff = sum(r["final_m_eff"] for r in all_final_stats) / len(all_final_stats)
    gold_in_sample = sum(r["gold_in_sample"] for r in all_final_stats) / len(all_final_stats)
    gold_missed = sum(r["gold_missed"] for r in all_final_stats) / len(all_final_stats)
    avg_metrics = average_metrics([r["metrics"] for r in all_final_stats])
    aucs = [r["auc"] for r in all_final_stats if r["auc"] is not None]
    avg_auc = sum(aucs) / len(aucs) if aucs else float("nan")

    row = [f"{calls:.1f}", f"{latency:.2f}", f"{tflops:.2f}", f"{final_m_eff:.1f}", f"{gold_in_sample:.1f}", f"{gold_missed:.1f}"]
    for name in ("recall", "precision", "ndcg"):
        row += [f"{avg_metrics[name][k]:.3f}" for k in EVAL_KS]
    row += [f"{avg_auc:.3f}"]
    print("".join(v.ljust(COL) for v in row))


def main():
    for config in CONFIGS:
        print(f"\n{'#'*70}\n TASK BEING TESTED: {config.upper()}\n{'#'*70}")
        corpus, queries, gold_by_query = load_data(config)

        query_ids = list(queries)[:N_QUERIES]
        gold_ids = set().union(*(gold_by_query.get(q, set()) for q in query_ids))
        sample_ids = build_sample(list(corpus), gold_ids)
        batches = build_batches(sample_ids)
        print(f"Sample: {len(sample_ids)} docs -> {len(batches)} batches of {BATCH_SIZE}, {N_QUERIES} queries\n")

        all_batch_stats, all_final_stats = [], []
        for qid in query_ids:
            q_start = time.perf_counter()
            batch_stats, final_stats = run_query_batched(qid, corpus, queries, gold_by_query, sample_ids, batches)
            all_batch_stats.append(batch_stats)
            all_final_stats.append(final_stats)
            print(f"=== Query {qid} done in {time.perf_counter() - q_start:.2f}s ===")

        print(f"\n{'='*70}\n BATCHED M_EFF TEST — {config} — avg over {N_QUERIES} queries\n{'='*70}")
        print_batch_table(all_batch_stats, len(batches))
        print_final_table(all_final_stats)

        results_path = "level1_batched_meff_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({"batches": all_batch_stats, "final": all_final_stats}, f, indent=2, default=str)
        print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()

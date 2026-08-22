from IPython.display import clear_output
#!pip install transformers datasets huggingface_hub torch accelerate scipy bitsandbytes
import json
import math
import random
import re
import time
import torch
import numpy as np
from scipy.stats import skew, kurtosis
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
clear_output()

# ---- copied as-is from hierarchy_experiment.py ----
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
WORDS_PER_PARAGRAPH = 12# strategy 1
SENTENCES_PER_PARAGRAPH = 1 # strategy 2
SENTENCES_PER_SECTION = 2   # strategy 3 (begin/mid/end)
DECAY_START_WORDS = 24  # strategy 4 (decaying words per paragraph)
DECAY_FACTOR = 0.7
DECAY_MIN_WORDS = 4
SLIDING_WINDOW_COUNT = 8# strategy 5 (evenly spaced windows across the whole doc)
SLIDING_WINDOW_WORDS = 8
GAMMA_0 = 0.25
SCORE_TEMPERATURE = 8.0
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


# ---- the 5 skim strategies under test ----

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def skim_words_per_paragraph(doc_text):
    paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [doc_text.strip()]
    snippets = []
    for p in paragraphs:
        words = p.split()[:WORDS_PER_PARAGRAPH]
        if words:
            snippets.append(" ".join(words))
    return " / ".join(snippets)


def skim_sentences_per_paragraph(doc_text):
    paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [doc_text.strip()]
    snippets = []
    for p in paragraphs:
        sentences = SENTENCE_SPLIT_RE.split(p)
        snippet = " ".join(sentences[:SENTENCES_PER_PARAGRAPH]).strip()
        if snippet:
            snippets.append(snippet)
    return " / ".join(snippets)


def skim_begin_mid_end_sentences(doc_text):
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(doc_text) if s.strip()]
    n = len(sentences)
    if n <= SENTENCES_PER_SECTION * 3:
        return " ".join(sentences)
    mid = n // 2
    start = sentences[:SENTENCES_PER_SECTION]
    middle = sentences[mid - SENTENCES_PER_SECTION // 2 : mid + SENTENCES_PER_SECTION // 2]
    end = sentences[-SENTENCES_PER_SECTION:]
    return " / ".join(" ".join(part) for part in (start, middle, end))


def skim_decaying_words(doc_text):
    paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [doc_text.strip()]
    snippets = []
    budget = DECAY_START_WORDS
    for p in paragraphs:
        n_words = max(DECAY_MIN_WORDS, round(budget))
        words = p.split()[:n_words]
        if words:
            snippets.append(" ".join(words))
        budget = max(DECAY_MIN_WORDS, budget * DECAY_FACTOR)
    return " / ".join(snippets)


def skim_sliding_window(doc_text):
    words = doc_text.split()
    n = len(words)
    if n <= SLIDING_WINDOW_COUNT * SLIDING_WINDOW_WORDS:
        return " ".join(words)
    snippets = []
    for i in range(SLIDING_WINDOW_COUNT):
        start = round(i * (n - SLIDING_WINDOW_WORDS) / max(1, SLIDING_WINDOW_COUNT - 1))
        snippets.append(" ".join(words[start : start + SLIDING_WINDOW_WORDS]))
    return " / ".join(snippets)


STRATEGIES = {
    "words_per_paragraph": skim_words_per_paragraph,
    "sentences_per_paragraph": skim_sentences_per_paragraph,
    "begin_mid_end_sentences": skim_begin_mid_end_sentences,
    "decaying_words": skim_decaying_words,
    "sliding_window": skim_sliding_window,
}


# ---- Level 1 only, run once per strategy ----

def run_level1(skim_fn, query_text, sample_ids, corpus):
    t0 = time.perf_counter()
    scores = {}
    total_tflops = 0.0
    n = len(sample_ids)
    for doc_id in sample_ids:
        skim_text = skim_fn(corpus[doc_id])
        score, tflops = score_relevance(query_text, format_chunks([skim_text]))
        scores[doc_id] = score
        total_tflops += tflops
    latency = time.perf_counter() - t0

    sigma, m_eff = compute_sigma_meff(list(scores.values()), k=round(math.sqrt(n)))
    survivors = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:m_eff]]
    return scores, survivors, sigma, m_eff, latency, total_tflops


def run_query_all_strategies(qid, corpus, queries, gold_by_query, sample_ids):
    gold = gold_by_query.get(qid, set()) - {qid}
    query_text = queries[qid]
    gold_in_sample = gold & set(sample_ids)

    results = {}
    for name, skim_fn in STRATEGIES.items():
        scores, survivors, sigma, m_eff, latency, tflops = run_level1(skim_fn, query_text, sample_ids, corpus)
        metrics, auc = evaluate_doc_scores(scores, gold)
        gold_missed = len(gold_in_sample - set(survivors))
        results[name] = {
            "metrics": metrics, "auc": auc, "calls": len(sample_ids), "latency": latency, "tflops": tflops,
            "sigma": sigma, "m_eff": m_eff, "survivors": len(survivors), "gold_missed": gold_missed,
        }
    return results, len(gold_in_sample)


def aggregate_strategy(results_list):
    avg = average_metrics([r["metrics"] for r in results_list])
    aucs = [r["auc"] for r in results_list if r["auc"] is not None]
    avg["auc"] = sum(aucs) / len(aucs) if aucs else float("nan")
    avg["calls"] = sum(r["calls"] for r in results_list) / len(results_list)
    avg["latency"] = sum(r["latency"] for r in results_list) / len(results_list)
    avg["tflops"] = sum(r["tflops"] for r in results_list) / len(results_list)
    avg["m_eff"] = sum(r["m_eff"] for r in results_list) / len(results_list)
    avg["gold_missed"] = sum(r["gold_missed"] for r in results_list) / len(results_list)
    return avg


NAME_COL = 24
COL = 14


def print_table(rows):
    headers = ["Strategy", "Calls", "Latency(s)", "Lat/Call(s)", "TFLOPs"]
    for name in ("Recall", "Precision", "NDCG"):
        headers += [f"{name}@{k}" for k in EVAL_KS]
    headers += ["AUC", "M_eff(->L2)", "GoldMissed"]
    print(headers[0].ljust(NAME_COL) + "".join(h.ljust(COL) for h in headers[1:]))
    for label, m in rows:
        lat_per_call = m["latency"] / m["calls"] if m["calls"] else 0.0
        row = [f"{m['calls']:.1f}", f"{m['latency']:.2f}", f"{lat_per_call:.2f}", f"{m['tflops']:.2f}"]
        for name in ("recall", "precision", "ndcg"):
            row += [f"{m[name][k]:.3f}" for k in EVAL_KS]
        row += [f"{m['auc']:.3f}", f"{m['m_eff']:.1f}", f"{m['gold_missed']:.1f}"]
        print(label.ljust(NAME_COL) + "".join(v.ljust(COL) for v in row))


def main():
    for config in CONFIGS:
        print(f"\n{'#'*70}\n TASK BEING TESTED: {config.upper()}\n{'#'*70}")
        corpus, queries, gold_by_query = load_data(config)

        query_ids = list(queries)[:N_QUERIES]
        gold_ids = set().union(*(gold_by_query.get(q, set()) for q in query_ids))
        sample_ids = build_sample(list(corpus), gold_ids)
        print(f"Sample: {len(sample_ids)} docs, {N_QUERIES} queries\n")

        per_strategy = {name: [] for name in STRATEGIES}
        for qid in query_ids:
            q_start = time.perf_counter()
            results, gold_in_sample = run_query_all_strategies(qid, corpus, queries, gold_by_query, sample_ids)
            for name in STRATEGIES:
                per_strategy[name].append(results[name])
            print(f"=== Query {qid} done in {time.perf_counter() - q_start:.2f}s (gold_in_sample={gold_in_sample}) ===")

        agg = {name: aggregate_strategy(rs) for name, rs in per_strategy.items()}

        print(f"\n{'='*70}\n LEVEL 1 SKIM STRATEGY COMPARISON — {config} — avg over {N_QUERIES} queries\n{'='*70}")
        print_table([(name, agg[name]) for name in STRATEGIES])

        results_path = "level1_skim_ablation_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
        print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()

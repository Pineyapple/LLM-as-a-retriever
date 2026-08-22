from IPython.display import clear_output, display, Javascript
#!pip install transformers datasets huggingface_hub torch accelerate bitsandbytes
import base64
import json
import random
import string
import math
import time
import torch
from datasets import load_dataset
import transformers, jinja2
from huggingface_hub import hf_hub_download
import bitsandbytes
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
K = 8
N_QUERIES = 2
EVAL_KS = [5, 10, 50, 100]
SAMPLE_SIZE = 100
SEED = 67
MAX_NEW_TOKENS = 5
MAX_CHUNK_TOKENS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


def chunk_rows(sample_ids):
    return [sample_ids[i : i + K] for i in range(0, len(sample_ids), K)]


def prompt_inputs(prompt, enable_thinking=True):
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


def score_row(query_text, doc_texts):
    chunks_text = "\n\n".join(f"chunk {i+1}: {truncate_chunk(t)}" for i, t in enumerate(doc_texts))
    prompt = f"""You are an expert relevance judge for OBLIQUE queries — queries where relevance is latent. and you have to be confident in your answers
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
    inputs = prompt_inputs(prompt, enable_thinking=False)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    option_ids = yesno_token_ids()
    generated = out.sequences[0][inputs["input_ids"].shape[1] :]
    for step, token_id in enumerate(generated):
        if token_id.item() in option_ids:
            logits = out.scores[step][0]
            ids = list(option_ids)
            probs = torch.softmax(logits[ids], dim=0)
            yes_prob = sum(p for tid, p in zip(ids, probs) if option_ids[tid] == "Yes")
            return float(yes_prob)
    return 0.0


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


def trigger_download(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    js = f"""
    var a = document.createElement('a');
    a.href = "data:application/json;base64,{data}";
    a.download = "{path}";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    """
    display(Javascript(js))


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


def format_metrics_line(metrics):
    parts = []
    for name in ("recall", "precision", "ndcg"):
        parts += [f"{name.capitalize()}@{k}: {metrics[name][k]:.2f}" for k in EVAL_KS]
    return "  ".join(parts)


def average_metrics(query_metrics_list):
    avg = {"recall": {}, "precision": {}, "ndcg": {}}
    for name in ("recall", "precision", "ndcg"):
        for k in EVAL_KS:
            vals = [m[name][k] for m in query_metrics_list if m[name][k] is not None]
            avg[name][k] = sum(vals) / len(vals) if vals else float("nan")
    return avg


def run_config(config):
    task_start = time.perf_counter()
    print(f"\n{'#'*70}\n TASK BEING TESTED: {config.upper()}\n{'#'*70}")
    corpus, queries, gold_by_query = load_data(config)

    query_ids = list(queries)[:N_QUERIES]
    gold_ids = set().union(*(gold_by_query.get(q, set()) for q in query_ids))
    sample_ids = build_sample(list(corpus), gold_ids)
    rows = chunk_rows(sample_ids)

    print(f"Sample: {len(sample_ids)} docs -> {len(rows)} rows\n")

    query_metrics = []
    for qid in query_ids:
        query_start = time.perf_counter()
        gold = gold_by_query.get(qid, set()) - {qid}
        query_text = queries[qid]
        print(f"=== Query {qid} === (gold docs: {len(gold)})\n")
        print(f"{'Row':<5}{'Score':<8}{'HasGold':<9}Doc IDs")

        row_results = []
        for i, row_ids in enumerate(rows):
            score = score_row(query_text, [corpus[cid] for cid in row_ids])
            has_gold = any(cid in gold for cid in row_ids)
            row_results.append((score, has_gold))
            print(f"{i:<5}{score:<8.2f}{str(has_gold):<9}{row_ids}")

        query_latency = time.perf_counter() - query_start
        ranked = [has_gold for _, has_gold in sorted(row_results, key=lambda x: x[0], reverse=True)]
        metrics = compute_metrics(ranked)
        auc = auc_score(row_results)
        metric_str = (
            f"{format_metrics_line(metrics)}  AUC: {auc:.2f}"
            if metrics["recall"][EVAL_KS[0]] is not None
            else "No gold rows to evaluate"
        )
        print(f"\n{metric_str}  Latency: {query_latency:.2f}s\n")
        query_metrics.append({**metrics, "auc": auc, "latency": query_latency})

    task_latency = time.perf_counter() - task_start
    print(f"Task total latency: {task_latency:.2f}s")
    return query_metrics, task_latency


def main():
    summary = {}
    for config in CONFIGS:
        query_metrics, task_latency = run_config(config)
        avg = average_metrics(query_metrics)
        aucs = [m["auc"] for m in query_metrics if m["auc"] is not None]
        avg["auc"] = sum(aucs) / len(aucs) if aucs else float("nan")
        avg["task_latency"] = task_latency
        summary[config] = avg

    print(f"\n{'='*70}\n SUMMARY — avg over {N_QUERIES} queries per task\n{'='*70}")
    col = 12
    headers = ["Task"]
    for name in ("Recall", "Precision", "NDCG"):
        headers += [f"{name}@{k}" for k in EVAL_KS]
    headers += ["AUC", "Latency(s)"]
    print("".join(h.ljust(col) for h in headers))
    for config, avg in summary.items():
        row = [config]
        for name in ("recall", "precision", "ndcg"):
            row += [f"{avg[name][k]:.3f}" for k in EVAL_KS]
        row += [f"{avg['auc']:.3f}", f"{avg['task_latency']:.2f}"]
        print("".join(v.ljust(col) for v in row))

    results_path = "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to {results_path}")
    trigger_download(results_path)


if __name__ == "__main__":
    main()

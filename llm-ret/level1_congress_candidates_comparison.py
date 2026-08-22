from IPython.display import clear_output
#!pip install datasets huggingface_hub torch accelerate scipy bitsandbytes
#!pip install -U transformers
import gc
import json
import math
import random
import re
import time
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import skew, kurtosis
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
clear_output()

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

clear_output()

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)

# Every model tested so far on the OBLIQ-Bench "writing" full-comparison run (3 old + 2 new),
# all in one pool per the request to re-test everything on this new, smaller candidate set.
MODELS = [
    {"id": "Qwen/Qwen3.5-0.8B", "quantize": False, "n_active_params": 0.8e9, "score_temperature": 0.7},
    {"id": "google/gemma-3-4b-it", "quantize": False, "n_active_params": 4e9, "score_temperature": 1.0},
    {"id": "google/gemma-3-270m-it", "quantize": False, "n_active_params": 270e6, "score_temperature": 1.0},
    {"id": "google/gemma-3-1b-it", "quantize": False, "n_active_params": 1e9, "score_temperature": 1.0},
    # gemma-4-E2B-it is MatFormer-based: "2.3B effective" params but a larger on-disk footprint
    # (Gemma 3n's E2B precedent was 5B total on disk) -- quantized for memory headroom accordingly.
    {"id": "google/gemma-4-E2B-it", "quantize": True, "n_active_params": 2.3e9, "score_temperature": 1.0},
]

# Point this at the file's location as a Kaggle input, e.g.
# "/kaggle/input/<your-dataset-slug>/candidates_congress_top300_goldinjected.jsonl"
DATA_PATH = "candidates_congress_top300_goldinjected.jsonl"

# This file has 20 queries total, each with its own ~300-doc hard-candidate pool (already far
# smaller than the 500-doc OBLIQ sample used elsewhere) -- so unlike the other scripts there is
# no SAMPLE_SIZE/build_sample step, every candidate in a query's pool is used as-is.
# Start with a small slice (e.g. 3) to sanity-check timing before committing to all 20.
N_QUERIES = 20

GAMMA_0 = 0.1

WORDS_PER_PARAGRAPH = 12          # strategy 1

SENTENCES_PER_PARAGRAPH = 1       # strategy 2

SENTENCES_PER_SECTION = 2         # strategy 3 (begin/mid/end)

DECAY_START_WORDS = 24            # strategy 4 (decaying words per paragraph)
DECAY_FACTOR = 0.7
DECAY_MIN_WORDS = 4

SLIDING_WINDOW_COUNT = 8          # strategy 5 (evenly spaced windows across the whole doc)
SLIDING_WINDOW_WORDS = 8

BATCH_SIZE = 10                   # strategy 6 (batched listwise)

SEED = 67

# One model copy per visible GPU (e.g. ["cuda:0", "cuda:1"] on Kaggle's T4 x2), so the query loop
# below can hand each device its own share of the work and run them concurrently. Falls back to a
# single device (GPU or CPU) automatically if only one is visible.
if torch.cuda.is_available():
    DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
else:
    DEVICES = ["cpu"]
print(f"Detected devices: {DEVICES}")


def get_model_dims(m):
    # Some newer architectures (e.g. Gemma3Config) nest the real language-model settings under
    # a text_config sub-object instead of exposing num_hidden_layers/hidden_size directly.
    config = m.config
    text_config = getattr(config, "text_config", config)
    return text_config.num_hidden_layers, text_config.hidden_size


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


def compute_sigma_meff(values, k):
    arr = np.array(values, dtype=float)
    sigma = 0.0
    if arr.var() > 1e-9:
        sigma = abs(skew(arr)) + 0.5 * max(0.0, kurtosis(arr, fisher=True))
    n = len(values)
    m_eff = max(1, min(n, math.ceil(n / (1 + GAMMA_0 * k * sigma))))
    return sigma, m_eff


def load_data(path):
    # Each line is a full query with its own pre-built candidate pool -- gold docs stage-1 missed
    # were injected (rank=None, injected=True) so is_gold is the only reliable gold signal; the
    # README explicitly says not to compute retrieval metrics on this file, so we only track
    # Level-1 call/latency/TFLOPs cost and whether the gold doc(s) survive the skim (gold_missed).
    queries_data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            queries_data.append(json.loads(line))
    return queries_data


# ---- strategies 1-5: single-document skim text builders (pure text, no model -- shared) ----

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


SIMPLE_STRATEGIES = {
    "words_per_paragraph": skim_words_per_paragraph,
    "sentences_per_paragraph": skim_sentences_per_paragraph,
    "begin_mid_end_sentences": skim_begin_mid_end_sentences,
    "decaying_words": skim_decaying_words,
    "sliding_window": skim_sliding_window,
}

ALL_NAMES = list(SIMPLE_STRATEGIES.keys()) + ["batches"]


def build_batches(sample_ids, batch_size=BATCH_SIZE):
    shuffled = list(sample_ids)
    random.Random(SEED).shuffle(shuffled)
    return [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]


# ---- Worker: one full model copy pinned to one device, all model-touching state lives here so
# that N devices can each run their own worker concurrently without stepping on each other ----

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
        self.letter_option_ids = None

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
        # Yes/No token ids are tokenizer-specific -- must be rebuilt fresh for each new model.
        self.option_ids = None
        self.letter_option_ids = None

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
        # enable_thinking=False matters for the Qwen3.5 model in this run (defaults to thinking
        # mode, which would break the single-token/single-letter logit read) -- harmless no-op
        # for Gemma.
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

    def yesno_letter_token_ids(self):
        ids = {}
        for label, variants in (("Yes", ["Y", " Y", "y", " y"]), ("No", ["N", " N", "n", " n"])):
            for variant in variants:
                ids[self.tokenizer.encode(variant, add_special_tokens=False)[-1]] = label
        return ids

    def score_batch_listwise(self, query_text, doc_ids, corpus):
        if self.letter_option_ids is None:
            self.letter_option_ids = self.yesno_letter_token_ids()

        skim_texts = [skim_words_per_paragraph(corpus[d]) for d in doc_ids]
        prompt = build_listwise_prompt(query_text, skim_texts)
        inputs = self.prompt_inputs(prompt)
        prompt_len = inputs["input_ids"].shape[1]
        max_new_tokens = len(doc_ids) * 2

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
            )

        generated_ids = out.sequences[0][prompt_len:]
        total_context = out.sequences.shape[1]
        tflops = self.estimate_tflops(total_context)

        ids = list(self.letter_option_ids)
        scores = {}
        doc_idx = 0
        for step, token_id in enumerate(generated_ids):
            if doc_idx >= len(doc_ids):
                break
            tid = token_id.item()
            if tid in self.letter_option_ids:
                logits = out.scores[step][0]
                probs = torch.softmax(logits[ids] / self.score_temperature, dim=0)
                yes_prob = sum(p for t, p in zip(ids, probs) if self.letter_option_ids[t] == "Yes")
                scores[doc_ids[doc_idx]] = float(yes_prob)
                doc_idx += 1

        for d in doc_ids:
            if d not in scores:
                scores[d] = 0.5

        return scores, tflops


def run_simple_strategy(worker, skim_fn, query_text, sample_ids, corpus, gold_in_sample):
    t0 = time.perf_counter()
    scores = {}
    total_tflops = 0.0
    for doc_id in sample_ids:
        skim_text = skim_fn(corpus[doc_id])
        score, tflops = worker.score_relevance(query_text, format_chunks([skim_text]))
        scores[doc_id] = score
        total_tflops += tflops
    latency = time.perf_counter() - t0

    n = len(sample_ids)
    sigma, m_eff = compute_sigma_meff(list(scores.values()), k=round(math.sqrt(n)))
    survivors = [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:m_eff]]
    gold_missed = len(gold_in_sample - set(survivors))

    return {
        "calls": len(sample_ids), "latency": latency, "tflops": total_tflops,
        "m_eff": m_eff, "gold_missed": gold_missed,
    }


def run_batches_strategy(worker, query_text, sample_ids, corpus, gold_in_sample):
    batches = build_batches(sample_ids)
    t0 = time.perf_counter()
    total_tflops = 0.0
    total_calls = 0
    survivor_scores = {}

    for batch_ids in batches:
        scores, tflops = worker.score_batch_listwise(query_text, batch_ids, corpus)
        total_tflops += tflops
        total_calls += 1

        sigma, m_eff = compute_sigma_meff(list(scores.values()), k=round(math.sqrt(len(batch_ids))))
        survivors = [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:m_eff]]
        for d in survivors:
            survivor_scores[d] = scores[d]

    latency = time.perf_counter() - t0
    total_survivors = len(survivor_scores)
    gold_missed = len(gold_in_sample - set(survivor_scores))

    return {
        "calls": total_calls, "latency": latency, "tflops": total_tflops,
        "m_eff": total_survivors, "gold_missed": gold_missed,
    }


def run_query_all(worker, query_obj):
    query_text = query_obj["query"]
    candidates = query_obj["candidates"]
    corpus = {c["doc_id"]: c["text"] for c in candidates}
    sample_ids = list(corpus)
    gold_in_sample = {c["doc_id"] for c in candidates if c["is_gold"]}

    results = {}
    for name, skim_fn in SIMPLE_STRATEGIES.items():
        results[name] = run_simple_strategy(worker, skim_fn, query_text, sample_ids, corpus, gold_in_sample)
    results["batches"] = run_batches_strategy(worker, query_text, sample_ids, corpus, gold_in_sample)

    return results, len(gold_in_sample)


def split_round_robin(items, n):
    return [items[i::n] for i in range(n)]


def run_worker_queries(worker, queries_chunk):
    # Runs on its own thread; torch.cuda.set_device is per-thread, so this pins every CUDA op
    # this thread issues (including ones without an explicit device= arg) to worker.device.
    if worker.device.startswith("cuda"):
        torch.cuda.set_device(worker.device)

    per_strategy = {name: [] for name in ALL_NAMES}
    logs = []
    for query_obj in queries_chunk:
        q_start = time.perf_counter()
        results, gold_in_sample = run_query_all(worker, query_obj)
        for name in ALL_NAMES:
            per_strategy[name].append(results[name])
        qid = query_obj["query_id"]
        logs.append(f"[{worker.device}] Query {qid} done in {time.perf_counter() - q_start:.2f}s (gold_in_sample={gold_in_sample})")
    return per_strategy, logs


def aggregate(results_list):
    return {
        "calls": sum(r["calls"] for r in results_list) / len(results_list),
        "latency": sum(r["latency"] for r in results_list) / len(results_list),
        "tflops": sum(r["tflops"] for r in results_list) / len(results_list),
        "m_eff": sum(r["m_eff"] for r in results_list) / len(results_list),
        "gold_missed": sum(r["gold_missed"] for r in results_list) / len(results_list),
    }


def print_table(rows):
    # GoldMissed% = share of queries where the gold doc was filtered out -- valid as a straight
    # percentage here because every query in this file has exactly one gold doc, guaranteed
    # present (injected), so the raw gold_missed average is already a 0..1 fraction per query.
    headers = ["Strategy", "Calls", "Latency(s)", "Lat/Call(s)", "TFLOPs", "M_eff (docs to next level)", "GoldMissed%"]
    formatted_rows = []
    for label, m in rows:
        lat_per_call = m["latency"] / m["calls"] if m["calls"] else 0.0
        gold_missed_pct = 100.0 * m["gold_missed"]
        formatted_rows.append([
            label, f"{m['calls']:.1f}", f"{m['latency']:.2f}", f"{lat_per_call:.2f}",
            f"{m['tflops']:.2f}", f"{m['m_eff']:.1f}", f"{gold_missed_pct:.1f}%",
        ])
    # Column widths are derived from the actual header/value lengths (+2 padding) instead of a
    # fixed constant, so a long header like the one above can never overrun into the next column.
    widths = [max(len(headers[i]), max((len(r[i]) for r in formatted_rows), default=0)) + 2 for i in range(len(headers))]
    print("".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in formatted_rows:
        print("".join(v.ljust(w) for v, w in zip(r, widths)))


def main():
    all_queries = load_data(DATA_PATH)
    queries_to_run = all_queries[:N_QUERIES]
    print(f"Loaded {len(all_queries)} queries from {DATA_PATH}, running {len(queries_to_run)} across {len(DEVICES)} device(s)")

    for model_cfg in MODELS:
        model_name = model_cfg["id"]
        safe_model_name = model_name.replace("/", "_")

        # Load sequentially (avoids two threads racing on the same HF cache download), one full
        # copy of the model per device.
        workers = [Worker(d) for d in DEVICES]
        for w in workers:
            w.load(model_cfg)

        print(f"\n{'#'*70}\n MODEL: {model_name}  |  DATASET: congress top-300 gold-injected\n{'#'*70}")

        chunks = split_round_robin(queries_to_run, len(workers))
        with ThreadPoolExecutor(max_workers=len(workers)) as ex:
            futures = [ex.submit(run_worker_queries, w, chunk) for w, chunk in zip(workers, chunks)]
            per_worker_results = [f.result() for f in futures]

        per_strategy = {name: [] for name in ALL_NAMES}
        for worker_per_strategy, logs in per_worker_results:
            for line in logs:
                print(line)
            for name in ALL_NAMES:
                per_strategy[name].extend(worker_per_strategy[name])

        agg = {name: aggregate(per_strategy[name]) for name in ALL_NAMES}

        print(f"\n{'='*70}\n FULL STRATEGY COMPARISON — {model_name} — congress candidates — avg over {len(queries_to_run)} queries\n{'='*70}")
        print_table([(name, agg[name]) for name in ALL_NAMES])

        results_path = f"congress_candidates_comparison_{safe_model_name}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
        print(f"\nSaved results to {results_path}")

        for w in workers:
            w.unload()

    print(f"\n{'#'*70}\n ALL MODELS DONE\n{'#'*70}")


if __name__ == "__main__":
    main()

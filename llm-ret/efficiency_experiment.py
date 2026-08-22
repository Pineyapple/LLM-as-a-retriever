from IPython.display import clear_output
#!pip install transformers datasets huggingface_hub torch accelerate bitsandbytes matplotlib
import heapq
import json
import math
import random
import time
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
clear_output()

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
DATASET_ID = "dianetc/OBLIQ-Bench"
CONFIGS = ["math", "writing", "twitter", "wildchat", "congress"]

LONGEST_PER_TASK = 50          # guaranteed long-tail coverage: the N longest docs per task, by raw char length
RANDOM_PER_TASK = 200          # broad random sample per task, for short/medium coverage
N_DOCS = 50                    # final number of documents actually scored, log-spaced shortest -> longest
TARGET_CHUNK_TOKENS = 100      # GridProbe sub-chunk size target (chunk COUNT grows with doc length, not chunk size)
FIXED_QUERY = "Find content that is relevant to recent developments and discussions."
SEED = 67
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, quantization_config=quant_config).to(DEVICE)

N_LAYERS = model.config.num_hidden_layers
D_MODEL = model.config.hidden_size
N_ACTIVE_PARAMS = 3e9  # approximate -- from the model's "A3B" (~3B active params) naming, not the full 35B


def estimate_tflops(context_length):
    # FLOPs ~= 2 * N_active * L (weight matmuls) + 2 * n_layers * d_model * L^2 (attention QK^T + AV)
    flops = 2 * N_ACTIVE_PARAMS * context_length + 2 * N_LAYERS * D_MODEL * context_length**2
    return flops / 1e12


def collect_candidates():
    candidates = []
    for config in CONFIGS:
        print(f"Sampling candidate docs from {config}...")
        texts = load_dataset(DATASET_ID, config)["corpus"]["text"]

        # guaranteed long-tail coverage: the actual longest docs, by raw char length (fast, no tokenizing needed)
        longest_idx = heapq.nlargest(LONGEST_PER_TASK, range(len(texts)), key=lambda i: len(texts[i]))
        # broad random sample for short/medium coverage
        random_idx = random.Random(SEED).sample(range(len(texts)), min(RANDOM_PER_TASK, len(texts)))

        for i in set(longest_idx) | set(random_idx):
            length = len(tokenizer.encode(texts[i], add_special_tokens=False))
            if length > 0:
                candidates.append((length, texts[i]))
    return candidates


def select_log_spaced(candidates, n_docs=N_DOCS):
    candidates.sort(key=lambda x: x[0])
    lo, hi = candidates[0][0], candidates[-1][0]
    targets = [lo * (hi / lo) ** (i / (n_docs - 1)) for i in range(n_docs)]

    selected, used = [], set()
    for target in targets:
        best = min((i for i in range(len(candidates)) if i not in used), key=lambda i: abs(candidates[i][0] - target))
        used.add(best)
        selected.append(candidates[best])
    selected.sort(key=lambda x: x[0])
    return selected


def split_document(text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    total = len(token_ids)
    k = max(2, math.ceil(math.sqrt(total / TARGET_CHUNK_TOKENS)))
    k = min(k, max(1, math.isqrt(total)))  # never ask for more chunks than tokens available
    n_chunks = k * k
    base, remainder = divmod(total, n_chunks)
    chunks, start = [], 0
    for i in range(n_chunks):
        size = base + (1 if i < remainder else 0)
        chunks.append(tokenizer.decode(token_ids[start : start + size], skip_special_tokens=True))
        start += size
    return k, chunks


def run_forward_pass(query_text, chunks_text):
    prompt = f"""Judge whether any chunk below is relevant to the query. Relevance may be implicit — a tone, structure, or indirect similarity — not just keyword overlap.

Query: {query_text}

Chunks:
{chunks_text}

Relevant? Answer Yes or No."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    context_length = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        model(**inputs)
    latency = time.perf_counter() - t0
    return latency, context_length


def score_naive(query_text, doc_text):
    return run_forward_pass(query_text, f"chunk 1: {doc_text}")


def score_gridprobe(query_text, doc_text):
    k, chunks = split_document(doc_text)
    rows = [chunks[r * k : (r + 1) * k] for r in range(k)]
    cols = [chunks[c::k] for c in range(k)]

    total_latency, total_tflops = 0.0, 0.0
    for group in rows + cols:
        chunks_text = "\n\n".join(f"chunk {i+1}: {c}" for i, c in enumerate(group))
        latency, context_length = run_forward_pass(query_text, chunks_text)
        total_latency += latency
        total_tflops += estimate_tflops(context_length)
    return k, total_latency, total_tflops


def find_crossover(results, naive_key, grid_key):
    for i in range(1, len(results)):
        prev_diff = results[i - 1][naive_key] - results[i - 1][grid_key]
        curr_diff = results[i][naive_key] - results[i][grid_key]
        if prev_diff == 0 or (prev_diff * curr_diff < 0):
            return results[i - 1]["tokens"], results[i]["tokens"]
    return None


def plot_results(results):
    valid = [r for r in results if r["naive_latency"] is not None]
    tokens = [r["tokens"] for r in valid]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(tokens, [r["naive_latency"] for r in valid], marker="o", label="Naive (plain LLM call)")
    axes[0].plot(tokens, [r["gridprobe_latency"] for r in valid], marker="o", label="GridProbe")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Document length (tokens)")
    axes[0].set_ylabel("Latency (s)")
    axes[0].set_title("Latency vs Document Length")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(tokens, [r["naive_tflops"] for r in valid], marker="o", label="Naive (plain LLM call)")
    axes[1].plot(tokens, [r["gridprobe_tflops"] for r in valid], marker="o", label="GridProbe")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Document length (tokens)")
    axes[1].set_ylabel("Estimated TFLOPs")
    axes[1].set_title("TFLOPs vs Document Length")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("efficiency_plot.png", dpi=150)
    plt.show()
    print("Saved plot to efficiency_plot.png")


def main():
    print("Collecting candidate documents across all tasks...")
    candidates = collect_candidates()
    docs = select_log_spaced(candidates)
    print(f"Selected {len(docs)} documents spanning {docs[0][0]} to {docs[-1][0]} tokens\n")

    results = []
    for length, text in docs:
        try:
            naive_latency, naive_ctx = score_naive(FIXED_QUERY, text)
            naive_tflops = estimate_tflops(naive_ctx)
        except Exception as e:
            print(f"tokens={length:<7} naive FAILED ({e})")
            naive_latency, naive_tflops = None, None

        k, grid_latency, grid_tflops = score_gridprobe(FIXED_QUERY, text)

        results.append({
            "tokens": length,
            "naive_latency": naive_latency,
            "naive_tflops": naive_tflops,
            "gridprobe_k": k,
            "gridprobe_latency": grid_latency,
            "gridprobe_tflops": grid_tflops,
        })

        naive_str = f"{naive_latency:.2f}s ({naive_tflops:.2f} TF)" if naive_latency is not None else "FAILED"
        print(
            f"tokens={length:<7} naive={naive_str:<22} "
            f"gridprobe(K={k})={grid_latency:.2f}s ({grid_tflops:.2f} TF)"
        )

    valid = [r for r in results if r["naive_latency"] is not None]
    latency_cross = find_crossover(valid, "naive_latency", "gridprobe_latency")
    tflops_cross = find_crossover(valid, "naive_tflops", "gridprobe_tflops")

    print("\n--- Crossover ---")
    print(f"Latency crossover between {latency_cross[0]} and {latency_cross[1]} tokens" if latency_cross else "No latency crossover found in sampled range")
    print(f"TFLOPs crossover between {tflops_cross[0]} and {tflops_cross[1]} tokens" if tflops_cross else "No TFLOPs crossover found in sampled range")

    results_path = "efficiency_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    plot_results(results)


if __name__ == "__main__":
    main()

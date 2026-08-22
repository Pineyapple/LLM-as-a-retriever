 

MODELS = [
    # Qwen3.5-4B is a HYBRID architecture, not a uniform transformer stack: per its model card,
    # it's 8 blocks of (3 Gated-DeltaNet linear-attention layers + 1 regular gated-attention
    # layer), 32 layers / 2560 hidden total, "4B parameters" (active-vs-total isn't broken out
    # separately on the card despite the "sparse MoE" description, so 4e9 is used as stated, same
    # approximation treatment as gemma-4-E2B-it's MatFormer figure earlier in this project).
    # Two honest caveats this creates: (1) estimate_tflops's quadratic term assumes every layer
    # pays full self-attention cost -- for this model 24 of 32 layers are linear-attention
    # (DeltaNet), so that term is a real overestimate of attention cost, not a calibrated one; (2)
    # the logit-lens early-exit read is best-validated on uniform self-attention stacks -- how
    # well it behaves on DeltaNet layers is genuinely untested here, not just unlisted.
    # enable_thinking=False (already set in prompt_inputs) is required -- the card says this model
    # thinks by default.
    {"id": "Qwen/Qwen3.5-4B", "quantize": True, "n_active_params": 4e9, "score_temperature": 1.0},
]

# Full congress sample: all 20 queries, each with its full ~300-doc candidate pool -- no
# sub-sampling. Point this at the file's location as a Kaggle input, e.g.
# "/kaggle/input/<your-dataset-slug>/candidates_congress_top300_goldinjected.jsonl"
DATA_PATH = "candidates_congress_top300_goldinjected.jsonl"
N_QUERIES = 20   # all queries in the file

EVAL_KS = [5, 10]
TARGET_CHUNK_TOKENS = 100
ROW_AGGREGATION = "peak"

# Patience-based early exit: after each layer, logit-lens the current hidden state into a Yes/No
# read; once PATIENCE consecutive layers agree on the same answer, stop -- for real this time (see
# Worker.score_relevance_early_exit): a forward hook raises out of the model's own layer loop the
# moment patience fires, so the remaining layers never execute.
PATIENCE = 2

# One model copy per visible GPU (e.g. ["cuda:0", "cuda:1"] on Kaggle's T4 x2).
if torch.cuda.is_available():
    DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
else:
    DEVICES = ["cpu"]
print(f"Detected devices: {DEVICES}")


def get_model_dims(m):
    config = m.config
    text_config = getattr(config, "text_config", config)
    return text_config.num_hidden_layers, text_config.hidden_size


def find_final_norm(model):
    # The pre-lm_head RMSNorm module lives at a different attribute path depending on
    # architecture. Try the common paths in order; fall back to no norm (raw hidden state
    # straight into lm_head) rather than guessing wrong and crashing -- the patience signal
    # (which layers AGREE with each other) still works without it, just noisier.
    backbone = getattr(model, "model", model)
    for path in ("norm", "language_model.norm", "final_layernorm", "ln_f"):
        obj = backbone
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    return None


def find_decoder_layers(model):
    # Same idea as find_final_norm, for the actual ModuleList of decoder layers we need to hook.
    # Tried in order across the paths seen so far in this project (Gemma3/Qwen-style flat
    # model.model.layers, Gemma4's nested model.model.language_model.layers).
    backbone = getattr(model, "model", model)
    for path in ("layers", "language_model.layers"):
        obj = backbone
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("Could not find the decoder layers ModuleList on this model -- check its architecture.")


class EarlyExitSignal(Exception):
    # Raised from inside a forward hook to unwind out of the model's own internal layer loop the
    # moment the patience condition fires -- the layers after the exit point are never called.
    def __init__(self, yes_prob, exit_layer):
        self.yes_prob = yes_prob
        self.exit_layer = exit_layer


def load_data(path):
    # Each line is a full query with its own pre-built ~300-doc candidate pool -- gold docs
    # stage-1 missed were injected (rank=None, injected=True) so is_gold is the only reliable
    # gold signal.
    queries_data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            queries_data.append(json.loads(line))
    return queries_data


def build_query_pool(query_obj):
    # Full pool, no sub-sampling -- every candidate in this query's ~300-doc pool is used.
    candidates = query_obj["candidates"]
    corpus = {c["doc_id"]: c["text"] for c in candidates}
    gold_ids = {c["doc_id"] for c in candidates if c["is_gold"]}
    sample_ids = sorted(corpus)
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
        self.final_norm = None
        self.decoder_layers = None

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
        self.final_norm = find_final_norm(self.model)
        self.decoder_layers = find_decoder_layers(self.model)
        print(
            f"[{self.device}] final_norm found: {self.final_norm is not None}, "
            f"n_layers={self.n_layers}, decoder_layers found: {len(self.decoder_layers)}"
        )

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

    def score_relevance_early_exit(self, query_text, chunks_text):
        # Real early exit: registers a forward hook on every decoder layer. Each hook logit-lenses
        # that layer's output (final norm + lm_head, restricted softmax over Yes/No token ids --
        # the exact same read score_relevance uses, just applied to an intermediate hidden state)
        # and tracks patience. The moment PATIENCE consecutive layers agree, the hook RAISES --
        # that exception unwinds out of the model's own internal layer loop, so the remaining
        # layers are never invoked. If patience never fires, the model's forward() completes
        # normally and we read its real final logits, identical to the non-early-exit path.
        if self.option_ids is None:
            self.option_ids = self.yesno_token_ids()

        prompt = build_relevance_prompt(query_text, chunks_text)
        inputs = self.prompt_inputs(prompt)
        context_length = inputs["input_ids"].shape[1]
        ids = list(self.option_ids)
        output_embeddings = self.model.get_output_embeddings()
        n_layers_total = len(self.decoder_layers)

        state = {"last_answer": None, "streak": 0}

        def make_hook(depth):
            def hook_fn(module, inp, output):
                hidden = output[0] if isinstance(output, tuple) else output
                h = hidden[:, -1:, :]
                if self.final_norm is not None:
                    h = self.final_norm(h)
                logits = output_embeddings(h)[0, -1, :]
                probs = torch.softmax(logits[ids] / self.score_temperature, dim=0)
                yes_prob = float(sum(p for tid, p in zip(ids, probs) if self.option_ids[tid] == "Yes"))
                answer = yes_prob >= 0.5
                if answer == state["last_answer"]:
                    state["streak"] += 1
                else:
                    state["streak"] = 1
                    state["last_answer"] = answer
                if state["streak"] >= PATIENCE:
                    raise EarlyExitSignal(yes_prob, depth)
            return hook_fn

        handles = [layer.register_forward_hook(make_hook(i + 1)) for i, layer in enumerate(self.decoder_layers)]
        try:
            with torch.no_grad():
                try:
                    outputs = self.model(**inputs)
                except EarlyExitSignal as sig:
                    full_tflops = self.estimate_tflops(context_length)
                    estimated_tflops = full_tflops * (sig.exit_layer / n_layers_total)
                    return sig.yes_prob, estimated_tflops, sig.exit_layer, n_layers_total

            # Patience never fired -- fell through to the model's own real final output.
            next_token_logits = outputs.logits[0, -1, :]
            probs = torch.softmax(next_token_logits[ids] / self.score_temperature, dim=0)
            yes_prob = float(sum(p for tid, p in zip(ids, probs) if self.option_ids[tid] == "Yes"))
            full_tflops = self.estimate_tflops(context_length)
            return yes_prob, full_tflops, n_layers_total, n_layers_total
        finally:
            for handle in handles:
                handle.remove()

    def split_document(self, text):
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


# ---- naive / per_chunks / gridprobe -- early-exit ONLY, no normal-path comparison this run ----

def run_naive(worker, query_text, sample_ids, corpus):
    t0 = time.perf_counter()
    scores = {}
    total_tflops = 0.0
    exit_fracs = []
    for doc_id in sample_ids:
        score, tflops, exit_layer, n_layers = worker.score_relevance_early_exit(query_text, format_chunks([corpus[doc_id]]))
        exit_fracs.append(exit_layer / n_layers)
        scores[doc_id] = score
        total_tflops += tflops
    latency = time.perf_counter() - t0
    avg_exit_frac = sum(exit_fracs) / len(exit_fracs) if exit_fracs else 1.0
    return {"scores": scores, "calls": len(sample_ids), "latency": latency, "tflops": total_tflops, "avg_exit_frac": avg_exit_frac}


def run_per_chunks_and_gridprobe(worker, query_text, sample_ids, corpus):
    row_t0 = time.perf_counter()
    doc_grids = {}
    row_scores_by_doc = {}
    row_calls = 0
    row_tflops = 0.0
    exit_fracs = []
    for doc_id in sample_ids:
        k, chunks = worker.split_document(corpus[doc_id])
        rows, cols = build_rows_cols(k, chunks)
        row_scores = []
        for row in rows:
            score, tflops, exit_layer, n_layers = worker.score_relevance_early_exit(query_text, format_chunks(row))
            exit_fracs.append(exit_layer / n_layers)
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
            score, tflops, exit_layer, n_layers = worker.score_relevance_early_exit(query_text, format_chunks(col))
            exit_fracs.append(exit_layer / n_layers)
            col_scores.append(score)
            col_tflops += tflops
            col_calls += 1
        grid_scores[doc_id] = max(grid["row_scores"][r] * col_scores[c] for r in range(k) for c in range(k))
    col_latency = time.perf_counter() - col_t0

    avg_exit_frac = sum(exit_fracs) / len(exit_fracs) if exit_fracs else 1.0
    per_chunks = {
        "scores": row_scores_by_doc, "calls": row_calls, "latency": row_latency, "tflops": row_tflops,
        "avg_exit_frac": avg_exit_frac,
    }
    gridprobe = {
        "scores": grid_scores,
        "calls": row_calls + col_calls, "latency": row_latency + col_latency, "tflops": row_tflops + col_tflops,
        "avg_exit_frac": avg_exit_frac,
    }
    return per_chunks, gridprobe


ALL_METHODS = ["naive_early_exit", "per_chunks_early_exit", "gridprobe_early_exit"]


def run_query(worker, query_obj):
    sample_ids, corpus, gold_in_sample = build_query_pool(query_obj)
    query_text = query_obj["query"]

    naive_ee = run_naive(worker, query_text, sample_ids, corpus)
    per_chunks_ee, gridprobe_ee = run_per_chunks_and_gridprobe(worker, query_text, sample_ids, corpus)

    results = {}
    for name, r in (
        ("naive_early_exit", naive_ee),
        ("per_chunks_early_exit", per_chunks_ee),
        ("gridprobe_early_exit", gridprobe_ee),
    ):
        metrics, auc = evaluate_doc_scores(r["scores"], gold_in_sample)
        results[name] = {
            "calls": r["calls"], "latency": r["latency"], "tflops": r["tflops"],
            "avg_exit_frac": r["avg_exit_frac"], "metrics": metrics, "auc": auc,
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
        results, gold_in_sample = run_query(worker, query_obj)
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
        "avg_exit_frac": sum(r["avg_exit_frac"] for r in results_list) / len(results_list),
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
    # Unlike the previous version of this script, Latency here IS real -- the hook genuinely
    # aborts the model's forward pass once patience fires, so fewer layers actually execute on
    # the GPU. TFLOPs is still (full_cost * exit_layer/n_layers), which is now consistent with
    # what was actually computed rather than just an estimate layered on top of a full run.
    # AvgExit% is the average fraction of layers actually used before exiting.
    headers = ["Method", "Calls", "Latency(s)", "Lat/Call(s)", "TFLOPs", "AvgExit%"]
    for k in EVAL_KS:
        headers += [f"Recall@{k}", f"NDCG@{k}"]
    headers.append("AUC")
    rows = []
    for name in ALL_METHODS:
        m = agg[name]
        lat_per_call = m["latency"] / m["calls"] if m["calls"] else 0.0
        row = [
            name, f"{m['calls']:.1f}", f"{m['latency']:.2f}", f"{lat_per_call:.2f}",
            f"{m['tflops']:.2f}", f"{100.0 * m['avg_exit_frac']:.1f}%",
        ]
        for k in EVAL_KS:
            row += [f"{m['recall'][k]:.3f}", f"{m['ndcg'][k]:.3f}"]
        row.append(f"{m['auc']:.3f}")
        rows.append(row)
    print_grid(headers, rows)


def main():
    for model_cfg in MODELS:
        model_name = model_cfg["id"]
        safe_model_name = model_name.replace("/", "_")

        workers = [Worker(d) for d in DEVICES]
        for w in workers:
            w.load(model_cfg)

        print(f"\n{'#'*70}\n MODEL: {model_name}  |  DATASET: congress top-300 gold-injected (sampled)\n{'#'*70}")
        all_queries = load_data(DATA_PATH)
        queries_to_run = all_queries[:N_QUERIES]
        print(f"Loaded {len(all_queries)} queries from {DATA_PATH}, running {len(queries_to_run)} across {len(workers)} device(s)")

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

        print(f"\n{'='*70}\n EARLY-EXIT ONLY (patience={PATIENCE}) — {model_name} — congress candidates — avg over {len(queries_to_run)} queries\n{'='*70}")
        print_results_table(agg)

        results_path = f"early_exit_only_congress_{safe_model_name}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
        print(f"\nSaved results to {results_path}")

        for w in workers:
            w.unload()

    print(f"\n{'#'*70}\n ALL DONE\n{'#'*70}")


if __name__ == "__main__":
    main()

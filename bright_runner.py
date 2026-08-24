#!/usr/bin/env python3
"""BRIGHT long_documents: FullRead vs ChunkMax vs GridProbe vs FirstP on identical pools.

  python bright_runner.py --domain biology --queries 10 --pool-size 100

Runs every reading mode in ONE process so the pools, the model, and the machine are held equal;
the only variable is how the document reaches the judge.

The OOM wall is hardware-dependent, so it is a parameter here, not a constant: --max-model-len
sets the context budget and FullRead is recorded as FAILED (scored last, counted in Success%)
whenever a document does not fit. Report the wall together with the GPU it was measured on.
Set --max-model-len 8192 to emulate a 16GB-class card, 32768 for an 80GB card.

Outputs runs_bright/<domain>_<model>_<len>.pkl plus a per-domain table.
"""
import argparse, json, math, os, pickle, random, time
import numpy as np
from tqdm.auto import tqdm

p = argparse.ArgumentParser()
p.add_argument("--domain", required=True,
               choices=["biology", "earth_science", "economics", "pony", "psychology",
                        "robotics", "stackoverflow", "sustainable_living", "ALL"])
p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
p.add_argument("--cache-dir", default="/ibex/user/hamidme/cache_dir")
p.add_argument("--queries", type=int, default=10, help="queries per domain (0 = all)")
p.add_argument("--pool-size", type=int, default=100,
               help="0 = FULL CORPUS retrieval (every document scored, nothing injected). "
                    ">0 builds a pool of this size with the GOLDS INJECTED, which is reranking "
                    "with recall 1.0 by construction: comparable to a teammate's pooled run, NOT "
                    "to published BRIGHT numbers, and it cannot support a retrieval claim.")
p.add_argument("--modes", default="naive,rows,grid,firstp",
               help="naive=whole document | perchunk=one probe per chunk (true MaxP) | "
                    "rows=sqrt(N) local spans | grid=local+strided | firstp=first chunk only")
p.add_argument("--chunk-size", type=int, default=100, help="tokens per chunk")
p.add_argument("--max-model-len", type=int, default=8192,
               help="context budget = the OOM wall being emulated. FullRead fails above it.")
p.add_argument("--max-num-seqs", type=int, default=256)
p.add_argument("--gpu-mem", type=float, default=0.90)
p.add_argument("--gen-batch", type=int, default=512)
p.add_argument("--server", default="",
               help="e.g. http://localhost:8000 . Score through a running `vllm serve` instead "
                    "of loading the model in-process, so several domains can share one GPU and "
                    "one client's CPU work overlaps another's GPU work.")
p.add_argument("--outdir", default="runs_bright")
p.add_argument("--overwrite", action="store_true",
               help="ignore existing checkpoints and rescore from scratch")
args = p.parse_args()
os.makedirs(args.outdir, exist_ok=True)
os.environ.setdefault("HF_HOME", args.cache_dir)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(args.cache_dir, "datasets"))
SEED = 42

# Qwen3-4B geometry for the analytic FLOPs model (non-embedding params, layers, hidden).
P_NE, N_LAYERS, D_MODEL = 3.633e9, 36, 2560

def flops(tokens):
    """2*P*T linear + 2*L*d*T^2 attention, per call."""
    return 2 * P_NE * tokens + 2 * N_LAYERS * D_MODEL * tokens ** 2

from datasets import load_dataset
from transformers import AutoTokenizer
if not args.server:
    from vllm import LLM, SamplingParams

DOMAINS = ([args.domain] if args.domain != "ALL" else
           ["biology", "earth_science", "economics", "pony", "psychology",
            "robotics", "stackoverflow", "sustainable_living"])

POS = ("Query: {q}\nDocument: {d}\nIs this document relevant to the query? Answer Yes or No:")

tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)

def build_prompt(q, d):
    return tok.apply_chat_template([{"role": "user", "content": POS.format(q=q, d=d)}],
                                   tokenize=False, add_generation_prompt=True)

_PROBE = build_prompt("telescoping sums", "1/k - 1/(k+1) cancels.")

def _logodds(topmap, yes_key, no_key):
    """Four cases, no default clamp: a fixed default collapses everything below it into one
    exact tie block. topmap maps key -> logprob; keys are ids offline, strings over HTTP."""
    if not topmap:
        return -1e9
    fl = min(topmap.values()) - 2.0
    hy, hn = yes_key in topmap, no_key in topmap
    if hy and hn:  return float(topmap[yes_key] - topmap[no_key])
    if hy:         return float(topmap[yes_key] - fl)
    if hn:         return float(fl - topmap[no_key])
    return float(fl - 50.0)

if args.server:
    # ---- HTTP path: share one `vllm serve` across several domain processes ----
    import requests
    URL = args.server.rstrip("/") + "/v1/completions"
    def _post(batch):
        r = requests.post(URL, json={"model": args.model, "prompt": batch, "max_tokens": 1,
                                     "logprobs": 20, "temperature": 1.0, "top_p": 1.0},
                          timeout=1800)
        r.raise_for_status()
        ch = sorted(r.json()["choices"], key=lambda c: c["index"])
        # the completions API returns token TEXT, not ids, so keys here are strings
        return [(c.get("logprobs") or {}).get("top_logprobs", [{}])[0] or {} for c in ch]
    _top = _post([_PROBE])[0]
    YES = NO = None
    _best = -1e9
    for yf, nf in [("Yes", "No"), (" Yes", " No"), ("YES", "NO"), ("yes", "no")]:
        if yf in _top and _top[yf] > _best:
            _best, YES, NO = _top[yf], yf, nf
    YES, NO = YES or "Yes", NO or "No"
    print(f"answer tokens (server): yes={YES!r} logprob {_best:.2f}  no={NO!r}", flush=True)
    def score_batch(prompts):
        out = []
        for i in tqdm(range(0, len(prompts), args.gen_batch), desc="  gen", leave=False):
            for m in _post(prompts[i:i + args.gen_batch]):
                out.append(_logodds(m, YES, NO))
        return out
else:
    # ---- in-process path ----
    llm = LLM(model=args.model, dtype="bfloat16", enable_prefix_caching=True,
              gpu_memory_utilization=args.gpu_mem, download_dir=args.cache_dir,
              max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs)
    SP = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
                        max_tokens=1, logprobs=20)
    def single(c):
        e = tok.encode(c, add_special_tokens=False)
        return e[0] if len(e) == 1 else None
    _top = llm.generate([_PROBE], SP, use_tqdm=False)[0].outputs[0].logprobs[0]
    YES = NO = None
    _best = -1e9
    for yf, nf in [("Yes", "No"), (" Yes", " No"), ("YES", "NO"), ("yes", "no")]:
        t = single(yf)
        if t is not None and t in _top and _top[t].logprob > _best:
            _best, YES, NO = _top[t].logprob, t, single(nf)
    print(f"answer tokens: yes={YES} {tok.decode([YES])!r}  no={NO} {tok.decode([NO])!r}")
    def score_batch(prompts):
        out = []
        for i in tqdm(range(0, len(prompts), args.gen_batch), desc="  gen", leave=False):
            res = llm.generate(prompts[i:i + args.gen_batch], SP, use_tqdm=False)
            for r in res:
                lp = r.outputs[0].logprobs[0] if r.outputs[0].logprobs else {}
                out.append(_logodds({k: v.logprob for k, v in lp.items()}, YES, NO))
        return out

_CHUNK_CACHE = {}
def chunks_of(text, key=None):
    """Chunk a document once. BRIGHT pools overlap heavily across queries and tok.decode is
    called once per chunk, so re-chunking every query was the dominant CPU cost (a 100k-token
    document is ~1000 decode calls). Cache by document id."""
    if key is not None and key in _CHUNK_CACHE:
        return _CHUNK_CACHE[key]
    ids = tok.encode(text, add_special_tokens=False)
    # batch_decode is one call instead of len(ids)/chunk_size calls
    pieces = [ids[i:i + args.chunk_size] for i in range(0, len(ids), args.chunk_size)]
    out = [t for t in (x.strip() for x in tok.batch_decode(pieces, skip_special_tokens=True)) if t]
    if key is not None:
        _CHUNK_CACHE[key] = out
    return out

def lines_for(text, mode, key=None):
    """(list of probe texts, kinds) for one document under one reading mode."""
    if mode == "naive":
        return [text], ["r"]
    ch = chunks_of(text, key) or [text[:200] or "empty"]
    if mode == "firstp":
        return [ch[0]], ["r"]
    if mode == "perchunk":
        # TRUE MaxP (Dai & Callan): one probe per chunk, max over chunks. N probes of
        # chunk_size each. NOTE this is NOT the same as `rows`, which sends sqrt(N) probes
        # of sqrt(N) consecutive chunks joined. Same total tokens, different granularity:
        # perchunk sees each chunk in isolation, rows sees local context around it.
        return ch, ["r"] * len(ch)
    K = math.ceil(math.sqrt(len(ch)))
    padded = ch + [""] * (K * K - len(ch))
    grid = [padded[i * K:(i + 1) * K] for i in range(K)]
    rows = [r for i in range(K) if (r := " ".join(c for c in grid[i] if c)).strip()]
    if mode == "rows":
        return rows, ["r"] * len(rows)
    cols = [c for j in range(K)
            if (c := " ".join(grid[i][j] for i in range(K) if grid[i][j])).strip()]
    return rows + cols, ["r"] * len(rows) + ["c"] * len(cols)

def ndcg(gold, ranked, k):
    dcg = sum((1.0 if d in gold else 0.0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0

def recall(gold, ranked, k):
    return len(gold & set(ranked[:k])) / len(gold) if gold else 0.0

def auc_of(scores, gold, ids):
    g = np.array([scores[d] for d in ids if d in gold])
    n = np.array([scores[d] for d in ids if d not in gold])
    if not len(g) or not len(n):
        return float("nan")
    allv = np.concatenate([g, n]); order = np.argsort(allv)
    ranks = np.empty(len(allv)); ranks[order] = np.arange(len(allv))
    return float((ranks[:len(g)].sum() - len(g) * (len(g) - 1) / 2) / (len(g) * len(n)))

MODES = args.modes.split(",")
PROMPT_OVERHEAD = 64   # chat template + question, measured empirically below
_ov = len(tok.encode(build_prompt("x", ""), add_special_tokens=False))
PROMPT_OVERHEAD = max(PROMPT_OVERHEAD, _ov)

all_rows = []
for domain in DOMAINS:
    docs_ds = load_dataset("xlangai/BRIGHT", "long_documents",
                           cache_dir=os.environ["HF_DATASETS_CACHE"])[domain]
    ex_ds = load_dataset("xlangai/BRIGHT", "examples",
                         cache_dir=os.environ["HF_DATASETS_CACHE"])[domain]
    corpus = {x["id"]: x["content"] for x in docs_ds}
    exs = list(ex_ds)
    if args.queries:
        exs = exs[:args.queries]
    all_ids = sorted(corpus)
    rng = random.Random(SEED)
    print(f"\n=== {domain}: {len(corpus):,} docs, {len(exs)} queries, pool {args.pool_size} ===")

    # identical candidate sets for every mode, so the only variable is the reading strategy
    pools = {}
    for e in exs:
        gold = [g for g in e.get("gold_ids_long", e.get("gold_ids", [])) if g in corpus]
        if args.pool_size <= 0:
            pools[e["id"]] = (all_ids, set(gold))      # FULL CORPUS, nothing injected
        else:
            pool = set(gold)                           # golds injected, then random fill
            while len(pool) < args.pool_size:
                pool.add(all_ids[rng.randrange(len(all_ids))])
            pools[e["id"]] = (sorted(pool), set(gold))
    if args.pool_size <= 0:
        print(f"FULL-CORPUS retrieval: {len(all_ids):,} docs/query, no gold injection. "
              f"Comparable to published BRIGHT retrieval numbers.", flush=True)
    else:
        print(f"POOLED reranking: {args.pool_size} docs/query WITH GOLDS INJECTED "
              f"(recall 1.0 by construction). This measures the judge, not retrieval, and is "
              f"NOT comparable to published BRIGHT rows.", flush=True)

    need = sorted({d for ids, _ in pools.values() for d in ids})
    print(f"tokenizing {len(need)} unique pool documents "
          f"({sum(len(corpus[d]) for d in need)/1e6:.1f}M chars)", flush=True)
    # one batched call, not one per document
    enc = tok([corpus[d] for d in need], add_special_tokens=False)["input_ids"]
    doclen = {d: len(e) for d, e in zip(need, enc)}
    dl = np.array(list(doclen.values()))
    print(f"  doc tokens: p50 {np.percentile(dl,50):.0f}  p90 {np.percentile(dl,90):.0f}  "
          f"max {dl.max():.0f}", flush=True)
    # realistic overhead = template + a typical BRIGHT query (these are long, so a fixed
    # constant badly underestimates it and would inflate the predicted fit rate)
    qov = int(np.median([len(tok.encode(build_prompt(e["query"], ""), add_special_tokens=False))
                         for e in exs]))
    print(f"  template+query overhead (median) {qov} tok | predicted FullRead fit in "
          f"{args.max_model_len}: {100*np.mean(dl + qov <= args.max_model_len):.1f}% "
          f"(exact per-prompt check happens at scoring time)", flush=True)

    for mode in MODES:
        # ---- resume: one checkpoint per mode, updated after every query ----
        mpath = os.path.join(args.outdir, f"{domain}_{mode}_{args.model.split(chr(47))[-1]}"
                                          f"_{args.max_model_len}_p{args.pool_size}.pkl")
        res, fails, calls, tflops, elapsed = {}, 0, 0, 0.0, 0.0
        if os.path.exists(mpath) and not args.overwrite:
            try:
                _prev = pickle.load(open(mpath, "rb"))
            except Exception:
                _prev = {}
            if _prev.get("complete"):
                print(f"skip {domain}/{mode}: already complete ({mpath})", flush=True)
                all_rows.append(_prev["metrics"])
                continue
            res = _prev.get("scores", {}) or {}
            fails, calls = _prev.get("fails", 0), _prev.get("calls", 0)
            tflops, elapsed = _prev.get("tflops", 0.0), _prev.get("elapsed", 0.0)
            if res:
                print(f"resuming {domain}/{mode} from {len(res)} completed queries", flush=True)

        def _ckpt(complete=False, metrics=None):
            tmp = mpath + ".tmp"
            pickle.dump({"scores": res, "fails": fails, "calls": calls, "tflops": tflops,
                         "elapsed": elapsed + (time.time() - t0),
                         "pools": {q: sorted(g) for q, (_, g) in pools.items()},
                         "doclen": doclen, "complete": complete, "metrics": metrics},
                        open(tmp, "wb"))
            os.replace(tmp, mpath)          # atomic: a crash mid-write cannot corrupt the file

        t0 = time.time()
        for qid, (ids, gold) in tqdm(pools.items(), desc=f"{domain}/{mode}"):
            if qid in res:
                continue                     # already scored in an earlier attempt
            q = next(e["query"] for e in exs if e["id"] == qid)
            sc = {}
            texts, owner, kinds, fails_q = [], [], [], 0
            for d in ids:
                t, k = lines_for(corpus[d], mode, key=d)
                texts.extend(t); owner.extend([d] * len(t)); kinds.extend(k)
            if texts:
                # Measure the ACTUAL prompt length rather than estimating it. The old estimate
                # used a fixed overhead that excluded the query, so a document could pass the
                # fit check and still overflow once a 300-token BRIGHT query was prepended --
                # which both crashed the run and biased Success%, the number this measures.
                prompts = [build_prompt(q, t) for t in texts]
                plen = [len(e) for e in tok(prompts, add_special_tokens=False)["input_ids"]]
                keep = [i for i, L in enumerate(plen) if L <= args.max_model_len]
                dropped = [i for i, L in enumerate(plen) if L > args.max_model_len]
                for i in dropped:
                    # naive: the document does not fit -> that IS the failure being measured.
                    # rows/grid: one over-long probe is skipped; the doc keeps its other probes.
                    if mode in ("naive", "firstp"):
                        sc[owner[i]] = -1e9
                        fails += 1; fails_q += 1
                print(f"  {qid[:14]}: {len(ids)} docs -> {len(keep)} probes sent, "
                      f"{len(dropped)} over {args.max_model_len} tok, "
                      f"~{sum(plen[i] for i in keep)//1000}k tokens", flush=True)
                prompts = [prompts[i] for i in keep]
                texts = [texts[i] for i in keep]
                owner = [owner[i] for i in keep]
                kinds = [kinds[i] for i in keep]
                vals = score_batch(prompts) if prompts else []
                calls += len(prompts)
                tflops += sum(flops(plen[i]) for i in keep)   # exact, plen already measured
                rs, cs = {}, {}
                for v, d, k in zip(vals, owner, kinds):
                    tgt = rs if k == "r" else cs
                    if d not in tgt or v > tgt[d]:
                        tgt[d] = v
                for d in set(owner):
                    sc[d] = rs.get(d, -1e9) + (cs.get(d, -1e9) if mode == "grid" else 0.0)
            res[qid] = sc
            _ckpt()                          # crash now loses at most this one query
        lat = elapsed + (time.time() - t0)
        n_docs = sum(len(ids) for ids, _ in pools.values())
        m = {"domain": domain, "mode": mode, "protocol": ("full-corpus" if args.pool_size <= 0 else f"pooled{args.pool_size}+goldinjected"),
             "Success%": 100 * (1 - fails / max(n_docs, 1)),
             "Calls": calls / max(len(pools), 1),
             "TFLOPs": tflops / 1e12 / max(len(pools), 1),
             "Latency(s)": lat / max(len(pools), 1)}
        for k in (5, 10):
            m[f"NDCG@{k}"] = float(np.mean([
                ndcg(g, sorted(res[q], key=res[q].get, reverse=True), k)
                for q, (_, g) in pools.items() if g]))
            m[f"Recall@{k}"] = float(np.mean([
                recall(g, sorted(res[q], key=res[q].get, reverse=True), k)
                for q, (_, g) in pools.items() if g]))
        m["AUC"] = float(np.nanmean([auc_of(res[q], g, ids)
                                     for q, (ids, g) in pools.items() if g]))
        all_rows.append(m)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
        _ckpt(complete=True, metrics=m)

import pandas as pd
df = pd.DataFrame(all_rows)
print("\n" + df.to_markdown(index=False))
out = os.path.join(args.outdir, f"table_{args.domain}_{args.model.split(chr(47))[-1]}_{args.max_model_len}_p{args.pool_size}.csv")
df.to_csv(out, index=False)
print(f"\nsaved {out}")
print(f"NOTE: Success% is relative to the {args.max_model_len}-token budget; the OOM wall is "
      f"hardware-dependent and must be reported with the GPU it was measured on.")

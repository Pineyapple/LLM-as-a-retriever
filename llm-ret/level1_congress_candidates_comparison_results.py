# Hardcoded results from level1_congress_candidates_comparison.py (avg over 20 queries,
# congress top-300 gold-injected hard-candidate pools). Add remaining models here as their
# runs finish -- Qwen/Qwen3.5-0.8B, google/gemma-3-1b-it.

RESULTS = {
    "google/gemma-3-270m-it": {
        "words_per_paragraph":     {"calls": 300.6, "latency": 36.98,  "tflops": 61.24,  "m_eff": 153.0, "gold_missed": 0.5},
        "sentences_per_paragraph": {"calls": 300.6, "latency": 35.65,  "tflops": 58.18,  "m_eff": 128.4, "gold_missed": 0.8},
        "begin_mid_end_sentences": {"calls": 300.6, "latency": 43.32,  "tflops": 73.04,  "m_eff": 62.1,  "gold_missed": 0.8},
        "decaying_words":          {"calls": 300.6, "latency": 37.41,  "tflops": 61.42,  "m_eff": 105.8, "gold_missed": 0.7},
        "sliding_window":          {"calls": 300.6, "latency": 38.64,  "tflops": 64.95,  "m_eff": 78.0,  "gold_missed": 0.7},
        "batches":                 {"calls": 30.6,  "latency": 14.69,  "tflops": 18.01,  "m_eff": 299.9, "gold_missed": 0.0},
    },
    "google/gemma-3-4b-it": {
        "words_per_paragraph":     {"calls": 300.6, "latency": 455.99, "tflops": 899.35,  "m_eff": 32.5,  "gold_missed": 0.9},
        "sentences_per_paragraph": {"calls": 300.6, "latency": 436.93, "tflops": 854.86,  "m_eff": 92.8,  "gold_missed": 0.8},
        "begin_mid_end_sentences": {"calls": 300.6, "latency": 542.85, "tflops": 1071.34, "m_eff": 47.3,  "gold_missed": 0.8},
        "decaying_words":          {"calls": 300.6, "latency": 460.62, "tflops": 902.27,  "m_eff": 42.2,  "gold_missed": 0.9},
        "sliding_window":          {"calls": 300.6, "latency": 479.67, "tflops": 954.19,  "m_eff": 62.3,  "gold_missed": 0.8},
        "batches":                 {"calls": 30.6,  "latency": 131.88, "tflops": 261.03,  "m_eff": 200.3, "gold_missed": 0.3},
    },
    "google/gemma-4-E2B-it": {
        "words_per_paragraph":     {"calls": 300.6, "latency": 273.09, "tflops": 518.89, "m_eff": 11.3,  "gold_missed": 0.9},
        "sentences_per_paragraph": {"calls": 300.6, "latency": 261.49, "tflops": 493.27, "m_eff": 9.8,   "gold_missed": 0.9},
        "begin_mid_end_sentences": {"calls": 300.6, "latency": 325.69, "tflops": 617.91, "m_eff": 17.9,  "gold_missed": 0.9},
        "decaying_words":          {"calls": 300.6, "latency": 276.73, "tflops": 520.55, "m_eff": 10.7,  "gold_missed": 0.8},
        "sliding_window":          {"calls": 300.6, "latency": 289.91, "tflops": 550.42, "m_eff": 13.7,  "gold_missed": 0.9},
        "batches":                 {"calls": 30.6,  "latency": 95.54,  "tflops": 151.27, "m_eff": 186.3, "gold_missed": 0.3},
    },
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


for model_name, strategies in RESULTS.items():
    print(f"\n{'='*70}\n FULL STRATEGY COMPARISON — {model_name} — congress candidates\n{'='*70}")
    print_table(list(strategies.items()))

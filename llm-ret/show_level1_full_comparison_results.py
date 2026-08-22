RESULTS = {
    "Qwen/Qwen3.5-0.8B": {
        "words_per_paragraph":     {"calls": 500.0, "latency": 221.35, "tflops": 397.29, "m_eff": 224.0, "gold_missed": 6.0},
        "sentences_per_paragraph": {"calls": 500.0, "latency": 230.26, "tflops": 412.45, "m_eff": 202.0, "gold_missed": 4.0},
        "begin_mid_end_sentences": {"calls": 500.0, "latency": 278.97, "tflops": 509.12, "m_eff": 318.0, "gold_missed": 3.0},
        "decaying_words":          {"calls": 500.0, "latency": 225.32, "tflops": 406.64, "m_eff": 201.0, "gold_missed": 7.0},
        "sliding_window":          {"calls": 500.0, "latency": 248.83, "tflops": 448.03, "m_eff": 116.0, "gold_missed": 7.0},
        "batches":                 {"calls": 50.0,  "latency": 52.02,  "tflops": 66.07,  "m_eff": 490.0, "gold_missed": 1.0},
    },
    "google/gemma-3-4b-it": {
        "words_per_paragraph":     {"calls": 500.0, "latency": 981.35,  "tflops": 1965.66, "m_eff": 92.0,  "gold_missed": 5.0},
        "sentences_per_paragraph": {"calls": 500.0, "latency": 1004.80, "tflops": 2040.69, "m_eff": 69.0,  "gold_missed": 0.0},
        "begin_mid_end_sentences": {"calls": 500.0, "latency": 1240.22, "tflops": 2517.21, "m_eff": 19.0,  "gold_missed": 0.0},
        "decaying_words":          {"calls": 500.0, "latency": 982.40,  "tflops": 2011.73, "m_eff": 126.0, "gold_missed": 5.0},
        "sliding_window":          {"calls": 500.0, "latency": 1090.59, "tflops": 2215.43, "m_eff": 186.0, "gold_missed": 2.0},
        "batches":                 {"calls": 50.0,  "latency": 179.32,  "tflops": 326.20,  "m_eff": 229.0, "gold_missed": 9.0},
    },
    "google/gemma-3-270m-it": {
        "words_per_paragraph":     {"calls": 500.0, "latency": 75.92, "tflops": 134.03, "m_eff": 344.0, "gold_missed": 0.0},
        "sentences_per_paragraph": {"calls": 500.0, "latency": 77.62, "tflops": 139.21, "m_eff": 374.0, "gold_missed": 0.0},
        "begin_mid_end_sentences": {"calls": 500.0, "latency": 95.36, "tflops": 172.11, "m_eff": 116.0, "gold_missed": 7.0},
        "decaying_words":          {"calls": 500.0, "latency": 75.98, "tflops": 137.19, "m_eff": 212.0, "gold_missed": 1.0},
        "sliding_window":          {"calls": 500.0, "latency": 84.74, "tflops": 151.23, "m_eff": 87.0,  "gold_missed": 6.0},
        "batches":                 {"calls": 50.0,  "latency": 24.34, "tflops": 22.36,  "m_eff": 480.0, "gold_missed": 0.0},
    },
}

NAME_COL = 26
COL = 15


def print_table(rows):
    headers = ["Strategy", "Calls", "Latency(s)", "Lat/Call(s)", "TFLOPs", "M_eff(->L2)", "GoldMissed"]
    print(headers[0].ljust(NAME_COL) + "".join(h.ljust(COL) for h in headers[1:]))
    for label, m in rows:
        lat_per_call = m["latency"] / m["calls"] if m["calls"] else 0.0
        row = [
            f"{m['calls']:.1f}", f"{m['latency']:.2f}", f"{lat_per_call:.2f}",
            f"{m['tflops']:.2f}", f"{m['m_eff']:.1f}", f"{m['gold_missed']:.1f}",
        ]
        print(label.ljust(NAME_COL) + "".join(v.ljust(COL) for v in row))


for model_name, strategies in RESULTS.items():
    print(f"\n{'='*70}\n FULL STRATEGY COMPARISON — {model_name}\n{'='*70}")
    print_table(list(strategies.items()))

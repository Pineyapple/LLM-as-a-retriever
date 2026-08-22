# Hardcoded results from level1_full_comparison.py (avg over 20 queries, congress top-300
# gold-injected hard-candidate pools), reformatted into the 2-table presentation layout:
# Table 1 = cost/efficiency, Table 2 = every strategy x every selection method in one grid.
# Add more models here as their runs finish.

ALL_NAMES = [
    "words_per_paragraph", "sentences_per_paragraph", "begin_mid_end_sentences",
    "decaying_words", "sliding_window", "batches", "batches-global",
]

SELECTION_METHOD_ORDER = ["kurtosis", "cut_90", "cut_80", "cut_70", "cut_60", "cut_50", "cut_40", "cut_30"]

RESULTS = {
    "google/gemma-4-E2B-it": {
        "agg": {
            "words_per_paragraph":     {"calls": 300.6, "latency": 270.10, "tflops": 516.07},
            "sentences_per_paragraph": {"calls": 300.6, "latency": 260.00, "tflops": 490.45},
            "begin_mid_end_sentences": {"calls": 300.6, "latency": 324.24, "tflops": 615.09},
            "decaying_words":          {"calls": 300.6, "latency": 274.46, "tflops": 517.74},
            "sliding_window":          {"calls": 300.6, "latency": 288.42, "tflops": 547.61},
            "batches":                 {"calls": 30.6,  "latency": 102.65, "tflops": 150.98},
            "batches-global":          {"calls": 30.6,  "latency": 102.65, "tflops": 150.98},
        },
        # docs_passing, gold_missed_pct (already a percentage, straight from the pasted run)
        "selection_agg": {
            "words_per_paragraph": {
                "kurtosis": (10.7, 95.0), "cut_90": (270.6, 10.0), "cut_80": (240.6, 20.0), "cut_70": (210.6, 20.0),
                "cut_60": (180.6, 35.0), "cut_50": (150.0, 35.0), "cut_40": (120.0, 50.0), "cut_30": (90.0, 55.0),
            },
            "sentences_per_paragraph": {
                "kurtosis": (8.8, 100.0), "cut_90": (270.6, 10.0), "cut_80": (240.6, 30.0), "cut_70": (210.6, 45.0),
                "cut_60": (180.6, 50.0), "cut_50": (150.0, 65.0), "cut_40": (120.0, 65.0), "cut_30": (90.0, 70.0),
            },
            "begin_mid_end_sentences": {
                "kurtosis": (15.5, 85.0), "cut_90": (270.6, 5.0), "cut_80": (240.6, 15.0), "cut_70": (210.6, 20.0),
                "cut_60": (180.6, 25.0), "cut_50": (150.0, 40.0), "cut_40": (120.0, 45.0), "cut_30": (90.0, 50.0),
            },
            "decaying_words": {
                "kurtosis": (11.7, 90.0), "cut_90": (270.6, 0.0), "cut_80": (240.6, 20.0), "cut_70": (210.6, 30.0),
                "cut_60": (180.6, 45.0), "cut_50": (150.0, 55.0), "cut_40": (120.0, 55.0), "cut_30": (90.0, 60.0),
            },
            "sliding_window": {
                "kurtosis": (11.9, 90.0), "cut_90": (270.6, 0.0), "cut_80": (240.6, 5.0), "cut_70": (210.6, 15.0),
                "cut_60": (180.6, 30.0), "cut_50": (150.0, 35.0), "cut_40": (120.0, 40.0), "cut_30": (90.0, 50.0),
            },
            "batches": {
                "kurtosis": (183.7, 35.0), "cut_90": (270.6, 10.0), "cut_80": (240.6, 20.0), "cut_70": (210.6, 30.0),
                "cut_60": (180.6, 30.0), "cut_50": (150.6, 35.0), "cut_40": (120.5, 45.0), "cut_30": (90.5, 70.0),
            },
            "batches-global": {
                "kurtosis": (8.1, 100.0), "cut_90": (270.6, 10.0), "cut_80": (240.6, 20.0), "cut_70": (210.6, 25.0),
                "cut_60": (180.6, 25.0), "cut_50": (150.0, 40.0), "cut_40": (120.0, 55.0), "cut_30": (90.0, 65.0),
            },
        },
    },
}


def print_grid(headers, rows):
    widths = [max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) + 2 for i in range(len(headers))]
    print("".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("".join(v.ljust(w) for v, w in zip(r, widths)))


def print_cost_table(agg):
    headers = ["Strategy", "Calls", "Latency(s)", "Lat/Call(s)", "TFLOPs"]
    rows = []
    for name in ALL_NAMES:
        m = agg[name]
        lat_per_call = m["latency"] / m["calls"] if m["calls"] else 0.0
        rows.append([name, f"{m['calls']:.1f}", f"{m['latency']:.2f}", f"{lat_per_call:.2f}", f"{m['tflops']:.2f}"])
    print_grid(headers, rows)


def print_selection_grid(selection_agg):
    headers = ["Strategy"]
    for method in SELECTION_METHOD_ORDER:
        headers += [f"{method} Docs", f"{method} Miss%"]
    rows = []
    for name in ALL_NAMES:
        row = [name]
        for method in SELECTION_METHOD_ORDER:
            docs_passing, gold_missed_pct = selection_agg[name][method]
            row += [f"{docs_passing:.1f}", f"{gold_missed_pct:.1f}%"]
        rows.append(row)
    print_grid(headers, rows)


for model_name, data in RESULTS.items():
    print(f"\n{'='*70}\n TABLE 1: COST/EFFICIENCY — {model_name} — congress candidates\n{'='*70}")
    print_cost_table(data["agg"])

    print(f"\n{'='*70}\n TABLE 2: SELECTION OUTCOMES (DocsPassing/GoldMissed%) — {model_name}\n{'='*70}")
    print_selection_grid(data["selection_agg"])

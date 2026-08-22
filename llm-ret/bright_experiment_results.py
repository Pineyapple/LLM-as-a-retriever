# Hardcoded results from bright_experiment.py (avg over 10 queries, BRIGHT biology domain,
# mixed short/long candidate pool). Reformatted with dynamic column widths so it renders cleanly
# regardless of screen width -- the live table can look shifted in a narrow Kaggle output cell.
# Add more domains/models here as their runs finish.

EVAL_KS = [5, 10]
ALL_METHODS = ["naive", "row_verifier", "gridprobe"]

RESULTS = {
    "google/gemma-4-E2B-it": {
        "biology": {
            "naive": {
                "calls": 100.0, "failed": 22.1, "latency": 609.14, "tflops": 503.50,
                "recall": {5: 0.436, 10: 0.679}, "ndcg": {5: 0.542, 10: 0.629}, "auc": 0.907,
            },
            "row_verifier": {
                "calls": 519.6, "failed": 0.0, "latency": 1516.87, "tflops": 2702.75,
                "recall": {5: 0.336, 10: 0.518}, "ndcg": {5: 0.458, 10: 0.509}, "auc": 0.881,
            },
            "gridprobe": {
                "calls": 1039.2, "failed": 0.0, "latency": 3033.13, "tflops": 5405.50,
                "recall": {5: 0.388, 10: 0.543}, "ndcg": {5: 0.489, 10: 0.527}, "auc": 0.894,
            },
        },
    },
    "Qwen/Qwen3-4B": {
        "biology": {
            "naive": {
                "calls": 20.9, "failed": 6.0, "latency": 85.96, "tflops": 123.93,
                "recall": {5: 0.775, 10: 0.945}, "ndcg": {5: 0.902, 10: 0.924}, "auc": 0.948,
            },
            "row_verifier": {
                "calls": 114.5, "failed": 0.0, "latency": 585.46, "tflops": 1074.61,
                "recall": {5: 0.693, 10: 0.931}, "ndcg": {5: 0.911, 10: 0.952}, "auc": 0.935,
            },
            "gridprobe": {
                "calls": 229.0, "failed": 0.0, "latency": 1170.66, "tflops": 2149.23,
                "recall": {5: 0.718, 10: 0.919}, "ndcg": {5: 0.951, 10: 0.970}, "auc": 0.967,
            },
        },
        "psychology": {
            "naive": {
                "calls": 21.5, "failed": 5.0, "latency": 92.53, "tflops": 147.22,
                "recall": {5: 0.683, 10: 0.880}, "ndcg": {5: 0.853, 10: 0.883}, "auc": 0.849,
            },
            "row_verifier": {
                "calls": 105.3, "failed": 0.0, "latency": 515.26, "tflops": 953.12,
                "recall": {5: 0.625, 10: 0.865}, "ndcg": {5: 0.834, 10: 0.875}, "auc": 0.868,
            },
            "gridprobe": {
                "calls": 210.6, "failed": 0.0, "latency": 1030.45, "tflops": 1906.24,
                "recall": {5: 0.745, 10: 0.869}, "ndcg": {5: 0.916, 10: 0.905}, "auc": 0.898,
            },
        },
    },
}


def print_grid(headers, rows):
    widths = [max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) + 2 for i in range(len(headers))]
    print("".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("".join(v.ljust(w) for v, w in zip(r, widths)))


def print_results_table(agg):
    headers = ["Method", "Calls", "Failed", "Latency(s)", "Lat/Call(s)", "TFLOPs"]
    for k in EVAL_KS:
        headers += [f"Recall@{k}", f"NDCG@{k}"]
    headers.append("AUC")
    rows = []
    for name in ALL_METHODS:
        m = agg[name]
        lat_per_call = m["latency"] / m["calls"] if m["calls"] else 0.0
        row = [
            name, f"{m['calls']:.1f}", f"{m['failed']:.1f}",
            f"{m['latency']:.2f}", f"{lat_per_call:.2f}", f"{m['tflops']:.2f}",
        ]
        for k in EVAL_KS:
            row += [f"{m['recall'][k]:.3f}", f"{m['ndcg'][k]:.3f}"]
        row.append(f"{m['auc']:.3f}")
        rows.append(row)
    print_grid(headers, rows)


for model_name, domains in RESULTS.items():
    for domain, agg in domains.items():
        print(f"\n{'='*70}\n NAIVE vs ROW-VERIFIER vs GRIDPROBE — {model_name} — {domain}\n{'='*70}")
        print_results_table(agg)

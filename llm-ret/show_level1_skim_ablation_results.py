import json

NAME_COL = 26
COL = 15


def print_table(rows, eval_ks):
    headers = ["Strategy", "Calls", "Latency(s)", "Lat/Call(s)", "TFLOPs"]
    for name in ("Recall", "Precision", "NDCG"):
        headers += [f"{name}@{k}" for k in eval_ks]
    headers += ["AUC", "M_eff(->L2)", "GoldMissed"]
    print(headers[0].ljust(NAME_COL) + "".join(h.ljust(COL) for h in headers[1:]))
    for label, m in rows:
        lat_per_call = m["latency"] / m["calls"] if m["calls"] else 0.0
        row = [f"{m['calls']:.1f}", f"{m['latency']:.2f}", f"{lat_per_call:.2f}", f"{m['tflops']:.2f}"]
        for name in ("recall", "precision", "ndcg"):
            row += [f"{m[name][k]:.3f}" for k in eval_ks]
        row += [f"{m['auc']:.3f}", f"{m['m_eff']:.1f}", f"{m['gold_missed']:.1f}"]
        print(label.ljust(NAME_COL) + "".join(v.ljust(COL) for v in row))


with open("level1_skim_ablation_results.json", encoding="utf-8") as f:
    agg = json.load(f)

eval_ks = sorted(int(k) for k in next(iter(agg.values()))["recall"].keys())
for name, m in agg.items():
    for metric_name in ("recall", "precision", "ndcg"):
        m[metric_name] = {int(k): v for k, v in m[metric_name].items()}

print(f"\n{'='*90}\n LEVEL 1 SKIM STRATEGY COMPARISON\n{'='*90}")
print_table(list(agg.items()), eval_ks)

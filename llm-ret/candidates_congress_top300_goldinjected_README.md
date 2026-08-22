# candidates_congress_top300_goldinjected.jsonl

Top-300 candidates per query for OBLIQ-Bench **congress**, retrieved by CorpusProbe
stage 1 (`fullcorpus`) over the full
213,650-document corpus.

## Format
One JSON object per line:

```json
{
  "query_id": "...", "query": "...",
  "n_gold_total": 1, "n_gold_in_pool": 1, "gold_doc_ids": ["..."],
  "candidates": [
    {"rank": 1, "doc_id": "...", "score": -12.3, "is_gold": false,
      "injected": false, "text": "..."}
  ]
}
```

`score` is the probe log-odds logP(Yes) - logP(No); higher is more relevant. Candidates are
already sorted by it, so `rank` 1 is stage 1's top pick.

## What this is
Gold documents were INJECTED where stage 1 missed them, so every query has the answer present. Use this to measure JUDGE quality (can a model recognise the gold among hard distractors). Do NOT compute retrieval metrics on it.

## Note on the negatives
These are the probe's own top-ranked non-gold documents, i.e. the passages that most confused
it. They are substantially harder than a random sample, so scores here will look worse than on
a randomly-built pool. That is expected and is the point.

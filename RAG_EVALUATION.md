# RAG Retrieval Evaluation

## Evaluation set

The evaluation set contains 25 hand-written questions against `company_employee_handbook.pdf`:

- 20 answerable questions with expected source/chunk IDs.
- 5 questions whose answers are intentionally not present in the document.

The retrieval metrics are calculated only on the 20 answerable questions.

## Metrics

- **Recall@k**: fraction of answerable questions where an expected source/chunk appears in the top-k retrieved results.
- **MRR (Mean Reciprocal Rank)**: average reciprocal rank of the first expected source/chunk.
- **Retrieval latency**: wall-clock time spent in the `retrieve()` function.

## Results

Run:

```powershell
python scripts/evaluate_retrieval.py
```

The script evaluates both `semantic` and `hybrid` modes and writes detailed results to:

```text
evaluation_results.json
```

Paste the measured numbers below after running the evaluation:

| Mode | Recall@1 | Recall@3 | Recall@4 | MRR | Mean latency (ms) |
|---|---:|---:|---:|---:|---:|
Semantic: 0.8500 / 1.0000 / 1.0000 / 0.9250 / 33.21 ms
Hybrid:   0.9000 / 1.0000 / 1.0000 / 0.9500 / 23.58 ms



## Recommendation

Choose the default retrieval mode based on the measured Recall/MRR and latency.

- If one mode has a consistent retrieval-quality advantage, use that mode as the default.
- If the differences are within normal run-to-run variance, keep the simpler/default mode and document that the evaluation did not show a meaningful difference.

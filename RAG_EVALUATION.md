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
python -m scripts.evaluate_retrieval
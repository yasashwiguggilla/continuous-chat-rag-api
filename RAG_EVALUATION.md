# RAG Retrieval Evaluation

## Evaluation Set

The evaluation set contains 25 hand-written questions against
`company_employee_handbook.pdf`.

- 20 answerable questions with expected source/chunk IDs.
- 5 out-of-document (OOD) questions whose answers are intentionally
  not present in the document.

The retrieval metrics are calculated only on the 20 answerable
questions.

The evaluation ground truth was updated to match the current
section-aware chunking structure.

## Retrieval Modes

The evaluation compares two retrieval modes.

### Semantic

Uses embedding-based semantic similarity between the user query
and document chunks.

### Hybrid

Combines semantic similarity with keyword matching.

The current application supports both modes and allows the retrieval
mode to be selected per request.

## Metrics

### Recall@k

The fraction of answerable questions for which an expected
source/chunk appears in the top-k retrieved results.

### MRR (Mean Reciprocal Rank)

The average reciprocal rank of the first relevant retrieved chunk.

### Retrieval Latency

The wall-clock time spent inside the `retrieve()` function.

### OOD Results

The number of out-of-document questions for which the retriever
still returned one or more results.

## Evaluation Command

Run the evaluation from the project root:

```powershell
python -m scripts.evaluate_retrieval
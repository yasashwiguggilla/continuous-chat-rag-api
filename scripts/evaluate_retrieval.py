from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from rag.retriever import retrieve


ROOT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_FILE = ROOT_DIR / "evaluation" / "evaluation_set.json"

MODES = ("semantic", "hybrid")
TOP_K = 4
MIN_SCORE = 0.35
RECALL_KS = (1, 3, 4)


def load_evaluation_set() -> list[dict]:
    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def is_relevant(result: dict, case: dict) -> bool:
    return (
        case["answerable"]
        and result.get("source") == case["expected_source"]
        and result.get("chunk_id") in case["expected_chunks"]
    )


def evaluate_case(case: dict, mode: str) -> dict:
    start = time.perf_counter()

    results = retrieve(
        case["question"],
        top_k=TOP_K,
        min_score=MIN_SCORE,
        mode=mode,
    )

    latency_ms = round(
        (time.perf_counter() - start) * 1000,
        2,
    )

    first_relevant_rank = None

    for rank, result in enumerate(results, start=1):
        if is_relevant(result, case):
            first_relevant_rank = rank
            break

    return {
        "id": case["id"],
        "question": case["question"],
        "answerable": case["answerable"],
        "latency_ms": latency_ms,
        "results": [
            {
                "source": result.get("source"),
                "chunk_id": result.get("chunk_id"),
                "score": result.get("score"),
            }
            for result in results
        ],
        "first_relevant_rank": first_relevant_rank,
    }


def calculate_metrics(cases: list[dict], evaluated: list[dict]) -> dict:
    answerable_count = sum(1 for case in cases if case["answerable"])

    metrics = {
        "answerable_questions": answerable_count,
        "recall_at_k": {},
        "mrr": 0.0,
        "retrieval_latency_ms": {},
        "out_of_document_cases_with_results": 0,
    }

    answerable_results = [
        item for item in evaluated if item["answerable"]
    ]

    for k in RECALL_KS:
        hits = sum(
            1
            for item in answerable_results
            if item["first_relevant_rank"] is not None
            and item["first_relevant_rank"] <= k
        )
        metrics["recall_at_k"][f"recall@{k}"] = round(
            hits / answerable_count,
            4,
        )

    reciprocal_ranks = [
        1 / item["first_relevant_rank"]
        for item in answerable_results
        if item["first_relevant_rank"] is not None
    ]

    if reciprocal_ranks:
        metrics["mrr"] = round(
            statistics.mean(reciprocal_ranks),
            4,
        )

    latencies = [item["latency_ms"] for item in evaluated]

    metrics["retrieval_latency_ms"] = {
        "mean": round(statistics.mean(latencies), 2),
        "median": round(statistics.median(latencies), 2),
        "min": round(min(latencies), 2),
        "max": round(max(latencies), 2),
    }

    metrics["out_of_document_cases_with_results"] = sum(
        1
        for item in evaluated
        if not item["answerable"] and item["results"]
    )

    return metrics


def main() -> None:
    cases = load_evaluation_set()

    all_results = {}

    for mode in MODES:
        print(f"\n=== {mode.upper()} RETRIEVAL ===")

        evaluated = [
            evaluate_case(case, mode)
            for case in cases
        ]

        metrics = calculate_metrics(cases, evaluated)

        all_results[mode] = {
            "metrics": metrics,
            "cases": evaluated,
        }

        print(f"Recall@1 : {metrics['recall_at_k']['recall@1']:.4f}")
        print(f"Recall@3 : {metrics['recall_at_k']['recall@3']:.4f}")
        print(f"Recall@4 : {metrics['recall_at_k']['recall@4']:.4f}")
        print(f"MRR      : {metrics['mrr']:.4f}")
        print(
            "Latency  : "
            f"mean={metrics['retrieval_latency_ms']['mean']} ms, "
            f"median={metrics['retrieval_latency_ms']['median']} ms"
        )
        print(
            "OOD cases returning results: "
            f"{metrics['out_of_document_cases_with_results']}"
        )

    output_file = ROOT_DIR / "evaluation_results.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()

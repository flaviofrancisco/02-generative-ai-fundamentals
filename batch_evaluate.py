#!/usr/bin/env python3
"""
Batch evaluation of the NASA RAG system over evaluation_dataset.txt.

For every question in the dataset it runs the full pipeline
(retrieve -> generate -> RAGAS evaluate), prints a per-question summary and an
aggregate (mean / min / max) per metric, and optionally writes the results to
CSV / JSON.

Usage:
    python batch_evaluate.py --openai-key $OPENAI_API_KEY
    python batch_evaluate.py --dataset evaluation_dataset.txt --chroma-dir ./chroma_db_openai \
        --collection-name nasa_space_missions_text --n-results 3 --output results.csv
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

import llm_client
import rag_client
import ragas_evaluator

load_dotenv()

METRIC_ORDER = ["answer_relevancy", "faithfulness", "context_precision", "bleu_score", "rouge_score"]


def load_evaluation_dataset(path: str) -> List[Dict[str, str]]:
    """Parse evaluation_dataset.txt into a list of records.

    Record format: "Field: value" lines, multi-line values continue on the next
    lines, records separated by a "---" line, "#" lines are comments.
    """
    text = Path(path).read_text(encoding="utf-8")
    records: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    last_field: Optional[str] = None

    def flush():
        nonlocal current, last_field
        if current.get("question"):
            current.setdefault("category", "general")
            current.setdefault("mission", "all")
            current.setdefault("expected", "")
            records.append({k: v.strip() for k, v in current.items()})
        current, last_field = {}, None

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("#"):
            continue
        if line.strip() == "---":
            flush()
            continue
        if not line.strip():
            continue
        head, sep, value = line.partition(":")
        key = head.strip().lower()
        if sep and key in ("category", "mission", "question", "expected"):
            current[key] = value.strip()
            last_field = key
        elif last_field:  # continuation line
            current[last_field] += " " + line.strip()
    flush()
    return records


def evaluate_record(record: Dict[str, str], collection, openai_key: str,
                    model: str, n_results: int, use_mission_filter: bool) -> Dict:
    mission_filter = record["mission"] if use_mission_filter else None
    docs_result = rag_client.retrieve_documents(collection, record["question"], n_results, mission_filter)

    documents, metadatas, distances = [], [], None
    if docs_result and docs_result.get("documents"):
        documents = docs_result["documents"][0]
        metadatas = docs_result["metadatas"][0]
        distances = (docs_result.get("distances") or [None])[0]

    context = rag_client.format_context(documents, metadatas, distances)
    contexts = rag_client.unique_documents(documents)

    answer = llm_client.generate_response(openai_key, record["question"], context, [], model)
    scores = ragas_evaluator.evaluate_response_quality(
        record["question"], answer, contexts, reference=record.get("expected") or None, openai_key=openai_key
    )
    return {
        **record,
        "n_contexts": len(contexts),
        "answer": answer,
        "scores": scores,
    }


def print_summary(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    print("\n" + "=" * 78)
    print("PER-QUESTION RESULTS")
    print("=" * 78)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] ({r['category']} / {r['mission']}) {r['question']}")
        print(f"    contexts retrieved: {r['n_contexts']}")
        print(f"    answer: {r['answer'][:220].replace(chr(10), ' ')}{'...' if len(r['answer']) > 220 else ''}")
        scores = r["scores"]
        if "error" in scores:
            print(f"    evaluation error: {scores['error']}")
            continue
        line = ", ".join(f"{m}={scores[m]:.3f}" for m in METRIC_ORDER if m in scores)
        print(f"    scores: {line}")
        for w in scores.get("warnings", []):
            print(f"    warning: {w}")

    aggregate: Dict[str, Dict[str, float]] = {}
    for metric in METRIC_ORDER:
        values = [r["scores"][metric] for r in results if isinstance(r["scores"].get(metric), (int, float))]
        if values:
            aggregate[metric] = {
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }

    print("\n" + "=" * 78)
    print("AGGREGATE")
    print("=" * 78)
    print(f"{'metric':<20}{'mean':>8}{'min':>8}{'max':>8}{'n':>5}")
    for metric, agg in aggregate.items():
        print(f"{metric:<20}{agg['mean']:>8.3f}{agg['min']:>8.3f}{agg['max']:>8.3f}{agg['n']:>5}")
    errors = sum(1 for r in results if "error" in r["scores"])
    if errors:
        print(f"\n{errors} question(s) could not be evaluated (see errors above).")
    return aggregate


def save_results(results: List[Dict], aggregate: Dict, output: str) -> None:
    out = Path(output)
    if out.suffix.lower() == ".json":
        out.write_text(json.dumps({"results": results, "aggregate": aggregate}, indent=2), encoding="utf-8")
    else:
        import pandas as pd
        rows = []
        for r in results:
            row = {k: v for k, v in r.items() if k != "scores"}
            row.update({m: r["scores"].get(m) for m in METRIC_ORDER})
            row["error"] = r["scores"].get("error", "")
            rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nResults written to {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch RAGAS evaluation over a test set")
    parser.add_argument("--dataset", default="evaluation_dataset.txt")
    parser.add_argument("--openai-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--chroma-dir", default="./chroma_db_openai")
    parser.add_argument("--collection-name", default="nasa_space_missions_text")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--n-results", type=int, default=3)
    parser.add_argument("--no-mission-filter", action="store_true",
                        help="Search all missions instead of the record's Mission field")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N questions")
    parser.add_argument("--output", help="Write results to .csv or .json")
    args = parser.parse_args()

    if not args.openai_key:
        print("No OpenAI API key: pass --openai-key or set OPENAI_API_KEY.", file=sys.stderr)
        return 2
    os.environ["OPENAI_API_KEY"] = args.openai_key
    os.environ["CHROMA_OPENAI_API_KEY"] = args.openai_key

    records = load_evaluation_dataset(args.dataset)
    if not records:
        print(f"No questions found in {args.dataset}", file=sys.stderr)
        return 2
    if args.limit:
        records = records[: args.limit]
    print(f"Loaded {len(records)} questions from {args.dataset}")

    collection, ok, error = rag_client.initialize_rag_system(args.chroma_dir, args.collection_name)
    if not ok:
        print(f"Could not open ChromaDB collection: {error}\nRun embedding_pipeline.py first.", file=sys.stderr)
        return 2
    print(f"Collection {args.collection_name} in {args.chroma_dir}: {collection.count()} chunks")

    results = []
    start = time.time()
    for i, record in enumerate(records, 1):
        print(f"\n({i}/{len(records)}) {record['question']}")
        try:
            results.append(evaluate_record(record, collection, args.openai_key, args.model,
                                           args.n_results, not args.no_mission_filter))
        except Exception as e:
            results.append({**record, "n_contexts": 0, "answer": "", "scores": {"error": str(e)}})

    aggregate = print_summary(results)
    print(f"\nTotal time: {time.time() - start:.1f}s")
    if args.output:
        save_results(results, aggregate, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

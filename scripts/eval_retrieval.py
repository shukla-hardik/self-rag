"""
Retrieval precision evaluation script.

Measures Precision@K, Recall@K, MRR, and NDCG@K against a test set of
(query, relevant_chunk_ids_or_keywords) pairs.

Usage:
    # With a JSONL test set (recommended):
    python scripts/eval_retrieval.py --test-file scripts/eval_testset.jsonl --user-id <uuid>

    # Quick smoke-test with a single query:
    python scripts/eval_retrieval.py --query "your query" --user-id <uuid>

Test-set JSONL format (one JSON object per line):
    {"query": "What is X?", "relevant": ["exact chunk text fragment", ...]}

    "relevant" entries are matched as substrings of retrieved chunk content
    (case-insensitive). Alternatively, pass chunk UUIDs if you know them:
    {"query": "What is X?", "relevant_ids": ["uuid1", "uuid2"]}

Environment:
    Reads DB/embedding settings from .env (same as the app).
    Requires GEMINI_API_KEY and GEMINI_EMBEDDING_MODEL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from uuid import UUID

# ── project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db.client import DBClient  # noqa: E402
# Import retriever directly to avoid loading the full LangGraph pipeline
import importlib.util, types  # noqa: E401
_spec = importlib.util.spec_from_file_location(
    "app.rag.retriever",
    ROOT / "app" / "rag" / "retriever.py",
)
_mod = types.ModuleType("app.rag.retriever")
# Pre-register as app.rag.retriever so its own imports resolve correctly
import sys as _sys  # noqa: E401
_sys.modules.setdefault("app.rag", types.ModuleType("app.rag"))
_sys.modules["app.rag.retriever"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
Retriever = _mod.Retriever


# ── metrics ──────────────────────────────────────────────────────────────────

def _is_relevant(chunk_content: str, chunk_id: str, relevant: list[str],
                 relevant_ids: list[str]) -> bool:
    if relevant_ids and chunk_id in relevant_ids:
        return True
    return any(kw.lower() in chunk_content.lower() for kw in relevant)


def precision_at_k(hits: list[bool]) -> float:
    return sum(hits) / len(hits) if hits else 0.0


def recall_at_k(hits: list[bool], total_relevant: int) -> float:
    if total_relevant == 0:
        return 1.0
    return sum(hits) / total_relevant


def reciprocal_rank(hits: list[bool]) -> float:
    for i, h in enumerate(hits, start=1):
        if h:
            return 1.0 / i
    return 0.0


def ndcg_at_k(hits: list[bool]) -> float:
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    ideal_hits = sorted(hits, reverse=True)
    idcg = sum(h / math.log2(i + 2) for i, h in enumerate(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ── core eval ────────────────────────────────────────────────────────────────

async def evaluate_query(
    db,
    user_id: UUID,
    query: str,
    relevant: list[str],
    relevant_ids: list[str],
    top_k: int,
    use_hybrid: bool,
    use_reranker: bool,
) -> dict:
    docs = await Retriever.get(
        db,
        user_id=user_id,
        query=query,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
    )

    hits = [
        _is_relevant(
            d.page_content,
            str(d.metadata.get("chunk_id", "")),
            relevant,
            relevant_ids,
        )
        for d in docs
    ]

    total_relevant = max(len(relevant) + len(relevant_ids), 1)

    return {
        "query": query,
        "retrieved": len(docs),
        "hits": hits,
        "precision": precision_at_k(hits),
        "recall": recall_at_k(hits, total_relevant),
        "mrr": reciprocal_rank(hits),
        "ndcg": ndcg_at_k(hits),
        "retrieved_snippets": [d.page_content[:120] for d in docs],
    }


async def run_eval(
    user_id: UUID,
    test_cases: list[dict],
    top_k: int,
    use_hybrid: bool,
    use_reranker: bool,
    verbose: bool,
) -> None:
    Retriever.init()
    if use_reranker:
        Retriever.init_reranker()

    results = []
    async with DBClient.get_session() as db:
        for i, tc in enumerate(test_cases, start=1):
            query = tc["query"]
            relevant = tc.get("relevant", [])
            relevant_ids = tc.get("relevant_ids", [])

            print(f"[{i}/{len(test_cases)}] {query[:80]}", flush=True)
            r = await evaluate_query(
                db, user_id, query, relevant, relevant_ids,
                top_k, use_hybrid, use_reranker,
            )
            results.append(r)

            if verbose:
                print(f"  P@{top_k}={r['precision']:.2f}  "
                      f"R@{top_k}={r['recall']:.2f}  "
                      f"MRR={r['mrr']:.2f}  NDCG@{top_k}={r['ndcg']:.2f}")
                for j, (hit, snip) in enumerate(
                        zip(r["hits"], r["retrieved_snippets"]), start=1):
                    mark = "✓" if hit else "✗"
                    print(f"    {mark} [{j}] {snip!r}")

    _print_summary(results, top_k)


def _print_summary(results: list[dict], top_k: int) -> None:
    n = len(results)
    if n == 0:
        print("No results.")
        return

    avg_p = sum(r["precision"] for r in results) / n
    avg_r = sum(r["recall"] for r in results) / n
    avg_mrr = sum(r["mrr"] for r in results) / n
    avg_ndcg = sum(r["ndcg"] for r in results) / n

    # per-query hit counts for a histogram
    hit_counts = [sum(r["hits"]) for r in results]
    zero_hit = sum(1 for h in hit_counts if h == 0)

    sep = "─" * 50
    print(f"\n{sep}")
    print(f"  Queries evaluated : {n}")
    print(f"  Top-K             : {top_k}")
    print(f"  Precision@{top_k:<5}   : {avg_p:.4f}")
    print(f"  Recall@{top_k:<8}  : {avg_r:.4f}")
    print(f"  MRR               : {avg_mrr:.4f}")
    print(f"  NDCG@{top_k:<9}   : {avg_ndcg:.4f}")
    print(f"  Queries w/ 0 hits : {zero_hit} / {n}")
    print(sep)

    # worst queries (zero hits) for debugging
    if zero_hit:
        print("\nQueries with no relevant results retrieved:")
        for r in results:
            if sum(r["hits"]) == 0:
                print(f"  - {r['query']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_testset(path: Path) -> list[dict]:
    cases = []
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"Bad JSON on line {lineno} of {path}: {e}")
            if "query" not in obj:
                sys.exit(f"Line {lineno}: missing 'query' field")
            if "relevant" not in obj and "relevant_ids" not in obj:
                sys.exit(f"Line {lineno}: need 'relevant' (keywords) or 'relevant_ids' (UUIDs)")
            cases.append(obj)
    return cases


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure retrieval precision on a test set")
    p.add_argument("--user-id", required=True, help="UUID of the user whose docs to query")
    p.add_argument("--test-file", type=Path,
                   help="JSONL file with test cases (one per line)")
    p.add_argument("--query", help="Single ad-hoc query (requires --relevant or --relevant-ids)")
    p.add_argument("--relevant", nargs="*", default=[],
                   help="Keyword substrings that mark a chunk as relevant (single-query mode)")
    p.add_argument("--relevant-ids", nargs="*", default=[],
                   help="Exact chunk UUIDs that are relevant (single-query mode)")
    p.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default 5)")
    p.add_argument("--no-hybrid", action="store_true", help="Use dense-only retrieval")
    p.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    p.add_argument("--verbose", "-v", action="store_true", help="Print per-query hit breakdown")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.test_file and args.query:
        sys.exit("Pass either --test-file or --query, not both.")

    if args.test_file:
        test_cases = _load_testset(args.test_file)
    elif args.query:
        if not args.relevant and not args.relevant_ids:
            sys.exit("Single-query mode requires --relevant or --relevant-ids.")
        test_cases = [{"query": args.query, "relevant": args.relevant,
                       "relevant_ids": args.relevant_ids}]
    else:
        sys.exit("Provide --test-file or --query.")

    try:
        user_id = UUID(args.user_id)
    except ValueError:
        sys.exit(f"Invalid UUID: {args.user_id!r}")

    asyncio.run(run_eval(
        user_id=user_id,
        test_cases=test_cases,
        top_k=args.top_k,
        use_hybrid=not args.no_hybrid,
        use_reranker=args.rerank,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    main()
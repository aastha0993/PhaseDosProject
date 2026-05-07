#!/usr/bin/env python3
"""
rag_engines.py  —  The Parallel RAG Strike
Three strictly typed retrieval functions. No routing logic. Error-as-Information.
Designed to be fired simultaneously via ThreadPoolExecutor in compiler.py.
"""

from __future__ import annotations

import os
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Paths & config ───────────────────────────────────────────────────
SQLITE_PATH = Path("sandbox_data/creators.db")
FAISS_PATH  = Path("sandbox_data/drama.faiss")
META_PATH   = Path("sandbox_data/drama_meta.pkl")
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 5

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "dramapassword")


# ── Lazy singletons — loaded once, reused across calls ───────────────
_faiss_index: faiss.Index | None = None
_faiss_meta:  List[Dict]  | None = None
_embed_model: SentenceTransformer | None = None


def _load_faiss_resources() -> tuple[faiss.Index, List[Dict], SentenceTransformer]:
    global _faiss_index, _faiss_meta, _embed_model
    if _faiss_index is None:
        _faiss_index = faiss.read_index(str(FAISS_PATH))
    if _faiss_meta is None:
        with open(META_PATH, "rb") as fh:
            _faiss_meta = pickle.load(fh)
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _faiss_index, _faiss_meta, _embed_model


# ════════════════════════════════════════════════════════════════════
#  ENGINE 1  —  SQLite  (The Math)
# ════════════════════════════════════════════════════════════════════

def query_sqlite(creator_name: str) -> str:
    """
    Pulls structured metrics for creator_name from SQLite.

    Returns a formatted stats string. On any failure — missing creator,
    corrupt DB, wrong path — returns a tagged error string so the
    synthesizer can treat absence-of-data as a signal, not a crash.
    """
    try:
        if not SQLITE_PATH.exists():
            return (
                f"[SQLITE_ERROR] Database file not found at '{SQLITE_PATH}'. "
                "Run generate_sandbox.py first."
            )

        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM creators WHERE name = ? COLLATE NOCASE",
                (creator_name,),
            ).fetchone()

        if row is None:
            # Partial match attempt — helps with typos / shortened names
            with sqlite3.connect(SQLITE_PATH) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM creators WHERE name LIKE ? COLLATE NOCASE LIMIT 1",
                    (f"%{creator_name}%",),
                ).fetchone()

        if row is None:
            return (
                f"[SQLITE_MISS] No creator matching '{creator_name}' exists in the database. "
                "Either a brand-new creator with zero paper trail, or a ghost account."
            )

        real_followers = int(row["follower_count"] * (1 - row["bot_percentage"]))
        roi_flag = "UNDERWATER" if row["avg_sponsorship_roi"] < 0 else (
            "EXCEPTIONAL" if row["avg_sponsorship_roi"] > 8 else "NOMINAL"
        )

        return (
            f"[SQLITE_STATS] {row['name']} ({row['niche']})\n"
            f"  Followers        : {row['follower_count']:,}  "
            f"(est. real: {real_followers:,} | bot%: {row['bot_percentage']:.0%})\n"
            f"  Avg Sponsor ROI  : {row['avg_sponsorship_roi']:.1f}x  [{roi_flag}]\n"
            f"  Apology Videos   : {row['apology_video_count']}\n"
            f"  Drama Score      : {row['drama_score']}/10"
        )

    except Exception as exc:
        return f"[SQLITE_ERROR] Unhandled exception querying '{creator_name}': {exc}"


# ════════════════════════════════════════════════════════════════════
#  ENGINE 2  —  FAISS  (The Lore)
# ════════════════════════════════════════════════════════════════════

def query_faiss(creator_name: str) -> str:
    """
    Retrieves the most relevant drama chunks for creator_name from FAISS.

    Strategy:
      1. Pull all chunks whose 'creator' field exactly matches the name.
      2. If that yields fewer than TOP_K results, augment with semantic
         nearest-neighbours from the full index.
    Returns a formatted lore string. Errors are returned as tagged strings.
    """
    try:
        if not FAISS_PATH.exists() or not META_PATH.exists():
            return (
                f"[FAISS_ERROR] Index files not found. "
                "Run generate_sandbox.py first."
            )

        index, meta, model = _load_faiss_resources()

        # Step 1: exact creator match from metadata
        exact: List[Dict] = [
            chunk for chunk in meta
            if chunk["creator"].lower() == creator_name.lower()
        ]

        # Step 2: semantic augmentation if needed
        semantic_extras: List[Dict] = []
        if len(exact) < TOP_K:
            query_vec = model.encode(
                [creator_name],
                normalize_embeddings=True,
            ).astype(np.float32)
            scores, indices = index.search(query_vec, TOP_K * 3)
            seen_texts = {c["text"] for c in exact}
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                chunk = meta[idx]
                if chunk["text"] not in seen_texts:
                    semantic_extras.append({**chunk, "_score": float(score)})
                    seen_texts.add(chunk["text"])

        combined = exact[:TOP_K] + semantic_extras[: max(0, TOP_K - len(exact))]

        if not combined:
            return (
                f"[FAISS_MISS] No drama corpus entries found for '{creator_name}'. "
                "Either squeaky clean or hasn't done anything interesting yet."
            )

        blocks: List[str] = []
        for i, chunk in enumerate(combined, 1):
            score_tag = f" | similarity={chunk['_score']:.3f}" if "_score" in chunk else ""
            blocks.append(
                f"  [{i}] type={chunk['type'].upper()}{score_tag}\n"
                f"  {chunk['text']}"
            )

        header = (
            f"[FAISS_LORE] {len(combined)} drama chunk(s) retrieved for '{creator_name}' "
            f"({len(exact)} direct, {len(combined) - len(exact)} semantic):"
        )
        return header + "\n\n" + "\n\n".join(blocks)

    except Exception as exc:
        return f"[FAISS_ERROR] Unhandled exception querying '{creator_name}': {exc}"


# ════════════════════════════════════════════════════════════════════
#  ENGINE 3  —  Neo4j  (The Beef)
# ════════════════════════════════════════════════════════════════════

def query_neo4j(creator_name: str) -> str:
    """
    Pulls the full relationship map for creator_name from Neo4j.
    Captures outgoing edges (what they did) and incoming edges (what was done to them).
    Also fetches second-degree connections for hidden management chains.
    Errors returned as tagged strings.
    """
    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        with driver.session() as session:
            # Direct relationships — outgoing
            out_rows = session.run(
                """
                MATCH (a:SandboxNode {name: $name})-[r]->(b:SandboxNode)
                RETURN type(r) AS rel, b.name AS target,
                       labels(b) AS target_labels, properties(r) AS props
                ORDER BY type(r)
                """,
                name=creator_name,
            ).data()

            # Direct relationships — incoming
            in_rows = session.run(
                """
                MATCH (a:SandboxNode)-[r]->(b:SandboxNode {name: $name})
                RETURN type(r) AS rel, a.name AS source,
                       labels(a) AS source_labels, properties(r) AS props
                ORDER BY type(r)
                """,
                name=creator_name,
            ).data()

            # Second-degree: who else shares the same manager?
            shared_mgr_rows = session.run(
                """
                MATCH (a:SandboxNode {name: $name})-[:SECRETLY_MANAGED_BY]->(mgr:SandboxNode)
                MATCH (other:SandboxNode)-[:SECRETLY_MANAGED_BY]->(mgr)
                WHERE other.name <> $name
                RETURN mgr.name AS manager, other.name AS co_client
                """,
                name=creator_name,
            ).data()

        if not out_rows and not in_rows:
            return (
                f"[NEO4J_MISS] '{creator_name}' has zero mapped relationships in the graph. "
                "No documented beefs, no brand terminations, no hidden handlers. "
                "Either genuinely clean or not yet indexed."
            )

        lines: List[str] = [f"[NEO4J_BEEF] Relationship map for '{creator_name}':"]

        if out_rows:
            lines.append("  — Outgoing —")
            for row in out_rows:
                target_type = next(
                    (l for l in row["target_labels"] if l != "SandboxNode"), "Node"
                )
                props = {k: v for k, v in row["props"].items() if v is not None}
                prop_str = "  |  ".join(f"{k}: {v}" for k, v in props.items())
                lines.append(
                    f"    [{row['rel']}] → {row['target']} ({target_type})"
                    + (f"\n        {prop_str}" if prop_str else "")
                )

        if in_rows:
            lines.append("  — Incoming —")
            for row in in_rows:
                source_type = next(
                    (l for l in row["source_labels"] if l != "SandboxNode"), "Node"
                )
                props = {k: v for k, v in row["props"].items() if v is not None}
                prop_str = "  |  ".join(f"{k}: {v}" for k, v in props.items())
                lines.append(
                    f"    [{row['rel']}] ← {row['source']} ({source_type})"
                    + (f"\n        {prop_str}" if prop_str else "")
                )

        if shared_mgr_rows:
            lines.append("  — Hidden Shared Management —")
            for row in shared_mgr_rows:
                lines.append(
                    f"    {creator_name} and {row['co_client']} are both "
                    f"secretly managed by {row['manager']} — neither knows."
                )

        return "\n".join(lines)

    except Exception as exc:
        return f"[NEO4J_ERROR] Unhandled exception querying '{creator_name}': {exc}"

    finally:
        if driver:
            driver.close()


# ════════════════════════════════════════════════════════════════════
#  PARALLEL STRIKE  —  fires all three engines simultaneously
# ════════════════════════════════════════════════════════════════════

def parallel_rag_strike(creator_name: str) -> Dict[str, str]:
    """
    Submits query_sqlite, query_faiss, query_neo4j to a thread pool
    and collects results. Each engine is independent — a failure in one
    does not block the others. All errors surface as tagged strings.

    Returns:
        {
            "sql_context":   str,   # structured metrics
            "faiss_context": str,   # unstructured lore
            "graph_context": str,   # relationship map
        }
    """
    task_map = {
        "sql_context":   query_sqlite,
        "faiss_context": query_faiss,
        "graph_context": query_neo4j,
    }

    results: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        future_to_key = {
            pool.submit(fn, creator_name): key
            for key, fn in task_map.items()
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                results[key] = f"[EXECUTOR_ERROR] {key} thread crashed: {exc}"

    return results


# ════════════════════════════════════════════════════════════════════
#  SMOKE TEST
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    TEST_CREATORS = [
        "CosmicKai",          # high drama, all three DBs populated
        "LarpingLorenzo",     # clean record, edge case
        "SatoshiSleeper",     # crypto villain, Neo4j beef + FAISS lore
        "NonExistentCreator", # should return graceful MISS strings
    ]

    for name in TEST_CREATORS:
        print(f"\n{'═' * 64}")
        print(f"  PARALLEL RAG STRIKE  →  {name}")
        print("═" * 64)
        ctx = parallel_rag_strike(name)
        for key in ("sql_context", "faiss_context", "graph_context"):
            print(f"\n── {key.upper()} ──")
            print(ctx[key])
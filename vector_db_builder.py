"""
Vector DB Builder & MVP Retrieval QA — Security Paper RAG
=========================================================

Workflow:
  1. Load up to 10,000 chunks from output/rag_chunks_sample.jsonl
  2. Embed with BAAI/bge-m3 (multilingual, 1024-dim)
  3. Store in local ChromaDB (output/chroma_db)
  4. Launch interactive terminal QA (top-3 retrieval)

Usage:
  python vector_db_builder.py                  # auto-detect: build or query
  python vector_db_builder.py --rebuild        # force rebuild the DB
  python vector_db_builder.py --query-only     # skip build, go straight to QA
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

# =============================================================================
# Config
# =============================================================================

CHUNKS_FILE = Path("output/rag_chunks_sample.jsonl")
CHROMA_DB_DIR = Path("output/chroma_db")
COLLECTION_NAME = "security_papers"
MAX_RECORDS = 10_000
MODEL_NAME = "BAAI/bge-m3"

# =============================================================================
# Check dependencies
# =============================================================================

_DEP_ERRORS: list[str] = []

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    _DEP_ERRORS.append("chromadb  →  pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    _DEP_ERRORS.append("sentence-transformers  →  pip install sentence-transformers")

try:
    from tqdm import tqdm
except ImportError:
    _DEP_ERRORS.append("tqdm  →  pip install tqdm")

if _DEP_ERRORS:
    print("Missing dependencies:")
    for err in _DEP_ERRORS:
        print(f"  - {err}")
    print("\nInstall with: pip install sentence-transformers chromadb tqdm")
    sys.exit(1)


# =============================================================================
# Utility: terminal formatting
# =============================================================================

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_CYAN = "\033[96m"
C_RED = "\033[91m"


def _maybe_color(text: str, color: str) -> str:
    """Apply ANSI color if stdout is a TTY, otherwise plain text."""
    if sys.stdout.isatty():
        return f"{color}{text}{C_RESET}"
    return text


def _print_header(title: str) -> None:
    print(f"\n{_maybe_color('=' * 62, C_CYAN)}")
    print(_maybe_color(f"  {title}", C_BOLD + C_CYAN))
    print(_maybe_color('=' * 62, C_CYAN))
    print()


def _print_chunk_result(idx: int, dist: float, metadata: dict, content: str) -> None:
    """Pretty-print a single retrieved chunk."""
    score = 1.0 - dist if dist <= 1.0 else 0.0
    score_pct = f"{score * 100:.1f}%"
    tag_str = ", ".join(metadata.get("tags", [])) or "(no tags)"

    print(_maybe_color(f"  ┌─ Result #{idx+1}  [similarity: {score_pct}]", C_GREEN + C_BOLD))
    print(_maybe_color(f"  ├─ Title: {metadata.get('paper_title', '?')[:90]}", C_YELLOW))
    print(_maybe_color(f"  ├─ Tags:  {tag_str}", C_BLUE))
    print(_maybe_color(f"  └─ Snippet:", C_DIM))

    # Wrap snippet to 80 columns for readability
    snippet = content[:250].replace("\n", " ")
    wrapped = textwrap.fill(snippet, width=80, initial_indent="      ", subsequent_indent="      ")
    print(wrapped)
    print()


# =============================================================================
# Module A: Embedding & Ingestion
# =============================================================================


def _load_chunks(path: Path, max_records: int) -> list[dict[str, Any]]:
    """Load up to max_records from a JSONL chunk file."""
    if not path.exists():
        print(f"{_maybe_color('[ERROR]', C_RED)} Chunks file not found: {path}")
        print("  Run eda.py first to generate the file.")
        sys.exit(1)

    records: list[dict[str, Any]] = []
    print(f"Loading records from {path} ...")
    with open(str(path), "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="  Reading JSONL", unit=" records", colour="blue"):
            if len(records) >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip malformed lines
    print(f"  Loaded {len(records):,} records.\n")
    return records


def build_vector_db(rebuild: bool = False) -> bool:
    """Embed records and store in ChromaDB. Returns True if successful."""
    # ---- Check existing DB ----
    if CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir()):
        if not rebuild:
            print(f"{_maybe_color('[INFO]', C_CYAN)} Existing ChromaDB found at {CHROMA_DB_DIR}")
            return True
        else:
            import shutil
            print(f"{_maybe_color('[INFO]', C_CYAN)} Removing existing DB for rebuild ...")
            shutil.rmtree(str(CHROMA_DB_DIR))

    # ---- Load model ----
    _print_header("Module A: Embedding & Ingestion")
    print(f"Loading embedding model: {_maybe_color(MODEL_NAME, C_BOLD)}")
    print("  (first run will download ~2 GB model weights — this may take a few minutes)\n")

    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"{_maybe_color('[ERROR]', C_RED)} Failed to load model: {e}")
        print("  Check network connection. Model: https://huggingface.co/BAAI/bge-m3")
        return False

    print(f"  Model loaded. Dimension: {model.get_sentence_embedding_dimension()}")
    print(f"  Max sequence length: {model.max_seq_length}\n")

    # ---- Load chunks ----
    records = _load_chunks(CHUNKS_FILE, MAX_RECORDS)
    if not records:
        print(f"{_maybe_color('[ERROR]', C_RED)} No records to embed.")
        return False

    # ---- Prepare data for ChromaDB ----
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for rec in records:
        ids.append(rec["id"])
        documents.append(rec["content"])
        # ChromaDB metadata values must be str, int, float, or bool
        meta_clean = {
            "paper_title": str(rec["metadata"].get("paper_title", ""))[:200],
            "source": str(rec["metadata"].get("source", "")),
            "chunk_index": int(rec["metadata"].get("chunk_index", 0)),
            "word_count": int(rec["metadata"].get("word_count", 0)),
            "tags": ", ".join(rec["metadata"].get("tags", [])),
        }
        metadatas.append(meta_clean)

    # ---- Embed in batches ----
    print("Generating embeddings (bge-m3, 1024-dim) ...")
    BATCH_SIZE = 64
    all_embeddings: list[list[float]] = []

    for i in tqdm(range(0, len(documents), BATCH_SIZE),
                   desc="  Embedding", unit=" batch", colour="green"):
        batch = documents[i:i + BATCH_SIZE]
        # bge-m3: normalize for cosine similarity
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.extend(embeddings.tolist())

    print(f"  Generated {len(all_embeddings):,} embeddings.")

    # ---- Create ChromaDB & ingest ----
    print(f"\nCreating ChromaDB at {CHROMA_DB_DIR} ...")
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    # Drop + recreate if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Dropped existing collection.")
    except (ValueError, Exception):
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Security paper RAG chunks — bge-m3 embeddings"},
    )

    # Ingest in sub-batches (ChromaDB can handle large batches)
    print("  Ingesting into ChromaDB ...")
    INGEST_BATCH = 500
    for i in tqdm(range(0, len(ids), INGEST_BATCH),
                   desc="  Storing", unit=" records", colour="yellow"):
        slice_end = min(i + INGEST_BATCH, len(ids))
        collection.add(
            ids=ids[i:slice_end],
            embeddings=all_embeddings[i:slice_end],
            documents=documents[i:slice_end],
            metadatas=metadatas[i:slice_end],
        )

    print(f"\n{_maybe_color('[OK]', C_GREEN)} Vector DB built successfully!")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Records:    {collection.count():,}")
    print(f"  Location:   {CHROMA_DB_DIR.resolve()}")
    return True


# =============================================================================
# Module B: Interactive Retrieval QA
# =============================================================================


def run_qa_loop() -> None:
    """Load existing ChromaDB and launch interactive QA loop."""
    _print_header("Module B: Interactive Security QA")

    # ---- Load model ----
    print(f"Loading embedding model: {_maybe_color(MODEL_NAME, C_BOLD)} ...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"{_maybe_color('[ERROR]', C_RED)} Failed to load model: {e}")
        return

    # ---- Load ChromaDB ----
    if not CHROMA_DB_DIR.exists():
        print(f"{_maybe_color('[ERROR]', C_RED)} ChromaDB not found at {CHROMA_DB_DIR}")
        print("  Run build first: python vector_db_builder.py --rebuild")
        return

    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"{_maybe_color('[ERROR]', C_RED)} Collection '{COLLECTION_NAME}' not found.")
        print("  Run build first: python vector_db_builder.py --rebuild")
        return

    count = collection.count()
    print(f"Collection ready: {count:,} records in '{COLLECTION_NAME}'\n")

    # ---- Instructions ----
    print(_maybe_color("  " + "-" * 58, C_DIM))
    print(f"  {_maybe_color('How to use:', C_BOLD)}")
    print("    Type a security-related question (English or Chinese).")
    print("    The system retrieves the top-3 most relevant paper chunks.")
    print(f"    Type {_maybe_color('exit', C_YELLOW)} or {_maybe_color('quit', C_YELLOW)} to leave.")
    print(_maybe_color("  " + "-" * 58, C_DIM))
    print()

    # ---- QA Loop ----
    while True:
        try:
            query = input(_maybe_color("[Query] → ", C_BOLD + C_GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        # Embed query
        print(_maybe_color("  Retrieving ...", C_DIM), end="\r")
        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # Query ChromaDB (use cosine distance — bge-m3 normalized)
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        # Clear "Retrieving..." line
        print(" " * 40, end="\r")

        # Display results
        print()
        if not results["ids"] or not results["ids"][0]:
            print(_maybe_color("  No results found. Try a different query.", C_YELLOW))
            print()
            continue

        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            dist = results["distances"][0][i] if results["distances"] else 1.0
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            document = results["documents"][0][i] if results["documents"] else ""

            # Unpack tags stored as comma-separated string
            tags_str = metadata.get("tags", "")
            metadata["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]

            _print_chunk_result(i, dist, metadata, document)

        print(_maybe_color("  " + "-" * 58, C_DIM))
        print()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Vector DB Builder & MVP Retrieval QA"
    )
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild the ChromaDB from scratch.")
    parser.add_argument("--query-only", action="store_true",
                        help="Skip build, go directly to QA loop.")
    args = parser.parse_args()

    # ---- Banner ----
    print(_maybe_color(r"""
   _____                     _            ____
  / ___/___  _______  _______(_)___  _____/ __ \___  ____ _____ ____
  \__ \/ _ \/ ___/ / / / ___/ / __ \/ ___/ /_/ / _ \/ __ '/ __ `/ _ \
 ___/ /  __/ /__/ /_/ / /  / / /_/ / /  / ____/  __/ /_/ / /_/ /  __/
/____/\___/\___/\__,_/_/  /_/\__,_/_/  /_/    \___/\__,_/\__, /\___/
                      Vector DB Builder & MVP QA         /____/
""", C_CYAN))

    db_exists = CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir())

    # ---- Decide mode ----
    if args.query_only:
        build_ok = db_exists
    elif args.rebuild:
        build_ok = build_vector_db(rebuild=True)
    elif db_exists:
        # Interactive choice
        print(f"{_maybe_color('[INFO]', C_CYAN)} Existing ChromaDB detected.\n")
        try:
            choice = input(
                "  [R] Rebuild from scratch\n"
                "  [T] Test / QA  (use existing DB)\n"
                "  Choice [T]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            sys.exit(0)

        if choice in ("r", "rebuild"):
            build_ok = build_vector_db(rebuild=True)
        else:
            build_ok = True  # use existing
    else:
        # No DB — auto-build
        print(f"{_maybe_color('[INFO]', C_CYAN)} No existing DB found. Starting build ...")
        build_ok = build_vector_db(rebuild=False)

    if not build_ok:
        print(f"\n{_maybe_color('[ERROR]', C_RED)} DB build failed. Cannot start QA.")
        sys.exit(1)

    # ---- QA Loop ----
    run_qa_loop()


if __name__ == "__main__":
    main()

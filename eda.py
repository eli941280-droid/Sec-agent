"""
EDA + Full Batch Processing -- Security Paper RAG Pipeline.
Data: Hugging Face `clouditera/security-paper-datasets` (428,155 rows).

Features:
  - Bilingual (EN/ZH) taxonomy tagging with 16 security domains
  - Resumable checkpointing via output/progress.json
  - Multiprocessing with ProcessPoolExecutor for CPU-bound full runs

Usage:
  python eda.py                           # EDA on 2000-sample
  python eda.py --sample 500              # EDA on 500 papers
  python eda.py --full                    # Full dataset (resumable, multiprocess)
  python eda.py --full --workers 8        # Full with 8 worker processes
  python eda.py --sample 1000 --skip-full # EDA only, skip full processing
"""

from __future__ import annotations

import warnings
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14",
)

import argparse
import json
import os
import sys
import time
import textwrap
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless -- no GUI required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from datasets import load_dataset

from data_pipeline import (
    SecurityPaperPipeline,
    AcademicPaperCleaner,
    TagExtractor,
    compute_pipeline_stats,
    SECURITY_TAXONOMY,
)

# =============================================================================
# Config
# =============================================================================

OUTPUT_DIR = Path("output")
CHARTS_DIR = OUTPUT_DIR / "charts"
CHUNKS_FILE = OUTPUT_DIR / "rag_chunks.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# Matplotlib style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.titlesize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


# =============================================================================
# EDA result containers
# =============================================================================


@dataclass
class EDAResult:
    """Aggregated EDA statistics across all processed papers."""

    total_papers: int = 0
    total_raw_chars: int = 0
    total_raw_words: int = 0
    total_cleaned_chars: int = 0
    total_cleaned_words: int = 0
    total_chunks: int = 0
    empty_after_clean: int = 0

    hyphenation_fixes: int = 0
    ref_sections_stripped: int = 0

    raw_lengths: list[int] = field(default_factory=list)
    cleaned_lengths: list[int] = field(default_factory=list)
    chunks_per_paper: list[int] = field(default_factory=list)
    words_per_chunk: list[int] = field(default_factory=list)

    category_counts: Counter[str] = field(default_factory=Counter)

    tag_counter: Counter[str] = field(default_factory=Counter)
    papers_with_tags: int = 0
    papers_without_tags: int = 0

    elapsed_sec: float = 0.0


# =============================================================================
# Helper: derive title from raw text
# =============================================================================


def _derive_title(raw_text: str, category: str, fallback_idx: int) -> str:
    """Extract a readable title from the first meaningful line of text."""
    for line in raw_text.split("\n"):
        s = line.strip()
        if len(s) > 10 and not s.startswith(("http", "DOI", "©", "ISBN")):
            return s[:120]
    return f"{category}_{fallback_idx}" if category else f"paper_{fallback_idx}"


# =============================================================================
# Checkpoint helpers (progress.json)
# =============================================================================


def _load_progress() -> tuple[set[int], int]:
    """Return (set_of_processed_indices, total_chunks_so_far)."""
    if not PROGRESS_FILE.exists():
        return set(), 0
    try:
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        return set(data.get("processed_indices", [])), data.get("total_chunks", 0)
    except (json.JSONDecodeError, KeyError):
        return set(), 0


def _save_progress(processed: set[int], total_chunks: int) -> None:
    """Atomically write progress to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    data = {
        "processed_indices": sorted(processed),
        "total_chunks": total_chunks,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(PROGRESS_FILE)  # atomic on same filesystem


# =============================================================================
# Phase 1 -- Raw Data Exploration (before pipeline)
# =============================================================================


def explore_raw_dataset(ds, sample_size: int | None = None) -> dict[str, Any]:
    """Run descriptive statistics on the raw dataset before any processing."""
    total = len(ds)
    n = min(sample_size or total, total)

    raw_lengths: list[int] = []
    category_counts: Counter[str] = Counter()
    hyphenation_lines = 0
    ref_header_lines = 0
    empty_texts = 0
    noise_line_hits = 0

    hyphen_pat = AcademicPaperCleaner._HYPHEN_BREAK
    ref_pat = AcademicPaperCleaner._REF_HEADER
    noise_pat = AcademicPaperCleaner._NOISE_LINE

    for i in range(n):
        row = ds[i]
        text = row.get("text") or ""
        cat = row.get("category") or "unknown"
        category_counts[cat] += 1

        if not text.strip():
            empty_texts += 1
            raw_lengths.append(0)
            continue

        raw_lengths.append(len(text))
        hyphenation_lines += len(hyphen_pat.findall(text))

        for line in text.split("\n"):
            if ref_pat.match(line):
                ref_header_lines += 1
                break

        for line in text.split("\n"):
            if noise_pat.match(line.strip()):
                noise_line_hits += 1

    arr = np.array(raw_lengths)
    return {
        "dataset_total": total,
        "sample_size": n,
        "empty_texts": empty_texts,
        "category_counts": category_counts,
        "raw_lengths": arr,
        "raw_length_mean": float(arr.mean()) if len(arr) > 0 else 0,
        "raw_length_median": float(np.median(arr)) if len(arr) > 0 else 0,
        "raw_length_p5": float(np.percentile(arr, 5)) if len(arr) > 0 else 0,
        "raw_length_p95": float(np.percentile(arr, 95)) if len(arr) > 0 else 0,
        "hyphenation_breaks_total": hyphenation_lines,
        "ref_header_lines_total": ref_header_lines,
        "noise_line_hits_total": noise_line_hits,
    }


# =============================================================================
# Phase 2 -- Pipeline (single-threaded, for EDA sample)
# =============================================================================


def run_pipeline_sequential(
    ds,
    pipeline: SecurityPaperPipeline,
    indices: list[int],
    save_chunks: bool = False,
    chunks_path: Path | None = None,
) -> EDAResult:
    """Run pipeline on a list of indices (single-threaded, used for EDA sample)."""
    result = EDAResult()
    n = len(indices)
    result.total_papers = n
    t0 = time.perf_counter()

    if save_chunks:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(chunks_path or CHUNKS_FILE)

    for batch_i, idx in enumerate(indices):
        row = ds[idx]
        raw_text = row.get("text") or ""
        category = row.get("category") or "unknown"
        result.category_counts[category] += 1
        title = _derive_title(raw_text, category, idx)

        if not raw_text.strip():
            result.empty_after_clean += 1
            result.raw_lengths.append(0)
            result.cleaned_lengths.append(0)
            result.chunks_per_paper.append(0)
            continue

        result.raw_lengths.append(len(raw_text))
        result.total_raw_chars += len(raw_text)
        result.total_raw_words += len(raw_text.split())
        result.hyphenation_fixes += len(
            AcademicPaperCleaner._HYPHEN_BREAK.findall(raw_text)
        )

        for line in raw_text.split("\n"):
            if AcademicPaperCleaner._REF_HEADER.match(line):
                result.ref_sections_stripped += 1
                break

        records = pipeline.run(raw_text, paper_title=title)
        cleaned = pipeline.cleaner.clean(raw_text)

        result.cleaned_lengths.append(len(cleaned))
        result.total_cleaned_chars += len(cleaned)
        result.total_cleaned_words += sum(
            r["metadata"]["word_count"] for r in records
        )
        result.total_chunks += len(records)
        result.chunks_per_paper.append(len(records))

        for r in records:
            result.words_per_chunk.append(r["metadata"]["word_count"])
            for tag in r["metadata"]["tags"]:
                result.tag_counter[tag] += 1

        if any(r["metadata"]["tags"] for r in records):
            result.papers_with_tags += 1
        else:
            result.papers_without_tags += 1

        if save_chunks:
            with open(output_path, "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if (batch_i + 1) % 100 == 0 or batch_i == n - 1:
            pct = (batch_i + 1) / n * 100
            elapsed = time.perf_counter() - t0
            rate = (batch_i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - batch_i - 1) / rate if rate > 0 else 0
            print(f"\r  Processing: {batch_i + 1}/{n} ({pct:.0f}%) | "
                  f"{rate:.0f} papers/s | ETA {eta:.0f}s", end="", file=sys.stderr)

    result.elapsed_sec = time.perf_counter() - t0
    print(file=sys.stderr)
    return result


# =============================================================================
# Worker function -- runs in a subprocess (multiprocessing pool)
# =============================================================================
# Each worker loads the dataset independently (Arrow mmap -- memory is shared
# at the OS level). The worker builds its own SecurityPaperPipeline instance
# to avoid pickling issues with langchain objects on Windows (spawn mode).


def _process_batch_worker(
    batch_indices: list[int],
    chunk_size: int,
    chunk_overlap: int,
    strip_refs: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Worker function: process a batch of papers by index.

    Args:
        batch_indices: list of dataset indices to process.
        chunk_size / chunk_overlap / strip_refs: pipeline params.

    Returns:
        (chunk_records, batch_stats_dict)
        chunk_records: JSON-serializable list of {id, content, metadata}.
        batch_stats_dict: partial EDAResult fields for aggregation.
    """
    pipeline = SecurityPaperPipeline(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strip_references=strip_refs,
    )
    # Each worker loads dataset independently; Arrow mmap avoids data duplication
    ds = load_dataset("clouditera/security-paper-datasets", split="train")

    all_records: list[dict[str, Any]] = []
    stats = {
        "papers": 0,
        "raw_chars": 0,
        "raw_words": 0,
        "cleaned_chars": 0,
        "cleaned_words": 0,
        "chunks": 0,
        "empty_after_clean": 0,
        "hyphenation_fixes": 0,
        "ref_stripped": 0,
        "raw_lengths": [],
        "cleaned_lengths": [],
        "chunks_per_paper": [],
        "words_per_chunk": [],
        "category_counts": {},
        "tag_counter": {},
        "papers_with_tags": 0,
        "papers_without_tags": 0,
    }

    for idx in batch_indices:
        row = ds[idx]
        raw_text = row.get("text") or ""
        category = row.get("category") or "unknown"
        title = _derive_title(raw_text, category, idx)

        stats["category_counts"][category] = stats["category_counts"].get(category, 0) + 1
        stats["papers"] += 1

        if not raw_text.strip():
            stats["empty_after_clean"] += 1
            stats["raw_lengths"].append(0)
            stats["cleaned_lengths"].append(0)
            stats["chunks_per_paper"].append(0)
            continue

        stats["raw_lengths"].append(len(raw_text))
        stats["raw_chars"] += len(raw_text)
        stats["raw_words"] += len(raw_text.split())
        stats["hyphenation_fixes"] += len(
            AcademicPaperCleaner._HYPHEN_BREAK.findall(raw_text)
        )

        for line in raw_text.split("\n"):
            if AcademicPaperCleaner._REF_HEADER.match(line):
                stats["ref_stripped"] += 1
                break

        records = pipeline.run(raw_text, paper_title=title)
        cleaned = pipeline.cleaner.clean(raw_text)

        stats["cleaned_lengths"].append(len(cleaned))
        stats["cleaned_chars"] += len(cleaned)
        stats["cleaned_words"] += sum(r["metadata"]["word_count"] for r in records)
        stats["chunks"] += len(records)
        stats["chunks_per_paper"].append(len(records))

        for r in records:
            stats["words_per_chunk"].append(r["metadata"]["word_count"])
            for tag in r["metadata"]["tags"]:
                stats["tag_counter"][tag] = stats["tag_counter"].get(tag, 0) + 1
            all_records.append(r)

        if any(r["metadata"]["tags"] for r in records):
            stats["papers_with_tags"] += 1
        else:
            stats["papers_without_tags"] += 1

    return all_records, stats


# =============================================================================
# Phase 2b -- Multiprocessing Pipeline (used for --full)
# =============================================================================


def _merge_stats(target: EDAResult, batch_stats: dict[str, Any]) -> None:
    """Merge a batch_stats dict into an EDAResult (mutates target)."""
    target.total_papers += batch_stats["papers"]
    target.total_raw_chars += batch_stats["raw_chars"]
    target.total_raw_words += batch_stats["raw_words"]
    target.total_cleaned_chars += batch_stats["cleaned_chars"]
    target.total_cleaned_words += batch_stats["cleaned_words"]
    target.total_chunks += batch_stats["chunks"]
    target.empty_after_clean += batch_stats["empty_after_clean"]
    target.hyphenation_fixes += batch_stats["hyphenation_fixes"]
    target.ref_sections_stripped += batch_stats["ref_stripped"]
    target.raw_lengths.extend(batch_stats["raw_lengths"])
    target.cleaned_lengths.extend(batch_stats["cleaned_lengths"])
    target.chunks_per_paper.extend(batch_stats["chunks_per_paper"])
    target.words_per_chunk.extend(batch_stats["words_per_chunk"])
    for cat, cnt in batch_stats["category_counts"].items():
        target.category_counts[cat] += cnt
    for tag, cnt in batch_stats["tag_counter"].items():
        target.tag_counter[tag] += cnt
    target.papers_with_tags += batch_stats["papers_with_tags"]
    target.papers_without_tags += batch_stats["papers_without_tags"]


def run_pipeline_multiprocess(
    ds,
    chunk_size: int,
    chunk_overlap: int,
    strip_refs: bool,
    num_workers: int,
) -> EDAResult:
    """Run pipeline on the FULL dataset using multiprocessing with checkpointing.

    Architecture:
      - Main process loads progress, computes pending indices, splits into batches.
      - Worker processes each handle a batch: load dataset, run pipeline, return results.
      - Main process writes chunks (single-threaded I/O) and updates progress.json
        after each batch completes, enabling crash-safe resume.
    """
    total = len(ds)
    processed_set, total_chunks = _load_progress()

    # Compute pending indices
    all_indices = set(range(total))
    pending = sorted(all_indices - processed_set)

    if not pending:
        print(f"All {total:,} papers already processed. Nothing to do.")
        result = EDAResult()
        result.total_papers = total
        result.total_chunks = total_chunks
        return result

    skipped = total - len(pending)
    if skipped > 0:
        print(f"Resuming from checkpoint: {skipped:,} already processed, "
              f"{len(pending):,} remaining.")

    print(f"Starting multiprocessing with {num_workers} workers "
          f"on {len(pending):,} papers...")

    # Split pending into batches (~500 per batch for good granularity)
    BATCH_SIZE = 500
    batches: list[list[int]] = []
    for i in range(0, len(pending), BATCH_SIZE):
        batches.append(pending[i:i + BATCH_SIZE])

    print(f"  Batches: {len(batches)} (up to {BATCH_SIZE} papers each)")

    merged = EDAResult()
    merged.total_papers = skipped  # start from what was already done
    merged.total_chunks = total_chunks
    t0 = time.perf_counter()
    completed_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _process_batch_worker,
                batch,
                chunk_size,
                chunk_overlap,
                strip_refs,
            ): batch_idx
            for batch_idx, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            batch_idx = futures[future]
            batch_indices = batches[batch_idx]
            try:
                records, batch_stats = future.result()
            except Exception as exc:
                print(f"\n  [ERROR] Batch {batch_idx} (indices "
                      f"{batch_indices[0]}..{batch_indices[-1]}) failed: {exc}",
                      file=sys.stderr)
                continue

            # ---- Single-threaded I/O: write chunks to JSONL ----
            if records:
                with open(str(CHUNKS_FILE), "a", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # ---- Update checkpoint ----
            for idx in batch_indices:
                processed_set.add(idx)
            _save_progress(processed_set, merged.total_chunks + batch_stats["chunks"])

            # ---- Merge stats ----
            _merge_stats(merged, batch_stats)

            # ---- Progress ----
            completed_count += len(batch_indices)
            elapsed = time.perf_counter() - t0
            rate = completed_count / elapsed if elapsed > 0 else 0
            remaining = len(pending) - completed_count
            eta = remaining / rate if rate > 0 else 0
            print(f"\r  [{completed_count + skipped:,}/{total:,} | "
                  f"{rate:.0f} p/s | ETA {eta/60:.0f}m{eta%60:.0f}s]",
                  end="", file=sys.stderr)

    merged.elapsed_sec = time.perf_counter() - t0
    print(file=sys.stderr)
    return merged


# =============================================================================
# Phase 3 -- Charts
# =============================================================================


def generate_charts(raw_explore: dict, eda: EDAResult) -> None:
    """Generate PNG charts saved to output/charts/."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Chart 1: Text length distribution (raw vs cleaned) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    raw_lens = np.array(eda.raw_lengths)
    raw_lens = raw_lens[raw_lens > 0]
    if len(raw_lens) > 0:
        ax.hist(raw_lens, bins=80, color="#3182bd", alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(np.median(raw_lens), color="red", linestyle="--", linewidth=1.2,
                   label=f"Median: {np.median(raw_lens):,.0f} chars")
        ax.set_title("Raw Text Length Distribution")
        ax.set_xlabel("Characters")
        ax.set_ylabel("Papers")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax = axes[1]
    clean_lens = np.array(eda.cleaned_lengths)
    clean_lens = clean_lens[clean_lens > 0]
    if len(clean_lens) > 0:
        ax.hist(clean_lens, bins=80, color="#31a354", alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(np.median(clean_lens), color="red", linestyle="--", linewidth=1.2,
                   label=f"Median: {np.median(clean_lens):,.0f} chars")
        ax.set_title("Cleaned Text Length Distribution")
        ax.set_xlabel("Characters")
        ax.set_ylabel("Papers")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.suptitle("Effect of Cleaning Pipeline on Text Length", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "01_text_length_distribution.png")
    plt.close(fig)

    # ---- Chart 2: Chunks per paper ----
    fig, ax = plt.subplots(figsize=(10, 5))
    chunks_arr = np.array(eda.chunks_per_paper)
    chunks_arr = chunks_arr[chunks_arr > 0]
    if len(chunks_arr) > 0:
        ax.hist(chunks_arr, bins=50, color="#756bb1", alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(np.median(chunks_arr), color="red", linestyle="--", linewidth=1.2,
                   label=f"Median: {np.median(chunks_arr):.1f} chunks")
        ax.set_title("Chunks per Paper Distribution")
        ax.set_xlabel("Number of Chunks")
        ax.set_ylabel("Papers")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "02_chunks_per_paper.png")
    plt.close(fig)

    # ---- Chart 3: Words per chunk ----
    fig, ax = plt.subplots(figsize=(10, 5))
    wpc = np.array(eda.words_per_chunk)
    if len(wpc) > 0:
        ax.hist(wpc, bins=60, color="#de9e4f", alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(np.median(wpc), color="red", linestyle="--", linewidth=1.2,
                   label=f"Median: {np.median(wpc):.1f} words")
        ax.set_title("Words per Chunk Distribution")
        ax.set_xlabel("Words")
        ax.set_ylabel("Chunks")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "03_words_per_chunk.png")
    plt.close(fig)

    # ---- Chart 4: Category distribution ----
    fig, ax = plt.subplots(figsize=(10, 6))
    cats = eda.category_counts.most_common(20)
    if cats:
        names, counts = zip(*cats)
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))
        ax.barh(range(len(names)), counts, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title("Top 20 Categories in Dataset")
        ax.set_xlabel("Number of Papers")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "04_category_distribution.png")
    plt.close(fig)

    # ---- Chart 5: Tag distribution ----
    fig, ax = plt.subplots(figsize=(10, 7))
    tags = eda.tag_counter.most_common()
    if tags:
        names, counts = zip(*tags)
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(names)))
        ax.barh(range(len(names)), counts, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title("Security Domain Tag Distribution (all chunks)")
        ax.set_xlabel("Number of Chunks Tagged")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "05_tag_distribution.png")
    plt.close(fig)

    # ---- Chart 6: Reduction ratio scatter ----
    fig, ax = plt.subplots(figsize=(8, 5))
    raw_arr = np.array(eda.raw_lengths)
    clean_arr = np.array(eda.cleaned_lengths)
    mask = (raw_arr > 0) & (clean_arr > 0)
    raw_arr = raw_arr[mask]
    clean_arr = clean_arr[mask]
    if len(raw_arr) > 0:
        reductions = (1 - clean_arr / raw_arr) * 100
        sample = np.random.choice(len(reductions), min(2000, len(reductions)), replace=False)
        ax.scatter(raw_arr[sample], reductions[sample], alpha=0.3, s=8,
                   c="#3182bd", edgecolors="none")
        ax.axhline(np.median(reductions), color="red", linestyle="--", linewidth=1.2,
                   label=f"Median reduction: {np.median(reductions):.1f}%")
        ax.set_xlabel("Raw Text Length (chars)")
        ax.set_ylabel("Reduction After Cleaning (%)")
        ax.set_title("Cleaning Reduction vs. Raw Text Length")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "06_cleaning_reduction.png")
    plt.close(fig)


# =============================================================================
# Phase 4 -- Console Report
# =============================================================================


def print_report(raw_explore: dict, eda: EDAResult,
                 chunks_path: Path | None = None) -> None:
    """Print a formatted EDA report to stdout."""
    sep = "=" * 72
    report_chunks = chunks_path or CHUNKS_FILE

    print(f"\n{sep}")
    print("  SECURITY PAPER RAG PIPELINE -- EDA REPORT")
    print(sep)

    # 4.1 Dataset overview
    print(f"\n{'-' * 60}")
    print("  1. DATASET OVERVIEW")
    print(f"{'-' * 60}")
    print(f"  Total rows in HF dataset : {raw_explore['dataset_total']:,}")
    print(f"  Papers processed (EDA)   : {eda.total_papers:,}")
    print(f"  Empty texts              : {raw_explore['empty_texts']} "
          f"({raw_explore['empty_texts']/max(raw_explore['sample_size'],1)*100:.2f}%)")
    cats_count = len(raw_explore['category_counts'])
    print(f"  Unique categories        : {cats_count}")

    print(f"\n  Raw text length stats:")
    print(f"    Mean   : {raw_explore['raw_length_mean']:,.0f} chars")
    print(f"    Median : {raw_explore['raw_length_median']:,.0f} chars")
    print(f"    P5     : {raw_explore['raw_length_p5']:,.0f} chars")
    print(f"    P95    : {raw_explore['raw_length_p95']:,.0f} chars")

    # 4.2 Cleaning diagnostics
    print(f"\n{'-' * 60}")
    print("  2. CLEANING DIAGNOSTICS")
    print(f"{'-' * 60}")
    print(f"  Hyphenation breaks found : {raw_explore['hyphenation_breaks_total']:,}")
    pct_ref = raw_explore['ref_header_lines_total'] / max(raw_explore['sample_size'], 1) * 100
    print(f"  Reference sections found : {raw_explore['ref_header_lines_total']:,} "
          f"({pct_ref:.1f}% of sample)")
    print(f"  Noise symbol lines       : {raw_explore['noise_line_hits_total']:,}")
    print(f"  Papers empty after clean : {eda.empty_after_clean}")

    char_red = (1 - eda.total_cleaned_chars / max(eda.total_raw_chars, 1)) * 100
    print(f"\n  Aggregate reduction:")
    print(f"    Raw chars           : {eda.total_raw_chars:,}")
    print(f"    Cleaned chars       : {eda.total_cleaned_chars:,}  ({char_red:.1f}% reduction)")
    print(f"    Raw words           : {eda.total_raw_words:,}")
    print(f"    Chunk total words   : {eda.total_cleaned_words:,}  (incl. overlap)")

    # 4.3 Chunking stats
    print(f"\n{'-' * 60}")
    print("  3. CHUNKING STATISTICS")
    print(f"{'-' * 60}")
    print(f"  Total chunks produced    : {eda.total_chunks:,}")
    avg_cpp = eda.total_chunks / max(eda.total_papers, 1)
    print(f"  Avg chunks per paper     : {avg_cpp:.1f}")
    chunks_arr = np.array(eda.chunks_per_paper)
    chunks_arr = chunks_arr[chunks_arr > 0]
    if len(chunks_arr) > 0:
        print(f"  Median chunks per paper  : {np.median(chunks_arr):.1f}")
    wpc = np.array(eda.words_per_chunk)
    if len(wpc) > 0:
        print(f"  Avg words per chunk      : {wpc.mean():.1f}")
        print(f"  Median words per chunk   : {np.median(wpc):.1f}")
        print(f"  P5 / P95 words per chunk : {np.percentile(wpc,5):.0f} "
              f"/ {np.percentile(wpc,95):.0f}")

    # 4.4 Tag analytics
    print(f"\n{'-' * 60}")
    print("  4. TAG ANALYTICS")
    print(f"{'-' * 60}")
    tag_rate = eda.papers_with_tags / max(eda.total_papers, 1) * 100
    no_tag_rate = eda.papers_without_tags / max(eda.total_papers, 1) * 100
    print(f"  Papers with tags    : {eda.papers_with_tags} ({tag_rate:.1f}%)")
    print(f"  Papers without tags : {eda.papers_without_tags} ({no_tag_rate:.1f}%)")
    print(f"  Unique tags matched : {len(eda.tag_counter)}")
    if eda.tag_counter:
        print(f"\n  Top 16 tags:")
        top_val = eda.tag_counter.most_common(1)[0][1]
        for tag, count in eda.tag_counter.most_common(16):
            bar = "#" * int(count / max(top_val, 1) * 30)
            print(f"    {tag:<42s} {count:>6,}  {bar}")

    # 4.5 Taxonomy coverage
    print(f"\n{'-' * 60}")
    print("  5. TAXONOMY COVERAGE (16 categories)")
    print(f"{'-' * 60}")
    for tag in SECURITY_TAXONOMY:
        hits = eda.tag_counter.get(tag, 0)
        status = "[Y]" if hits > 0 else "[N]"
        print(f"    [{status}] {tag:<42s} {hits:>8,} chunks")

    # 4.6 Timing
    print(f"\n{'-' * 60}")
    print("  6. RUNTIME")
    print(f"{'-' * 60}")
    print(f"  Processing time : {eda.elapsed_sec:.1f}s")
    rate = eda.total_papers / eda.elapsed_sec if eda.elapsed_sec > 0 else 0
    print(f"  Throughput      : {rate:.1f} papers/s")
    print(f"  Output chunks   : {report_chunks}")

    # 4.7 Example records
    print(f"\n{'-' * 60}")
    print("  7. SAMPLE OUTPUT RECORDS (first 3)")
    print(f"{'-' * 60}")
    if report_chunks.exists():
        with open(str(report_chunks), "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                rec = json.loads(line)
                content_preview = textwrap.shorten(
                    rec["content"], width=120, placeholder="..."
                )
                title = rec['metadata']['paper_title'][:60]
                print(f"\n  [{i}] id={rec['id'][:8]}... | title={title}...")
                print(f"      words={rec['metadata']['word_count']} | "
                      f"tags={rec['metadata']['tags']}")
                print(f"      content: {content_preview}")

    print(f"\n{sep}")
    print(f"  Charts saved to: {CHARTS_DIR}/")
    print(f"  Chunks saved to: {report_chunks}")
    print(sep)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Security Paper RAG Pipeline -- EDA + Batch Processing"
    )
    parser.add_argument("--sample", type=int, default=2000,
                        help="Number of papers for EDA sampling (default: 2000)")
    parser.add_argument("--full", action="store_true",
                        help="Process the ENTIRE dataset (428,155 papers) with multiprocessing.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes for --full (default: cpu_count).")
    parser.add_argument("--skip-full", action="store_true",
                        help="Skip full processing, only run EDA on sample.")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--no-strip-refs", action="store_true",
                        help="Keep reference sections.")
    parser.add_argument("--reset", action="store_true",
                        help="Clear checkpoint and chunks, start fresh.")
    args = parser.parse_args()

    # ---- Reset ----
    if args.reset:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        if CHUNKS_FILE.exists():
            CHUNKS_FILE.unlink()
        print("Checkpoint and chunks reset.")

    print("Loading dataset from HuggingFace...")
    ds = load_dataset("clouditera/security-paper-datasets", split="train")
    print(f"Dataset loaded: {len(ds):,} rows | columns: {ds.column_names}")

    # ---- Phase 1: Raw data exploration ----
    print(f"\n{'='*72}")
    print("  PHASE 1 -- Raw Data Exploration")
    print(f"{'='*72}")
    raw_explore = explore_raw_dataset(ds, sample_size=args.sample)
    print(f"  Sample: {raw_explore['sample_size']} papers")
    print(f"  Mean length: {raw_explore['raw_length_mean']:,.0f} chars")
    print(f"  Median length: {raw_explore['raw_length_median']:,.0f} chars")
    print(f"  Hyphenation breaks: {raw_explore['hyphenation_breaks_total']:,}")
    print(f"  Empty texts: {raw_explore['empty_texts']}")

    strip_refs = not args.no_strip_refs

    # ---- Phase 2: Run pipeline on sample (for EDA stats + charts) ----
    print(f"\n{'='*72}")
    print("  PHASE 2 -- Pipeline Processing (EDA sample, single-threaded)")
    print(f"{'='*72}")

    pipeline = SecurityPaperPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        strip_references=strip_refs,
    )

    # For the EDA sample, write to a separate file to avoid conflating
    # with full-run chunks. The full multiprocessing run uses CHUNKS_FILE.
    sample_chunks_file = OUTPUT_DIR / "rag_chunks_sample.jsonl"
    if sample_chunks_file.exists():
        sample_chunks_file.unlink()

    sample_indices = list(range(args.sample))
    eda = run_pipeline_sequential(
        ds, pipeline,
        indices=sample_indices,
        save_chunks=True,
        chunks_path=sample_chunks_file,
    )

    # ---- Phase 3: Charts ----
    print(f"\n{'='*72}")
    print("  PHASE 3 -- Generating Charts")
    print(f"{'='*72}")
    generate_charts(raw_explore, eda)
    print(f"  Charts saved to {CHARTS_DIR}/")

    # ---- Phase 4: Report (uses sample chunks for example records) ----
    print_report(raw_explore, eda, sample_chunks_file)

    # ---- Phase 5: Full processing (optional, multiprocess) ----
    if args.full and not args.skip_full:
        print(f"\n{'='*72}")
        print("  PHASE 5 -- Full Dataset Processing (multiprocessing)")
        print(f"{'='*72}")

        num_workers = args.workers or os.cpu_count() or 4
        print(f"  Total papers: {len(ds):,}")
        print(f"  Workers: {num_workers}")
        print(f"  Progress on stderr.", file=sys.stderr)

        full_eda = run_pipeline_multiprocess(
            ds,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            strip_refs=strip_refs,
            num_workers=num_workers,
        )
        print(f"\n  Full processing done: {full_eda.total_chunks:,} chunks "
              f"in {full_eda.elapsed_sec:.0f}s "
              f"({full_eda.total_papers / full_eda.elapsed_sec:.0f} papers/s)")


if __name__ == "__main__":
    main()

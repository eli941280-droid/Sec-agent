import json, os, sys, time, math, uuid, warnings, traceback
from collections import defaultdict
from pathlib import Path

# Force UTF-8 stdout to avoid UnicodeEncodeError on Windows GBK consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# -- Load .env --
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

if not DASHSCOPE_API_KEY:
    raise RuntimeError("Missing DASHSCOPE_API_KEY in .env file at project root")

# -- Global Config --
ROOT_DIR = Path(__file__).resolve().parent
CHROMA_PATH = ROOT_DIR / "output" / "chroma_db"
JSONL_PATH = ROOT_DIR / "output" / "rag_chunks_sample.jsonl"
COLLECTION_NAME = "security_papers"
MAX_SAMPLE = 8000          # HDBSCAN sampling upper bound (balance memory & compute)
TOP_K = 3                  # RAG retrieval window size
LLM_MODEL = "qwen-plus"    # Bailian model
EMBED_MODEL = "BAAI/bge-m3"  # Local embedding model
MIN_CLUSTER = 5            # HDBSCAN min_cluster_size
BATCH_SLEEP = 0.3          # API call interval (seconds)

warnings.filterwarnings("ignore")



def create_openai_client():
    """Create OpenAI-compatible client targeting Alibaba Bailian (SDK >= 1.0)."""
    from openai import OpenAI
    return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=BASE_URL)


def llm_chat(client, messages: list, use_json: bool = False, max_tokens: int = 1024) -> str:
    """Generic LLM call wrapper with optional JSON mode."""
    kwargs = dict(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=max_tokens,
    )
    if use_json:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def emit_line(char: str = "=", width: int = 66) -> None:
    print(char * width)


class ModuleA:

    def __init__(self):
        self.client = create_openai_client()
        self.embed_model = None
        self.collection = None

    # -- Vector DB Loading --

    def load_or_build_collection(self):
        """Load existing chroma_db collection, or build from JSONL."""
        import chromadb
        from chromadb.config import Settings

        chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False),
        )

        existing = [c.name for c in chroma_client.list_collections()]
        if COLLECTION_NAME in existing:
            print(f"  [LOAD] Found existing collection: {COLLECTION_NAME}")
            self.collection = chroma_client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=None,
            )
        else:
            print(f"  [BUILD] Collection '{COLLECTION_NAME}' not found; building from JSONL ...")
            self.collection = chroma_client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"},
            )
            self._ingest_jsonl()

    def _ingest_jsonl(self):
        """Batch-embed JSONL chunks and write into ChromaDB."""
        if not JSONL_PATH.exists():
            raise FileNotFoundError(f"JSONL data file not found: {JSONL_PATH}")

        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer(EMBED_MODEL)

        records = []
        with open(JSONL_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))

        batch_size = 64
        total = len(records)
        for start in range(0, total, batch_size):
            batch = records[start:start + batch_size]
            ids = [r["id"] for r in batch]
            contents = [r["content"] for r in batch]
            metas = [r.get("metadata", {}) for r in batch]
            embs = self.embed_model.encode(
                contents, batch_size=batch_size, show_progress_bar=False
            ).tolist()

            self.collection.add(ids=ids, embeddings=embs, documents=contents, metadatas=metas)
            print(f"\r  [..] Ingest progress: {min(start + batch_size, total)}/{total}", end="")
        print()

    # -- HDBSCAN Clustering --

    def run_clustering(self, n_samples: int = MAX_SAMPLE) -> tuple:
        """
        Returns: (embeddings_np, labels, cluster_centroids, valid_cluster_ids)
          - embeddings_np: (N, dim) all sampled vectors
          - labels: (N,) cluster labels (-1 = noise)
          - cluster_centroids: dict {cid: centroid_vector}
          - valid_cluster_ids: list of cid
        """
        from sklearn.cluster import HDBSCAN
        from sklearn.preprocessing import normalize

        print(f"\n  [SAMPLE] Fetching up to {n_samples} embeddings from ChromaDB ...")
        all_data = self.collection.get(
            limit=n_samples, include=["embeddings", "metadatas", "documents"]
        )
        embs = np.array(all_data["embeddings"], dtype=np.float32)
        docs = all_data["documents"]
        ids = all_data["ids"]
        n_total = len(embs)
        print(f"  [DATA] Actual count: {n_total} vectors, dim: {embs.shape[1]}")

        embs_norm = normalize(embs, norm="l2")

        print(f"  [CLUSTER] HDBSCAN density clustering (min_cluster_size={MIN_CLUSTER}) ...")
        clusterer = HDBSCAN(
            min_cluster_size=MIN_CLUSTER,
            metric="euclidean",
            cluster_selection_epsilon=0.05,
            min_samples=2,
        )
        labels = clusterer.fit_predict(embs_norm)

        unique_labels = set(labels)
        n_noise = int(np.sum(labels == -1))
        valid_clusters = sorted([int(c) for c in unique_labels if c >= 0])
        K = len(valid_clusters)

        print(f"  [DONE] Clustering complete: K={K} valid clusters, noise={n_noise} "
              f"({n_noise / n_total * 100:.1f}%)")

        centroids = {}
        for cid in valid_clusters:
            mask = labels == cid
            centroids[cid] = embs_norm[mask].mean(axis=0)

        self._cluster_meta = {
            "all_ids": ids,
            "all_docs": docs,
            "all_embs_norm": embs_norm,
            "labels": labels,
        }
        return embs_norm, labels, centroids, valid_clusters


    def reverse_generate_questions(self, centroids: dict, valid_clusters: list) -> list:
        """
        For each cluster: find nearest chunk to centroid -> LLM generates question.
        Returns: gold_testset: list[dict] with question/ground_truth/eval_type/source_chunk_id/cluster_id
        """
        meta = self._cluster_meta
        all_ids = meta["all_ids"]
        all_docs = meta["all_docs"]
        all_embs = meta["all_embs_norm"]
        labels = meta["labels"]

        gold_testset = []
        prompt_tpl = (
            "You are a senior cybersecurity exam designer. Based on the following security literature excerpt, "
            "generate a high-quality evaluation question.\n\n"
            "## Source Excerpt\n{chunk}\n\n"
            "## Requirements\n"
            "1. `question`: A question targeting the core knowledge point of the excerpt. "
            "The answer must have clear support in the text.\n"
            "2. `ground_truth`: The standard answer, concise and accurate (max 150 words).\n"
            "3. `eval_type`: Use \"exact_match\" if the answer involves precise identifiers "
            "(CVE numbers, tool names, commands, version numbers, etc.). "
            "Use \"semantic\" if the answer requires understanding and summarization.\n\n"
            "Output STRICT JSON only, no extra text. Format:\n"
            '{{"question": "...", "ground_truth": "...", "eval_type": "exact_match|semantic"}}'
        )

        for cid in valid_clusters:
            mask = labels == cid
            indices = np.where(mask)[0]
            centroid = centroids[cid]

            # Find nearest chunk by Euclidean distance
            cluster_vecs = all_embs[indices]
            dists = np.linalg.norm(cluster_vecs - centroid, axis=1)
            best_local_idx = int(np.argmin(dists))
            best_global_idx = indices[best_local_idx]

            chunk_id = all_ids[best_global_idx]
            chunk_text = all_docs[best_global_idx]
            print(f"\n  [CENTROID] Cluster[{cid}] (size={len(indices)}) -> "
                  f"chunk: {chunk_id[:12]}... (dist={dists[best_local_idx]:.4f})")

            # LLM question generation
            try:
                raw = llm_chat(
                    self.client,
                    messages=[{"role": "user", "content": prompt_tpl.format(chunk=chunk_text[:2800])}],
                    use_json=True,
                    max_tokens=800,
                )
                q_data = json.loads(raw)
                for k in ("question", "ground_truth", "eval_type"):
                    if k not in q_data:
                        raise KeyError(f"Missing field: {k}")

                item = {
                    "question": q_data["question"].strip(),
                    "ground_truth": q_data["ground_truth"].strip(),
                    "eval_type": q_data["eval_type"].strip(),
                    "source_chunk_id": chunk_id,
                    "cluster_id": cid,
                    "cluster_size": int(len(indices)),
                }
                gold_testset.append(item)
                print(f"     [Q] {item['question'][:80]}...")
                print(f"     [T] eval_type={item['eval_type']}")

            except (json.JSONDecodeError, KeyError, Exception) as e:
                print(f"     [SKIP] Cluster[{cid}] question generation failed: "
                      f"{type(e).__name__}: {e}. Skipping this cluster.")
                continue

            time.sleep(BATCH_SLEEP)

        return gold_testset


class ModuleB:
    """
    For each gold question, execute two inference tracks:
      Track 1: Ask LLM directly (no context) -> answer_pre
      Track 2: Retrieve Top-K from ChromaDB -> assemble prompt -> answer_post
    Also record cluster_hit.
    """

    def __init__(self, collection, gold_testset: list, ckpt_path: Path = None):
        self.client = create_openai_client()
        self.collection = collection
        self.gold = gold_testset
        self.ckpt_path = ckpt_path

        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer(EMBED_MODEL)

    def run(self) -> list:
        # Resume from checkpoint if exists
        if self.ckpt_path and self.ckpt_path.exists():
            with open(self.ckpt_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            done = len(results)
            print(f"  [RESUME] Loaded {done} existing results from {self.ckpt_path.name}, "
                  f"continuing from question {done + 1}/{len(self.gold)}")
        else:
            results = []
            done = 0

        total = len(self.gold)

        for i in range(done, total):
            item = self.gold[i]
            q = item["question"]
            source_id = item["source_chunk_id"]

            print(f"\n  -- [{i + 1}/{total}] Q: {q[:60]}...")

            try:
                ans_pre = llm_chat(
                    self.client,
                    messages=[{"role": "user", "content": q}],
                    use_json=False,
                    max_tokens=512,
                )
            except Exception as e:
                print(f"     [WARN] Pre-RAG inference failed: {e}")
                ans_pre = "[ERROR]"

            q_emb = self.embed_model.encode([q]).tolist()
            retrieved = self.collection.query(
                query_embeddings=q_emb,
                n_results=TOP_K,
                include=["documents", "metadatas"],
            )
            retrieved_ids = retrieved["ids"][0]
            retrieved_docs = retrieved["documents"][0]

            # --- Cluster Hit ---
            cluster_hit = 1 if source_id in retrieved_ids else 0

            # --- Track 2: Post-RAG ---
            context_str = ""
            for j, (rid, rdoc) in enumerate(zip(retrieved_ids, retrieved_docs)):
                context_str += f"\n[Reference {j + 1}] (ID={rid[:12]}...)\n{rdoc[:800]}\n"

            rag_prompt = (
                "You are a rigorous cybersecurity expert. Answer the user's question "
                "based on the reference materials below. If the references are insufficient, "
                "state so honestly.\n\n"
                f"## Reference Materials\n{context_str}\n"
                f"## User Question\n{q}\n\n"
                "Provide a direct answer without restating the question."
            )

            try:
                ans_post = llm_chat(
                    self.client,
                    messages=[{"role": "user", "content": rag_prompt}],
                    use_json=False,
                    max_tokens=512,
                )
            except Exception as e:
                print(f"     [WARN] Post-RAG inference failed: {e}")
                ans_post = "[ERROR]"

            entry = {
                **item,
                "answer_pre": ans_pre,
                "answer_post": ans_post,
                "retrieved_ids": retrieved_ids,
                "retrieved_docs": retrieved_docs,
                "cluster_hit": cluster_hit,
            }
            results.append(entry)
            print(f"     [HIT] cluster_hit={cluster_hit} | "
                  f"pre_len={len(ans_pre)} | post_len={len(ans_post)}")

            # Incremental save to checkpoint
            if self.ckpt_path:
                with open(self.ckpt_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"     [SAVE] Checkpoint updated: {len(results)}/{total}")

            time.sleep(BATCH_SLEEP)

        return results


class ModuleC:
    """
    1. Exact Match: for eval_type=="exact_match" questions,
       check if answer string contains the ground_truth substring.
    2. RAGAS semantic evaluation: context_precision / faithfulness / answer_relevancy
    """

    def __init__(self, results: list):
        self.results = results
        self.client = create_openai_client()


    def compute_exact_match(self) -> dict:
        em_items = [r for r in self.results if r["eval_type"] == "exact_match"]
        if not em_items:
            return {"count": 0, "pre_correct": 0, "post_correct": 0,
                    "pre_rate": 0.0, "post_rate": 0.0}

        pre_correct = 0
        post_correct = 0
        for r in em_items:
            gt = r["ground_truth"].lower().strip()
            if gt in r["answer_pre"].lower():
                pre_correct += 1
            if gt in r["answer_post"].lower():
                post_correct += 1

        n = len(em_items)
        return {
            "count": n,
            "pre_correct": pre_correct,
            "post_correct": post_correct,
            "pre_rate": pre_correct / n,
            "post_rate": post_correct / n,
        }


    def compute_ragas(self) -> dict:
        """
        Assemble RAGAS Dataset -> evaluate -> return mean of 3 metrics.
        ragas 0.4.x API: instantiate metric classes with llm kwarg.
        """
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI

        # RAGAS expects: question, answer, contexts, ground_truth
        records = []
        for r in self.results:
            records.append({
                "question": r["question"],
                "answer": r["answer_post"],
                "contexts": r["retrieved_docs"],
                "ground_truth": r["ground_truth"],
            })

        if not records:
            print("  [WARN] No valid data for RAGAS evaluation")
            return {}

        ds = Dataset.from_list(records)

        eval_llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=DASHSCOPE_API_KEY,
            base_url=BASE_URL,
            temperature=0.0,
        )
        wrapped_llm = LangchainLLMWrapper(eval_llm)
        metrics = [
            ContextPrecision(llm=wrapped_llm),
            Faithfulness(llm=wrapped_llm),
            AnswerRelevancy(llm=wrapped_llm),
        ]

        try:
            result = evaluate(ds, metrics=metrics)
            scores = result.to_pandas().to_dict(orient="list")
            avg_scores = {}
            for k, vals in scores.items():
                try:
                    avg_scores[k] = float(np.nanmean([v for v in vals if v is not None]))
                except Exception:
                    avg_scores[k] = None
            return avg_scores
        except Exception:
            print(f"  [WARN] RAGAS evaluate failed: {traceback.format_exc()}")
            return {"error": traceback.format_exc()}

    # -- Cluster Hit Summary --

    def compute_cluster_hit(self) -> dict:
        hits = [r["cluster_hit"] for r in self.results]
        total = len(hits)
        hit_count = sum(hits)
        return {
            "total_questions": total,
            "cluster_hits": hit_count,
            "cluster_hit_rate": hit_count / total if total > 0 else 0.0,
        }

class ModuleD:
    """Terminal report (structured) + JSON dump to disk."""

    def __init__(self, gold_testset, results, em_scores, ragas_scores, cluster_hit):
        self.gold = gold_testset
        self.results = results
        self.em = em_scores
        self.ragas = ragas_scores
        self.chit = cluster_hit

    def terminal_report(self):
        emit_line("=")
        print("  ***  RAG Full-Auto Quantitative Evaluation -- Scorecard  ***")
        emit_line("=")

        K = len(set(r["cluster_id"] for r in self.results))
        n_em = self.em.get("count", 0)
        n_sem = len(self.results) - n_em

        print(f"  [STAT] Dynamic clusters: {K}   |   Total questions: {len(self.results)}")
        print(f"  [STAT] Exact Match questions: {n_em}   |   Semantic questions: {n_sem}")
        emit_line()

        # -- Cluster Hit --
        print(f"  [CLUSTER-HIT] Knowledge Cluster Hit Rate")
        print(f"     Hits: {self.chit['cluster_hits']}/{self.chit['total_questions']} "
              f"  ->  {self.chit['cluster_hit_rate']:.2%}")
        emit_line()

        # -- Exact Match --
        print(f"  [EXACT-MATCH] Objective Fact Accuracy")
        print(f"     Pre-RAG  accuracy:  {self.em['pre_rate']:.2%}  "
              f"({self.em['pre_correct']}/{self.em['count']})")
        print(f"     Post-RAG accuracy:  {self.em['post_rate']:.2%}  "
              f"({self.em['post_correct']}/{self.em['count']})")
        delta_em = self.em['post_rate'] - self.em['pre_rate']
        print(f"     Delta (Post - Pre): {delta_em:+.2%}")
        emit_line()

        # -- RAGAS Semantic --
        if self.ragas and "error" not in self.ragas:
            print(f"  [RAGAS] Semantic Judge Scores (Post-RAG)")
            for metric, val in self.ragas.items():
                if val is not None:
                    print(f"     {metric:25s}: {val:.4f}")
                else:
                    print(f"     {metric:25s}: N/A")
        elif self.ragas.get("error"):
            print(f"  [WARN] RAGAS error: {self.ragas['error'][:100]}")
        emit_line("=")

    def save_json(self, path: Path = None):
        if path is None:
            path = ROOT_DIR / "eval_final_report.json"

        report = {
            "meta": {
                "model": LLM_MODEL,
                "embed_model": EMBED_MODEL,
                "top_k": TOP_K,
                "total_questions": len(self.results),
                "total_clusters": len(set(r["cluster_id"] for r in self.results)),
            },
            "cluster_hit": self.chit,
            "exact_match": self.em,
            "ragas_scores": self.ragas,
            "details": [],
        }

        for r in self.results:
            report["details"].append({
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "eval_type": r["eval_type"],
                "cluster_id": r["cluster_id"],
                "source_chunk_id": r["source_chunk_id"],
                "answer_pre": r["answer_pre"],
                "answer_post": r["answer_post"],
                "retrieved_ids": r["retrieved_ids"],
                "cluster_hit": r["cluster_hit"],
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  [SAVE] Full evaluation report saved to: {path}")


def main():
    # Checkpoint files
    ckpt_testset = ROOT_DIR / "eval_checkpoint_testset.json"
    ckpt_results = ROOT_DIR / "eval_checkpoint_results.json"

    emit_line("=")
    print("  ***  RAG Full-Auto Quantitative Evaluation Pipeline  ***")
    emit_line("=")
    print(f"  LLM: {LLM_MODEL}  |  Embed: {EMBED_MODEL}  |  Top-K: {TOP_K}")
    print(f"  BaseURL: {BASE_URL}")
    emit_line()

    if ckpt_testset.exists():
        print("\n[MODULE A] Loading gold testset from checkpoint ...")
        with open(ckpt_testset, "r", encoding="utf-8") as f:
            gold_testset = json.load(f)
        print(f"  [LOAD] Loaded {len(gold_testset)} questions from {ckpt_testset.name}")
    else:
        print("\n[MODULE A] Dynamic Knowledge Cluster Detection & Reverse Question Generation")
        emit_line()
        mod_a = ModuleA()
        mod_a.load_or_build_collection()
        embs_norm, labels, centroids, valid_clusters = mod_a.run_clustering(n_samples=MAX_SAMPLE)

        if len(valid_clusters) < 2:
            print(f"  [ABORT] Too few valid clusters ({len(valid_clusters)}). "
                  f"Try lowering min_cluster_size or increasing data. Exiting.")
            return

        gold_testset = mod_a.reverse_generate_questions(centroids, valid_clusters)
        print(f"\n  [DONE] Gold testset generated: {len(gold_testset)} questions")

        if len(gold_testset) == 0:
            print("  [ABORT] No valid questions generated. Exiting pipeline.")
            return

        # Save checkpoint
        with open(ckpt_testset, "w", encoding="utf-8") as f:
            json.dump(gold_testset, f, indent=2, ensure_ascii=False)
        print(f"  [SAVE] Checkpoint saved: {ckpt_testset.name}")

    import chromadb
    from chromadb.config import Settings
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=None)

    print("\n[MODULE B] Dual-Track Inference Collection (Pre-RAG vs Post-RAG)")
    emit_line()
    mod_b = ModuleB(collection, gold_testset, ckpt_path=ckpt_results)
    results = mod_b.run()
    print(f"\n  [DONE] Inference complete: {len(results)} results, checkpoint at {ckpt_results.name}")
    print("\n[MODULE C] Quantitative Scoring (Exact Match + RAGAS)")
    emit_line()
    mod_c = ModuleC(results)

    print("  [..] Computing Exact Match ...")
    em_scores = mod_c.compute_exact_match()

    print("  [..] Running RAGAS semantic judge (may take several minutes) ...")
    ragas_scores = mod_c.compute_ragas()

    cluster_hit = mod_c.compute_cluster_hit()

    # ---- D: Report ----
    print("\n[MODULE D] Report Output")
    mod_d = ModuleD(gold_testset, results, em_scores, ragas_scores, cluster_hit)
    mod_d.terminal_report()
    mod_d.save_json()

    print("\n[DONE] Evaluation pipeline finished successfully.")


if __name__ == "__main__":
    main()

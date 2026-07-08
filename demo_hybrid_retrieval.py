"""
Hybrid-retrieval demo — shows FAISS (semantic) vs BM25 (keyword) vs RRF-fused,
side by side, on the live index. Reuses the real RAGEngine helpers so it matches
production exactly. Good for explaining WHY hybrid retrieval beats either half.

Prerequisite: a built FAISS index (upload a document or load scans in the app
first). If none exists, the script prints setup instructions instead of crashing.

Run (from the DocMindAI folder, in its venv):
    .venv/Scripts/python demo_hybrid_retrieval.py "how long until the fruit goes off?"
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows-console safe
except Exception:
    pass
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from rag_engine import RAGEngine, EMBED_MODEL, FAISS_DIR   # real config + helpers

TOPN = 5      # rows shown per method
CAND = 8      # candidates pulled from each method before fusion


def _key(doc):
    md = doc.metadata or {}
    return (md.get("doc_id", ""), md.get("chunk_index"))


def _label(doc):
    md = doc.metadata or {}
    did = str(md.get("doc_id", "?"))
    did = (did[:12] + "..") if len(did) > 13 else did
    snippet = " ".join(doc.page_content.split())[:50]
    return f"{did}#{md.get('chunk_index', '?')}", snippet


def _show(title, docs):
    print(f"\n--- {title} ---")
    if not docs:
        print("   (nothing)")
    for i, d in enumerate(docs, 1):
        lab, snip = _label(d)
        print(f"  {i}. {lab:<18} {snip}")


def _no_index_message():
    print(f"\nNo FAISS index found at: {FAISS_DIR}")
    print("This demo needs some indexed content first. To build one:\n")
    print("  1. Start DocMindAI:   .venv/Scripts/python app.py")
    print("     then open          http://localhost:5001\n")
    print("  2. Add content (either works):")
    print("     - Upload a document (PDF / DOCX / TXT). For a good hybrid demo, use a")
    print("       multi-page report, manual, or article - something with BOTH flowing")
    print("       prose (so semantic search has meaning to match) AND specific names,")
    print("       dates, codes or IDs (so keyword search has exact terms to match).")
    print("       A single short note won't show the difference between the two methods.")
    print("     - Or click 'Load Scans' to ingest fruit scan reports from the database.\n")
    print('  3. Re-run:  demo_hybrid_retrieval.py "your question"')


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "how long until the fruit goes off?"
    print(f'QUERY: "{query}"')

    if not (FAISS_DIR / "index.faiss").exists():
        _no_index_message()
        return

    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.load_local(str(FAISS_DIR), emb, allow_dangerous_deserialization=True)

    # 1) FAISS — semantic / embeddings
    faiss_docs = db.similarity_search(query, k=CAND)

    # 2) BM25 — keyword, built exactly like the app (_ensure_bm25)
    corpus = list(db.docstore._dict.values())
    bm25 = BM25Okapi([RAGEngine._tok(d.page_content) for d in corpus])
    scores = bm25.get_scores(RAGEngine._tok(query))
    bm25_docs = [corpus[i] for s, i in
                 sorted(zip(scores, range(len(corpus))), key=lambda x: x[0], reverse=True)
                 if s > 0][:CAND]

    # 3) RRF fusion — the real helper the system uses
    fused = RAGEngine._rrf_fuse([faiss_docs, bm25_docs], top_n=TOPN)

    _show("FAISS only   (semantic / embeddings)", faiss_docs[:TOPN])
    _show("BM25 only    (keyword)", bm25_docs[:TOPN])
    _show("RRF fused    (hybrid — what DocMindAI actually retrieves)", fused)

    # the "aha": where the two methods agree vs differ
    fset = {_key(d) for d in faiss_docs[:TOPN]}
    bset = {_key(d) for d in bm25_docs[:TOPN]}
    print(f"\n{len(fset & bset)} chunk(s) appear in BOTH top-{TOPN} lists — RRF floats those up.")
    print(f"FAISS alone surfaced {len(fset - bset)} that keyword missed; "
          f"BM25 alone surfaced {len(bset - fset)} the semantic search ranked lower.")
    print("Hybrid keeps the best of both.")


if __name__ == "__main__":
    main()

# rag/retriever.py
"""
Person 5: Document Retriever
Responsibility: Retrieve relevant chunks from ChromaDB using Hybrid (Dense + BM25) and Reranking.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from core.config import DB_DIR, TOP_K, EMBEDDING_MODEL, COLLECTION_NAME
import pickle
import os

# Modules loaded lazily to save startup time
_bm25 = None
_bm25_docs = None
_reranker = None

def _get_bm25(db):
    global _bm25, _bm25_docs
    if _bm25 is None:
        from rank_bm25 import BM25Okapi
        try:
            with open("bm25_docs.pkl", "rb") as f:
                _bm25_docs = pickle.load(f)
            texts = [doc.page_content for doc in _bm25_docs]
            _bm25 = BM25Okapi([t.lower().split() for t in texts])
        except Exception as e:
            print(f"BM25 cache miss, falling back to slow db.get(): {e}")
            raw = db.get()
            texts = raw.get("documents", [])
            metas = raw.get("metadatas", [])
            if texts:
                _bm25_docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metas)]
                _bm25 = BM25Okapi([t.lower().split() for t in texts])
    return _bm25, _bm25_docs

def _get_reranker():
    global _reranker
    from sentence_transformers import CrossEncoder
    try:
        from core.config import RERANKER_MODEL
        model_name = RERANKER_MODEL
    except ImportError:
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
    if _reranker is None:
        try:
            _reranker = CrossEncoder(model_name)
        except Exception:
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def retrieve_chunks(query, db=None, top_k=None):
    """
    Retrieve the most relevant document chunks using Hybrid Search and Reranking.
    """
    if top_k is None:
        top_k = TOP_K

    if db is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

    # 1. Semantic Search
    sem_docs = db.similarity_search(query, k=top_k * 2)
    
    # 2. BM25 Search (Keyword search)
    bm25_top_docs = []
    try:
        bm25, bm25_docs = _get_bm25(db)
        if bm25 and bm25_docs:
            tokens = query.lower().split()
            bm25_scores = bm25.get_scores(tokens)
            top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]
            # Only include if score > 0
            bm25_top_docs = [bm25_docs[i] for i in top_idx if bm25_scores[i] > 0]
    except Exception as e:
        print(f"BM25 Error: {e}")

    # 3. Combine and Deduplicate
    unique_docs = {}
    for d in sem_docs + bm25_top_docs:
        if d.page_content not in unique_docs:
            unique_docs[d.page_content] = d
    combined_docs = list(unique_docs.values())

    if not combined_docs:
        return []

    # 4. Cross-Encoder Reranking
    try:
        reranker = _get_reranker()
        pairs = [(query, d.page_content) for d in combined_docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(combined_docs, scores), key=lambda x: x[1], reverse=True)
        
        # Calculate max rerank score to inject for metrics
        for doc, score in ranked:
            doc.metadata["reranker_score"] = float(score)
            
        final_docs = [d for d, s in ranked][:top_k]
    except Exception as e:
        print(f"Reranking Error: {e}. Falling back to default ordering.")
        final_docs = combined_docs[:top_k]

    return final_docs

# Alias for pipeline.py compatibility
retrieve_docs = retrieve_chunks
import os
import json
import hashlib
from typing import Optional, Dict, Any

class MockDoc:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata

class SemanticCache:
    """
    A simple file-based semantic cache to store verified high-confidence answers.
    Prevents redundant LLM calls for repeated identical queries.
    """
    def __init__(self, cache_file: str = "semantic_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=4)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode("utf-8")).hexdigest()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        q_hash = self._hash_query(query)
        cached_data = self.cache.get(q_hash)
        
        if cached_data:
            print(f"--- SEMANTIC CACHE HIT for: '{query}' ---")
            # Reconstruct MockDocs so they have .metadata and .page_content attributes
            # which app.py expects.
            result = cached_data.copy()
            if "docs_data" in result:
                reconstructed_docs = []
                for d_data in result["docs_data"]:
                    reconstructed_docs.append(MockDoc(d_data["content"], d_data["metadata"]))
                result["docs"] = reconstructed_docs
                result["context"] = reconstructed_docs
            return result
            
        return None

    def set(self, query: str, result: Dict[str, Any]):
        q_hash = self._hash_query(query)
        serializable_result = result.copy()
        
        # Convert Document objects to plain dictionaries for json serialization
        if "docs" in result:
            docs_data = []
            for d in result["docs"]:
                docs_data.append({
                    "content": d.page_content if hasattr(d, "page_content") else str(d),
                    "metadata": d.metadata if hasattr(d, "metadata") else {}
                })
            serializable_result["docs_data"] = docs_data
            
        # Remove un-serializable LangChain objects to prevent JSON errors
        serializable_result.pop("docs", None)
        serializable_result.pop("context", None)
        
        self.cache[q_hash] = serializable_result
        self._save_cache()

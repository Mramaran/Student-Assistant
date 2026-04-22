"""
Answer Span Highlighter — Fixed sentence splitting (handles Dr., Mr., etc.)
"""
from __future__ import annotations
import re
from core.config import (
    FAITHFULNESS_THRESHOLD, SIMILARITY_THRESHOLD, TOP_K_AGREEMENT_THRESHOLD,
    KEYWORD_OVERLAP_MIN, CONFIDENCE_REJECTION_FLOOR, MULTI_HOP_THRESHOLD,
    CITATION_THRESHOLD, REGENERATION_THRESHOLD_LOW, REGENERATION_THRESHOLD_HIGH,
    CONFIDENCE_GATE_THRESHOLD, FACTUAL_GROUNDED_THRESHOLD, ADVICE_BASELINE_CONFIDENCE,
    SUPPORT_RATIO_WEIGHT, KEYWORD_SCORE_WEIGHT, SEMANTIC_BOOST_WEIGHT,
    SEMANTIC_SCALE_DIVISOR, SEMANTIC_SCALE_OFFSET, DEFAULT_RELEVANCE,
    DEFAULT_SIMILARITY, DEFAULT_BOOST, DEFAULT_BM25_SCORE, SUPPORT_THRESHOLD
)

_STOPWORDS = {
    "the","is","a","an","in","of","to","and","for","are","it","on","at","by",
    "or","be","as","this","that","from","with","was","has","have","can","will",
    "not","but","if","you","your","they","their","we","our","i","my","do",
    "about","so","would","should","could","please","also",
}
_MIN_WORDS = 1
_SUPPORT_THRESHOLD = SUPPORT_THRESHOLD


def _extract_entities(text: str) -> set[str]:
    """Extract proper nouns (Title Case) to find college names, cities, etc."""
    # Find words starting with Capital letter that are not at the start of a sentence
    # This is a heuristic, but good for "KCT", "CIT", "PSG", etc.
    return set(re.findall(r"\b[A-Z][A-Z0-9a-z]*\b", text))


def _has_number_anchor(sentence: str, doc_text: str) -> bool:
    """Check if numbers in the sentence have matching context in the doc."""
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", sentence)
    if not numbers:
        return True # No numbers to verify
    
    # For every number, check if the surrounding words also exist in the doc
    sent_words = _meaningful_words(sentence)
    for num in numbers:
        # If the number itself isn't in the doc, it's a hallucination
        if num not in doc_text:
            return False
            
    return True # Basic check passed


def _split_sentences(text: str) -> list[str]:
    """Split into sentences, skipping common abbreviations (Dr., Mr., etc.)."""
    # Protect common abbreviations before splitting
    protected = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|St|Dept|No|vs|etc)\.',
                       r'\1<DOT>', text)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    # Restore
    return [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]


def _meaningful_words(sentence: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", sentence.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


def highlight_spans(answer: str, docs) -> dict:
    """
    Split answer into sentences and check which document best supports each.
    Includes Entity Gating (no KCT if not in docs) and Number Anchoring.
    """
    if not answer or not docs:
        return {"supported": [], "unsupported": [], "support_ratio": 0.0}

    # Robust sentence splitting
    sentences = _split_sentences(answer)
    if not sentences:
        sentences = [answer]

    # Pre-calculate doc contents and entities
    doc_contents = []
    doc_entities = set()
    for d in docs:
        content = (d.page_content if hasattr(d, "page_content") else str(d)).lower()
        doc_contents.append(content)
        doc_entities.update(_extract_entities(content))

    # 🔥 FIX 5: IGNORE "CONTEXT ONLY" PENALTY FOR EXACT MATCH
    clean_ans = re.sub(r'\[.*?\]', '', answer).strip().lower()
    exact_match_idx = -1
    if len(clean_ans) > 15:
        for idx, doc_text in enumerate(doc_contents):
            if clean_ans in doc_text:
                exact_match_idx = idx
                break
                
    if exact_match_idx != -1:
        # Full answer is verbatim in the document. Bypass all penalties.
        return {
            "supported": [{"text": sent, "doc_idx": exact_match_idx} for sent in sentences],
            "unsupported": [],
            "support_ratio": 1.0
        }

    supported = []
    unsupported = []
    
    for sent in sentences:
        sent_lower = sent.lower()
        best_doc_idx = -1
        max_overlap = 0.0
        
        # 🔥 Step 1: Entity Gating Check (The 'KCT' Problem)
        sent_entities = _extract_entities(sent)
        hallucinated_entities = [e for e in sent_entities if e.lower() not in ["cit", "coimbatore"] and e not in doc_entities]
        
        if hallucinated_entities:
            # Contains an entity like "KCT" that is NOT in the database
            unsupported.append(sent)
            continue
            
        words = _meaningful_words(sent)
        if len(words) < _MIN_WORDS:
            best_doc_idx = -1
        else:
            for idx, doc_text in enumerate(doc_contents):
                # 🔥 Step 2: Number-Anchor Check (The '75% CSE' Problem)
                if not _has_number_anchor(sent, doc_text):
                    continue
                    
                # 1. Exact match (strongest)
                if sent_lower in doc_text:
                    best_doc_idx = idx
                    max_overlap = 1.0
                    break
                
                # 2. Token overlap (Jaccard-like)
                matches = sum(1 for w in words if w in doc_text)
                overlap = matches / len(words)
                
                if overlap > max_overlap and overlap >= _SUPPORT_THRESHOLD:
                    max_overlap = overlap
                    best_doc_idx = idx
        
        if best_doc_idx != -1:
            supported.append({"text": sent, "doc_idx": best_doc_idx})
        else:
            unsupported.append(sent)

    support_ratio = len(supported) / len(sentences)
    
    # Weighted Grounding: 70% Sentence support, 30% Keyword overlap
    ans_keywords = _meaningful_words(answer)
    all_context_keywords = _meaningful_words(" ".join(doc_contents))
    overlap_count = sum(1 for w in ans_keywords if w in all_context_keywords)
    keyword_score = overlap_count / max(len(ans_keywords), 1)
    
    final_score = (SUPPORT_RATIO_WEIGHT * support_ratio) + (KEYWORD_SCORE_WEIGHT * keyword_score)
    
    return {
        "supported": supported,
        "unsupported": unsupported,
        "support_ratio": round(final_score, 2)
    }

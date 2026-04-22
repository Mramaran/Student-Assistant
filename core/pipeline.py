"""
👑 Pipeline Master — Industry Production Level
===============================================
Dual-mode Hybrid Pipeline:

  TUTORING / MATCHING queries
  [1] Guard Layer  [2] Intent Classification (TUTORING)
  [3] Matching Engine  [4] Edge Handler  [5] Explainer

  RAG / POLICY queries
  [1] Guard Layer  [2] Intent Classification (FACTUAL/VAGUE)
  [3] Retrieval  [4] Generation  [5] Verification  [6] Confidence
"""

import time
from core.config import (
    TOP_K, FACTUAL_GROUNDED_THRESHOLD, CONFIDENCE_GATE_THRESHOLD, DB_DIR,
    DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    FAITHFULNESS_THRESHOLD, SIMILARITY_THRESHOLD, TOP_K_AGREEMENT_THRESHOLD,
    KEYWORD_OVERLAP_MIN, CONFIDENCE_REJECTION_FLOOR, MULTI_HOP_THRESHOLD,
    CITATION_THRESHOLD, REGENERATION_THRESHOLD_LOW, REGENERATION_THRESHOLD_HIGH,
    ADVICE_BASELINE_CONFIDENCE, SUPPORT_RATIO_WEIGHT, KEYWORD_SCORE_WEIGHT,
    SEMANTIC_BOOST_WEIGHT, SEMANTIC_SCALE_DIVISOR, SEMANTIC_SCALE_OFFSET,
    DEFAULT_RELEVANCE, DEFAULT_SIMILARITY, DEFAULT_BOOST, DEFAULT_BM25_SCORE,
    SIMILARITY_GATE_THRESHOLD
)
from ingestion.loader import load_documents
from ingestion.cleaner import clean_text
from ingestion.chunker import split_into_chunks
from embeddings.embedder import get_embedding_function
from embeddings.vector_store import store_chunks, load_vector_db
from rag.retriever import retrieve_docs
from rag.generator import generate_answer
from verification.verifier import verify_answer
from verification.confidence import (
    compute_confidence_with_breakdown,
    CONF_HIGH,
    CONF_LOW,
)
from core.guard import guard_layer
from core.cache import SemanticCache

from core.guard import guard_layer
from core.cache import SemanticCache

# Decision Layer [NEW]
from decision.advisor import AcademicAdvisor
from decision.rules_engine import engine as rules_engine, CITRulesEngine


# ── Legacy Hybrid Service Logic Removed ───────────────────────────────────────


def run_ingestion_pipeline():
    """Runs the full ingestion pipeline to build the vector DB."""
    print("1. Loading documents...")
    docs = load_documents(DATA_DIR)
    if not docs:
        print("No documents found in", DATA_DIR)
        return
    print("2. Cleaning text...")
    cleaned_docs = clean_text(docs)
    print("3. Splitting into chunks...")
    chunks = split_into_chunks(cleaned_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print("4. Embedding and storing...")
    embed_func = get_embedding_function()
    store_chunks(chunks, embed_func, DB_DIR)
    print("Pipeline complete!")





def classify_decision_intent(query: str) -> bool:
    """
    Returns True if user asks for eligibility or mentions numerical stats.
    """
    import re
    q = query.lower()
    decision_keywords = ["eligible", "can i", "allow", "attendance", "detain", "condonation", "pass", "placement", "arrear", "minimum"]
    
    # Auto-detect if numbers are present (enhanced detection)
    has_numbers = bool(re.search(r"\d+\.?\d*", query))
    
    return any(k in q for k in decision_keywords) or has_numbers



<<<<<<< HEAD
    try:
        (
            match_tutors, StudentProfile, TutorProfile,
            explain_match, format_for_ui,
            edge_handle, build_student_profile, get_tutor_pool
        ) = _get_matching_services()

        # Build student profile from context (or defaults)
        student = build_student_profile(query, student_context)
        tutors  = get_tutor_pool()

        if not tutors:
            fallback = edge_handle("no_tutor_available", {"student_id": student.student_id, "subjects": student.subjects, "candidates": []})
            return {
                "answer":            fallback["message"],
                "status":            "fallback",
                "edge_response":     fallback,
                "strategy":          "matching",
                "metrics":           {"latency": round(time.time() - start_time, 2)},
            }

        # Run matching
        match_result = match_tutors(student, tutors, top_k=3)

        # Edge: no match
        if not match_result.get("ranked_tutors"):
            fallback = edge_handle("no_tutor_available", {"student_id": student.student_id, "subjects": student.subjects, "candidates": [t.__dict__ for t in tutors]})
            return {"answer": fallback["message"], "status": "fallback", "edge_response": fallback, "strategy": "matching", "metrics": {"latency": round(time.time() - start_time, 2)}}

        # Explainability layer
        report = explain_match(match_result)
        ui     = format_for_ui(report)

        # Adaptive study plan (Next-Level #5) — attached to UI data so the
        # Streamlit renderer can surface a week-by-week tutor timeline.
        try:
            from services.study_plan import build as build_study_plan
            ui["study_plan"] = build_study_plan(query, student, tutors)
        except Exception:
            ui["study_plan"] = []

        # Intelligent triage (Next-Level #7) — urgency-aware routing.
        try:
            from services.triage import triage as _triage
            triage_result = _triage(
                student, match_result.get("ranked_tutors", []), tutors=tutors
            )
        except Exception:
            triage_result = {"triage_path": "standard"}

        answer_text = (
            f"🏫 **{ui['header']}**\n\n"
            f"{ui['top_explanation']}\n\n"
            f"**Confidence:** {ui['top_confidence']}% — {ui['top_badge']}\n\n"
        )
        if ui.get("top_strengths"):
            answer_text += f"**Strengths:** {', '.join(ui['top_strengths'])}\n\n"
        if ui.get("alternatives"):
            answer_text += "**Alternatives:**\n" + "\n".join(
                f"- {a['name']} ({a['confidence']}% — {a['badge']})"
                for a in ui["alternatives"]
            )

        return {
            "answer":            answer_text,
            "status":            "matched",
            "match_result":      match_result,
            "explainability":    report,
            "ui_data":           ui,
            "strategy":          "matching",
            "confidence":        match_result["top_pick"]["score"] if match_result.get("top_pick") else 0.0,
            "confidence_pct":    match_result["top_pick"]["confidence"] if match_result.get("top_pick") else 0,
            "confidence_level":  "High" if (match_result.get("top_pick") or {}).get("confidence", 0) >= 75 else "Low",
            "intent":            "TUTORING",
            "metrics":           {"latency": round(time.time() - start_time, 2), "retrieval_hits": len(tutors), "cache_hit": False},
            "docs":              [],
            "citations":         [],
            "verification":      {},
            "sentence_validations": [],
            "grade_distribution": {},
            "source_confidence_map": {},
            "confidence_breakdown":  {},
            **triage_result,
        }

    except Exception as exc:
        return {
            "answer":  f"❌ Matching engine error: {type(exc).__name__}: {exc}",
            "status":  "error",
            "strategy": "matching",
            "metrics": {"latency": round(time.time() - start_time, 2)},
        }
=======
>>>>>>> a505e53286153f69354753588cc3e2494853f001


def classify_query_intent(query: str, llm=None) -> str:
    """
    Query Understanding Layer — classifies query BEFORE retrieval.
    Replaced slow LLM inference with ultra-fast heuristics to fix latency.

    Categories and adaptive retrieval depth:
      FACTUAL   → top-3 chunks  (precise single-fact answer)
      VAGUE     → top-6 chunks  (broader context for summarisation)
      MULTI-PART→ top-5 chunks  (covers all sub-questions)
    """
    q = query.lower()
    
    # 1. Multi-part intent
    if " and " in q or " also " in q or "," in q:
        wh_count = q.count("what") + q.count("how") + q.count("where") + q.count("who") + q.count("when")
        if wh_count > 1 or len(q.split()) > 15:
            return "MULTI-PART"
            
    # 2. Vague intent
    vague_patterns = ["tell me about", "explain", "details on", "how is", "what is the general"]
    if any(p in q for p in vague_patterns) or len(q.split()) <= 4:
        return "VAGUE"
        
    return "FACTUAL"


def run_fast_path(query: str, stream_placeholder=None, student_context: dict | None = None) -> dict:
    """Stage 1: Fast Path — RAG Retrieval."""

    start_time = time.time()
    embed_func = get_embedding_function()

    db = load_vector_db(embed_func, DB_DIR)
    
    if db is None:
        return _error_result("Database not initialized. Please run ingestion first.", "no_database", start_time)

    guard = guard_layer(query)
    if guard["status"] == "blocked":
        return _error_result(guard["message"], guard["stage"], start_time)
        
    # Semantic Cache Layer
    cache = SemanticCache()
    cached_result = cache.get(query)
    if cached_result:
        cached_result["metrics"]["latency"] = round(time.time() - start_time, 2)
        cached_result["metrics"]["cache_hit"] = True
        return cached_result

    intent = classify_query_intent(query)
    
    from rag.query_rewriter import rewrite_query
    search_query = rewrite_query(query)
    
    # Fast path forces top_k=3 to optimize speed as requested
    dynamic_k = 3
    
    context = retrieve_docs(search_query, db, top_k=dynamic_k)
    if not context:
        return _error_result("I could not find this information in the available documents.\n\nYou can:\n- Refer to the official handbook\n- Contact your department office\n- Ask a more specific question", "escalated", start_time, intent=intent)

    # Max similarity gate (Retrieval Top-Chunk Guarantee)
    from verification.confidence import compute_embedding_similarity
    top_chunk_similarity = compute_embedding_similarity(search_query, context[0].page_content, embed_func)
    if top_chunk_similarity < SIMILARITY_GATE_THRESHOLD:
        return _error_result("I could not find this information in the available documents.\n\nYou can:\n- Refer to the official handbook\n- Contact your department office\n- Ask a more specific question", "escalated", start_time, intent=intent)

    # Conflict Detection Layer
    from verification.conflict_detector import detect_conflicts
    conflict_report = detect_conflicts(query, context)

    # Do not halt completely on conflict. Let generator proceed but append the conflict.
    if conflict_report and stream_placeholder:
        # Prompt generator to summarize the agreed facts while keeping conflict separate
        query = f"{query}\n[SYSTEM: Conflicting rules detected in documents. Give a general summary keeping in mind this conflict.]"
        
    raw_answer = generate_answer(query, context, stream_placeholder)

    return {
        "fast_path": True,
        "answer": raw_answer,
        "context": context,
        "docs": context,
        "intent": intent,
        "start_time": start_time,
        "status": "unverified",
        "conflict_panel": conflict_report,
        "dynamic_k": dynamic_k,
        "confidence_level": "Pending",
        "metrics": {"latency": round(time.time() - start_time, 2)}
    }

def run_slow_path(query: str, raw_answer: str, context: list, intent: str, start_time: float, dynamic_k: int = 3, student_context: dict | None = None, attempt: int = 1, conflict_panel: str | None = None) -> dict:
    """Stage 2: Slow Path - Verification, Decision, Recommendation"""
    embed_func = get_embedding_function()
    
    # Verifier
    verification = verify_answer(raw_answer, context, query=query, embed_model=embed_func)
    verdict = verification.get("verdict", "Not Supported")
    
    # Regeneration Loop
    if verdict == "Not Supported" and attempt == 1:
        from rag.generator import generate_answer
        correction_query = f"{query}\n[SYSTEM NOTE: Your previous answer contained unsupported logic. Strictly stick to the exact text in the provided documents and use [Source X] format for every sentence.]"
        regenerated_answer = generate_answer(correction_query, context, stream_placeholder=None) # Silent regeneration
        return run_slow_path(query, regenerated_answer, context, intent, start_time, dynamic_k, student_context, attempt=2, conflict_panel=conflict_panel)
    
    if verdict == "Not Supported" and not (raw_answer.startswith("⚠️") or raw_answer.startswith("❌")):
        raw_answer = "⚠️ Answer not supported by original documents. " + raw_answer

    verifier_score = verification.get("verifier_score_override", 0.2)
    reranker_score = max([float(doc.metadata.get("reranker_score", 0.0)) for doc in context] + [0.0])

    conf_result = compute_confidence_with_breakdown(
        raw_answer, context, reranker_score,
        embed_model=embed_func,
        verifier_score=verifier_score,
        intent=intent,
        verdict=verdict,
        max_k=dynamic_k,
        name_match_flag=verification.get("name_match_flag", 0),
        coverage_ratio=verification.get("coverage_ratio", 1.0)
    )

    confidence = conf_result["confidence"]
    conf_pct = conf_result["confidence_pct"]
    level = conf_result["level"]
    should_answer = conf_result["should_answer"]

    if not should_answer or confidence < 0.20:
        return _error_result(
            "I could not find this information in the available documents.\n\nYou can:\n- Refer to the official handbook\n- Contact your department office\n- Ask a more specific question",
            "escalated", start_time,
            intent=intent,
            confidence=confidence,
            conf_pct=conf_pct,
            conf_breakdown=conf_result["breakdown"],
            verification=verification
        )

    final_answer = verification.get("cited_answer", raw_answer) if verdict == "Supported" else raw_answer
    final_status = "grounded" if level == "High" else "restricted" if level == "Low" else "rejected"

    citations = []
    for doc in context:
        citations.append({
            "source":  doc.metadata.get("source", "unknown"),
            "content": doc.page_content[:200] + "...",
            "page":    doc.metadata.get("page", "?"),
            "section": doc.metadata.get("section", "General"),
        })

    latency = time.time() - start_time

    final_result = {
        "answer": final_answer,
        "confidence": confidence,
        "confidence_pct": conf_pct,
        "confidence_level": level,
        "formatted_confidence": conf_result["formatted"],
        "confidence_breakdown": conf_result["breakdown"],
        "status": final_status,
        "verdict": verdict,
        "intent": intent,
        "conflict_panel": conflict_panel,
        "citations": citations,
        "context": context,
        "docs": context,
        "verification": verification,
        "source_confidence_map": verification.get("source_confidence_map", {}),
        "grade_distribution": verification.get("grade_distribution", {}),
        "sentence_validations": verification.get("sentence_validations", []),
        "metrics": {
            "latency": round(latency, 2),
            "token_usage": len(final_answer.split()),
            "retrieval_hits": len(context),
            "dynamic_k": dynamic_k,
            "cache_hit": False
        }
    }
    
    # ── Stage 4 & 5: Decision & Recommendation Layers ────────────────────────
    advisor_result = None
    if classify_decision_intent(query):
        # We use the cited_answer (Stage 3) for decision logic to ensure grounding
        target_answer = final_result.get("answer", raw_answer)
        advisor = AcademicAdvisor()
        # The 'advise' method extracts numbers from query if student_context is missing
        advisor_result = advisor.advise(query, target_answer, student_context)
        final_result["advisor"] = advisor_result

    if final_status == "grounded" and confidence >= 0.5:
        cache = SemanticCache()
        cache.set(query, final_result)
        
    return final_result

def run_qa_pipeline(query: str, stream_placeholder=None, student_context: dict | None = None) -> dict:
    """Full 7-stage legacy wrapper."""
    fast = run_fast_path(query, stream_placeholder, student_context)
    if not fast.get("fast_path") or fast.get("status") == "conflict":
        return fast
        
    return run_slow_path(
        query=query,
        raw_answer=fast["answer"],
        context=fast["context"],
        intent=fast["intent"],
        start_time=fast["start_time"],
        dynamic_k=fast["dynamic_k"],
        student_context=student_context,
        conflict_panel=fast.get("conflict_panel")
    )


def _error_result(
    message: str,
    status: str,
    start_time: float,
    intent: str = "UNKNOWN",
    confidence: float = 0.0,
    conf_pct: int = 0,
    conf_breakdown: dict = None,
    verification: dict = None,
) -> dict:
    """Consistent structured error response with full schema."""
    return {
        "answer":               message,
        "confidence":           confidence,
        "confidence_pct":       conf_pct,
        "confidence_level":     "None" if confidence < 0.2 else "Low",
        "formatted_confidence": f"{conf_pct}% (None)" if confidence < 0.2 else f"{conf_pct}% (Low)",
        "confidence_breakdown": conf_breakdown or {
            "similarity_score": 0.0, "verifier_score": 0.0, "source_score": 0.0,
            "num_sources": 0,
            "weights": {"similarity": 0.5, "verifier": 0.3, "sources": 0.2}
        },
        "status":               status,
        "verdict":              "Not Supported",
        "intent":               intent,
        "citations":            [],
        "context":              [],
        "docs":                 [],
        "verification":         verification or {},
        "source_confidence_map": {},
        "grade_distribution":   {},
        "sentence_validations": [],
        "metrics": {
            "latency":        round(time.time() - start_time, 2),
            "token_usage":    0,
            "retrieval_hits": 0,
            "dynamic_k":      TOP_K,
            "cache_hit":      False
        }
    }

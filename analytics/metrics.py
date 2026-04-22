"""
analytics/metrics.py — Decision and RAG Quality Metrics
=======================================================
Tracks the accuracy and performance of the Intelligent Academic Assistant.
"""

from typing import Dict, Any
from feedback.logger import log_feedback # Reusing existing logger foundation

def log_advisor_decision(query: str, decision: Dict[str, Any], confidence: float):
    """
    Specifically logs whether the advisor made a decision and the confidence.
    """
    # Simply delegates but adds metadata for Advisor logic
    metadata = {
        "is_decision": True,
        "decision_count": len(decision.get("decisions", [])),
        "recommendation_count": len(decision.get("recommendations", []))
    }
    
    # We log using the existing system but with higher priority on decision-related fields
    log_feedback(
        query=query,
        answer=decision.get("rag_context", ""),
        confidence=confidence,
        verified=True,
        metadata=metadata
    )

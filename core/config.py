"""
🔧 Global Configuration — Single Source of Truth
================================================
All modules MUST import constants from here.
No hardcoded thresholds anywhere else in the codebase.
"""

# ── Document Processing ────────────────────────────────────────────────────────
CHUNK_SIZE    = 400    # optimal semantic unit for academic docs
CHUNK_OVERLAP = 80     # maintains continuity across chunks
TOP_K         = 5      # best balance: k=4 or 5 avoids noise

# ── Database Paths ─────────────────────────────────────────────────────────────
import os as _os
DB_DIR          = _os.getenv("DB_DIR", "./db")
DATA_DIR        = _os.getenv("DATA_DIR", "./data")
COLLECTION_NAME = "student_docs"
PROJECT_TITLE   = "Verification Assistant"
INSTITUTION_NAME = "CIT"
UI_HEADER       = "A Self-Verifying Retrieval-Augmented Generation System"

# ── Domain Configuration ───────────────────────────────────────────────────────
DOMAIN_KEYWORDS = [
    "cit", "coimbatore", "institute", "college", "campus", "attendance", 
    "exam", "examination", "arrear", "marks", "internal", "semester", 
    "cgpa", "sgpa", "grade", "result", "backlog", "reappear", "supplementary", 
    "condonation", "regulation", "syllabus", "timetable", "fees", "fee", 
    "tuition", "scholarship", "payment", "fine", "challan", "hostel", "mess", 
    "library", "lab", "laboratory", "wifi", "bus", "transport", "hod", 
    "faculty", "professor", "warden", "principal", "department", "staff", 
    "student", "admission", "placement", "cse", "ece", "eee", "it dept", 
    "mca", "mba", "ai", "mechanical", "civil"
]

REFUSAL_MESSAGE = (
    f"⚠️ This system only answers {INSTITUTION_NAME}-related academic queries.\n\n"
    "You can ask about: attendance, exams, fees, hostel, scholarships, "
    "faculty, departments, results, or placement."
)



EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384  # output dimension of all-MiniLM-L6-v2
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── LLM & UC4 operational constants ────────────────────────────────────────
GEMINI_MODEL     = _os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ESCALATION_EMAIL = _os.getenv("ESCALATION_EMAIL", "faculty@example.edu")
CONF_HIGH        = 0.75  # threshold for "High" confidence label (Task 3.1)
CONF_MEDIUM      = 0.45  # threshold for "Medium" confidence label (Task 3.1)
CACHE_TTL_DAYS   = 7     # semantic cache retention window

# ── Verification Thresholds ────────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD      = 0.4
SIMILARITY_THRESHOLD        = 0.3   # reduced from 0.75 — conversational queries have lower cosine sim
TOP_K_AGREEMENT_THRESHOLD   = 0.50  # reduced from 0.70 — more forgiving for broad questions
KEYWORD_OVERLAP_MIN         = 0.15  # reduced from 0.20 — short factual queries have low overlap naturally
CONFIDENCE_REJECTION_FLOOR  = 0.3   # hard floor — below this, answer is rejected regardless
MULTI_HOP_THRESHOLD         = 0.45
CITATION_THRESHOLD          = 0.58  # cosine sim needed for "Supported" grade in verifier
PARTIAL_THRESHOLD           = 0.45  # cosine sim for "Partially Supported" grade
REGENERATION_THRESHOLD_LOW  = 0.4
REGENERATION_THRESHOLD_HIGH = 0.0
CONFIDENCE_GATE_THRESHOLD   = 0.5
FACTUAL_GROUNDED_THRESHOLD  = 0.6
ADVICE_BASELINE_CONFIDENCE  = 0.5
SUPPORT_THRESHOLD           = 0.30

# ── Scoring Weights & Thresholds ───────────────────────────────────────────
# (Legacy names kept for v1 confidence; Task 3.1 will add the 4-factor set.)
W_SIMILARITY    = 0.40   # cosine sim (answer ↔ context)
W_EVIDENCE      = 0.35   # evidence-alignment grounding score (was W_RULE_MATCH)
W_SOURCES       = 0.15   # multi-source corroboration
W_COMPLETENESS  = 0.10   # chunk count vs max_k

# ── Gate Thresholds ────────────────────────────────────────────────────────
SIMILARITY_GATE_THRESHOLD = 0.25  # Lowered to 0.25 to catch acronyms like 'ai&ds'
CONFIDENCE_THRESHOLD_HIGH = 50.0
CONFIDENCE_THRESHOLD_LOW  = 20.0

SUPPORT_RATIO_WEIGHT  = 0.7
KEYWORD_SCORE_WEIGHT  = 0.3
SEMANTIC_BOOST_WEIGHT = 0.2
SEMANTIC_SCALE_DIVISOR = 1.2
SEMANTIC_SCALE_OFFSET  = 0.2

# ── Default Values ─────────────────────────────────────────────────────────────
DEFAULT_RELEVANCE    = 0.5
DEFAULT_SIMILARITY   = 0.5
DEFAULT_BOOST        = 0.0
DEFAULT_BM25_SCORE   = 0.5

RATIONALE = {
    "CHUNK_SIZE": (
        "400 chars. Optimal for academic docs: too small causes loss of meaning, "
        "too large introduces irrelevant noise. 300-500 is the optimal semantic unit."
    ),
    "CHUNK_OVERLAP": (
        "80 chars. Maintains continuity across chunks and prevents sentence splitting issues."
    ),
    "TOP_K_BASE": (
        "5 chunks. Best balance. k=1-2 misses info, k=3 is borderline, k>6 introduces noise."
    ),
    "EMBEDDING_MODEL": (
        "all-MiniLM-L6-v2 (SentenceTransformers): 80MB, 384-dim vectors. "
        "Chosen for: (1) strong BEIR benchmark scores for semantic search, "
        "(2) CPU-friendly — no GPU required, (3) widely used = reproducible results, "
        "(4) compatible with Chroma without additional dependencies."
    ),
    "RERANKER_MODEL": (
        "cross-encoder/ms-marco-MiniLM-L-6-v2: bidirectional cross-attention between "
        "query and document. Unlike bi-encoders (which embed independently), cross-encoders "
        "see both query and document simultaneously — 15-20% higher MRR on MSMARCO benchmark. "
        "Trade-off: slower (O(n) LLM calls) but used only on top-12 candidates."
    ),
    "BM25_HYBRID": (
        "BM25 (keyword search) catches exact matches that dense embeddings miss. "
        "Example: 'SGPA 7.5' — the embedding finds semantically similar text, but BM25 "
        "finds the exact number. Hybrid = best of both worlds. "
        "Final scores combined with cross-encoder reranking."
    ),
    "CITATION_THRESHOLD": (
        "0.58 cosine similarity for 'Supported' grade. Empirically tuned on our dataset: "
        "below 0.58, too many irrelevant sentences get cited; above 0.65, too many valid "
        "sentences are missed. 0.58 balances precision and recall."
    ),
    "CONFIDENCE_FORMULA": (
        "Composite: 0.40×similarity + 0.35×verifier + 0.15×source_score + 0.10×data_availability. "
        "Prioritizes answer-to-context alignment and factual entity matching (Name/Number)."
    ),
}

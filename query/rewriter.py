"""
LLM-Powered Query Rewriter
============================
Converts informal / vague student queries into formal retrieval-optimized queries.

Examples:
  "when is deadline?"    → "Scholarship application deadline eligibility criteria 2024"
  "bro attendance low"  → "Attendance shortage below 75 percent consequences policy"
  "hod details"         → "Head of department faculty name contact location"

Falls back to rule-based expansion if the LLM rewrite fails or produces garbage.
"""

from __future__ import annotations


_REWRITE_PROMPT = """You are a search query optimizer and INTENT classifier for a college student assistant.
Convert the student question into a crisp, 2-4 word KEYWORD search query. 
Categorize the INTENT into one of:
- POLICY: Rules/facts (e.g., "attendance", "fees").
- SCENARIO: "What if" cases (e.g., "fail subject").
- STRATEGY: Advice/tips (e.g., "high cgpa").
- OPINION: Comparisons (e.g., "CIT vs PSG").
- CASUAL: Greetings (e.g., "hello").

Rules:
1. QUERY must be ONLY 2-4 keywords (e.g., "CSE HOD name").
2. NO long sentences. NO fluff. NO polite phrases.

Format:
QUERY: [keywords]
INTENT: [LABEL]
"""

def rewrite_query(question: str, llm=None) -> tuple[str, str, str]:
    """
    Rewrite and classify student intent in a single pass.
    """
    base_expanded = _rule_based_expand(question)
    default_intent = "POLICY"

    if llm is None:
        return base_expanded, default_intent, _friendly_label(question)

    # Quick heuristic for common keywords
    q_low = question.lower()
    if len(question) < 15 and any(w in q_low for w in ["hod", "fee", "scholarship", "attendance"]):
        return base_expanded, "POLICY", _friendly_label(question)

    try:
        prompt = (
            f"{_REWRITE_PROMPT}\n\n"
            f"Student Question: {question}\n"
        )
        response = llm.invoke(prompt)
        
        lines = [line.strip() for line in response.split("\n") if ":" in line]
        optimized_q = base_expanded
        intent = default_intent
        
        for line in lines:
            if line.upper().startswith("QUERY:"):
                optimized_q = line.split(":", 1)[1].strip().strip('"\'')
            elif line.upper().startswith("INTENT:"):
                intent = line.split(":", 1)[1].strip().upper()

        if intent not in ["POLICY", "SCENARIO", "STRATEGY", "OPINION", "CASUAL"]:
            intent = "POLICY"

        return optimized_q, intent, _friendly_label(question)

    except Exception as e:
        print(f"⚠️  Rewriter fails: {e}")
        return base_expanded, default_intent, _friendly_label(question)


def _clean_rewrite(response: str, original: str) -> str:
    """Strip quotes, newlines, and common prefixes from LLM output."""
    text = response.strip()
    for prefix in ["rewritten query:", "query:", "answer:", "search query:", "optimized query:"]:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.strip('"\'')
    text = text.split("\n")[0].strip()
    return text if text else original


def _friendly_label(q: str) -> str:
    """Create a UI-friendly interpretation label from rule-based expansion."""
    q = q.strip()
    return q[0].upper() + q[1:] if q else q


def _extra_keywords(q: str) -> str:
    """Inject domain keywords for ALL detected topics in the query."""
    q_lower = q.lower()
    extras = []
    
    # Check for Placement/Job keywords
    if any(w in q_lower for w in ["placement", "company", "opportunity", "job", "hiring", "salary", "package", "placed", "unplaced", "job offers"]):
        extras.append("placement policy recruitment hiring procedure salary package job offers")
        
    # Check for Attendance/Academic keywords
    if any(w in q_lower for w in ["attendance", "75%", "present", "absent", "percentage"]):
        extras.append("minimum 75 percent requirement shortage condonation")
        
    # Check for Scholarship keywords
    if "scholarship" in q_lower:
        extras.append("eligibility application renewal merit income criteria")
        
    # Check for Exam/Arrear keywords
    if any(w in q_lower for w in ["exam", "fail", "backlog", "arrear", "supplementary", "reappear"]):
        extras.append("reappear supplementary arrear policy exam rules marks grades")
        
    # Check for Hostel keywords
    if "hostel" in q_lower:
        extras.append("hostel rules curfew timing visitor warden mess facilities")
        
    # Check for Fee keywords
    if any(w in q_lower for w in ["fee", "fees", "payment", "tuition"]):
        extras.append("fee structure tuition mess exam payment due date")
        
    # Check for Faculty/HOD keywords
    if any(w in q_lower for w in ["hod", "head", "faculty", "professor", "staff", "teacher"]):
        extras.append("faculty name contact location department CIT")
        
    if "deadline" in q_lower:
        extras.append("last date submission deadline academic calendar")
        
    return " ".join(extras)


def _rule_based_expand(query: str) -> str:
    """Rule-based query expansion — always runs as the base layer."""
    q = query.lower().strip()

    acronyms = {
        "ai & ds":  "artificial intelligence and data science",
        "ai ds":    "artificial intelligence and data science",
        "ai&ds":    "artificial intelligence and data science",
        "aids":     "artificial intelligence and data science",
        "hod":      "head of department",
        "cse":      "computer science and engineering",
        "ece":      "electronics and communication engineering",
        "eee":      "electrical and electronics engineering",
        "it dept":  "information technology",
        "mca":      "master of computer applications",
        "mba":      "master of business administration",
        "cgpa":     "cumulative grade point average",
        "sgpa":     "semester grade point average",
        "backlog":  "arrear failed subject reappear",
        "arrears":  "arrear failed subject reappear",
    }
    for key, val in acronyms.items():
        if key in q:
            q += " " + val

    slang = {
        "bro":              "",
        "any chance":       "is it possible eligibility",
        "what to do":       "procedure process steps",
        "now what":         "next steps process",
        "came late":        "late entry curfew timing",
        "low attendance":   "attendance shortage below 75",
        "attendance short": "attendance shortage below 75",
    }
    for key, val in slang.items():
        if key in q:
            q = q.replace(key, val)

    q += " " + _extra_keywords(query)
    return q.strip()

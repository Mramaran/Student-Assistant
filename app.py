"""UC4 Student Assistant — Streamlit entry point.

Wiring (Task 8.1):
    user query
        -> services.safety_handler.sanitize_query (Task 7.1)
        -> rag.chain.answer (retrieval + LCEL Gemini generation + confidence
                             + conflicts + auto-escalation on Low)
        -> UI render:
             * contradiction banner                        (Task 4.3)
             * answer body + citation markers
             * confidence badge + 'Why this score'         (Task 3.3)
             * auto / manual escalation link               (Task 5.3)
             * retrieved chunks expander

The imports below (`services.safety_handler`, `services.escalation`,
`verification.confidence`, `verification.conflict_detector`) are intentionally
kept at module scope — Task 8.1's AST check asserts they appear on `app.py`.
The last three are already wired into the chain, but the import surface here
makes the UC4 integration explicit and easy to audit.
"""

from __future__ import annotations

import html
import re

from dotenv import load_dotenv

# Load .env BEFORE any project imports so core.config picks up overrides
# like ESCALATION_EMAIL at module-import time.
load_dotenv()

import streamlit as st
from rag import chain
from services import escalation, safety_handler  # noqa: F401 — audit surface
from verification import confidence, conflict_detector  # noqa: F401 — audit surface


LABEL_COPY: dict[str, tuple[str, str]] = {
    "High":   ("✅ Verified",              "#16a34a"),
    "Medium": ("⚠️ Review before acting",  "#f59e0b"),
    "Low":    ("⛔ Insufficient evidence",  "#dc2626"),
}


_FACTOR_META: tuple[tuple[str, str, str], ...] = (
    ("cosine",             "Semantic similarity",
     "Cosine(answer embedding, retrieved context embedding)"),
    ("evidence_alignment", "Evidence alignment",
     "Proper-noun & numeric anchors in the answer that appear verbatim in context"),
    ("corroboration",      "Corroboration",
     "Fraction of retrieved sources that back at least one anchor"),
    ("availability",       "Availability",
     "Retrieved chunks / configured TOP_K"),
)


def _render_confidence(conf: dict) -> None:
    label = conf.get("label", "Low")
    copy, color = LABEL_COPY.get(label, LABEL_COPY["Low"])
    composite = float(conf.get("composite", 0.0))

    st.markdown(
        f"<div style='padding:6px 12px; border-radius:6px; background:{color}; "
        f"color:white; display:inline-block; font-weight:600;'>"
        f"{copy} · score {composite:.2f}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Why this score"):
        factors = conf.get("factors", {})
        weights = conf.get("weights", {})

        cards: list[str] = []
        for key, pretty, desc in _FACTOR_META:
            raw = float(factors.get(key, 0.0))
            weight = float(weights.get(key, 0.0))
            contrib = raw * weight
            pct = max(0, min(100, int(round(raw * 100))))
            cards.append(
                f"""
<div style="padding: 10px 14px; background: #fafafa;
            border-left: 3px solid {color}; border-radius: 4px;
            margin-bottom: 8px;">
  <div style="display: flex; justify-content: space-between;
              align-items: baseline; gap: 12px;">
    <span style="font-weight: 600; color: #1a1a1a; font-size: 0.9rem;">
      {html.escape(pretty)}
    </span>
    <span style="font-size: 0.80rem; color: #555; white-space: nowrap;">
      {raw:.3f}
      <span style="color: #999;">×</span>
      <b style="color: #333;">{weight:.2f}</b>
      <span style="color: #999;">=</span>
      <b style="color: #1a1a1a;">{contrib:.3f}</b>
    </span>
  </div>
  <div style="margin: 8px 0 6px 0; background: #ececec;
              border-radius: 3px; height: 6px; overflow: hidden;">
    <div style="width: {pct}%; background: {color}; height: 100%;"></div>
  </div>
  <div style="color: #777; font-size: 0.74rem;">{html.escape(desc)}</div>
</div>
"""
            )

        total_card = f"""
<div style="display: flex; justify-content: space-between;
            align-items: center; padding: 12px 14px;
            background: #ffffff; border: 2px solid {color};
            border-radius: 6px; margin-top: 10px;">
  <span style="font-weight: 700; color: #1a1a1a;
               font-size: 0.95rem; letter-spacing: 0.02em;
               text-transform: uppercase;">Composite</span>
  <span style="font-weight: 700; color: {color};
               font-size: 1.35rem;">{composite:.3f}</span>
</div>
"""
        st.markdown("".join(cards) + total_card, unsafe_allow_html=True)


def _render_conflicts(conflicts: list) -> None:
    if not conflicts:
        return
    banner = "⚠️ Sources disagree on " + ", ".join(c["anchor_key"] for c in conflicts)
    st.warning(banner)
    with st.expander("Source comparison"):
        for conflict in conflicts:
            st.markdown(f"**Anchor:** {conflict['anchor_key']}")
            cols = st.columns(len(conflict["values"]))
            for col, entry in zip(cols, conflict["values"]):
                with col:
                    st.markdown(f"**{entry['source']}**")
                    st.markdown(f"Value: `{entry['value']}`")
                    st.caption(entry["snippet"])
            st.markdown("---")


def _render_escalation(esc: dict | None, query: str, response: dict,
                       key_suffix: str = "") -> None:
    if esc:
        st.info(f"📨 Ticket {esc['ticket_id']} recorded — faculty review recommended.")
        st.markdown(
            f"[Escalate to faculty]({esc['mailto_url']}) — ticket `{esc['ticket_id']}`"
        )

    if st.button("Escalate anyway", key=f"escalate_manual_button_{key_suffix}"):
        conf = response.get("confidence") or {}
        docs = response.get("sources") or []
        manual = escalation.create_ticket(
            query=query,
            sources=[d.metadata.get("source", "unknown") for d in docs if hasattr(d, "metadata")],
            confidence=float(conf.get("composite", 0.0)),
            verdict={"verdict": "Manual"},
        )
        st.success(f"Manual ticket {manual['ticket_id']} recorded.")
        st.markdown(f"[Send to faculty]({manual['mailto_url']})")


def _render_metrics(response: dict) -> None:
    latency_ms = int(response.get("latency_ms", 0) or 0)
    prompt_tokens = int(response.get("prompt_tokens", 0) or 0)
    completion_tokens = int(response.get("completion_tokens", 0) or 0)
    sources = response.get("sources") or []

    c1, c2, c3 = st.columns(3)
    c1.metric("Latency", f"{latency_ms / 1000:.2f} s")
    c2.metric("Tokens", prompt_tokens + completion_tokens)
    c3.metric("Sources", len(sources))


_SOURCE_MARKER_RE = re.compile(r"\[\s*Source\s*(\d+)\s*\]", re.IGNORECASE)


def _rewrite_citations(answer: str, sources: list) -> str:
    """Replace ``[Source N]`` markers with the underlying filename.

    Unknown indices are left untouched so we never silently drop a citation.
    """
    if not answer or not sources:
        return answer

    def _sub(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if 1 <= idx <= len(sources):
            meta = getattr(sources[idx - 1], "metadata", {}) or {}
            name = str(meta.get("source", "")).strip()
            if name:
                return f"[{name}]"
        return match.group(0)

    return _SOURCE_MARKER_RE.sub(_sub, answer)


def _render_answer(response: dict, query: str, key_suffix: str = "") -> None:
    _render_conflicts(response.get("conflicts") or [])

    sources = response.get("sources") or []
    answer_text = _rewrite_citations(
        response.get("answer", "") or "",
        sources,
    )

    with st.container(border=True):
        st.markdown("### Answer")
        st.markdown(answer_text or "_(no answer returned)_")
        st.markdown("---")
        _render_confidence(response.get("confidence", {}) or {})
        _render_metrics(response)

    _render_escalation(response.get("escalation"), query, response, key_suffix=key_suffix)

    if sources:
        with st.expander("Explore Sources"):
            _render_source_cards(sources)


_SOURCE_BADGE_COLORS: tuple[str, ...] = (
    "#2563eb",  # blue
    "#7c3aed",  # purple
    "#db2777",  # pink
    "#ea580c",  # orange
    "#059669",  # green
    "#0891b2",  # teal
    "#ca8a04",  # amber
    "#4f46e5",  # indigo
)


def _render_source_cards(sources: list) -> None:
    """Badge + accent-line source cards inside the 'Explore Sources' expander."""
    for i, doc in enumerate(sources, start=1):
        color = _SOURCE_BADGE_COLORS[(i - 1) % len(_SOURCE_BADGE_COLORS)]
        meta = getattr(doc, "metadata", {}) or {}
        src = html.escape(str(meta.get("source", "Unknown")))
        page = meta.get("page")
        page_suffix = f" · Pg. {html.escape(str(page))}" if page is not None else ""

        raw_content = getattr(doc, "page_content", "") or ""
        trimmed = raw_content[:800] + ("…" if len(raw_content) > 800 else "")
        content = html.escape(trimmed).replace("\n", "<br/>")

        st.markdown(
            f"""
<div style="margin: 0 0 18px 0;">
  <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
    <span style="background: {color}; color: #fff; font-weight: 600;
                 padding: 4px 12px; border-radius: 4px; font-size: 0.80rem;
                 letter-spacing: 0.02em;">Source {i}</span>
    <span style="font-weight: 600; color: #1a1a1a; font-size: 0.92rem;">
      {src}{page_suffix}
    </span>
  </div>
  <div style="border-left: 3px solid {color}; padding: 2px 0 2px 14px;
              margin-left: 2px; color: #333; font-size: 0.92rem;
              line-height: 1.55;">
    {content}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


QUICK_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("Attendance",  "What is the minimum attendance percentage required?"),
    ("Scholarship", "What are the scholarship eligibility requirements?"),
    ("Placement",   "What are the placement training attendance rules?"),
    ("Exams",       "What are the rules for writing arrear exams?"),
)


def _set_query(template: str) -> None:
    st.session_state["pending_query"] = template


def _init_session_state() -> None:
    defaults = {
        "history":              [],    # list of recent queries (most recent first)
        "queries_count":        0,
        "escalations_count":    0,
        "confidence_sum":       0.0,
        "confidence_n":         0,
        "messages":             [],    # chat transcript (user + assistant turns)
        "pending_query":        None,  # set by chip/history click; consumed next run
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _render_message(msg: dict, idx: int) -> None:
    """Render a single chat turn inside an st.chat_message bubble."""
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg.get("content", ""))
        elif msg.get("error"):
            st.error(msg["error"])
        else:
            _render_answer(msg["response"], msg["query"], key_suffix=str(idx))


def _record_answer_stats(query: str, response: dict) -> None:
    st.session_state["queries_count"] += 1
    history = st.session_state["history"]
    if query not in history:
        history.insert(0, query)
        del history[8:]  # keep last 8

    conf = (response.get("confidence") or {}).get("composite")
    if isinstance(conf, (int, float)):
        st.session_state["confidence_sum"] += float(conf)
        st.session_state["confidence_n"] += 1

    if response.get("escalation"):
        st.session_state["escalations_count"] += 1


def _reset_session() -> None:
    for key in ("history", "queries_count", "escalations_count",
                "confidence_sum", "confidence_n",
                "messages", "pending_query"):
        if key in st.session_state:
            del st.session_state[key]


_DOC_TYPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("handbook",    ("handbook", "academic_info", "regulation", "conduct", "basic_facts")),
    ("exam",        ("exam", "grading", "grade_system")),
    ("scholarship", ("scholarship",)),
    ("placement",   ("placement",)),
    ("admission",   ("admission", "brochure")),
    ("facilities",  ("facilities", "hostel", "lab_facilities", "infrastructure", "library")),
)


def _classify_doc(name: str) -> str:
    """Light-weight filename → doc_type; mirrors ingestion.loader without heavy imports."""
    lower = name.lower()
    for doc_type, keywords in _DOC_TYPE_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return doc_type
    return "other"


def _corpus_snapshot() -> dict:
    """Read ``data/manifest.json`` (if present) and return a summary dict."""
    from collections import Counter
    from datetime import datetime, timezone
    import json
    from pathlib import Path

    manifest_path = Path("data") / "manifest.json"
    if not manifest_path.exists():
        return {"indexed": False}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"indexed": False}

    counts: Counter = Counter()
    for rel_path in manifest.keys():
        counts[_classify_doc(Path(rel_path).name)] += 1

    mtime = datetime.fromtimestamp(manifest_path.stat().st_mtime, tz=timezone.utc)
    return {
        "indexed":    True,
        "n_docs":     len(manifest),
        "by_type":    dict(counts),
        "last_build": mtime.strftime("%Y-%m-%d %H:%M UTC"),
    }


def _render_sidebar() -> None:
    with st.sidebar:
        # Brand lockup
        st.markdown(
            f"""
<div style='padding: 14px 14px 10px 14px; border-left: 4px solid {REVATURE_ORANGE};
            background: #fafafa; border-radius: 4px; margin-bottom: 18px;'>
  <div style='font-weight: 700; font-size: 1.1rem; color: {REVATURE_BLACK};
              letter-spacing: 0.02em;'>
    Student <span style='color: {REVATURE_ORANGE};'>Assistant</span>
  </div>
  <div style='font-size: 0.78rem; color: #666; margin-top: 4px;'>
    Say it. Source it.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Session stats
        st.markdown("##### Session")
        avg_conf = (
            st.session_state["confidence_sum"] / st.session_state["confidence_n"]
            if st.session_state["confidence_n"]
            else 0.0
        )
        c1, c2 = st.columns(2)
        c1.metric("Questions", st.session_state["queries_count"])
        c2.metric("Avg conf.", f"{avg_conf:.2f}")
        st.metric("Escalations", st.session_state["escalations_count"])

        st.markdown("---")

        # Corpus info
        st.markdown("##### Corpus")
        snap = _corpus_snapshot()
        if not snap.get("indexed"):
            st.caption("_Not indexed yet — run `python build_db.py`._")
        else:
            st.caption(f"**{snap['n_docs']}** documents · built {snap['last_build']}")
            if snap.get("by_type"):
                pretty = " · ".join(
                    f"{kind} {count}" for kind, count in sorted(snap["by_type"].items())
                )
                st.caption(pretty)

        st.markdown("---")

        # Session history
        st.markdown("##### Recent questions")
        if not st.session_state["history"]:
            st.caption("_No questions yet._")
        else:
            for i, q in enumerate(st.session_state["history"]):
                st.button(
                    q if len(q) <= 42 else q[:42] + "…",
                    key=f"hist_{i}",
                    on_click=_set_query,
                    args=(q,),
                    use_container_width=True,
                )

        st.markdown("---")

        # Reset
        st.button(
            "Reset session",
            key="reset_session_button",
            on_click=_reset_session,
            use_container_width=True,
        )


REVATURE_ORANGE = "#F47B20"
REVATURE_BLACK = "#1a1a1a"


def _inject_theme_css() -> None:
    """Revature-style visual polish: hide default chrome, add brand accents."""
    st.markdown(
        f"""
<style>
    /* Hide Streamlit chrome */
    footer {{ visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    /* Widen the content container (wide layout) but cap it on huge monitors */
    .block-container {{
        max-width: 1080px;
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-bottom: 2rem;
    }}

    /* Minimal two-tone hero */
    .rv-hero {{
        padding: 10px 0 4px 0;
        margin-bottom: 4px;
    }}
    .rv-hero h1 {{
        margin: 0;
        font-weight: 700;
        font-size: 2.2rem;
        letter-spacing: -0.01em;
        color: {REVATURE_BLACK};
        line-height: 1.15;
    }}
    .rv-hero h1 .accent {{ color: {REVATURE_ORANGE}; }}
    .rv-hero p {{
        margin: 8px 0 0 0;
        color: #555;
        font-size: 0.95rem;
        max-width: 640px;
    }}

    /* Thin orange accent rule under the hero */
    .rv-accent {{
        height: 2px;
        width: 48px;
        background: {REVATURE_ORANGE};
        border: 0;
        margin: 14px 0 22px 0;
    }}

    /* Chip buttons — orange hover border */
    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {{
        border-color: {REVATURE_ORANGE} !important;
        color: {REVATURE_ORANGE} !important;
    }}

    /* Brand footer strip */
    .rv-footer {{
        margin-top: 28px;
        padding: 10px 0;
        border-top: 2px solid {REVATURE_ORANGE};
        text-align: center;
        color: #666;
        font-size: 0.82rem;
    }}
    .rv-footer .dot {{ color: {REVATURE_ORANGE}; font-weight: 700; }}

    /* ─── Sidebar theme — match main page ─────────────────────────── */

    /* Clean white sidebar with a thin orange edge */
    section[data-testid="stSidebar"] {{
        background: #ffffff !important;
        border-right: 1px solid #eee;
    }}
    section[data-testid="stSidebar"] > div {{
        background: #ffffff !important;
        padding-top: 1rem;
    }}

    /* Sidebar typography — same family + color as hero */
    section[data-testid="stSidebar"] h5 {{
        color: {REVATURE_BLACK} !important;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 10px 0 8px 0;
        padding-bottom: 6px;
        border-bottom: 2px solid {REVATURE_ORANGE};
        display: inline-block;
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] {{
        color: #555 !important;
        font-size: 0.82rem;
    }}

    /* Sidebar dividers — match the thin orange rule on main page */
    section[data-testid="stSidebar"] hr {{
        border: 0 !important;
        height: 2px !important;
        width: 48px !important;
        background: {REVATURE_ORANGE} !important;
        margin: 18px 0 !important;
    }}

    /* Metric cards — subtle orange accent bar */
    section[data-testid="stSidebar"] div[data-testid="stMetric"] {{
        background: #fafafa;
        border-left: 3px solid {REVATURE_ORANGE};
        border-radius: 4px;
        padding: 8px 10px;
        margin-bottom: 6px;
    }}
    section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] {{
        color: #666 !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    section[data-testid="stSidebar"] div[data-testid="stMetricValue"] {{
        color: {REVATURE_BLACK} !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
    }}

    /* Sidebar buttons — same orange hover treatment as main-page chips */
    section[data-testid="stSidebar"] button {{
        background: #ffffff !important;
        color: {REVATURE_BLACK} !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 6px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        transition: all 0.15s ease-in-out;
    }}
    section[data-testid="stSidebar"] button:hover {{
        border-color: {REVATURE_ORANGE} !important;
        color: {REVATURE_ORANGE} !important;
        background: #fff8f2 !important;
    }}

    /* Reset-session button — solid orange (primary-ish look) */
    section[data-testid="stSidebar"] button[key="reset_session_button"],
    section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:last-child {{
        /* keep default for now — :hover already highlights it */
    }}
</style>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Student Assistant",
        page_icon="🎓",
        layout="wide",
    )
    _inject_theme_css()
    _init_session_state()
    _render_sidebar()

    st.markdown(
        """
<div class="rv-hero">
  <h1>Student <span class="accent">Assistant</span></h1>
  <p>Grounded answers on academic policies, exams, scholarships, placements,
     admissions, and campus facilities — every claim cites its source.</p>
</div>
<hr class="rv-accent" />
""",
        unsafe_allow_html=True,
    )

    chip_cols = st.columns(len(QUICK_QUESTIONS))
    for col, (label, template) in zip(chip_cols, QUICK_QUESTIONS):
        col.button(
            label,
            key=f"chip_{label.lower()}",
            on_click=_set_query,
            args=(template,),
            use_container_width=True,
        )

    # ── Chat transcript ──────────────────────────────────────────────
    for idx, msg in enumerate(st.session_state["messages"]):
        _render_message(msg, idx)

    st.markdown(
        '<div class="rv-footer">Powered by Student Assistant</div>',
        unsafe_allow_html=True,
    )

    # ── Chat input (Streamlit pins st.chat_input to the viewport bottom) ──
    pending = st.session_state.get("pending_query")
    if pending:
        st.session_state["pending_query"] = None

    chat_prompt = st.chat_input(
        placeholder="Ask about academic policies, exams, scholarships, placements..."
    )
    query = pending or chat_prompt

    if not query or not query.strip():
        return

    sanitized = safety_handler.sanitize_query(query.strip())
    if not sanitized["safe"]:
        st.session_state["messages"].append(
            {"role": "user", "content": query}
        )
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "error": (
                    f"⛔ Query blocked — {sanitized['reason'].replace('_', ' ')}. "
                    f"This system ignores instructions that try to override its rules."
                ),
            }
        )
        st.rerun()

    safe_query = sanitized["redacted_query"]
    st.session_state["messages"].append({"role": "user", "content": safe_query})

    # Render the new user turn immediately so the spinner has context.
    with st.chat_message("user"):
        st.markdown(safe_query)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                response = chain.answer(safe_query)
            except Exception as exc:
                st.session_state["messages"].append(
                    {"role": "assistant", "error": f"Chain error: {exc}"}
                )
                st.rerun()

    _record_answer_stats(safe_query, response)
    st.session_state["messages"].append(
        {"role": "assistant", "response": response, "query": safe_query}
    )
    st.rerun()


if __name__ == "__main__":
    main()
else:
    main()

"""GeoPol Bot - Streamlit Chat Interface (Step 5).

An interactive, evidence-grounded chat application for geopolitical intelligence.
Connects directly to the FastAPI backend (`/api/v1/chat/generate`).
Enforces domain gate status displays, clickable primary source citations,
epistemic answer distinctions, and clear abstention notices.

CRITICAL SECURITY: This frontend contains zero API keys. All validation,
domain gating, vector search, and LLM calls occur strictly on the backend.
"""

import os
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import streamlit as st
import httpx

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================

st.set_page_config(
    page_title="GeoPol Bot | Geopolitics Intelligence",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for state-of-the-art geopolitical intelligence UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .main-subtitle {
        font-size: 1.0rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    
    /* Status Badges */
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }
    .pill-answered {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .pill-rejected {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .pill-clarify {
        background-color: #fef9c3;
        color: #854d0e;
        border: 1px solid #fef08a;
    }
    .pill-abstained {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
    }
    .pill-error {
        background-color: #ffe4e6;
        color: #be123c;
        border: 1px solid #fecdd3;
    }
    
    /* Citation Cards */
    .citation-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .citation-title {
        font-weight: 600;
        color: #1e3a8a;
        font-size: 0.95rem;
    }
    .citation-meta {
        font-size: 0.8rem;
        color: #64748b;
    }
    .citation-quote {
        font-size: 0.85rem;
        color: #334155;
        font-style: italic;
        margin-top: 0.25rem;
    }
    
    /* Meta Pills in Sidebar */
    .health-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Backend URL configuration
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000/api/v1")


# ==============================================================================
# Backend API Helpers (Zero-Key Architecture)
# ==============================================================================

def check_backend_health(base_url: str) -> Dict[str, Any]:
    """Check connectivity and readiness of the FastAPI backend."""
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{base_url.rstrip('/')}/health")
            if resp.status_code == 200:
                return {"online": True, "data": resp.json()}
            return {"online": False, "error": f"HTTP {resp.status_code}"}
    except Exception as exc:
        return {"online": False, "error": str(exc)}


def fetch_corpus_sources(base_url: str) -> List[Dict[str, Any]]:
    """Retrieve list of verified corpus sources from backend."""
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{base_url.rstrip('/')}/knowledge/sources")
            if resp.status_code == 200:
                return resp.json().get("sources", [])
    except Exception:
        pass
    return []


def query_chat_generation(
    base_url: str,
    query: str,
    context: Optional[List[str]] = None,
    similarity_threshold: float = 0.25,
    top_k: int = 4,
    allow_foundational_fallback: bool = True,
) -> Dict[str, Any]:
    """Send question to the FastAPI /api/v1/chat/generate endpoint."""
    url = f"{base_url.rstrip('/')}/chat/generate"
    payload = {
        "query": query,
        "conversation_context": context or [],
        "similarity_threshold": similarity_threshold,
        "top_k": top_k,
        "allow_foundational_fallback": allow_foundational_fallback,
    }
    with httpx.Client(timeout=35.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


# ==============================================================================
# Session State Initialization
# ==============================================================================

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to GeoPol Bot.**\n\n"
                "I am an evidence-grounded research assistant strictly dedicated to **geopolitics, "
                "international relations, diplomacy, armed conflicts, sanctions, treaties, strategic geography, "
                "and geoeconomics**.\n\n"
                "All answers are grounded in our verified primary corpus (UN resolutions, NATO founding treaty, "
                "UNCLOS, AUKUS, Taiwan Relations Act). Queries outside this scope will be halted at the Domain Gate, "
                "and insufficient evidence will result in transparent abstention."
            ),
            "status": "ANSWERED",
            "citations": [],
            "evidence": [],
            "latency_ms": 0.0,
            "gate_category": "welcome",
        }
    ]

if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = None


# ==============================================================================
# Sidebar: System Monitor & Parameters
# ==============================================================================

with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1526778548025-fa2f459cd5c1?w=400&auto=format&fit=crop&q=60",
        caption="GeoPol Bot Intelligence Hub",
        use_container_width=True,
    )
    st.title("🌐 GeoPol Bot")
    st.caption("Evidence-Grounded Geopolitical Assistant")
    st.markdown("---")

    # Backend Connectivity
    backend_url = st.text_input("Backend API URL", value=BACKEND_API_URL)
    health = check_backend_health(backend_url)

    if health["online"]:
        h_data = health["data"]
        st.success(f"● Backend Connected (v{h_data.get('version', '0.1.0')})")
        comp = h_data.get("components", {})
        gate_status = comp.get("domain_gate", {}).get("status", "unknown")
        rag_status = comp.get("knowledge_pipeline", {}).get("status", "unknown")
        gen_status = comp.get("generation_engine", {}).get("status", "unknown")

        with st.expander("🧩 Component Readiness", expanded=False):
            st.write(f"- **Domain Gate:** `{gate_status.upper()}`")
            st.write(f"- **Knowledge Store:** `{rag_status.upper()}`")
            st.write(f"- **Generation Engine:** `{gen_status.upper()}`")
            st.write(f"- **OpenAI Model:** `{comp.get('openai_api', {}).get('details', '')}`")
    else:
        st.error(f"⚠️ Backend Disconnected: {health.get('error')}")
        st.info("Start backend: `uvicorn backend.app.main:app --reload`")

    st.markdown("---")
    st.markdown("### 🧠 Intelligence Scope")
    intel_mode = st.radio(
        "Response Mode",
        options=["Comprehensive Geopolitics (Default)", "Strict Local Corpus Only"],
        index=0,
        help="Comprehensive mode answers all global geopolitical inquiries (diplomacy, conflicts, treaties, alliances) using strategic statecraft intelligence. Strict mode requires exact matching in the local database.",
    )
    allow_fallback = (intel_mode == "Comprehensive Geopolitics (Default)")

    st.markdown("---")
    st.markdown("### ⚙️ Retrieval Parameters")
    similarity_thresh = st.slider(
        "Similarity Threshold",
        min_value=0.10,
        max_value=0.70,
        value=0.20,
        step=0.05,
        help="Higher values require stricter relevance before evidence is admitted.",
    )
    top_k_chunks = st.slider(
        "Top-K Evidence Chunks",
        min_value=1,
        max_value=6,
        value=3,
        help="Maximum primary source passages retrieved for context.",
    )

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state["messages"] = [st.session_state["messages"][0]]
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔒 Security Guarantee")
    st.info(
        "**Zero Credential Exposure**\n\n"
        "This frontend communicates exclusively over HTTP with the backend. "
        "`OPENAI_API_KEY` is never transmitted to or stored inside the client browser."
    )


# ==============================================================================
# Main Chat Area
# ==============================================================================

st.markdown('<div class="main-title">🌐 GeoPol Bot Chat</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Official Primary Evidence • Tri-State Domain Gating • Non-Hallucinatory Abstention</div>',
    unsafe_allow_html=True,
)

# Quick Prompts Carousel / Bar
st.markdown("**Quick Inquiries:**")
q_col1, q_col2, q_col3, q_col4 = st.columns(4)

with q_col1:
    if st.button("🇮🇳 🇷🇺 India-Russia Ties", use_container_width=True):
        st.session_state["pending_query"] = "What are the key diplomatic and strategic dynamics in India-Russia relations?"
with q_col2:
    if st.button("🇨🇳 🇹🇼 Taiwan Straits", use_container_width=True):
        st.session_state["pending_query"] = "What are the core geopolitical friction points between China and Taiwan?"
with q_col3:
    if st.button("🛡️ NATO Article 5", use_container_width=True):
        st.session_state["pending_query"] = "Under Article 5 of the North Atlantic Treaty, what constitutes an armed attack and what are the collective defense obligations?"
with q_col4:
    if st.button("💻 Quicksort (Reject)", use_container_width=True):
        st.session_state["pending_query"] = "Write a Python script to implement quicksort with recursion."


# Render Conversation History
for msg in st.session_state["messages"]:
    role = msg["role"]
    with st.chat_message(role):
        if role == "user":
            st.markdown(msg["content"])
        else:
            # Assistant Message with Status Pill
            status = msg.get("status", "ANSWERED")
            latency = msg.get("latency_ms", 0.0)
            category = msg.get("gate_category", "")
            source = msg.get("grounding_source", "primary_corpus")

            # Render Status Badge
            if status == "ANSWERED":
                if source == "foundational_geopolitical_knowledge":
                    badge_label = f"✅ Answered (Comprehensive Geopolitics • {latency:.0f}ms)"
                else:
                    badge_label = f"✅ Answered (Grounded Treaty Evidence • {latency:.0f}ms)"
                st.markdown(
                    f'<span class="status-pill pill-answered">{badge_label}</span>',
                    unsafe_allow_html=True,
                )
            elif status == "REJECTED_OUT_OF_DOMAIN":
                st.markdown(
                    f'<span class="status-pill pill-rejected">🚫 Rejected at Gate (Out of Domain: {category})</span>',
                    unsafe_allow_html=True,
                )
            elif status == "REQUIRES_CLARIFICATION":
                st.markdown(
                    f'<span class="status-pill pill-clarify">⚠️ Clarification Needed (Ambiguous Input)</span>',
                    unsafe_allow_html=True,
                )
            elif status == "INSUFFICIENT_EVIDENCE_ABSTENTION":
                st.markdown(
                    f'<span class="status-pill pill-abstained">⚖️ Abstained (Insufficient Evidence in Corpus)</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="status-pill pill-error">⚠️ Service Exception</span>',
                    unsafe_allow_html=True,
                )

            # Render Main Markdown Content
            st.markdown(msg["content"])

            # Render Clickable Citations if available
            citations = msg.get("citations", [])
            if citations:
                st.markdown("#### 🔗 Official Source Citations")
                for c in citations:
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-title">
                                <a href="{c['source_url']}" target="_blank" style="text-decoration: none; color: #1e3a8a;">
                                    🔗 {c['document_title']}
                                </a>
                            </div>
                            <div class="citation-meta">
                                <strong>Issuing Entity:</strong> {c['source_name']} &nbsp;|&nbsp; 
                                <strong>Date:</strong> {c['publication_date']} &nbsp;|&nbsp; 
                                <strong>Chunk ID:</strong> <code>{c['chunk_id']}</code>
                            </div>
                            {f'<div class="citation-quote">"{c["relevant_quote"]}"</div>' if c.get('relevant_quote') else ''}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Render Raw Retrieved Evidence Accordion
            evidence = msg.get("evidence", [])
            if evidence:
                with st.expander(f"📜 Inspect Raw Retrieved Evidence ({len(evidence)} passages)", expanded=False):
                    for ev in evidence:
                        st.markdown(f"**Rank #{ev['rank']} — [{ev['document_title']}]({ev['source_url']})** (Cosine Similarity: `{ev['score']:.4f}`)")
                        st.caption(f"Authority: {ev['source_name']} • Date: {ev['publication_date']} • Topic: {ev['topic']}")
                        st.text(ev["content"])
                        st.divider()


# Handle User Input (either from chat_input or clicked prompt button)
user_input = st.chat_input("Ask a geopolitical question (e.g. diplomacy, conflicts, treaties, chokepoints)...")
if st.session_state["pending_query"]:
    user_input = st.session_state["pending_query"]
    st.session_state["pending_query"] = None

if user_input:
    # 1. Append User Message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Query Backend with Loading State
    with st.chat_message("assistant"):
        with st.spinner("Evaluating domain policy, retrieving evidence & generating answer..."):
            # Collect last 3 user questions for context
            prev_context = [
                m["content"] for m in st.session_state["messages"][-4:-1] if m["role"] == "user"
            ]

            try:
                res_data = query_chat_generation(
                    base_url=backend_url,
                    query=user_input,
                    context=prev_context,
                    similarity_threshold=similarity_thresh,
                    top_k=top_k_chunks,
                    allow_foundational_fallback=allow_fallback,
                )

                status = res_data.get("status", "ANSWERED")
                markdown_out = res_data.get("markdown_output", "")
                citations = res_data.get("validated_citations", [])
                evidence = res_data.get("retrieved_evidence", [])
                latency = res_data.get("latency_ms", 0.0)
                gate_res = res_data.get("gate_result", {})
                category = gate_res.get("primary_category", "geopolitics")
                source = res_data.get("grounding_source", "primary_corpus")

                # Render Pill
                if status == "ANSWERED":
                    if source == "foundational_geopolitical_knowledge":
                        badge_label = f"✅ Answered (Comprehensive Geopolitics • {latency:.0f}ms)"
                    else:
                        badge_label = f"✅ Answered (Grounded Treaty Evidence • {latency:.0f}ms)"
                    st.markdown(
                        f'<span class="status-pill pill-answered">{badge_label}</span>',
                        unsafe_allow_html=True,
                    )
                elif status == "REJECTED_OUT_OF_DOMAIN":
                    st.markdown(
                        f'<span class="status-pill pill-rejected">🚫 Rejected at Gate (Out of Domain: {category})</span>',
                        unsafe_allow_html=True,
                    )
                elif status == "REQUIRES_CLARIFICATION":
                    st.markdown(
                        f'<span class="status-pill pill-clarify">⚠️ Clarification Needed (Ambiguous Input)</span>',
                        unsafe_allow_html=True,
                    )
                elif status == "INSUFFICIENT_EVIDENCE_ABSTENTION":
                    st.markdown(
                        f'<span class="status-pill pill-abstained">⚖️ Abstained (Insufficient Evidence in Corpus)</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<span class="status-pill pill-error">⚠️ Service Exception</span>',
                        unsafe_allow_html=True,
                    )

                st.markdown(markdown_out)

                if citations:
                    st.markdown("#### 🔗 Official Source Citations")
                    for c in citations:
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <div class="citation-title">
                                    <a href="{c['source_url']}" target="_blank" style="text-decoration: none; color: #1e3a8a;">
                                        🔗 {c['document_title']}
                                    </a>
                                </div>
                                <div class="citation-meta">
                                    <strong>Issuing Entity:</strong> {c['source_name']} &nbsp;|&nbsp; 
                                    <strong>Date:</strong> {c['publication_date']} &nbsp;|&nbsp; 
                                    <strong>Chunk ID:</strong> <code>{c['chunk_id']}</code>
                                </div>
                                {f'<div class="citation-quote">"{c["relevant_quote"]}"</div>' if c.get('relevant_quote') else ''}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                if evidence:
                    with st.expander(f"📜 Inspect Raw Retrieved Evidence ({len(evidence)} passages)", expanded=False):
                        for ev in evidence:
                            st.markdown(f"**Rank #{ev['rank']} — [{ev['document_title']}]({ev['source_url']})** (Cosine Similarity: `{ev['score']:.4f}`)")
                            st.caption(f"Authority: {ev['source_name']} • Date: {ev['publication_date']} • Topic: {ev['topic']}")
                            st.text(ev["content"])
                            st.divider()

                # Save to history
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": markdown_out,
                        "status": status,
                        "citations": citations,
                        "evidence": evidence,
                        "latency_ms": latency,
                        "gate_category": category,
                    }
                )

            except httpx.HTTPStatusError as err:
                error_msg = f"HTTP Error communicating with backend: {err.response.status_code} - {err.response.text}"
                st.error(error_msg)
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": f"⚠️ **Backend Error**: {error_msg}",
                        "status": "API_ERROR",
                        "citations": [],
                        "evidence": [],
                        "latency_ms": 0.0,
                        "gate_category": "error",
                    }
                )
            except Exception as err:
                error_msg = f"Failed to connect to FastAPI backend at `{backend_url}`: {err}"
                st.error(error_msg)
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": f"⚠️ **Connection Error**: {error_msg}\n\nPlease verify that the FastAPI backend is running.",
                        "status": "API_ERROR",
                        "citations": [],
                        "evidence": [],
                        "latency_ms": 0.0,
                        "gate_category": "error",
                    }
                )


# ==============================================================================
# Footer
# ==============================================================================

st.markdown("---")
f_col1, f_col2 = st.columns([3, 1])
with f_col1:
    st.caption(
        "GeoPol Bot • Evidence Grounded Geopolitical Intelligence • "
        "Strict Policy Invariants & Zero Hallucination Standard"
    )
with f_col2:
    st.caption(f"Local Time: {datetime.now().strftime('%H:%M:%S')}")

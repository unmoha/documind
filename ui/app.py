"""Streamlit interactive web interface for DocuMind."""

import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("DOCUMIND_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="DocuMind â€” Grounded Document Q&A",
    page_icon="ðŸ“„",
    layout="wide",
)

st.title("ðŸ“„ DocuMind")
st.caption("Grounded document question-answering with verifiable inline citations.")


def api_get(path: str, params: dict | None = None) -> dict | None:
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{API_BASE_URL}{path}", params=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        st.error(f"Backend error calling {path}: {exc}")
        return None


def api_post_json(path: str, payload: dict) -> dict | None:
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{API_BASE_URL}{path}", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        st.error(f"Backend error calling {path}: {exc}")
        return None


# Sidebar: Backend health & retrieval settings
with st.sidebar:
    st.header("Settings & Status")
    health = api_get("/health")
    if health:
        st.success(f"API Online â€¢ v{health['version']} ({health['environment']})")
    else:
        st.warning(
            "FastAPI backend offline. Start it with:\n"
            "`uv run uvicorn documind.api.main:app --reload`"
        )

    top_k = st.slider("Top-K Chunks to Retrieve", min_value=1, max_value=10, value=4)
    st.divider()
    st.markdown(
        "**Architecture**\n"
        "- Embeddings: `BAAI/bge-small-en-v1.5`\n"
        "- Vector Store: `ChromaDB`\n"
        "- Generator: `Gemini`\n"
        "- Audit Store: `SQLite`"
    )

tab_ask, tab_docs, tab_logs = st.tabs(
    ["ðŸ’¬ Ask Documents", "ðŸ“š Knowledge Base", "ðŸ“Š Audit Logs"]
)

# --- Tab 1: Ask Documents ---
with tab_ask:
    question = st.text_input(
        "Ask a question about your uploaded documents:",
        placeholder="e.g., What are the success metrics and target SLAs?",
    )
    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner(
            "Retrieving relevant passages and generating grounded answer..."
        ):
            data = api_post_json(
                "/api/v1/query",
                {"question": question.strip(), "top_k": top_k},
            )
        if data:
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Grounded", "Yes" if data["has_relevant_context"] else "Refused"
            )
            score_display = (
                f"{data['best_score']:.3f}" if data["best_score"] is not None else "N/A"
            )
            col2.metric("Top Similarity Score", score_display)
            col3.metric("Latency", f"{data['latency_ms']:.0f} ms")

            st.subheader("Answer")
            if data["has_relevant_context"]:
                st.markdown(data["answer"])
            else:
                st.warning(data["answer"])

            if data["citations"]:
                st.subheader("Verified Sources")
                for c in data["citations"]:
                    label = (
                        f"[{c['source_number']}] {c['filename']} â€” "
                        f"Page {c['page_number']}, Chunk #{c['chunk_index']} "
                        f"(score: {c['score']:.3f})"
                    )
                    with st.expander(label):
                        st.write(c["excerpt"])

# --- Tab 2: Knowledge Base ---
with tab_docs:
    st.subheader("Upload a Document")
    uploaded = st.file_uploader(
        "Supported formats: PDF, Markdown (.md), Plain Text (.txt)",
        type=["pdf", "md", "txt"],
    )
    if uploaded is not None and st.button("Ingest Document", type="primary"):
        with st.spinner(f"Indexing {uploaded.name}..."):
            try:
                with httpx.Client(timeout=120.0) as client:
                    resp = client.post(
                        f"{API_BASE_URL}/api/v1/documents/upload",
                        files={
                            "file": (
                                uploaded.name,
                                uploaded.getvalue(),
                                uploaded.type or "application/octet-stream",
                            )
                        },
                    )
                    resp.raise_for_status()
                    res = resp.json()
                st.success(
                    f"**{res['filename']}**: {res['status']} "
                    f"({res['chunks_indexed']} chunks indexed, "
                    f"{res['chunks_replaced']} stale chunks replaced)."
                )
            except httpx.HTTPError as exc:
                st.error(f"Upload failed: {exc}")

    st.divider()
    st.subheader("Indexed Documents")
    doc_data = api_get("/api/v1/documents")
    if doc_data:
        c1, c2 = st.columns(2)
        c1.metric("Documents", doc_data["total_documents"])
        c2.metric("Total Chunks", doc_data["total_chunks"])

        if not doc_data["documents"]:
            st.info("No documents indexed yet. Upload one above to get started.")
        else:
            for doc in doc_data["documents"]:
                col_name, col_meta, col_del = st.columns([3, 3, 1])
                doc_id = (
                    doc.get("document_id")
                    or doc.get("file_hash")
                    or doc.get("filename")
                )
                col_name.write(f"**{doc.get('filename', 'Unknown')}**")
                col_meta.write(
                    f"Chunks: `{doc.get('chunk_count', '?')}` â€¢ ID: `{str(doc_id)[:12]}...`"
                )
                if col_del.button("Delete", key=f"del_{doc_id}"):
                    try:
                        with httpx.Client(timeout=30.0) as client:
                            del_resp = client.delete(
                                f"{API_BASE_URL}/api/v1/documents/{doc_id}"
                            )
                            del_resp.raise_for_status()
                        st.rerun()
                    except httpx.HTTPError as exc:
                        st.error(f"Delete failed: {exc}")

# --- Tab 3: Audit Logs ---
with tab_logs:
    st.subheader("Query & Response Audit Trail")
    log_limit = st.number_input("Max entries", min_value=5, max_value=200, value=25)
    if st.button("Refresh Logs"):
        st.rerun()
    logs_data = api_get("/api/v1/logs", params={"limit": int(log_limit)})
    if logs_data:
        if not logs_data["logs"]:
            st.info("No queries logged yet.")
        else:
            st.dataframe(logs_data["logs"], use_container_width=True)

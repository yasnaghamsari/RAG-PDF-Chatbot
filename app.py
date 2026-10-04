"""
app.py
------
ScholarLens - the Streamlit user interface.

This file only handles the *interface*: layout, buttons, chat history and
session state. All the AI logic lives in:
    pdf_utils.py     -> reading PDFs and chunking text
    rag_pipeline.py  -> embeddings, FAISS, retrieval, prompt and LLM

Run with:  streamlit run app.py
"""

import html
from dataclasses import asdict
from datetime import datetime

import streamlit as st

from config import SETTINGS, get_api_key
from pdf_utils import PDFReport, chunk_documents, extract_pages
from rag_pipeline import RAGError, RAGPipeline
from styles import CUSTOM_CSS

st.set_page_config(
    page_title=f"{SETTINGS.app_name} · PDF Q&A",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================================================
# Session state
# ==========================================================================
# Streamlit re-runs this whole script on every click, so anything that must
# survive between runs (chat history, the FAISS index, ...) is stored in
# st.session_state.
def init_session_state() -> None:
    defaults = {
        "messages": [],  # conversation history
        "pipeline": None,  # RAGPipeline holding the FAISS index
        "reports": [],  # PDFReport per uploaded file
        "stats": {"files": 0, "pages": 0, "chunks": 0},
        "uploader_key": 0,  # changing this empties the file uploader
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_conversation() -> None:
    st.session_state.messages = []


def reset_everything() -> None:
    st.session_state.messages = []
    st.session_state.pipeline = None
    st.session_state.reports = []
    st.session_state.stats = {"files": 0, "pages": 0, "chunks": 0}
    st.session_state.uploader_key += 1


# ==========================================================================
# Document processing (indexing phase)
# ==========================================================================
def process_documents(uploaded_files, api_key: str, chunk_size: int, chunk_overlap: int) -> None:
    """Extract -> chunk -> embed -> index, with live progress in a status box."""
    reports: list[PDFReport] = []
    all_pages = []

    with st.status("Processing documents...", expanded=True) as status:
        # ---- 1. Text extraction ------------------------------------------
        for file in uploaded_files:
            st.write(f"📄 Reading **{file.name}** ...")
            if file.size > SETTINGS.max_file_size_mb * 1024 * 1024:
                reports.append(PDFReport(filename=file.name, error=f"File is larger than {SETTINGS.max_file_size_mb} MB."))
                continue
            pages, report = extract_pages(file.getvalue(), file.name)
            reports.append(report)
            all_pages.extend(pages)

        st.session_state.reports = reports
        if not all_pages:
            status.update(label="No usable text found in the uploaded PDFs.", state="error")
            return

        # ---- 2. Chunking -------------------------------------------------
        chunks = chunk_documents(all_pages, chunk_size, chunk_overlap)
        st.write(f"✂️ Split {len(all_pages)} pages into **{len(chunks)} chunks** "
                 f"(size {chunk_size}, overlap {chunk_overlap}).")

        # ---- 3. Embeddings + FAISS index ---------------------------------
        st.write(f"🧠 Creating embeddings with `{SETTINGS.embedding_model}` and building the FAISS index ...")
        pipeline = RAGPipeline(api_key)
        try:
            pipeline.build_index(chunks)
        except RAGError as exc:
            status.update(label="Embedding failed", state="error")
            st.error(str(exc))
            return

        # ---- Save results ------------------------------------------------
        usable = [r for r in reports if not r.error]
        st.session_state.pipeline = pipeline
        st.session_state.stats = {
            "files": len(usable),
            "pages": sum(r.total_pages for r in usable),
            "chunks": len(chunks),
        }
        status.update(label=f"Ready! Indexed {len(usable)} document(s) into {len(chunks)} chunks.",
                      state="complete", expanded=False)
    st.toast("Documents indexed - you can start asking questions.", icon="✅")


# ==========================================================================
# Sidebar
# ==========================================================================
def render_sidebar() -> dict:
    """Draw the sidebar and return the user's choices."""
    with st.sidebar:
        st.markdown(f'<p class="sidebar-brand">🔎 {SETTINGS.app_name}</p>'
                    f'<p class="sidebar-sub">{SETTINGS.app_tagline}</p>', unsafe_allow_html=True)

        # ---- API key -----------------------------------------------------
        api_key = get_api_key()
        if api_key:
            st.markdown('<span class="key-ok">● OpenAI API key loaded from .env</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="key-miss">● No API key found in .env</span>', unsafe_allow_html=True)
            api_key = st.text_input("OpenAI API key", type="password", placeholder="sk-...",
                                    help="Used only for this session. Add it to .env to skip this step.") or None

        st.divider()

        # ---- Upload ------------------------------------------------------
        st.subheader("1 · Upload documents")
        files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True,
                                 key=f"uploader_{st.session_state.uploader_key}",
                                 label_visibility="collapsed")

        with st.expander("Chunking options"):
            chunk_size = st.slider("Chunk size (characters)", 300, 2000, SETTINGS.default_chunk_size, 100)
            chunk_overlap = st.slider("Chunk overlap (characters)", 0, 500, SETTINGS.default_chunk_overlap, 50)

        process_clicked = st.button("⚙️ Process documents", type="primary", disabled=not files)

        st.divider()

        # ---- Retrieval / generation settings -----------------------------
        st.subheader("2 · Answer settings")
        top_k = st.slider("Top-K retrieved chunks", 1, SETTINGS.max_top_k, SETTINGS.default_top_k,
                          help="How many of the most similar chunks are given to the LLM as context.")
        temperature = st.slider("Model temperature", 0.0, 1.0, SETTINGS.default_temperature, 0.05,
                                help="0 = focused and deterministic, 1 = more creative and varied.")

        st.divider()

        # ---- Conversation controls ---------------------------------------
        st.subheader("3 · Conversation")
        col1, col2 = st.columns(2)
        col1.button("🧹 Clear chat", on_click=clear_conversation, width="stretch")
        col2.button("♻️ Reset all", on_click=reset_everything, width="stretch")
        if st.session_state.messages:
            st.download_button("⬇️ Export chat (.md)", data=conversation_to_markdown(),
                               file_name="scholarlens_chat.md", mime="text/markdown",
                               width="stretch")

        st.caption(f"Models: `{SETTINGS.chat_model}` · `{SETTINGS.embedding_model}`")

    return {
        "api_key": api_key,
        "files": files,
        "process_clicked": process_clicked,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "temperature": temperature,
    }


# ==========================================================================
# Main-area components
# ==========================================================================
def render_header() -> None:
    st.markdown(f"""
    <div class="hero">
        <h1>🔎 {SETTINGS.app_name}</h1>
        <p>{SETTINGS.app_tagline} - every answer comes with the file and page it was taken from.</p>
        <div class="pipeline">PDF → Text → Chunks → Embeddings → FAISS → Top-K Search → Context → LLM → Answer + Sources</div>
    </div>""", unsafe_allow_html=True)


def render_stats() -> None:
    stats = st.session_state.stats
    questions = sum(1 for m in st.session_state.messages if m["role"] == "user")
    cards = [("📚", "Documents", stats["files"]), ("📄", "Pages", stats["pages"]),
             ("🧩", "Chunks", stats["chunks"]), ("💬", "Questions", questions)]
    html_cards = "".join(
        f'<div class="stat-card"><span class="icon">{icon}</span>'
        f'<div class="label">{label}</div><div class="value">{value}</div></div>'
        for icon, label, value in cards
    )
    st.markdown(f'<div class="stat-grid">{html_cards}</div>', unsafe_allow_html=True)


def render_document_panel() -> None:
    st.markdown('<div class="panel-title">📁 Document library</div>', unsafe_allow_html=True)
    if not st.session_state.reports:
        st.caption("No documents processed yet.")
        return

    for r in st.session_state.reports:
        if r.error:
            meta, note = "Not indexed", r.error
        else:
            meta = f"{r.text_pages}/{r.total_pages} pages with text · {r.characters:,} characters"
            note = " ".join(r.warnings)
        icon = {"ok": "✅", "warning": "⚠️", "error": "❌"}[r.status]
        note_html = f'<div class="note">{html.escape(note)}</div>' if note else ""
        st.markdown(f'<div class="doc-item {r.status}"><div class="name">{icon} {html.escape(r.filename)}</div>'
                    f'<div class="meta">{meta}</div>{note_html}</div>', unsafe_allow_html=True)


def render_welcome() -> None:
    st.markdown("""
    <div class="welcome">
        <h3>👋 Welcome! Let's get started</h3>
        <ol>
            <li>Upload one or more PDF files in the sidebar.</li>
            <li>Click <b>Process documents</b> - the text is split into chunks and indexed with FAISS.</li>
            <li>Ask questions below. Each answer shows its <b>source citations</b> and the
                <b>retrieved context</b> so you can verify it.</li>
        </ol>
    </div>""", unsafe_allow_html=True)


def render_sources(message: dict) -> None:
    """Show the file + page citation chips under an assistant answer."""
    if message.get("error"):
        return
    if message["sources"]:
        chips = "".join(f'<span class="chip">📄 {html.escape(s["source"])} · p. {s["page"]}</span>'
                        for s in message["sources"])
    else:
        chips = '<span class="chip none">No supporting source found in the documents</span>'
    st.markdown(f'<div class="sources"><span class="label">Sources:</span>{chips}</div>', unsafe_allow_html=True)


def render_retrieved_context(message: dict) -> None:
    """Expandable section listing every passage the retriever returned."""
    retrieved = message.get("retrieved") or []
    if not retrieved:
        return
    with st.expander(f"🔍 Retrieved context ({len(retrieved)} passages)"):
        st.caption("Passages returned by the FAISS similarity search, ranked by cosine similarity. "
                   "The numbers match the [n] citations in the answer.")
        for c in retrieved:
            width = max(0, min(100, int(c["score"] * 100)))
            st.markdown(f"""
            <div class="passage">
                <div class="passage-head"><b>[{c["rank"]}]</b> {html.escape(c["source"])} · page {c["page"]}
                    · similarity {c["score"]:.3f}</div>
                <div class="score-bar"><div style="width:{width}%"></div></div>{html.escape(c["text"])}</div>""",
                        unsafe_allow_html=True)


def render_assistant_message(message: dict) -> None:
    if message.get("error"):
        st.error(message["content"])
        return
    st.markdown(message["content"])
    render_sources(message)
    meta = f'Top-K {message["top_k"]} · temperature {message["temperature"]}'
    if message.get("tokens"):
        meta += f' · {message["tokens"]} tokens'
    st.markdown(f'<div class="answer-meta">{meta} · {message["time"]}</div>', unsafe_allow_html=True)
    render_retrieved_context(message)


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍🎓" if message["role"] == "user" else "🔎"):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant_message(message)


def conversation_to_markdown() -> str:
    lines = [f"# {SETTINGS.app_name} - conversation export\n"]
    for m in st.session_state.messages:
        if m["role"] == "user":
            lines.append(f"**Question:** {m['content']}\n")
        else:
            lines.append(f"**Answer:** {m['content']}\n")
            for s in m.get("sources", []):
                lines.append(f"- Source: {s['source']}, page {s['page']}")
            lines.append("")
    return "\n".join(lines)


# ==========================================================================
# Question answering (query phase)
# ==========================================================================
def answer_question(question: str, top_k: int, temperature: float) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🔎"):
        with st.spinner("Searching your documents and writing an answer..."):
            try:
                result = st.session_state.pipeline.ask(question, top_k=top_k, temperature=temperature)
                message = {
                    "role": "assistant",
                    "content": result.answer,
                    "sources": result.sources,
                    "retrieved": [asdict(c) for c in result.retrieved],
                    "tokens": result.tokens_used,
                    "top_k": top_k,
                    "temperature": temperature,
                    "time": datetime.now().strftime("%H:%M"),
                }
            except RAGError as exc:
                message = {"role": "assistant", "content": str(exc), "error": True}
        render_assistant_message(message)

    st.session_state.messages.append(message)


# ==========================================================================
# Main
# ==========================================================================
def main() -> None:
    init_session_state()
    options = render_sidebar()

    render_header()

    # Indexing runs before the statistics are drawn so the numbers are fresh.
    if options["process_clicked"]:
        if not options["api_key"]:
            st.error("An OpenAI API key is required. Add OPENAI_API_KEY to your .env file "
                     "or enter it in the sidebar.")
        else:
            st.session_state.messages = []  # new documents -> new conversation
            st.session_state.pipeline = None
            process_documents(options["files"], options["api_key"],
                              options["chunk_size"], options["chunk_overlap"])

    render_stats()

    chat_col, side_col = st.columns([2.4, 1], gap="large")
    with side_col:
        render_document_panel()
    with chat_col:
        if st.session_state.pipeline is None:
            render_welcome()
        render_chat_history()

    ready = st.session_state.pipeline is not None
    question = st.chat_input("Ask a question about your documents..." if ready
                             else "Upload and process PDFs to start chatting", disabled=not ready)
    if question:
        with chat_col:
            answer_question(question, options["top_k"], options["temperature"])
        st.rerun()  # refresh the statistics cards and the export button


if __name__ == "__main__":
    main()

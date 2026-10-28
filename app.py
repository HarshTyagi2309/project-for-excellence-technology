import hashlib
import streamlit as st

from services.rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="DocuMind | AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- UI STYLING --------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f7f9ff 0%, #f1f5fb 100%);
    }
    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263449;
    }
    [data-testid="stSidebar"] * {
        color: #f3f4f6;
    }
    .hero {
        padding: 26px 28px;
        border-radius: 20px;
        color: white;
        background: linear-gradient(120deg, #172554, #3730a3, #6d28d9);
        margin-bottom: 22px;
        box-shadow: 0 12px 30px rgba(55, 48, 163, 0.18);
    }
    .hero h1 {
        font-size: 30px;
        margin: 0 0 8px 0;
        color: white;
    }
    .hero p {
        color: #e0e7ff;
        margin: 0;
        font-size: 15px;
    }
    .metric-card {
        background: white;
        border: 1px solid #e3e8f2;
        border-radius: 14px;
        padding: 15px 17px;
        min-height: 84px;
        box-shadow: 0 3px 12px rgba(31, 41, 55, 0.04);
    }
    .metric-label {
        color: #64748b;
        font-size: 12px;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #172554;
        font-weight: 700;
        font-size: 19px;
        overflow-wrap: anywhere;
    }
    .section-label {
        font-size: 12px;
        font-weight: 700;
        color: #64748b;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin: 18px 0 8px 0;
    }
    div[data-testid="stChatMessage"] {
        border: 1px solid #e6eaf2;
        border-radius: 14px;
        padding: 12px;
        background: rgba(255,255,255,0.88);
    }
    .stButton button {
        border-radius: 10px;
        font-weight: 600;
    }
    .footer-note {
        color: #64748b;
        font-size: 12px;
        text-align: center;
        padding: 20px 0 8px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    return RAGPipeline()


pipeline = get_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_fingerprint" not in st.session_state:
    st.session_state.active_fingerprint = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## 📚 DocuMind")
    st.caption("AI-powered document Q&A")
    st.divider()

    st.markdown("### 📄 Your document")
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"],
        help="Your document is processed automatically after upload.",
    )

    st.caption("PDF and TXT files are supported.")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### ⚙️ How it works")
    st.markdown(
        "1. Upload your document\n\n"
        "2. AI splits it into chunks\n\n"
        "3. FAISS finds relevant text\n\n"
        "4. Groq generates a grounded answer"
    )
    st.divider()
    st.caption("Embeddings: Sentence Transformers")
    st.caption("Vector search: FAISS")
    st.caption("LLM: Groq")


# -------------------- AUTO PROCESS UPLOAD --------------------
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    fingerprint = hashlib.sha256(file_bytes).hexdigest()

    if fingerprint != st.session_state.active_fingerprint:
        try:
            with st.spinner(
                "Preparing your document... First-time model download may take a while."
            ):
                chunk_count = pipeline.process_document(
                    uploaded_file.name,
                    file_bytes,
                )

            st.session_state.active_fingerprint = fingerprint
            st.session_state.document_name = uploaded_file.name
            st.session_state.chunk_count = chunk_count
            st.session_state.messages = []
            st.session_state.document_error = None

        except Exception as error:
            st.session_state.document_error = str(error)
            st.session_state.active_fingerprint = None
            st.session_state.document_name = None
            st.session_state.chunk_count = 0

if st.session_state.get("document_error"):
    st.error(
        "Could not process this document: "
        + st.session_state.document_error
    )


# -------------------- MAIN HEADER --------------------
st.markdown("""
<div class="hero">
    <h1>📚 Chat with your documents</h1>
    <p>Upload a document. Ask questions. Get answers grounded in its content.</p>
</div>
""", unsafe_allow_html=True)


# -------------------- DOCUMENT STATUS --------------------
metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">DOCUMENT</div>
        <div class="metric-value">{
            "Ready" if st.session_state.document_name else "Not uploaded"
        }</div></div>""",
        unsafe_allow_html=True,
    )

with metric2:
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">FILE NAME</div>
        <div class="metric-value" style="font-size:14px;">{
            st.session_state.document_name or "—"
        }</div></div>""",
        unsafe_allow_html=True,
    )

with metric3:
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">TEXT CHUNKS</div>
        <div class="metric-value">{st.session_state.chunk_count}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# -------------------- CHAT HISTORY --------------------
st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)

if not st.session_state.document_name:
    st.info(
        "👈 Upload a PDF or TXT file in the sidebar. "
        "The document will be processed automatically."
    )

    st.markdown("#### Try the sample document")
    st.caption("Use data/sample_faq.txt to test the app.")
else:
    st.success("Document ready — ask your first question below.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("🔎 View retrieved source chunks"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**Chunk {source['chunk_id']}** · "
                            f"Similarity: {source['score']:.3f}"
                        )
                        st.write(source["text"])
                        st.divider()


# -------------------- SUGGESTED QUESTIONS --------------------
if st.session_state.document_name and not st.session_state.messages:
    st.markdown("#### 💡 Suggested questions")
    suggestions = [
        "Summarize this document",
        "What are the main points?",
        "What information should I know?",
    ]
    cols = st.columns(len(suggestions))

    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion, use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()


# -------------------- QUESTION INPUT --------------------
typed_question = st.chat_input("Ask anything about your uploaded document...")

question = typed_question
if st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

if question:
    if pipeline.vector_store.index is None:
        st.warning("Please upload a readable PDF or TXT document first.")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the document and preparing an answer..."):
                    result = pipeline.ask(question)

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                with st.expander("🔎 View retrieved source chunks"):
                    if not sources:
                        st.caption("No source chunks were retrieved.")
                    for source in sources:
                        st.markdown(
                            f"**Chunk {source['chunk_id']}** · "
                            f"Similarity: {source['score']:.3f}"
                        )
                        st.write(source["text"])
                        st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as error:
                st.error(f"Could not generate an answer: {error}")

st.markdown(
    '<div class="footer-note">'
    'DocuMind · Retrieval-Augmented Generation · '
    'Answers are based on retrieved document context'
    '</div>',
    unsafe_allow_html=True,
)

# DOCUMIND_READABILITY_FIX
st.markdown("""
<style>
/* Improve readability across the complete app */
.stApp,
.stApp p,
.stApp label,
.stApp span,
.stApp li,
.stApp div[data-testid="stMarkdownContainer"] {
    color: #172033;
}

/* Sidebar text */
section[data-testid="stSidebar"] {
    background-color: #111827;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #F9FAFB !important;
}

/* Uploaded filename and uploader controls */
div[data-testid="stFileUploader"],
div[data-testid="stFileUploader"] * {
    color: #172033 !important;
}
div[data-testid="stFileUploader"] section {
    background-color: #FFFFFF !important;
}

/* Input field and placeholder */
.stTextInput input,
.stTextArea textarea,
.stChatInput textarea {
    color: #172033 !important;
    background-color: #FFFFFF !important;
    opacity: 1 !important;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder,
.stChatInput textarea::placeholder {
    color: #667085 !important;
    opacity: 1 !important;
}

/* Buttons, including disabled-looking controls */
.stButton button,
.stDownloadButton button,
div[data-testid="stFileUploader"] button {
    color: #172033 !important;
    opacity: 1 !important;
}
.stButton button p,
.stDownloadButton button p {
    color: inherit !important;
    opacity: 1 !important;
}

/* Keep chat messages readable */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
    color: #172033 !important;
}

/* Avoid faded secondary text */
.stApp small,
.stApp [data-testid="stCaptionContainer"] {
    color: #475467 !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# DOCUMIND_TARGETED_UI_FIX
st.markdown("""
<style>
/* Make the file-upload area and its controls readable */
.stApp [data-testid="stFileUploader"] section {
    background: #FFFFFF !important;
    border: 1px solid #D0D5DD !important;
}
.stApp [data-testid="stFileUploader"] section,
.stApp [data-testid="stFileUploader"] section * {
    opacity: 1 !important;
}
.stApp [data-testid="stFileUploader"] section button {
    background: #F2F4F7 !important;
    color: #172033 !important;
    border: 1px solid #98A2B3 !important;
}
.stApp [data-testid="stFileUploader"] section button * {
    color: #172033 !important;
}
.stApp [data-testid="stFileUploader"] small,
.stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
.stApp [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] {
    color: #475467 !important;
}
.stApp [data-testid="stFileUploader"] section span,
.stApp [data-testid="stFileUploader"] section p {
    color: #344054 !important;
}

/* Restore white text on gradient hero/banner */
.stApp div[style*="linear-gradient"] p,
.stApp div[style*="linear-gradient"] span {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

/* Keep normal content and form controls readable */
.stApp input,
.stApp textarea {
    color: #172033 !important;
    background-color: #FFFFFF !important;
}
.stApp input::placeholder,
.stApp textarea::placeholder {
    color: #667085 !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

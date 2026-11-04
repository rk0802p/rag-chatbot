"""
Streamlit Web Interface for RAG Chatbot.

Provides a robust, multi-turn conversational UI for chatting with PDF documents,
complete with session state caching, grounded LLM responses, and verifiable citations.
"""

import os
import hashlib
import streamlit as st
from dotenv import load_dotenv

import backend
from backend import (
    PDFProcessor,
    TextChunker,
    EmbeddingManager,
    FAISSVectorStore,
    LLMClient,
    PDFExtractionError,
    RAGError
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for cards and badges
st.markdown(
    """
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .citation-box {
        border-left: 3px solid #4CAF50;
        padding-left: 10px;
        margin-bottom: 12px;
        font-size: 0.9em;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================================
# Session State Initialization
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None

if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}


# ============================================================================
# Sidebar Configuration & Document Ingestion
# ============================================================================

with st.sidebar:
    st.title("⚙️ RAG Settings & PDF")
    st.markdown("Upload a document to extract text, build a local FAISS index, and query with grounded AI.")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    # Document Ingestion Handler (Runs only when a NEW file is detected)
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]

        if st.session_state.current_file_hash != file_hash:
            with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                try:
                    # Ingest directly from memory bytes
                    pages = PDFProcessor.extract_pages(file_bytes)
                    chunks = TextChunker.chunk_pages(
                        pages=pages,
                        document_name=uploaded_file.name,
                        document_id=file_hash
                    )

                    texts = [c.text for c in chunks]
                    embeddings = EmbeddingManager.encode_texts(texts)

                    store = FAISSVectorStore(dimension=embeddings.shape[1])
                    store.add_chunks(chunks, embeddings)

                    # Update session state
                    st.session_state.vector_store = store
                    st.session_state.current_file_hash = file_hash
                    st.session_state.current_file_name = uploaded_file.name
                    st.session_state.doc_stats = {
                        "pages": len(pages),
                        "chunks": len(chunks),
                        "chars": sum(len(p["text"]) for p in pages)
                    }
                    # Reset chat history for the new document
                    st.session_state.messages = []
                    st.success(f"Indexed '{uploaded_file.name}' ({len(pages)} pages, {len(chunks)} chunks)")

                except PDFExtractionError as pe:
                    st.error(f"PDF Extraction Failed: {str(pe)}")
                    st.session_state.vector_store = None
                    st.session_state.current_file_hash = None
                except Exception as e:
                    st.error(f"Processing Error: {str(e)}")
                    st.session_state.vector_store = None
                    st.session_state.current_file_hash = None

    elif st.session_state.current_file_hash is not None and uploaded_file is None:
        # User removed the file
        st.session_state.vector_store = None
        st.session_state.current_file_hash = None
        st.session_state.current_file_name = None
        st.session_state.doc_stats = {}
        st.session_state.messages = []
        st.info("Document cleared.")

    # Document Info Display
    if st.session_state.vector_store is not None:
        st.markdown("### 📄 Active Document")
        stats = st.session_state.doc_stats
        st.markdown(
            f"""
            - **File:** `{st.session_state.current_file_name}`
            - **Pages:** `{stats.get('pages', 0)}`
            - **Chunks:** `{stats.get('chunks', 0)}`
            - **Vectors Indexed:** `{st.session_state.vector_store.index.ntotal}`
            """
        )

    st.divider()

    # Retrieval Tuning Parameters
    st.markdown("### 🔍 Retrieval Parameters")
    top_k = st.slider("Top-K Passages", min_value=1, max_value=8, value=4, help="Number of context passages retrieved")
    score_threshold = st.slider(
        "Min Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.05,
        help="Cosine similarity score cutoff (-1.0 to 1.0)"
    )

    st.divider()

    # LLM Provider Configuration
    st.markdown("### 🤖 AI Model & Provider")
    provider = st.selectbox(
        "AI Engine",
        options=[
            "Built-in (Free / No Key Required)",
            "Groq (Llama 3.3 70B - Free Key)",
            "OpenAI (Cloud)",
            "Local (Ollama / LM Studio)",
            "Custom Endpoint"
        ],
        index=0,
        help="Choose between the built-in free document engine, Groq, OpenAI, or local models"
    )

    env_api_key = os.getenv("OPENAI_API_KEY", "")
    env_base_url = os.getenv("OPENAI_BASE_URL", "")
    env_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    active_base_url = None
    active_api_key = None
    selected_model = "builtin"

    if provider == "Built-in (Free / No Key Required)":
        selected_model = "builtin-extractive-synthesizer"
        active_api_key = None
        active_base_url = None
        st.caption("🟢 Built-in semantic document engine active (100% free, zero setup required)")

    elif provider == "Groq (Llama 3.3 70B - Free Key)":
        groq_key = st.text_input(
            "Groq API Key (Free at console.groq.com)",
            value="",
            type="password",
            help="Free API key from console.groq.com (no credit card required)"
        )
        active_api_key = groq_key.strip() or env_api_key.strip() or None
        active_base_url = "https://api.groq.com/openai/v1"
        selected_model = "llama-3.3-70b-versatile"
        if active_api_key:
            st.caption("🟢 Groq Llama 3.3 70B configured")
        else:
            st.caption("🟡 Paste your free Groq key or use Built-in engine")

    elif provider == "OpenAI (Cloud)":
        api_key_input = st.text_input(
            "OpenAI API Key",
            value="",
            type="password",
            help="Leave blank to use OPENAI_API_KEY from environment"
        )
        active_api_key = api_key_input.strip() or env_api_key.strip() or None
        selected_model = st.selectbox(
            "Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0
        )
        active_base_url = env_base_url.strip() or None
        if active_api_key:
            st.caption(f"🟢 OpenAI `{selected_model}` active")
        else:
            st.caption("🟡 Paste OpenAI key or use Built-in engine")

    elif provider == "Local (Ollama / LM Studio)":
        local_base = st.text_input(
            "Local Base URL",
            value=env_base_url or "http://localhost:11434/v1",
            help="Ollama is http://localhost:11434/v1, LM Studio is http://localhost:1234/v1"
        )
        selected_model = st.text_input(
            "Model Name",
            value="llama3.2",
            help="Ensure this model is installed locally (e.g. ollama run llama3.2)"
        )
        active_base_url = local_base.strip()
        active_api_key = "local-key-not-required"
        st.caption(f"🟢 Local endpoint: `{selected_model}`")

    elif provider == "Custom Endpoint":
        custom_base = st.text_input(
            "Base URL",
            value=env_base_url or "https://api.groq.com/openai/v1"
        )
        custom_key = st.text_input("API Key", type="password")
        selected_model = st.text_input("Model Identifier", value="llama-3.3-70b-versatile")
        active_base_url = custom_base.strip()
        active_api_key = custom_key.strip() or env_api_key.strip() or None

    st.divider()

    # Chat Actions
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================================
# Main Chat Interface
# ============================================================================

st.title("📄💬 Chat with Your PDF")
st.markdown("Ask precise questions and receive answers grounded strictly in your document's text with page citations.")

# Active Document Warning / Banner
if st.session_state.vector_store is None:
    st.info("👈 **Upload a PDF in the sidebar** to start asking questions.")
else:
    st.caption(f"Active Document: **{st.session_state.current_file_name}** | Chunks: **{st.session_state.doc_stats.get('chunks', 0)}**")

# Render Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display expandable source citations for assistant messages
        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"📚 View {len(sources)} Verified Source Citations"):
                for idx, src in enumerate(sources, 1):
                    chunk = src["chunk"]
                    st.markdown(
                        f"""
                        <div class="citation-box">
                            <b>Citation #{idx}</b> — <i>Page {chunk['page_number']}</i> (Relevance: <code>{src['score']:.3f}</code>)<br/>
                            <small>Chunk ID: <code>{chunk['chunk_id']}</code></small><br/>
                            <p style="margin-top: 6px;">"{chunk['text']}"</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# Chat Input & Response Generation
user_input = st.chat_input(
    "Ask anything about your document...",
    disabled=(st.session_state.vector_store is None)
)

if user_input:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Initialize LLM client
    llm_client = LLMClient(
        api_key=active_api_key,
        model=selected_model,
        base_url=active_base_url
    )

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching document passages and synthesizing answer..."):
            try:
                rag_response = backend.query_pdf(
                    question=user_input,
                    vector_store=st.session_state.vector_store,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    llm_client=llm_client,
                    chat_history=st.session_state.messages
                )

                st.markdown(rag_response.answer)

                # Display citation verification banner if hallucinated citations detected
                if rag_response.citation_validation and not rag_response.citation_validation.get("is_valid", True):
                    hallucinated = rag_response.citation_validation.get("hallucinated_citations", [])
                    st.warning(
                        f"⚠️ **Citation Alert:** The assistant referenced Page(s) {hallucinated}, "
                        "which were not in the retrieved passages. Use caution regarding unverified page claims."
                    )

                # Format source data for session state
                serialized_sources = [
                    {
                        "chunk": res.chunk.to_dict(),
                        "score": res.score
                    }
                    for res in rag_response.sources
                ]

                # Render sources
                if serialized_sources:
                    with st.expander(f"📚 View {len(serialized_sources)} Verified Source Citations"):
                        for idx, src in enumerate(serialized_sources, 1):
                            chunk = src["chunk"]
                            st.markdown(
                                f"""
                                <div class="citation-box">
                                    <b>Citation #{idx}</b> — <i>Page {chunk['page_number']}</i> (Relevance: <code>{src['score']:.3f}</code>)<br/>
                                    <small>Chunk ID: <code>{chunk['chunk_id']}</code></small><br/>
                                    <p style="margin-top: 6px;">"{chunk['text']}"</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                # Append assistant response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": rag_response.answer,
                    "sources": serialized_sources
                })

            except RAGError as re:
                error_msg = f"RAG Pipeline Error: {str(re)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
            except Exception as e:
                error_msg = f"An unexpected error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})

import streamlit as st
import os
import uuid
import json
from pathlib import Path
from dotenv import load_dotenv

from src.embeddings import LocalEmbedder
from src.ingestion import (
    load_documents_from_paths,
    load_documents_from_uploads,
    chunk_documents,
)
from src.models import Document
from src.store import EmbeddingStore

# --- INITIALIZATION ---
load_dotenv(override=True)

# Custom CSS for a professional light theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap');
    
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
    }
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #3B82F6, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 1rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Message styling */
    div[data-testid="stChatMessage"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        padding: 1rem;
    }
    
    /* Button and Input refinement */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stMetric {
        background: #FFFFFF;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "store_a" not in st.session_state:
    st.session_state.store_a = None
if "store_b" not in st.session_state:
    st.session_state.store_b = None
if "is_built" not in st.session_state:
    st.session_state.is_built = False
if "source_docs" not in st.session_state:
    st.session_state.source_docs = []
if "embedder" not in st.session_state:
    st.session_state.embedder = None

# --- UTILS ---
def get_embedder():
    if st.session_state.embedder is None:
        provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
        try:
            with st.spinner(f"Initializing {provider.capitalize()} Embedder..."):
                if provider == "openai":
                    from src.embeddings import OpenAIEmbedder
                    st.session_state.embedder = OpenAIEmbedder()
                else:
                    st.session_state.embedder = LocalEmbedder()
        except Exception as e:
            st.error(f"Failed to load {provider} embedder: {e}. Falling back to demo mode.")
            from src.embeddings import MockEmbedder
            st.session_state.embedder = MockEmbedder()
    return st.session_state.embedder

def _demo_llm(prompt: str) -> str:
    preview = prompt[:420].replace("\n", " ")
    return f"[DEMO ANSWER] Prompt preview: {preview}..."

def _call_real_llm(question: str, retrieved: list[dict]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
    
    context_lines = [f"[{i+1}] {item.get('content', '').strip()}" for i, item in enumerate(retrieved)]
    context = "\n".join(context_lines) if context_lines else "(no relevant context found)"
    
    system_prompt = (
        "You are a professional grounded RAG assistant.\n"
        "Strict Rule: Answer ONLY using the provided context. If the context does not contain the answer, "
        "politely state that you don't have enough information.\n"
        "Maintain a helpful and objective tone."
    )
    user_message = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    if not api_key:
        return _demo_llm(user_message)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content or "No response from AI."
    except Exception as exc:
        return f"[LLM ERROR] {exc}\n{_demo_llm(user_message)}"

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.markdown('<div class="main-header" style="font-size: 1.5rem;">⚡ Config</div>', unsafe_allow_html=True)
    
    st.subheader("Global Settings")
    compare_mode = st.toggle("Compare Mode", value=False)
    use_sample_data = st.checkbox("Include Sample Data", value=True)
    top_k = st.slider("Top-K Retrieval", 1, 10, 3)
    llm_mode = st.selectbox("LLM Mode", ["Real LLM", "Demo Mode"])
    
    st.divider()
    
    st.subheader("Controls")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_c2:
        if st.button("🔄 Reset App", use_container_width=True):
            st.session_state.clear()
            st.rerun()
            
    st.divider()
    
    # Workspace A
    st.markdown("**WORKSPACE A**")
    method_a = st.selectbox("Chunking Method A", ["fixed_size", "recursive", "by_sentences"], key="method_a")
    params_a = {}
    if method_a == "fixed_size":
        params_a["chunk_size"] = st.number_input("Chunk Size A", 100, 2000, 500, step=100)
        params_a["overlap"] = st.number_input("Overlap A", 0, 500, 50, step=10)
    elif method_a == "recursive":
        params_a["chunk_size"] = st.number_input("Chunk Size A", 100, 2000, 500, step=100)
        params_a["separators"] = st.text_input("Separators A (CSV)", "\\n\\n,\\n,. , , ").split(',')
        params_a["separators"] = [s.strip().replace('\\n', '\n') for s in params_a["separators"]]
    else:
        params_a["max_sentences_per_chunk"] = st.number_input("Max Sentences A", 1, 10, 3)

    if compare_mode:
        st.divider()
        st.markdown("**WORKSPACE B**")
        method_b = st.selectbox("Chunking Method B", ["fixed_size", "recursive", "by_sentences"], key="method_b")
        params_b = {}
        if method_b == "fixed_size":
            params_b["chunk_size"] = st.number_input("Chunk Size B", 100, 2000, 500, step=100)
            params_b["overlap"] = st.number_input("Overlap B", 0, 500, 50, step=10)
        elif method_b == "recursive":
            params_b["chunk_size"] = st.number_input("Chunk Size B", 100, 2000, 500, step=100)
            params_b["separators"] = st.text_input("Separators B (CSV)", "\\n\\n,\\n,. , , ").split(',')
            params_b["separators"] = [s.strip().replace('\\n', '\n') for s in params_b["separators"]]
        else:
            params_b["max_sentences_per_chunk"] = st.number_input("Max Sentences B", 1, 10, 3)

# --- MAIN UI ---
st.markdown('<div class="main-header">RAG Studio</div>', unsafe_allow_html=True)

# Status Bar using columns & metrics for a premium feel
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("System Status", "🟢 Online" if st.session_state.is_built else "⚪ Idle")
with m2:
    st.metric("Indexed Docs", len(st.session_state.source_docs))
with m3:
    st.metric("Compare Mode", "ON ⚖️" if compare_mode else "OFF")
with m4:
    st.metric("LLM Provider", "Demo Mode 🤖" if llm_mode == "Demo Mode" else "Real LLM ⚡")

st.divider()

# Ingestion Section
with st.expander("📁 Document Ingestion", expanded=not st.session_state.is_built):
    uploaded_files = st.file_uploader("Upload local files (.txt, .md, .pdf, .docx)", accept_multiple_files=True)
    if st.button("Build Index", type="primary", use_container_width=True):
        if not uploaded_files and not use_sample_data:
            st.error("Please provide at least one source (upload files or use sample data).")
        else:
            with st.spinner("Building RAG Index..."):
                st.write("Initializing embedder...")
                embedder = get_embedder()
                
                all_docs = []
                if use_sample_data:
                    st.write("Loading sample data...")
                    SAMPLE_FILES = [
                        "data/python_intro.txt", "data/vector_store_notes.md", 
                        "data/rag_system_design.md", "data/customer_support_playbook.txt",
                        "data/chunking_experiment_report.md", "data/vi_retrieval_notes.md"
                    ]
                    all_docs.extend(load_documents_from_paths(SAMPLE_FILES))
                
                if uploaded_files:
                    st.write(f"Processing {len(uploaded_files)} uploaded files...")
                    all_docs.extend(load_documents_from_uploads(uploaded_files))
                
                st.session_state.source_docs = all_docs
                
                st.write("Syncing Workspace A...")
                chunk_a = chunk_documents(all_docs, method=method_a, params=params_a)
                st.write(f"Workspace A: Created {len(chunk_a)} chunks.")
                store_a = EmbeddingStore(collection_name="rag_a", embedding_fn=embedder)
                
                # Batch processing for high performance
                BATCH_SIZE = 200
                with st.empty():
                    for i in range(0, len(chunk_a), BATCH_SIZE):
                        batch = chunk_a[i:i+BATCH_SIZE]
                        store_a.add_documents(batch)
                        st.info(f"Workspace A: Indexed {min(i+BATCH_SIZE, len(chunk_a))}/{len(chunk_a)} chunks...")
                st.session_state.store_a = store_a
                
                if compare_mode:
                    st.write("Syncing Workspace B...")
                    chunk_b = chunk_documents(all_docs, method=method_b, params=params_b)
                    st.write(f"Workspace B: Created {len(chunk_b)} chunks.")
                    store_b = EmbeddingStore(collection_name="rag_b", embedding_fn=embedder)
                    with st.empty():
                        for i in range(0, len(chunk_b), BATCH_SIZE):
                            batch = chunk_b[i:i+BATCH_SIZE]
                            store_b.add_documents(batch)
                            st.success(f"Workspace B: Indexed {min(i+BATCH_SIZE, len(chunk_b))}/{len(chunk_b)} chunks...")
                    st.session_state.store_b = store_b
                
                st.session_state.is_built = True
                st.success("Index Built Successfully!")
                import time
                time.sleep(0.5)
                st.rerun()

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            if "panel_b" in msg:
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("WORKSPACE A")
                    st.write(msg["panel_a"]["answer"])
                    with st.expander("Retrieved Chunks"):
                        for res in msg["panel_a"]["results"]:
                            st.info(f"Source: {res['metadata'].get('source')} | Score: {res['score']:.2f}\n\n{res['content']}")

                with c2:
                    st.caption("WORKSPACE B")
                    st.write(msg["panel_b"]["answer"])
                    with st.expander("Retrieved Chunks"):
                        for res in msg["panel_b"]["results"]:
                            st.success(f"Source: {res['metadata'].get('source')} | Score: {res['score']:.2f}\n\n{res['content']}")
            else:
                st.write(msg["panel_a"]["answer"])
                with st.expander("Retrieved Chunks"):
                    for res in msg["panel_a"]["results"]:
                        st.info(f"Source: {res['metadata'].get('source')} | Score: {res['score']:.2f}\n\n{res['content']}")

if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.is_built:
        st.warning("Please build the index first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            answer_fn = _call_real_llm if llm_mode == "Real LLM" else lambda q, r: _demo_llm(f"Question: {q}\nRetrieved: {len(r)} chunks")
            
            with st.spinner("Analyzing context..."):
                # Panel A
                results_a = st.session_state.store_a.search(prompt, top_k=top_k)
                ans_a = answer_fn(prompt, results_a)
                panel_a = {"query": prompt, "answer": ans_a, "results": results_a}
                
                # Panel B
                panel_b = None
                if compare_mode and st.session_state.store_b:
                    results_b = st.session_state.store_b.search(prompt, top_k=top_k)
                    ans_b = answer_fn(prompt, results_b)
                    panel_b = {"query": prompt, "answer": ans_b, "results": results_b}
                
            msg = {"role": "assistant", "panel_a": panel_a}
            if panel_b:
                msg["panel_b"] = panel_b
                
            st.session_state.messages.append(msg)
            st.rerun()

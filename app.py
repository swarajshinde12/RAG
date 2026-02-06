import os
import io
import time
import numpy as np
import faiss
import streamlit as st
import PyPDF2
import google.generativeai as genai

# HYBRID SEARCH IMPORTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- IMPORT CONFIG ----------
try:
    import config
except ImportError:
    raise RuntimeError("config.py not found. Make sure it exists.")

# ---------- GEMINI SETUP ----------
genai.configure(api_key=config.GEMINI_API_KEY)

CHAT_MODEL_NAME = config.GEMINI_CHAT_MODEL          # "models/gemini-2.5-flash"
EMBED_MODEL_NAME = config.GEMINI_EMBED_MODEL        # "models/text-embedding-004"
CHUNK_SIZE = getattr(config, "CHUNK_SIZE", 800)
CHUNK_OVERLAP = getattr(config, "CHUNK_OVERLAP", 150)
TOP_K = getattr(config, "TOP_K", 5)


# ============================================================
#                PDF EXTRACTION + CHUNKING
# ============================================================
def extract_text_from_pdf(file_obj):
    reader = PyPDF2.PdfReader(file_obj)
    text = ""
    for page in reader.pages:
        try:
            text += page.extract_text() + "\n"
        except:
            continue
    return text


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.replace("\n", " ")
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start = end - overlap
        if start < 0:
            start = 0

    return chunks


# ============================================================
#                    EMBEDDINGS (GEMINI)
# ============================================================
def embed_texts(texts, task_type="retrieval_document"):
    vectors = []
    for t in texts:
        if not t.strip():
            vectors.append(np.zeros((768,), dtype="float32"))
            continue

        emb = genai.embed_content(
            model=EMBED_MODEL_NAME,
            content=t,
            task_type=task_type
        )
        vectors.append(np.array(emb["embedding"], dtype="float32"))

    return np.vstack(vectors)


# ============================================================
#           BUILD FAISS INDEX + BUILD TF-IDF MATRIX
# ============================================================
def build_index_from_pdfs(files):
    all_chunks = []
    all_sources = []

    st.write("📄 Processing uploaded PDFs...")

    for f in files:
        st.write(f"➡ Extracting: **{f.name}**")
        raw_bytes = f.read()
        text = extract_text_from_pdf(io.BytesIO(raw_bytes))

        chunks = chunk_text(text)
        st.write(f"• {len(chunks)} chunks created")

        all_chunks.extend(chunks)
        all_sources.extend([f.name] * len(chunks))

    # Dense embeddings
    st.write("🧠 Generating embeddings (Gemini)...")
    embeddings = embed_texts(all_chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # TF-IDF sparse retrieval
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_chunks)

    st.success(f"✅ Knowledge Base Ready! Total chunks: {len(all_chunks)}")

    return index, all_chunks, all_sources, vectorizer, tfidf_matrix


# ============================================================
#                         RETRIEVAL
# ============================================================
def retrieve_dense(query, index, chunks, sources, k=TOP_K):
    q_vec = embed_texts([query], task_type="retrieval_query")
    D, I = index.search(q_vec, k)

    results = []
    for idx, dist in zip(I[0], D[0]):
        results.append({
            "index": int(idx),
            "chunk": chunks[idx],
            "source": sources[idx],
            "distance": float(dist)
        })

    return results


def retrieve_sparse(query, vectorizer, tfidf_matrix, k=TOP_K):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()

    top_idx = sims.argsort()[::-1][:k]
    return [
        {"index": int(i), "score": float(sims[i])}
        for i in top_idx
    ]


def merge_dense_sparse(dense, sparse, chunks, sources, alpha=0.65):
    sparse_dict = {r["index"]: r["score"] for r in sparse}

    hybrid = []
    for dr in dense:
        idx = dr["index"]
        dense_score = -dr["distance"]              # convert distance → similarity
        sparse_score = sparse_dict.get(idx, 0)

        final_score = alpha * dense_score + (1 - alpha) * sparse_score

        hybrid.append({
            "index": idx,
            "chunk": chunks[idx],
            "source": sources[idx],
            "distance": dr["distance"],
            "hybrid_score": final_score
        })

    hybrid = sorted(hybrid, key=lambda x: x["hybrid_score"], reverse=True)
    return hybrid[:TOP_K]


# ============================================================
#                        GENERATE ANSWER
# ============================================================
def generate_answer(query, hybrid_chunks):
    context = ""
    for r in hybrid_chunks:
        context += (
            f"[Source: {r['source']} | Chunk #{r['index']} | Score={r['hybrid_score']:.4f}]\n"
            f"{r['chunk']}\n\n---\n\n"
        )

    prompt = f"""
You are KnowItBot. Answer only using the context below.
If info is missing, say: "I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION:
{query}
"""

    model = genai.GenerativeModel(CHAT_MODEL_NAME)
    response = model.generate_content(prompt)

    return response.text


# ============================================================
#                       STREAMLIT UI
# ============================================================
def main():
    st.set_page_config(page_title="KnowItBot RAG", page_icon="🤖", layout="wide")

    # Inject ChatGPT-style CSS
    chat_css = """
    <style>
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        border-radius: 12px;
        background-color: #f0f2f6;
    }
    .user-msg {
        background-color: #d1e7ff;
        padding: 10px 16px;
        border-radius: 12px;
        margin: 8px 0;
        max-width: 75%;
        align-self: flex-end;
    }
    .bot-msg {
        background-color: #ffffff;
        padding: 10px 16px;
        border-radius: 12px;
        margin: 8px 0;
        border-left: 4px solid #6a5acd;
        max-width: 75%;
    }
    .avatar {
        height: 35px;
        width: 35px;
        border-radius: 50%;
        margin-right: 8px;
    }
    </style>
    """
    st.markdown(chat_css, unsafe_allow_html=True)

    st.title("🤖 KnowItBot — Hybrid RAG Chatbot")
    st.caption("FAISS + TF-IDF Hybrid Search | Gemini 2.5 Flash")

    # Initialize session state
    if "index" not in st.session_state:
        st.session_state.index = None
        st.session_state.chunks = None
        st.session_state.sources = None
        st.session_state.vectorizer = None
        st.session_state.tfidf_matrix = None
        st.session_state.chat_history = []

    # ---------------- PDF Upload ----------------
    st.subheader("📚 Upload PDFs")
    uploaded = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if st.button("🚀 Build Knowledge Base"):
        if not uploaded:
            st.error("Please upload at least one PDF.")
            return

        with st.spinner("Building Knowledge Base..."):
            index, chunks, sources, vectorizer, tfidf_matrix = build_index_from_pdfs(uploaded)

            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.sources = sources
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = tfidf_matrix

    st.markdown("---")

    # ---------------- Chat UI ----------------
    st.subheader("💬 Chat with KnowItBot")

    chat_box = st.container()
    with chat_box:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='display:flex; justify-content:flex-end;'>"
                    f"<div class='user-msg'>{msg['content']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='display:flex; align-items:flex-start;'>"
                    f"<img class='avatar' src='https://cdn-icons-png.flaticon.com/512/4712/4712102.png'>"
                    f"<div class='bot-msg'>{msg['content']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    query = st.chat_input("Ask something…")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        dense = retrieve_dense(query, st.session_state.index, st.session_state.chunks, st.session_state.sources)
        sparse = retrieve_sparse(query, st.session_state.vectorizer, st.session_state.tfidf_matrix)
        hybrid = merge_dense_sparse(dense, sparse, st.session_state.chunks, st.session_state.sources)

        answer = generate_answer(query, hybrid)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        st.subheader("📄 Retrieved Context")
        for h in hybrid:
            with st.expander(f"{h['source']} | Chunk {h['index']} | Score={h['hybrid_score']:.4f}"):
                st.write(h["chunk"])

        st.rerun()


if __name__ == "__main__":
    main()

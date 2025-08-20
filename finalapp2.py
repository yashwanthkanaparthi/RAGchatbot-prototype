# app_single.py
# One-file Streamlit Document QA app (UI + VectorStore + Chatbot)
# - Streamlit UI with "+" uploader and filename pill
# - VectorStore: PyMuPDF -> text, semantic chunking, sentence-transformer embeddings, optional FAISS, optional OCR, BGE reranker
# - Chatbot: Groq chat completion (streamed), answers strictly from retrieved docs
#
# Requirements (typical):
#   streamlit
#   python-dotenv
#   groq
#   pymupdf
#   sentence-transformers
#   faiss-cpu                (optional; falls back to numpy search if missing)
#   pillow, pytesseract      (optional; for OCR on scanned PDFs)
#
# Env (.env or system):
#   GROQ_API_KEY=...
#   GROQ_CHAT_MODEL=llama-3.3-70b-versatile   (default)
#   EMBED_MODEL=pritamdeka/S-PubMedBert-MS-MARCO
#   RERANK_MODEL=BAAI/bge-reranker-v2-m3
#
# Run:
#   streamlit run app_single.py

import os
import re
import uuid
from typing import List, Dict, Any, Optional

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from textwrap import dedent

# ---- PDF text extraction ----
import fitz  # PyMuPDF

# ---- Optional OCR (for scanned PDFs) ----
try:
    import pytesseract
    from PIL import Image
    import io
    _OCR_AVAILABLE = True
except Exception:
    _OCR_AVAILABLE = False

# ---- Embedding + Reranker (HuggingFace) ----
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---- FAISS (vector index) ----
try:
    import faiss  # faiss-cpu
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

# ---- Groq LLM ----
from groq import Groq


# ============================
# VectorStore (merged)
# ============================
class VectorStore:
    """
    Open-source RAG backend:
      • Semantic chunking (sentence-level cosine dips)
      • PubMedBERT embeddings (configurable via EMBED_MODEL env)
      • FAISS index (cosine via normalized vectors)
      • BGE cross-encoder reranker (configurable via RERANK_MODEL env)
    """
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

        # Configurable (via env) model IDs
        self.embed_model_id = "abhinand/MedEmbed-large-v0.1"
        self.rerank_model_id = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

        # Init models
        self._embedder = SentenceTransformer(self.embed_model_id)
        self._reranker = None   # lazy load

        # State
        self.page_texts: List[str] = []
        self.page_count: int = 0
        self.pdf_text: str = ""
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None  # shape: (N, d)
        self.index = None

        # Retrieval knobs
        self.retrieve_top_k = 12
        self.rerank_top_k = 5

        # Pipeline
        self.load_pdf()
        # Prefer semantic chunking; fallback to simple length-based if needed
        try:
            self.split_text_semantic()
            if not self.chunks:
                self.split_text_fallback()
        except Exception:
            self.split_text_fallback()
        self.embed_chunks()
        self.index_chunks()

    # ---------------- PDF → text ----------------
    def load_pdf(self) -> None:
        text_all = []
        with fitz.open(self.pdf_path) as pdf:
            self.page_count = pdf.page_count
            for i in range(self.page_count):
                page = pdf.load_page(i)
                t = page.get_text("text")
                if not t.strip() and _OCR_AVAILABLE:
                    try:
                        pix = page.get_pixmap()
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        t = pytesseract.image_to_string(img)
                    except Exception:
                        t = ""
                self.page_texts.append(t or "")
                text_all.append(t or "")
        self.pdf_text = "\n".join(text_all)

    # ---------------- Chunking ----------------
    def split_text_fallback(self, chunk_size: int = 1000) -> None:
        """Simple length-based chunking on sentence boundaries."""
        cur = []
        cur_len = 0
        sentences = [s.strip() for s in re.split(r'(?<=[\.!?])\s+', self.pdf_text) if s.strip()]
        self.chunks = []
        for s in sentences:
            s_len = len(s) + 1
            if cur_len + s_len > chunk_size and cur:
                self.chunks.append(" ".join(cur).strip())
                cur, cur_len = [], 0
            cur.append(s)
            cur_len += s_len
        if cur:
            self.chunks.append(" ".join(cur).strip())

    def split_text_semantic(
        self,
        desired_chunk_chars: int = 1200,
        min_chunk_chars: int = 600,
        max_chunk_chars: int = 2000,
        sim_threshold: float = 0.35,
        batch_size: int = 96,
    ) -> None:
        """
        Semantic chunking:
          1) Split into sentences
          2) Embed sentences
          3) Compute cosine similarity of adjacent sentences
          4) Break at low-sim points (with min/desired/max guards)
        """
        text = self.pdf_text or ""
        sentences = [s.strip() for s in re.split(r'(?<=[\.!?])\s+', text) if s.strip()]
        if not sentences:
            self.chunks = []
            return

        # Embed sentences
        embs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            arr = self._embedder.encode(batch, batch_size=min(64, batch_size),
                                        convert_to_numpy=True, normalize_embeddings=True)
            embs.append(arr)
        sent_vecs = np.vstack(embs)  # (S, d)

        # Cosine similarities of adjacent sentences
        sims = (sent_vecs[:-1] * sent_vecs[1:]).sum(axis=1)  # normalized → dot = cosine

        # Dynamic threshold (slightly below corpus mean)
        if sims.size > 0:
            mean_sim = float(sims.mean())
            dyn_thresh = min(sim_threshold, max(0.0, mean_sim - 0.15))
        else:
            dyn_thresh = sim_threshold

        chunks, cur, cur_len = [], [], 0

        def flush():
            nonlocal chunks, cur, cur_len
            if cur:
                chunks.append(" ".join(cur).strip())
                cur, cur_len = [], 0

        for i, sent in enumerate(sentences):
            s_len = len(sent) + 1
            # hard cap
            if cur and (cur_len + s_len > max_chunk_chars):
                flush()
            cur.append(sent)
            cur_len += s_len

            if i < len(sentences) - 1:
                boundary = sims[i] < dyn_thresh
                long_enough = cur_len >= desired_chunk_chars
                if (boundary and cur_len >= min_chunk_chars) or long_enough:
                    flush()

        flush()
        self.chunks = [c for c in chunks if c]

    # ---------------- Embeddings ----------------
    def embed_chunks(self, batch_size: int = 64) -> None:
        if not self.chunks:
            self.embeddings = None
            return
        arr = self._embedder.encode(
            self.chunks,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # enables inner product for cosine
        )
        self.embeddings = arr.astype(np.float32)

    # ---------------- Indexing (FAISS) ----------------
    def index_chunks(self) -> None:
        if self.embeddings is None or self.embeddings.size == 0:
            self.index = None
            return
        dim = self.embeddings.shape[1]
        if _FAISS_AVAILABLE:
            # Cosine via normalized vectors → use inner product
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        else:
            # Fallback: we'll brute-force in numpy in retrieve()
            self.index = None

    # ---------------- Retrieval + Rerank ----------------
    def _ensure_reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoder(self.rerank_model_id)

    def _search(self, qvec: np.ndarray, top_k: int) -> List[int]:
        """Return indices of top_k nearest neighbors."""
        if self.embeddings is None or self.embeddings.size == 0:
            return []
        if _FAISS_AVAILABLE and self.index is not None:
            D, I = self.index.search(qvec[None, :].astype(np.float32), top_k)
            return I[0].tolist()
        # Linear scan fallback
        sims = (self.embeddings @ qvec.astype(np.float32))
        return np.argsort(-sims)[:top_k].tolist()

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        if not self.chunks or self.embeddings is None:
            return []

        # Query embedding (normalized)
        qvec = self._embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)

        # Vector search
        cand_idx = self._search(qvec, self.retrieve_top_k)
        docs = [self.chunks[i] for i in cand_idx]

        if not docs:
            return []

        # Rerank with cross-encoder (optional but recommended)
        try:
            self._ensure_reranker()
            pairs = [(query, d) for d in docs]
            scores = self._reranker.predict(pairs)  # higher = more relevant
            order = np.argsort(-np.array(scores))[: self.rerank_top_k]
            return [{"text": docs[i]} for i in order]
        except Exception:
            # If reranker is missing or errors, just return vector hits
            return [{"text": d} for d in docs[: self.rerank_top_k]]


# ============================
# Chatbot (merged)
# ============================

# --- Global system prompt ---
DEFAULT_SYSTEM_PROMPT = dedent("""
You are a precise document QA assistant. Answer ONLY using the supplied documents.
- If the user asks to extract specific sections (e.g., Abstract, Methods, Results), find and return those spans verbatim where possible.
- For tables, reconstruct table content present in text as Markdown tables; if table text is not present, say so.
- For figures/images, report captions or references if present in text; otherwise explain they are not available as text.
- If the answer is not present in the provided documents, say you cannot find it.
Keep answers concise and faithful to the source text.""")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)


class Chatbot:
    def __init__(self, vectorstore: VectorStore, groq_api_key: Optional[str] = None):
        self.vectorstore = vectorstore
        self.client = Groq(api_key=groq_api_key or os.getenv("GROQ_API_KEY"))
        # Options: "llama-3.1-70b-versatile" (quality) or "llama-3.1-8b-instant" (speed)
        self.chat_model = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
        self.conversation_id = str(uuid.uuid4())

    def respond(self, user_message: str):
        # Retrieve context using the user message
        docs = self.vectorstore.retrieve(user_message) or []

        # Flatten docs into a single context block
        context_blocks = []
        for i, d in enumerate(docs, 1):
            t = d.get("text", "")
            if t:
                context_blocks.append(f"[DOC {i}]\\n{t}")
        context_text = "\\n\\n".join(context_blocks) if context_blocks else "No documents retrieved."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User request:\\n{user_message}\\n\\n"
                    f"Relevant documents:\\n{context_text}\\n\\n"
                    f"Answer the user strictly from the documents above."
                ),
            },
        ]

        # Stream completion and assemble final text
        stream = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            stream=True,
            temperature=0.2,
        )

        out = []
        for chunk in stream:
            delta = getattr(chunk.choices[0].delta, "content", None)
            if delta:
                out.append(delta)

        return "".join(out), docs, "rag"


# ============================
# Streamlit UI (merged)
# ============================
# --- Page config ---
st.set_page_config(page_title="Document QA Bot", page_icon="🤖", layout="wide")

# --- Env keys ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Global CSS: "+" uploader & polish (filename pill, no heading anchor) ---
GLOBAL_CSS = """
<style>
div.block-container h1 a, div.block-container h2 a { display: none !important; }
[data-testid="stFileUploader"] { margin: 0 !important; }
[data-testid="stFileUploader"] > label { display: none !important; }
[data-testid="stFileUploader"] {
  width: 44px !important; height: 44px !important; min-width: 44px !important; min-height: 44px !important;
  display: inline-flex !important; align-items: center !important; justify-content: center !important;
  padding: 0 !important; overflow: visible !important;
}
[data-testid="stFileUploader"] * { display: none !important; }
[data-testid="stFileUploaderDropzone"] {
  display: flex !important; align-items: center !important; justify-content: center !important;
  width: 44px !important; height: 44px !important; padding: 0 !important; border-radius: 9999px !important;
  border: none !important; background: rgba(255,255,255,0.06) !important; cursor: pointer !important;
}
[data-testid="stFileUploaderDropzone"] * { display: none !important; }
[data-testid="stFileUploaderDropzone"]::after { content: "+"; font-size: 20px; font-weight: 700; opacity: 0.95; line-height: 1; pointer-events: none; }
[data-testid="stFileUploaderDropzone"]:hover { background: rgba(255,255,255,0.10) !important; }
[data-testid="stFileUploaderDropzone"]:focus, [data-testid="stFileUploaderDropzone"]:focus-within { outline: none !important; }
.upload-name{ margin-top: 6px; font-size: 0.85rem; opacity: 0.85; max-width: 160px; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; border: 1px solid rgba(150,150,150,.35); padding: .2rem .5rem;
  border-radius: 9999px; display: inline-block; background: rgba(255,255,255,.04); }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

def _ensure_ss_keys():
    for k in ("uploaded_pdf_bytes", "uploaded_pdf_name"):
        if k not in st.session_state:
            st.session_state[k] = None

def main():
    _ensure_ss_keys()
    st.title("Document QA Bot 🤖")

    if not GROQ_API_KEY:
        st.error("Set GROQ_API_KEY in your environment or .env."); st.stop()

    # --- Top row: inline +uploader, text box, Send button ---
    c_plus, c_input, c_send = st.columns([0.055, 1, 0.15])

    with c_plus:
        f = st.file_uploader("upload_file:", type=["pdf"], label_visibility="collapsed",
                             accept_multiple_files=False, key="plus_uploader")
        if f is not None:
            st.session_state.uploaded_pdf_bytes = f.read()
            st.session_state.uploaded_pdf_name = f.name

        name = st.session_state.get("uploaded_pdf_name")
        if name:
            st.markdown(f'<div class="upload-name" title="{name}">{name}</div>', unsafe_allow_html=True)

    with c_input:
        q = st.text_input("Upload PDF and Ask anything",
                          placeholder="Upload PDF and Ask anything…",
                          label_visibility="collapsed", key="user_q")

    with c_send:
        send = st.button("Send", use_container_width=True)

    # --- Handle submit ---
    if not send:
        return
    if not q.strip():
        st.warning("Type a question first."); return
    if st.session_state.uploaded_pdf_bytes is None:
        st.warning("Please upload a PDF first using the “＋” button."); return

    with st.spinner("Processing your PDF..."):
        with open("uploaded_document.pdf", "wb") as out:
            out.write(st.session_state.uploaded_pdf_bytes)
        vs = VectorStore("uploaded_document.pdf")
        bot = Chatbot(vs, GROQ_API_KEY)

    with st.spinner("Generating answer..."):
        out, docs, _ = bot.respond(q)

    # --- Render output (pure RAG) ---
    st.write(f"**You:** {q}")
    if docs:
        with st.expander("Documents used"):
            for i, d in enumerate(docs, 1):
                txt = d.get("text", "")
                st.write(f"{i}. {txt[:400]}{'...' if len(txt) > 400 else ''}")
    else:
        st.warning("No documents were retrieved; the model is answering without context.")

    st.write(f"**Bot:** {out}")


if __name__ == "__main__":
    main()

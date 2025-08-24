import os
import re
import io
import json
import uuid
import hashlib
from pathlib import Path
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

# --- Medical spell-check (SymSpell) ---
from symspellpy import SymSpell, Verbosity

_SYMSPELL = None

def _get_symspell():
    """
    Singleton SymSpell using your merged MeSH dictionary.
    Set MEDICAL_VOCAB_PATH to point to medical_vocab_all.tsv (or the supp/desc TSV you want).
    """
    global _SYMSPELL
    if _SYMSPELL is not None:
        return _SYMSPELL

    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    vocab_path = os.getenv("MEDICAL_VOCAB_PATH", "medical_vocab_all.tsv")
    if not os.path.exists(vocab_path):
        # Fall back to local 'dicts' folder if you keep it there
        alt = os.path.join("dicts", "medical_vocab_all.tsv")
        if os.path.exists(alt):
            vocab_path = alt
        else:
            # If dictionary not found, we still return an empty SymSpell (no-ops)
            _SYMSPELL = sym
            return _SYMSPELL

    sym.load_dictionary(vocab_path, term_index=0, count_index=1, separator="\t")
    _SYMSPELL = sym
    return _SYMSPELL


def medical_spell_correct(text: str) -> str:
    """
    Conservative token-wise correction for alphabetic tokens (>=3 chars).
    Restores simple casing (Title/UPPER) after correction.
    """
    sym = _get_symspell()
    tokens = re.split(r'(\W+)', text)  # keep separators (spaces, punctuation)
    out = []
    for tok in tokens:
        tlo = tok.lower()
        if (not tok) or (not tok.isalpha()) or (len(tok) < 3):
            out.append(tok); continue

        suggestions = sym.lookup(tlo, Verbosity.TOP, max_edit_distance=2, include_unknown=True)
        cand = suggestions[0].term if suggestions else tlo

        # restore casing
        if tok.istitle():
            cand = cand.title()
        elif tok.isupper():
            cand = cand.upper()

        out.append(cand)
    return "".join(out)


# =========================================
# Config / Globals
# =========================================
load_dotenv()

# Hardcode models (as we agreed earlier)
EMBED_MODEL_ID = "abhinand/MedEmbed-large-v0.1"
RERANK_MODEL_ID = "BAAI/bge-reranker-v2-m3"
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ⚠️ System prompt restored to the plain, earlier style (same spirit as your original finalapp2.py)
GENERAL_SYSTEM_INSTRUCTIONS = """
You are a precise document QA assistant. Answer ONLY using the supplied documents.
- If the user asks to extract specific sections (e.g., Abstract, Methods, Results), find and return those spans verbatim where possible.
- For tables, reconstruct table content present in text as Markdown tables; if table text is not present, say so.
- For figures/images, report captions or references if present in text; otherwise explain they are not available as text.
- If the answer is not present in the provided documents, say you cannot find it.
Keep answers concise and faithful to the source text.
"""

DEFAULT_RETRIEVE_TOP_K = 12
DEFAULT_RERANK_TOP_K = 5

CACHE_ROOT = Path(".rag_cache")


# =========================================
# Streamlit Page
# =========================================
st.set_page_config(page_title="Document QA Bot", page_icon="🤖", layout="wide")

# Global CSS (includes the "+" uploader and small polish)
GLOBAL_CSS = """
<style>
div.block-container h1 a, div.block-container h2 a { display: none !important; }

/* + uploader button */
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

.upload-name{ margin-top: 6px; font-size: 0.85rem; opacity: 0.85; max-width: 220px; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; border: 1px solid rgba(150,150,150,.35); padding: .2rem .5rem;
  border-radius: 9999px; display: inline-block; background: rgba(255,255,255,.04); }

/* Sidebar history cards */
.history-item{ padding:.5rem .6rem; border-radius:.6rem; border:1px solid rgba(150,150,150,.25); margin-bottom:.4rem; }
.history-q{ font-weight:600; font-size:.9rem; margin-bottom:.2rem; }
.history-a{ font-size:.85rem; opacity:.9; }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

if not GROQ_API_KEY:
    st.error("Set GROQ_API_KEY in your environment or .env before running.")
    st.stop()


# =========================================
# Utility: hashing and cache paths
# =========================================
def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def cache_dir_for(pdf_path: str, embed_model_id: str) -> Path:
    CACHE_ROOT.mkdir(exist_ok=True)
    return CACHE_ROOT / f"{sha256_file(pdf_path)[:16]}_{embed_model_id.replace('/','__')}"


# =========================================
# Cache: models (singleton per process)
# =========================================
@st.cache_resource(show_spinner=False)
def get_embedder(model_id: str):
    return SentenceTransformer(model_id)

@st.cache_resource(show_spinner=False)
def get_reranker(model_id: str):
    return CrossEncoder(model_id)


# =========================================
# VectorStore
# =========================================
class VectorStore:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.embed_model_id = EMBED_MODEL_ID
        self.rerank_model_id = RERANK_MODEL_ID

        self._embedder = get_embedder(self.embed_model_id)
        self._reranker: Optional[CrossEncoder] = None

        self.page_texts: List[str] = []
        self.page_count: int = 0
        self.pdf_text: str = ""
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None

        self.retrieve_top_k = DEFAULT_RETRIEVE_TOP_K
        self.rerank_top_k = DEFAULT_RERANK_TOP_K

        # Per-PDF cache
        self.cache_dir = cache_dir_for(self.pdf_path, self.embed_model_id)
        if not self._try_load_cache():
            self.load_pdf()
            try:
                self.split_text_semantic()
                if not self.chunks:
                    self.split_text_fallback()
            except Exception:
                self.split_text_fallback()
            self.embed_chunks()
            self.index_chunks()
            self._save_cache()

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
        text = self.pdf_text or ""
        sentences = [s.strip() for s in re.split(r'(?<=[\.!?])\s+', text) if s.strip()]
        if not sentences:
            self.chunks = []
            return

        embs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            arr = self._embedder.encode(batch, batch_size=min(64, batch_size),
                                        convert_to_numpy=True, normalize_embeddings=True)
            embs.append(arr)
        sent_vecs = np.vstack(embs)

        # Cosine similarities of adjacent sentences (normalized vectors → dot=cosine)
        sims = (sent_vecs[:-1] * sent_vecs[1:]).sum(axis=1) if len(sent_vecs) > 1 else np.array([])

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
            if cur and (cur_len + s_len > max_chunk_chars):
                flush()
            cur.append(sent)
            cur_len += s_len

            if i < len(sentences) - 1:
                boundary = sims[i] < dyn_thresh if sims.size > i else False
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
            normalize_embeddings=True,
        )
        self.embeddings = arr.astype(np.float32)

    # ---------------- Indexing (FAISS) ----------------
    def index_chunks(self) -> None:
        if self.embeddings is None or self.embeddings.size == 0:
            self.index = None
            return
        dim = self.embeddings.shape[1]
        if _FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dim)  # cosine with normalized vectors
            self.index.add(self.embeddings)
        else:
            self.index = None

    # ---------------- Retrieval + Rerank ----------------
    def _ensure_reranker(self):
        if self._reranker is None:
            self._reranker = get_reranker(self.rerank_model_id)

    def _search(self, qvec: np.ndarray, top_k: int) -> List[int]:
        if self.embeddings is None or self.embeddings.size == 0:
            return []
        if _FAISS_AVAILABLE and self.index is not None:
            D, I = self.index.search(qvec[None, :].astype(np.float32), top_k)
            return I[0].tolist()
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

    # ---------------- Cache I/O ----------------
    def _try_load_cache(self) -> bool:
        try:
            meta_p = self.cache_dir / "meta.json"
            chunks_p = self.cache_dir / "chunks.json"
            emb_p = self.cache_dir / "embeddings.npy"
            faiss_p = self.cache_dir / "faiss.index"
            if not meta_p.exists() or not chunks_p.exists():
                return False
            meta = json.load(open(meta_p, "r", encoding="utf-8"))
            if meta.get("embed_model_id") != self.embed_model_id:
                return False
            self.chunks = json.load(open(chunks_p, "r", encoding="utf-8"))
            if emb_p.exists():
                self.embeddings = np.load(emb_p).astype(np.float32)
            if _FAISS_AVAILABLE and faiss_p.exists():
                self.index = faiss.read_index(str(faiss_p))
            self.page_count = meta.get("page_count", 0)
            return True
        except Exception:
            return False

    def _save_cache(self) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "embed_model_id": self.embed_model_id,
                "dim": int(self.embeddings.shape[1]) if self.embeddings is not None else None,
                "page_count": int(getattr(self, "page_count", 0)),
            }
            json.dump(meta, open(self.cache_dir / "meta.json", "w", encoding="utf-8"))
            if self.chunks:
                json.dump(self.chunks, open(self.cache_dir / "chunks.json", "w", encoding="utf-8"))
            if self.embeddings is not None:
                np.save(self.cache_dir / "embeddings.npy", self.embeddings)
            if _FAISS_AVAILABLE and self.index is not None:
                faiss.write_index(self.index, str(self.cache_dir / "faiss.index"))
        except Exception:
            pass


# =========================================
# Chatbot (restored answering behavior)
# =========================================
class Chatbot:
    def __init__(self, vectorstore: VectorStore, groq_api_key: Optional[str] = None):
        self.vectorstore = vectorstore
        self.client = Groq(api_key=groq_api_key or os.getenv("GROQ_API_KEY"))
        self.chat_model = GROQ_CHAT_MODEL
        self.conversation_id = str(uuid.uuid4())

    def respond(self, user_message: str):
        # Retrieve context using the user message
        corrected_query = medical_spell_correct(user_message)
        docs = self.vectorstore.retrieve(corrected_query) or []

        # Flatten docs into a single context block
        context_blocks = []
        for i, d in enumerate(docs, 1):
            t = d.get("text", "")
            if t:
                context_blocks.append(f"[DOC {i}]\n{t}")
        context_text = "\n\n".join(context_blocks) if context_blocks else "No documents retrieved."

        messages = [
            {"role": "system", "content": GENERAL_SYSTEM_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"User request:\n{user_message}\n\n"
                    f"Relevant documents:\n{context_text}\n\n"
                    f"Answer the user strictly from the documents above."
                ),
            },
        ]

        # Stream completion and assemble final text — temp=0.2 (like your original)
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


# =========================================
# Session State helpers (PDF hash, history)
# =========================================
def ensure_session_state():
    defaults = {
        "pdf_bytes": None,
        "pdf_name": None,
        "pdf_hash": None,
        "vectorstore": None,
        "chat_history": [],  # list of dicts: {"q":..., "a":...}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_for_new_pdf(pdf_bytes: bytes, file_name: str):
    st.session_state["pdf_bytes"] = pdf_bytes
    st.session_state["pdf_name"] = file_name
    st.session_state["pdf_hash"] = sha256_bytes(pdf_bytes)
    st.session_state["chat_history"] = []  # flush history for previous PDF
    st.session_state["vectorstore"] = None


# =========================================
# Build UI (single row: + uploader, input, Send)
# =========================================
ensure_session_state()

with st.sidebar:
    st.header("📄 Current PDF")
    if st.session_state["pdf_name"]:
        st.markdown(f'<div class="upload-name" title="{st.session_state["pdf_name"]}">{st.session_state["pdf_name"]}</div>', unsafe_allow_html=True)
    else:
        st.caption("Upload a PDF to start.")
    st.markdown("---")
    st.subheader("🕘 History (this PDF)")
    if st.session_state["chat_history"]:
        for i, item in enumerate(st.session_state["chat_history"], 1):
            with st.container():
                st.markdown(
                    f'<div class="history-item"><div class="history-q">{i}. {item["q"]}</div>'
                    f'<div class="history-a">{item["a"][:200]}{"..." if len(item["a"])>200 else ""}</div></div>',
                    unsafe_allow_html=True
                )
    else:
        st.caption("No questions yet for this document.")

st.title("Document QA Bot 🤖")

# Top row: + uploader, text input, Send — aligned horizontally
c_plus, c_input, c_send = st.columns([0.055, 1, 0.15])

with c_plus:
    f = st.file_uploader("upload_file:", type=["pdf"], label_visibility="collapsed",
                         accept_multiple_files=False, key="plus_uploader")
    if f is not None:
        pdf_bytes = f.read()
        file_hash = sha256_bytes(pdf_bytes)
        if st.session_state["pdf_hash"] != file_hash:
            reset_for_new_pdf(pdf_bytes, f.name)

    # Show current filename below the +
    name = st.session_state.get("pdf_name")
    if name:
        st.markdown(f'<div class="upload-name" title="{name}">{name}</div>', unsafe_allow_html=True)

with c_input:
    q = st.text_input("Upload PDF and Ask anything",
                      placeholder="Upload PDF and Ask anything…",
                      label_visibility="collapsed", key="user_q")

with c_send:
    send = st.button("Send", use_container_width=True)

# --- Handle submit ---
if send:
    if not q.strip():
        st.warning("Type a question first."); st.stop()
    if st.session_state["pdf_bytes"] is None:
        st.warning("Please upload a PDF first using the “＋” button."); st.stop()

    # Persist PDF to disk for vectorstore (cache key uses file hash+model id)
    tmp_path = "uploaded_document.pdf"
    with open(tmp_path, "wb") as out:
        out.write(st.session_state["pdf_bytes"])

    # Build or reuse vectorstore in-memory (per-session)
    if st.session_state["vectorstore"] is None:
        with st.spinner("Indexing your PDF..."):
            vs = VectorStore(tmp_path)
            st.session_state["vectorstore"] = vs
    else:
        vs = st.session_state["vectorstore"]

    bot = Chatbot(vs, GROQ_API_KEY)

    with st.spinner("Generating answer..."):
        out_text, docs, _ = bot.respond(q)

    # --- Render output (pure RAG) ---
    st.write(f"**You:** {q}")
    if docs:
        with st.expander("Documents used"):
            for i, d in enumerate(docs, 1):
                txt = d.get("text", "")
                st.write(f"{i}. {txt[:500]}{'...' if len(txt) > 500 else ''}")
    else:
        st.warning("No documents were retrieved; the model is answering without context.")

    st.write(f"**Bot:** {out_text}")

    # Save to sidebar history (current PDF only)
    st.session_state["chat_history"].append({"q": q, "a": out_text})

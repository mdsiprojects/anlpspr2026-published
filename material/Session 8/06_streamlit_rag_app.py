
# to run, use the following command from the root of the repository:
# streamlit run material/Session\ 8/06_streamlit_rag_app.py

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


load_dotenv()


CURRENT_FILE = Path(__file__).resolve()
SESSION_DIR = CURRENT_FILE.parent
HELPERS_DIR = SESSION_DIR / "helpers"
if str(HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(HELPERS_DIR))

from pdf_utils import extract_pdf_text, join_pages
from rag_utils import (
    build_faiss_index,
    build_grounded_prompt,
    chunk_text,
    package_chunks,
    search_index,
)


def setting(name: str, default=None):
    """Use local environment values or deployed Streamlit secrets."""
    if os.getenv(name):
        return os.environ[name]
    try:
        return st.secrets.get(name) or default
    except FileNotFoundError:
        return default


OPENAI_API_KEY = setting("OPENAI_API_KEY")
OPENAI_ORG_ID = setting("OPENAI_ORG_ID")
OPENAI_PROJECT_ID = setting("OPENAI_PROJECT_ID")
GENERATION_MODEL = setting("SESSION8_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = setting("SESSION8_EMBEDDING_MODEL", "text-embedding-3-small")

DEFAULT_SYSTEM_PROMPT = """\
# PURPOSE:
You are a friendly assistant that answers questions from the company's HR policy and information documents.
Answer the query using only the sources provided below in a friendly and concise bulleted manner.
Answer ONLY with the facts listed in the list of sources below.
If there isn't enough information below, say you don't know.
Do not generate answers that don't use the sources below.

# CITATION:
- Always include source in brackets in the response from the list of sources below.
- If you use multiple sources, cite them all in brackets in the response.
- Include a list of sources as a footnote in bullet-point format.
- Only include sources that were used to generate the answer.
- Don't display any sources section if nothing was cited in the response.\
"""


def get_generation_client() -> OpenAI | None:
    if not OPENAI_API_KEY:
        return None
    return OpenAI(
        api_key=OPENAI_API_KEY,
        organization=OPENAI_ORG_ID,
        project=OPENAI_PROJECT_ID,
    )


@st.cache_resource
def load_embedding_backend():
    if OPENAI_API_KEY:
        return "openai", OpenAI(
            api_key=OPENAI_API_KEY,
            organization=OPENAI_ORG_ID,
            project=OPENAI_PROJECT_ID,
        )
    return "local", SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]) -> np.ndarray:
    backend_name, backend = load_embedding_backend()
    if backend_name == "openai":
        response = backend.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return np.array([item.embedding for item in response.data], dtype="float32")
    return backend.encode(texts, convert_to_numpy=True).astype("float32")


@st.cache_resource
def build_session8_index():
    pdf_dir = SESSION_DIR / "data" / "pdfs"
    documents = []

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        pages = extract_pdf_text(pdf_path)
        full_text = join_pages(pages)
        chunks = chunk_text(full_text, chunk_size=500, overlap=100)
        documents.extend(package_chunks(chunks, pdf_path.name))

    chunk_texts = [doc["text"] for doc in documents]
    if not chunk_texts:
        raise ValueError("No extractable text found in data/pdfs.")
    embeddings = embed_texts(chunk_texts)
    index = build_faiss_index(embeddings)
    return documents, index


def retrieve_chunks(question: str, top_k: int = 4) -> list[dict]:
    documents, index = build_session8_index()
    query_embedding = embed_texts([question])[0]
    _, indices = search_index(index, query_embedding, top_k=top_k)
    return [documents[idx] for idx in indices[0]]


def stream_grounded_answer(question: str, retrieved_chunks: list[dict], system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    client = get_generation_client()
    if client is None:
        yield "OPENAI_API_KEY is required for the Streamlit RAG app."
        return

    prompt = build_grounded_prompt(question, retrieved_chunks)

    stream = client.responses.create(
        model=GENERATION_MODEL,
        instructions=system_prompt,
        input=prompt,
        stream=True,
    )

    for event in stream:
        if event.type == "response.output_text.delta":
            yield event.delta


st.set_page_config(page_title="Session 8 RAG App", page_icon=":books:")
st.title("Session 8 Streamlit RAG App")
st.caption("Streaming grounded answers over the Session 8 handbook and policy PDF corpus.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
top_k = st.sidebar.slider("Top-k retrieved chunks", min_value=2, max_value=6, value=4)
st.sidebar.markdown(f"Generation model: `{GENERATION_MODEL}`")

st.sidebar.divider()

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

def reset_system_prompt():
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

st.sidebar.text_area(
    "System Prompt",
    key="system_prompt",
    height=340,
    help="Edit the instructions sent to the model before every answer.",
)
st.sidebar.button("Reset to default", on_click=reset_system_prompt)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not OPENAI_API_KEY:
    st.info("Set OPENAI_API_KEY in your .env file or Streamlit secrets to enable chat.")

if prompt := st.chat_input("Ask about the handbook or benefits documents", disabled=not OPENAI_API_KEY):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    retrieved = retrieve_chunks(prompt, top_k=top_k)
    st.session_state.last_sources = retrieved

    with st.chat_message("assistant"):
        response_text = st.write_stream(
            stream_grounded_answer(prompt, retrieved, system_prompt=st.session_state.system_prompt)
        )

    st.session_state.messages.append({"role": "assistant", "content": response_text})

if st.session_state.last_sources:
    with st.expander("Retrieved sources from the latest answer", expanded=False):
        for chunk in st.session_state.last_sources:
            st.markdown(f"**{chunk['source_file']} | chunk {chunk['chunk_id']}**")
            st.write(chunk["text"][:800] + ("..." if len(chunk["text"]) > 800 else ""))
            st.divider()

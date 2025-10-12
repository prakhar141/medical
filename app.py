# app_clean.py
import os
import time
import json
import hashlib
import logging
from typing import List, Dict
from datetime import datetime

import requests
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =====================================================
# CONFIGURATION
# =====================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY")
MODEL_MAIN = "deepseek/deepseek-r1:free"
MODEL_FALLBACKS = [
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "tiiuae/falcon-180b-chat:free"
]
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
K_VAL = int(os.getenv("K_VAL", "4"))

MAX_RETRIES = 5
BASE_BACKOFF = 2
MAX_BACKOFF = 30
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

FAISS_INDEX_URL = "https://huggingface.co/datasets/prakhar146/derma/resolve/main/index.faiss"
FAISS_PKL_URL = "https://huggingface.co/datasets/prakhar146/derma/resolve/main/index.pkl"
LOCAL_FAISS_DIR = "./faiss_store"
os.makedirs(LOCAL_FAISS_DIR, exist_ok=True)

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("derma_consult")

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="Derma Consult", layout="wide", page_icon="💎")
st.markdown("""
<style>
body {background-color: #f9fafb; font-family: 'Inter', sans-serif;}
.main {padding: 2rem;}
h1 {text-align: center; font-weight: 700; color: #2d3436; letter-spacing: -0.5px;}
.user, .assistant {border-radius: 15px; padding: 1rem; margin: 0.6rem 0;}
.user {background-color: #e8f0fe;}
.assistant {background-color: #f1f8e9;}
hr {border: none; border-top: 1px solid #ddd; margin-top: 40px;}
</style>
""", unsafe_allow_html=True)

st.title("💎 Derma Consult")
st.markdown("#### AI-assisted dermatological reasoning— clear, precise, and reliable.")

# =====================================================
# UTILITIES
# =====================================================
def download_file(url: str, local_path: str):
    if os.path.exists(local_path):
        return
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        logger.info(f"Downloaded {url}")
    except Exception as e:
        st.error(f"Failed to download {url}: {e}")

def hash_prompt(model: str, messages: List[Dict[str, str]]):
    raw = model + json.dumps(messages, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()

# =====================================================
# VECTOR DATABASE
# =====================================================
download_file(FAISS_INDEX_URL, os.path.join(LOCAL_FAISS_DIR, "index.faiss"))
download_file(FAISS_PKL_URL, os.path.join(LOCAL_FAISS_DIR, "index.pkl"))

@st.cache_resource
def load_vector_db():
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.load_local(LOCAL_FAISS_DIR, emb, allow_dangerous_deserialization=True)
    return vectordb.as_retriever(search_type="similarity", k=K_VAL)

retriever = load_vector_db()

# =====================================================
# IN-MEMORY CACHE
# =====================================================
if "prompt_cache" not in st.session_state:
    st.session_state.prompt_cache = {}

CACHE_LIMIT = 2000
def cache_get(key): return st.session_state.prompt_cache.get(key)
def cache_set(key, value):
    st.session_state.prompt_cache[key] = value
    if len(st.session_state.prompt_cache) > CACHE_LIMIT:
        st.session_state.prompt_cache.pop(next(iter(st.session_state.prompt_cache)))

# =====================================================
# OPENROUTER CALLER
# =====================================================
def exponential_backoff(attempt):
    return min(MAX_BACKOFF, BASE_BACKOFF * (2 ** attempt))

def call_openrouter(model: str, messages: List[Dict[str, str]]) -> str:
    key = hash_prompt(model, messages)
    cached = cache_get(key)
    if cached:
        return cached

    backoff = BASE_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(OPENROUTER_URL, headers=HEADERS, json={"model": model, "messages": messages}, timeout=30)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", backoff))
                logger.warning(f"Rate limited — sleeping {retry_after}s")
                time.sleep(retry_after)
                backoff *= 2
                continue
            if resp.status_code >= 500:
                logger.warning(f"Server error {resp.status_code} — retrying in {backoff}s")
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            data = resp.json()
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"]
                cache_set(key, content)
                return content
            return json.dumps(data)
        except requests.RequestException as e:
            logger.error(f"Network error: {e}")
            time.sleep(backoff)
            backoff *= 2
            continue
    raise RuntimeError("OpenRouter request failed after retries")

def call_with_fallbacks(messages: List[Dict[str, str]]) -> str:
    for model in [MODEL_MAIN] + MODEL_FALLBACKS:
        try:
            return call_openrouter(model, messages)
        except Exception as e:
            logger.warning(f"{model} failed → {e}")
            continue
    return "⚠️ All models temporarily unavailable. Please retry later."

# =====================================================
# RAG PIPELINE
# =====================================================
SYSTEM_PROMPT = (
    "You are Derma Consult, a board-certified dermatologist and medical educator. "
    "Provide precise, evidence-based explanations and diagnostic reasoning. "
    "Be structured, concise, and clinically sound — as if medical doctor.give very short answers in one line only"
)

def rag_answer(question: str) -> str:
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([d.page_content for d in docs]) if docs else "No relevant context found."
    deep_mode = len(question.split()) > 25
    system_prompt = SYSTEM_PROMPT + (" Use deeper reasoning for multifactorial dermatologic conditions." if deep_mode else "")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    return call_with_fallbacks(messages)

# =====================================================
# CHAT INTERFACE
# =====================================================
def type_like_chatgpt(text: str, speed: float = 0.004):
    placeholder = st.empty()
    out = ""
    for c in text:
        out += c
        placeholder.markdown(out + " ▍")
        time.sleep(speed)
    placeholder.markdown(out)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "animate_last" not in st.session_state:
    st.session_state.animate_last = False

user_query = st.chat_input("Ask a dermatology-related question...")
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.spinner("Consulting dermatologic literature..."):
        answer = rag_answer(user_query)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.animate_last = True

# Render chat messages
for i, msg in enumerate(st.session_state.chat_history):
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        if i == len(st.session_state.chat_history) - 1 and msg["role"] == "assistant" and st.session_state.animate_last:
            type_like_chatgpt(msg["content"])
            st.session_state.animate_last = False
        else:
            st.markdown(msg["content"])

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<hr>
<div style='text-align:center; color:#666; font-size:13px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani <br>
    📬 <a href="mailto:prakhar.mathur2020@gmail.com">Contact me</a>
</div>
""", unsafe_allow_html=True)

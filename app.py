# app_conditional_context.py
import os
import time
import json
import hashlib
import logging
from typing import List, Dict
import re

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
    "openai/gpt-oss-20b:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
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
# FOLLOW-UP DETECTION
# =====================================================
FOLLOW_UP_KEYWORDS = [
    "explain", "why", "more", "details", "elaborate", "clarify", "expand", "further", "deeper",
    "reason", "rationale", "how come", "please explain", "give example", "examples", "can you detail",
    "step by step", "break down", "insight", "insights", "expand on", "expand further", "expand more",
    "go deeper", "go further", "justify", "explanation", "context", "elaboration", "tell me more",
    "help me understand", "more info", "additional info", "additional information", "further details",
    "more details", "what else", "continue", "keep going", "clarification", "please clarify",
    "illuminate", "shed light", "reasoning", "background", "expand reasoning", "deeper reasoning",
    "amplify", "add context", "give context", "exemplify", "examples please", "break it down",
    "what do you mean", "more explanation", "more clarity", "please elaborate", "expand explanation",
    "justify reasoning", "stepwise explanation", "stepwise reasoning", "further explanation",
    "dig deeper", "enlighten", "explain further", "more info please", "details please",
    "expand thoughts", "clarify further", "go into details", "additional clarification", "more insight",
    "expand insight", "give reasoning", "more reasoning", "please expand", "please provide details"
]

FOLLOW_UP_PATTERNS = [
    re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
    for keyword in sorted(FOLLOW_UP_KEYWORDS, key=lambda x: -len(x))
]

def is_follow_up(query: str) -> bool:
    query_clean = re.sub(r'[^\w\s]', '', query.lower())
    return any(pattern.search(query_clean) for pattern in FOLLOW_UP_PATTERNS)

# =====================================================
# RAG PIPELINE WITH CONDITIONAL CONTEXT
# =====================================================
SYSTEM_PROMPT_NORMAL = (
    "You are Derma Consult, a board-certified dermatologist and medical educator. "
    "Provide precise, evidence-based explanations and diagnostic reasoning. "
    "Be structured,Give detailed yet concise answers in one line or two."
)

SYSTEM_PROMPT_FOLLOWUP = (
    "You are Derma Consult, a board-certified dermatologist and medical educator. "
    "The user is asking a follow-up question. Provide precise, evidence-based explanations "
    "and reasoning based on the previous answer. Be structured, concise, and clinically sound. "
    "Give thinking insights at each step of answer"
)

def rag_answer_conditional(chat_history: List[Dict[str, str]]) -> str:
    last_user_query = chat_history[-1]["content"]
    docs = retriever.invoke(last_user_query)
    context = "\n".join([d.page_content for d in docs]) if docs else ""

    follow_up = len(chat_history) > 1 and is_follow_up(last_user_query)
    system_prompt = SYSTEM_PROMPT_FOLLOWUP if follow_up else SYSTEM_PROMPT_NORMAL
    messages = [{"role": "system", "content": system_prompt}]

    if follow_up:
        prev = chat_history[-2]
        if prev["role"] == "assistant" and prev["content"].strip():
            messages.append({"role": "assistant", "content": prev["content"]})
        messages.append({"role": "user", "content": f"Follow-up question: {last_user_query}"})
    else:
        messages.append({"role": "user", "content": last_user_query})

    if context:
        messages.append({"role": "user", "content": f"Context:\n{context}"})

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
        answer = rag_answer_conditional(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.animate_last = True

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

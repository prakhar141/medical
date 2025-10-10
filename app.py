# app.py
import os
import time
import json
import hashlib
import logging
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import requests
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------ CONFIG ------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek/deepseek-chat-v3.1:free")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
K_VAL = int(os.getenv("K_VAL", "4"))

# Retry / circuit-breaker tuning
MAX_API_ATTEMPTS = int(os.getenv("MAX_API_ATTEMPTS", "6"))
BASE_BACKOFF = float(os.getenv("BASE_BACKOFF", "1.0"))     # seconds
MAX_BACKOFF = float(os.getenv("MAX_BACKOFF", "30.0"))      # seconds
CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", "5"))  # failures to open circuit
CB_COOLDOWN_SECONDS = int(os.getenv("CB_COOLDOWN_SECONDS", "60"))  # cooldown after opening circuit
DEDUP_TTL_SECONDS = int(os.getenv("DEDUP_TTL_SECONDS", "20"))  # collapse duplicates within window

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

FAISS_INDEX_URL = "https://huggingface.co/datasets/prakhar146/derma/resolve/main/index.faiss"
FAISS_PKL_URL = "https://huggingface.co/datasets/prakhar146/derma/resolve/main/index.pkl"
LOCAL_FAISS_DIR = "./faiss_store"
os.makedirs(LOCAL_FAISS_DIR, exist_ok=True)

# ------------------ LOGGING ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dermaconsult")

# ------------------ STREAMLIT SETUP ------------------
st.set_page_config(page_title="DermaConsult", layout="wide")
st.title("🐾 DermaConsult – Your Skin & Paw-sitive Guide")
st.markdown("Helping dermatologists — robustly and reliably. 🐶✨")

# ------------------ UTILITIES ------------------
def download_file(url: str, local_path: str, timeout: int = 60):
    if os.path.exists(local_path):
        return
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        logger.info(f"Downloaded {url} -> {local_path}")
    except Exception as e:
        logger.exception(f"Failed to download {url}: {e}")
        st.error(f"Failed to download required resource: {e}")

def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

# ------------------ DOWNLOAD FAISS IF NEEDED ------------------
download_file(FAISS_INDEX_URL, os.path.join(LOCAL_FAISS_DIR, "index.faiss"))
download_file(FAISS_PKL_URL, os.path.join(LOCAL_FAISS_DIR, "index.pkl"))

# ------------------ VECTOR DB ------------------
@st.cache_resource
def load_vector_db():
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.load_local(
        LOCAL_FAISS_DIR,
        embedder,
        allow_dangerous_deserialization=True
    )
    return vectordb.as_retriever(search_type="similarity", k=K_VAL)

retriever = load_vector_db()

# ------------------ CIRCUIT BREAKER + DEDUP STORE ------------------
@dataclass
class CircuitBreaker:
    failure_count: int = 0
    opened_at: Optional[datetime] = None
    cooldown_seconds: int = CB_COOLDOWN_SECONDS
    threshold: int = CB_FAILURE_THRESHOLD

    def record_failure(self):
        self.failure_count += 1
        logger.warning(f"Circuit failure recorded: {self.failure_count}/{self.threshold}")
        if self.failure_count >= self.threshold and not self.is_open():
            self.opened_at = datetime.utcnow()
            logger.warning(f"Circuit opened at {self.opened_at.isoformat()}")

    def record_success(self):
        if self.failure_count > 0:
            logger.info("Circuit breaker success -> reset failure_count")
        self.failure_count = 0
        self.opened_at = None

    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if datetime.utcnow() - self.opened_at > timedelta(seconds=self.cooldown_seconds):
            # cooldown expired
            logger.info("Circuit cooldown expired -> closing circuit")
            self.failure_count = 0
            self.opened_at = None
            return False
        return True

# Dedup cache: prompt_hash -> (timestamp, response)
@dataclass
class DedupCache:
    store: Dict[str, Tuple[float, str]] = field(default_factory=dict)
    ttl_seconds: int = DEDUP_TTL_SECONDS

    def get(self, key: str) -> Optional[str]:
        entry = self.store.get(key)
        if not entry:
            return None
        ts, resp = entry
        if time.time() - ts > self.ttl_seconds:
            self.store.pop(key, None)
            return None
        return resp

    def set(self, key: str, response: str):
        self.store[key] = (time.time(), response)

# Singletons in session_state for persistence across reruns
if "circuit" not in st.session_state:
    st.session_state.circuit = CircuitBreaker()
if "dedup" not in st.session_state:
    st.session_state.dedup = DedupCache()

circuit: CircuitBreaker = st.session_state.circuit
dedup: DedupCache = st.session_state.dedup

# ------------------ OPENROUTER CLIENT (robust) ------------------
def parse_retry_after(resp: requests.Response) -> Optional[int]:
    # parse Retry-After header (seconds)
    ra = resp.headers.get("Retry-After")
    if not ra:
        return None
    try:
        return int(ra)
    except:
        try:
            # sometimes Retry-After is a date
            dt = parsedate_to_datetime(ra)
            return max(0, int((dt - datetime.utcnow()).total_seconds()))
        except Exception:
            return None

def exponential_backoff_with_jitter(attempt: int) -> float:
    # full jitter per AWS guidance: sleep = random(0, min(MAX_BACKOFF, base * 2**attempt))
    cap = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** attempt))
    return random.uniform(0, cap)

def call_openrouter_with_retries(
    model: str,
    messages: List[Dict[str, str]],
    max_attempts: int = MAX_API_ATTEMPTS,
) -> Tuple[bool, str]:
    """
    Returns (success: bool, content_or_error: str).
    Implements:
      - circuit breaker
      - deduplication
      - exponential backoff + jitter
      - respect Retry-After when provided
      - fallback message on persistent failures
    """
    # If circuit is open, short-circuit immediately
    if circuit.is_open():
        logger.warning("Circuit open -> skipping API call")
        return False, "OpenRouter temporarily unavailable (circuit open). Using local fallback."

    # Deduplicate identical prompts (prevent duplicate submission conflicts)
    payload = {"model": model, "messages": messages}
    prompt_dump = json.dumps(payload, sort_keys=True)
    prompt_hash = hash_prompt(prompt_dump)
    cached = dedup.get(prompt_hash)
    if cached:
        logger.info("Dedup hit -> returning cached response")
        return True, cached

    attempt = 0
    last_error = None

    while attempt < max_attempts:
        attempt += 1
        try:
            logger.info(f"OpenRouter attempt {attempt}/{max_attempts}")
            resp = requests.post(
                OPENROUTER_URL,
                headers=HEADERS,
                json=payload,
                timeout=30
            )
            # If success
            if resp.status_code == 200:
                data = resp.json()
                # safe parsing
                if "choices" in data and data["choices"]:
                    content = data["choices"][0]["message"]["content"]
                    # store in dedup cache
                    dedup.set(prompt_hash, content)
                    circuit.record_success()
                    return True, content
                else:
                    last_error = "Unexpected response structure from OpenRouter."
                    logger.error(last_error + f" raw: {resp.text}")
                    # treat as transient and retry
            else:
                # Handle common problematic

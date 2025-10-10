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
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/mistral-small-3.2-24b-instruct:free")
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
                # Handle common problematic codes
                if resp.status_code in (409, 429):
                    # Respect Retry-After if present
                    retry_after = parse_retry_after(resp) or exponential_backoff_with_jitter(attempt)
                    msg = f"Received {resp.status_code} from OpenRouter. Waiting {retry_after:.1f}s before retry."
                    logger.warning(msg)
                    circuit.record_failure()
                    time.sleep(retry_after)
                    continue
                elif 500 <= resp.status_code < 600:
                    # server error
                    wait = exponential_backoff_with_jitter(attempt)
                    logger.warning(f"Server error {resp.status_code}. Waiting {wait:.2f}s and retrying.")
                    circuit.record_failure()
                    time.sleep(wait)
                    continue
                else:
                    # client error (400s other than 409/429) treat as terminal
                    last_error = f"OpenRouter returned {resp.status_code}: {resp.text}"
                    logger.error(last_error)
                    # Do not retry on 400-level errors other than 409/429
                    circuit.record_failure()
                    return False, last_error

        except requests.exceptions.RequestException as e:
            last_error = f"Network error: {e}"
            logger.exception(last_error)
            circuit.record_failure()
            wait = exponential_backoff_with_jitter(attempt)
            time.sleep(wait)
            continue

    # Exhausted attempts -> open circuit and return failure
    circuit.record_failure()
    circuit.opened_at = datetime.utcnow()  # ensure circuit opens when exhausted
    logger.error(f"OpenRouter calls exhausted after {attempt} attempts. Last error: {last_error}")
    return False, f"OpenRouter unavailable after retries: {last_error or 'unknown error'}"

# ------------------ RAG + LOCAL FALLBACK ------------------
SYSTEM_PROMPT = (
    "You are Derma Consult. Summarize advanced dermatology concepts like "
    "inflammatory skin diseases, nail and hair disorders, dermatopathology, "
    "and dermatologic therapeutics in micro-learning chunks.\n\n"
    "Act as a gamified quizmaster, offering adaptive problem-solving levels, "
    "leaderboard challenges, and badges for clinical learning streaks.\n\n"
    "Suggest 'clinic hacks' or exam shortcuts based on common mistakes and "
    "best practices (ethically safe, medically accurate). "
    "Sprinkle dog-inspired analogies where fun, but keep answers clinically accurate. "
    "Answer in English and stick to dermatology topics only."
)

def build_prompt_with_context(question: str, context_docs: List[str]) -> List[Dict[str, str]]:
    context_text = "\n\n".join(context_docs) if context_docs else "No relevant context found."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
    ]
    return messages

def local_rag_answer(question: str) -> str:
    """A simple local fallback when API is down — returns a concise synthesis of retrieved docs."""
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "I couldn't reach the external API and I found no relevant documents locally. Please try again later."
    snippets = [doc.page_content.strip() for doc in docs if doc.page_content.strip()]
    summary = "Local summary based on stored documents:\n\n"
    for i, s in enumerate(snippets[:K_VAL], 1):
        snippet_clean = s[:800].replace("\n", " ")
        summary += f"{i}. {snippet_clean}\n\n"
    summary += "⚠️ Note: This is a local fallback response because the external API was unavailable."
    return summary

# ------------------ UI: Chat behavior ------------------
def type_like_chatgpt(text: str, speed: float = 0.004):
    placeholder = st.empty()
    animated = ""
    for c in text:
        animated += c
        placeholder.markdown(animated + " |")
        time.sleep(speed)
    placeholder.markdown(animated + " 🐾")

DOG_EMOJIS = ["🐶", "🐕", "🐩", "🐾", "🦴"]

def doggy_reaction():
    st.markdown(f"### {random.choice(DOG_EMOJIS)} Thanks for the question!")

def show_dog_pic():
    try:
        img = requests.get("https://dog.ceo/api/breeds/image/random", timeout=8).json().get("message")
        if img:
            st.image(img, caption="Here’s a little 🐶 break!", use_container_width=True)
    except Exception:
        logger.debug("Dog pic failed to load")

# ------------------ SESSION STATE ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_answer_animated" not in st.session_state:
    st.session_state.last_answer_animated = False

# Show circuit state to user (helpful)
if circuit.is_open():
    cool_until = (circuit.opened_at + timedelta(seconds=circuit.cooldown_seconds)).isoformat() if circuit.opened_at else "soon"
    st.warning(f"OpenRouter requests are temporarily paused due to repeated failures. We'll try again after cooldown (until ~{cool_until}). A local fallback will be used in the meantime.")

# Chat input
user_query = st.chat_input("Ask me about Dermatology 🐾")
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    # Build prompt with retrieved context
    docs = retriever.get_relevant_documents(user_query)
    context_docs = [d.page_content for d in docs] if docs else []
    messages = build_prompt_with_context(user_query, context_docs)

    # Call OpenRouter robust client
    with st.spinner("Consulting OpenRouter (robust) ..."):
        success, reply_or_error = call_openrouter_with_retries(MODEL_NAME, messages)

    if success:
        st.session_state.chat_history.append({"role": "assistant", "content": reply_or_error})
    else:
        # fallback: either local rag or an informative error
        if "temporary" in reply_or_error.lower() or "unavailable" in reply_or_error.lower():
            fallback = local_rag_answer(user_query)
            st.session_state.chat_history.append({"role": "assistant", "content": fallback})
        else:
            # Generic informative message + local fallback
            fallback = local_rag_answer(user_query)
            combined = f"⚠️ External model unavailable: {reply_or_error}\n\nUsing local fallback:\n\n{fallback}"
            st.session_state.chat_history.append({"role": "assistant", "content": combined})

    st.session_state.last_answer_animated = True
    st.rerun()

# Render chat
for i, chat in enumerate(st.session_state.chat_history):
    role = "user" if chat["role"] == "user" else "assistant"
    with st.chat_message(role):
        if i == len(st.session_state.chat_history) - 1 and chat["role"] == "assistant" and st.session_state.last_answer_animated:
            type_like_chatgpt(chat["content"])
            doggy_reaction()
            if random.random() < 0.20:
                show_dog_pic()
            st.session_state.last_answer_animated = False
        else:
            st.markdown(chat["content"])

# ------------------ FOOTER ------------------
st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ + 🐶 by <b>Prakhar Mathur</b> · BITS Pilani · 
    <br>📬 Email: <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)

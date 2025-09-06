import os
import json
import fitz
import requests
import streamlit as st
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import time

# ================== CONFIG ==================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_NAME = os.getenv("MODEL_NAME") or "deepseek/deepseek-r1-0528:free"
EMBED_MODEL = os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
K_VAL = int(os.getenv("K_VAL") or 4)

# Path to repo root (where notes.pdf and .txt files are located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = BASE_DIR

# ================== STREAMLIT PAGE SETUP ==================
st.set_page_config(page_title="DermaConsult", layout="wide")
st.title("🎓 DermaConsult")
st.markdown("Your Friendly neighbourhood bot")

def type_like_chatgpt(text, speed=0.004):
    """Types out the text character-by-character with a blinking cursor effect."""
    placeholder = st.empty()
    animated = ""
    for c in text:
        animated += c
        placeholder.markdown(animated + "|")  # add cursor
        time.sleep(speed)
    placeholder.markdown(animated)  # final text without cursor

# ================== VECTOR DB LOADING ==================
@st.cache_resource
def load_vector_db(folder: str):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if file.lower().endswith(".pdf"):
                with fitz.open(file_path) as doc:
                    text = "\n".join(page.get_text() for page in doc)
                    chunks = splitter.split_text(text)
                    docs.extend([Document(page_content=c, metadata={"source": file}) for c in chunks])
            elif file.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    chunks = splitter.split_text(text)
                    docs.extend([Document(page_content=c, metadata={"source": file}) for c in chunks])
        except Exception as e:
            st.warning(f"Could not read {file}: {e}")

    if not docs:
        st.warning("No documents found — retrieval will return nothing.")
        class EmptyRetriever:
            def get_relevant_documents(self, q): return []
        return EmptyRetriever()

    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.from_documents(docs, embedder)
    return vectordb.as_retriever(search_type="similarity", k=K_VAL)

# Create retriever at startup
retriever = load_vector_db(DATASET_FOLDER)

# ================== OPENROUTER HELPER ==================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

def query_openrouter(model: str, messages: List[Dict[str, str]]) -> str:
    payload = {"model": model, "messages": messages}
    r = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "choices" in data and data["choices"]:
        return data["choices"][0]["message"]["content"]
    return json.dumps(data)

# ================== VANILLA RAG PIPELINE ==================
def vanilla_rag_answer(question: str) -> str:
    try:
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."
        
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are Derma Buddy. Summarize advanced dermatology concepts like "
                    "inflammatory skin diseases, nail and hair disorders, dermatopathology, "
                    "and dermatologic therapeutics in micro-learning chunks.\n\n"
                    "Act as a gamified quizmaster, offering adaptive problem-solving levels, "
                    "leaderboard challenges, and badges for clinical learning streaks.\n\n"
                    "Suggest 'clinic hacks' or exam shortcuts based on common mistakes and "
                    "best practices (ethically safe, medically accurate). Answer in English."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]

        return query_openrouter(MODEL_NAME, prompt)

    except Exception as e:
        return f"⚠️ An error occurred: {e}"

# ================== CHAT INTERFACE ==================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_answer_animated" not in st.session_state:
    st.session_state.last_answer_animated = False

if user_query := st.chat_input("Ask me about Dermatology"):
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.spinner("Thinking..."):
        answer = vanilla_rag_answer(user_query)
    
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.last_answer_animated = True
    st.rerun()  # Force rerun so UI refreshes cleanly

# Show chat history
for i, chat in enumerate(st.session_state.chat_history):
    with st.chat_message("user" if chat["role"] == "user" else "assistant"):
        if (
            i == len(st.session_state.chat_history) - 1
            and chat["role"] == "assistant"
            and st.session_state.last_answer_animated
        ):
            type_like_chatgpt(chat["content"])
            st.session_state.last_answer_animated = False
        else:
            st.markdown(chat["content"])

st.markdown("""<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani · 
    <br>📬 Email: <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)  

import os
import streamlit as st
import requests
from huggingface_hub import list_repo_files, hf_hub_download
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ========== CONFIG ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_NAME = "deepseek/deepseek-chat:free"
HF_REPO_ID = "prakhar146/medical"
HF_REPO_TYPE = "dataset"

# ========== UI ==========
st.set_page_config(page_title="📄 Quiliffy", layout="wide")
st.title("🎓 Welcome to Quiliffy")
st.markdown("Ask anything from your friendly Medico")

if st.button("🔁 Reset Chat"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()

# ========== LIST FILES ==========
@st.cache_data(show_spinner="📂 Scanning Hugging Face repo...")
def get_text_files():
    try:
        all_files = list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)
        return [f for f in all_files if f.endswith(".txt")]
    except Exception as e:
        st.error(f"❌ Could not list files: {e}")
        return []

# ========== BUILD VECTOR DB ==========
@st.cache_resource(show_spinner="📚 Building vector database...")
def build_vector_db():
    docs = []
    text_files = get_text_files()
    if not text_files:
        st.warning("⚠️ No TXT files found in repo.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    for text_name in text_files:
        try:
            file_path = hf_hub_download(repo_id=HF_REPO_ID, filename=text_name, repo_type=HF_REPO_TYPE)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                chunks = splitter.split_text(text)
                docs.extend([Document(page_content=chunk, metadata={"source": text_name}) for chunk in chunks])
            st.markdown(f"✅ Loaded `{text_name}` ({len(chunks)} chunks)")
        except Exception as e:
            st.warning(f"⚠️ Error loading `{text_name}`: {e}")

    if not docs:
        st.error("❌ No documents extracted.")
        return None

    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    db = FAISS.from_documents(docs, embedder)
    return db.as_retriever(search_type="similarity", k=4)

# ========== API CALL ==========
def ask_deepseek(context, query):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com",
        "X-Title": "Text Chatbot"
    }
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    payload = {"model": MODEL_NAME, "messages": messages}
    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API Error: {e}"

# ========== LOAD RETRIEVER ==========
retriever = build_vector_db()
if not retriever:
    st.stop()

# ========== CHAT STATE ==========
if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("💬 Ask something medical…")

if query:
    with st.spinner("🤖 Thinking..."):
        try:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join(doc.page_content for doc in docs)
            answer = ask_deepseek(context, query)
        except Exception as e:
            answer = f"❌ Error: {e}"
        st.session_state.chat.append({
            "question": query,
            "answer": answer,
            "sources": list(set(doc.metadata['source'] for doc in docs))
        })

# ========== DISPLAY ==========
for chat in reversed(st.session_state.chat):
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])
        for src in chat["sources"]:
            st.caption(f"📄 Source: `{src}`")

# ========== FULL HISTORY ==========
with st.expander("📜 Full Chat History"):
    for i, chat in enumerate(st.session_state.chat):
        st.markdown(f"**Q{i+1}:** {chat['question']}")
        st.markdown(f"**A{i+1}:** {chat['answer']}")
        st.markdown("---")

# ========== FOOTER ==========
st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani · 
    <br>📬 <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)

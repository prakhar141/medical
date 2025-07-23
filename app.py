import os
import streamlit as st
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ===================== CONFIGURATION =====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_NAME = "deepseek/deepseek-chat:free"
DATASET_PATH = "The Gale Encyclopedia of Medicine.txt"
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 1000
EMBEDDING_MODEL = "BAAI/bge-base-en"
TOP_K = 4

# ===================== STREAMLIT UI =====================
st.set_page_config(page_title="🩺 Medical Chatbot", layout="wide")
st.title("🧠 Medical Chatbot from Encyclopedia 📄")
st.markdown("Ask questions based on **The Gale Encyclopedia of Medicine**.")

if st.button("🔁 Reset Chat"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()

# ===================== VECTOR DB =====================
@st.cache_resource(show_spinner="🔍 Indexing medical data... Please wait...")
def build_vector_db_from_txt(txt_path=DATASET_PATH):
    if not os.path.exists(txt_path):
        st.error(f"❌ `{txt_path}` not found.")
        st.stop()

    with open(txt_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(full_text)
    docs = [Document(page_content=chunk, metadata={"source": txt_path}) for chunk in chunks]

    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = FAISS.from_documents(docs, embedder)

    return vectordb.as_retriever(search_type="similarity", k=TOP_K)

retriever = build_vector_db_from_txt()

# ===================== LLM QUERY =====================
def ask_openrouter_llm(context, query):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com",  # Fake referer to comply
        "X-Title": "TXT Chatbot"
    }

    system_prompt = (
        "You are a medical assistant. Use the provided context strictly to answer the user’s query.\n"
        "Respond in a simple, helpful way. If context is insufficient, say 'I'm not sure based on available data.'"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    payload = {"model": MODEL_NAME, "messages": messages}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API Error: {e}"

# ===================== CHAT LOGIC =====================
if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("💬 Ask a medical question...")

if query:
    with st.spinner("🤖 Thinking..."):
        try:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            answer = ask_openrouter_llm(context, query)
        except Exception as e:
            answer = f"❌ Error: {e}"

        st.session_state.chat.append({
            "question": query,
            "answer": answer,
            "sources": list(set([doc.metadata['source'] for doc in docs]))
        })

# ===================== DISPLAY =====================
for chat in reversed(st.session_state.chat):
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])
        for src in chat["sources"]:
            st.caption(f"📄 Source: `{src}`")

# ===================== CHAT HISTORY =====================
with st.expander("📜 Chat History", expanded=False):
    for i, chat in enumerate(st.session_state.chat):
        st.markdown(f"**Q{i+1}:** {chat['question']}")
        st.markdown(f"**A{i+1}:** {chat['answer']}")
        st.markdown("---")

# ===================== FOOTER =====================
st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani <br>
    📬 <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)

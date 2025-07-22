import os
import fitz  # PyMuPDF
import streamlit as st
import requests
from huggingface_hub import list_repo_files, hf_hub_download
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ========== API Setup ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_NAME = "deepseek/deepseek-chat:free"

# Hugging Face Repo Config
HF_REPO_ID = "prakhar146/medical"  # Replace with your own repo
HF_REPO_TYPE = "dataset"

# ========== UI Setup ==========
st.set_page_config(page_title="📄 Quiliffy", layout="wide")
st.title("🎓 Welcome to Quiliffy")
st.markdown("Ask anything from your friendly Medico")

# ========== Reset Button ==========
if st.button("🔁 Reset Chat"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()

# ========== Load PDFs from Hugging Face ==========
@st.cache_resource(show_spinner="📚 Reading my books...")
def build_vector_db_from_huggingface(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

    try:
        all_files = list_repo_files(repo_id=repo_id, repo_type=repo_type)
        pdf_files = [f for f in all_files if f.endswith(".pdf")]

        if not pdf_files:
            st.warning("⚠️ No PDF files found in the Hugging Face repo.")
            return None

        for pdf_name in pdf_files:
            try:
                file_path = hf_hub_download(repo_id=repo_id, filename=pdf_name, repo_type=repo_type)
                with fitz.open(file_path) as doc:
                    text = "\n".join([page.get_text() for page in doc])
                    page_count = len(doc)

                chunks = splitter.split_text(text)
                file_docs = [Document(page_content=chunk, metadata={"source": pdf_name}) for chunk in chunks]
                docs.extend(file_docs)

                st.markdown(f"✅ Loaded `{pdf_name}` — {len(text)//1024} KB, {page_count} pages")

            except Exception as pdf_error:
                st.warning(f"⚠️ Could not process `{pdf_name}`: {pdf_error}")

        if not docs:
            st.error("❌ No usable documents found.")
            return None

        embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
        vectordb = FAISS.from_documents(docs, embedder)
        return vectordb.as_retriever(search_type="similarity", k=4)

    except Exception as e:
        st.error(f"❌ Error connecting to Hugging Face: {e}")
        return None

# ========== Ask Function ==========
def ask_deepseek(context, query):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com",
        "X-Title": "PDF Chatbot"
    }
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    payload = {"model": MODEL_NAME, "messages": messages}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API Error: {e}"

# ========== Build Retriever ==========
retriever = build_vector_db_from_huggingface()
if not retriever:
    st.stop()

# ========== Chat State ==========
if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("💬 Ask something about the Medical…")

if query:
    with st.spinner("🤖 Thinking..."):
        try:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            answer = ask_deepseek(context, query)
        except Exception as e:
            answer = f"❌ Error: {e}"
        st.session_state.chat.append({
            "question": query,
            "answer": answer,
            "sources": list(set([doc.metadata['source'] for doc in docs]))
        })

# ========== Display Chat ==========
for chat in reversed(st.session_state.chat):
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])
        for src in chat["sources"]:
            st.caption(f"📄 Source: `{src}`")

# ========== Expandable Chat History ==========
with st.expander("📜 Full Chat History"):
    for i, chat in enumerate(st.session_state.chat):
        st.markdown(f"**Q{i+1}:** {chat['question']}")
        st.markdown(f"**A{i+1}:** {chat['answer']}")
        st.markdown("---")

# ========== Footer ==========
st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani · 
    <br>📬 <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)

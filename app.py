import os
import streamlit as st
import requests
from huggingface_hub import list_repo_files, hf_hub_download
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import pipeline
from pinecone import Pinecone, ServerlessSpec

# ========== CONFIG ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_OPENROUTER_API_KEY"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "YOUR_PINECONE_API_KEY"
INDEX_NAME = "quiliffy-medical"
MODEL_NAME = "deepseek/deepseek-chat:free"
HF_REPO_ID = "prakhar146/medical"
HF_REPO_TYPE = "dataset"

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="🍺 Quiliffy Medical Bot", layout="wide")
st.title("🧠 Quiliffy Medical Assistant")

# ========== SIDEBAR ==========
st.sidebar.title("🗭 Select Mode")
mode = st.sidebar.radio("Choose what you want to do:", ["💬 Ask Questions", "📄 File Summary", "🧬 Extract Keywords"])

# ========== PINECONE INIT ==========
@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,  # for BAAI/bge-base-en
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )
    return pc.Index(INDEX_NAME)

# ========== HELPERS ==========
@st.cache_data(show_spinner="📂 Loading repo files...")
def get_text_files():
    try:
        return [f for f in list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE) if f.endswith(".txt")]
    except Exception as e:
        st.error(f"❌ Could not fetch files: {e}")
        return []

@st.cache_resource(show_spinner="🔧 Building Pinecone Vector DB...")
def build_vector_db():
    index = init_pinecone()
    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectorstore = LangchainPinecone(index, embedder, text_key="text")

    text_files = get_text_files()
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

    for fname in text_files:
        try:
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=fname, repo_type=HF_REPO_TYPE)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                chunks = splitter.split_text(text)
                docs.extend([Document(page_content=chunk, metadata={"source": fname}) for chunk in chunks])
            st.markdown(f"✅ `{fname}` loaded ({len(chunks)} chunks)")
        except Exception as e:
            st.warning(f"⚠️ Could not read `{fname}`: {e}")

    if docs:
        vectorstore.add_documents(docs)
        return vectorstore.as_retriever(search_type="similarity", k=4)
    else:
        st.error("❌ No documents found.")
        return None

def ask_llm(context, query):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com",
        "X-Title": "Quiliffy Bot"
    }
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant. Use the provided context to answer the question."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={"model": MODEL_NAME, "messages": messages})
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API Error: {e}"

@st.cache_resource
def get_keyword_pipeline():
    return pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# ========== MAIN EXECUTION ==========
retriever = build_vector_db()
if not retriever:
    st.stop()

# ========== CHAT ==========
if mode == "💬 Ask Questions":
    query = st.chat_input("💬 Ask anything medical...")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    if query:
        with st.spinner("🤖 Thinking..."):
            try:
                docs = retriever.get_relevant_documents(query)
                context = "\n\n".join(doc.page_content for doc in docs)
                answer = ask_llm(context, query)
                sources = list(set(doc.metadata["source"] for doc in docs))
                st.session_state.chat.append({"question": query, "answer": answer, "sources": sources})
            except Exception as e:
                st.error(f"❌ Error: {e}")

    for chat in reversed(st.session_state.chat):
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
            for src in chat["sources"]:
                st.caption(f"📄 Source: `{src}`")

# ========== FILE SUMMARY ==========
elif mode == "📄 File Summary":
    files = get_text_files()
    selected = st.selectbox("📂 Choose a file to summarize", files)

    if selected:
        with st.spinner("📝 Generating summary..."):
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=selected, repo_type=HF_REPO_TYPE)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                summary = ask_llm(text[:3000], "Give a summary of this medical content.")
                st.markdown(f"### 📝 Summary of `{selected}`")
                st.write(summary)

# ========== EXTRACT KEYWORDS ==========
elif mode == "🧬 Extract Keywords":
    files = get_text_files()
    selected = st.selectbox("🧬 Choose a file to extract keywords", files)

    if selected:
        with st.spinner("🔍 Extracting medical keywords..."):
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=selected, repo_type=HF_REPO_TYPE)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                ner_pipeline = get_keyword_pipeline()
                entities = ner_pipeline(text[:2000])
                unique_keywords = sorted(set(e['word'] for e in entities))
                st.markdown(f"### 🧪 Keywords in `{selected}`")
                st.write(", ".join(unique_keywords))

# ========== FOOTER ==========
st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani · 
    <br>📬 <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)

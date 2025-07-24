import os
import re
import asyncio
import httpx
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor

# ===================== CONFIGURATION =====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
MODEL_NAME = "deepseek/deepseek-r1-0528:free"
TEXT_FILES = ["The Gale Encyclopedia of Medicine.txt"]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
TOP_K = 5
MAX_WORKERS = 4
FILE_CHUNK_SIZE = 50000

# Validate API key
if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_API_KEY":
    st.error("❌ OpenRouter API key missing. Please set it via environment or Streamlit secrets.")
    st.stop()

# ===================== UI SETUP =====================
st.set_page_config(page_title="🩺 Quiliffy Medico", layout="wide")
st.title("Medico Assistant 📄")
st.markdown("Ask questions based on **Multiple Trusted Medical Sources**.")

if st.button("🔁 Reset Chat"):
    for key in list(st.session_state.keys()):
        if key != "vector_db":
            del st.session_state[key]
    st.rerun()

# ===================== TEXT CLEANING =====================
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\u201c|\u201d', '"', text)
    text = re.sub(r'\u2019', "'", text)
    return text.strip()

def read_file_in_chunks(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        while True:
            chunk = f.read(FILE_CHUNK_SIZE)
            if not chunk:
                break
            yield chunk

# ===================== EMBEDDING IN PARALLEL =====================
@st.cache_resource(show_spinner="🔍 Indexing medical data... This might take a while...")
def build_vector_db_from_txts(txt_paths=TEXT_FILES):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
        keep_separator=True
    )

    all_chunks = []
    for path in txt_paths:
        if not os.path.exists(path):
            st.error(f"❌ File not found: `{path}`")
            st.stop()
        st.info(f"📖 Processing {os.path.basename(path)}...")
        buffer = ""

        for chunk in read_file_in_chunks(path):
            buffer += chunk
            if len(buffer) >= FILE_CHUNK_SIZE * 10:
                buffer = clean_text(buffer)
                chunks = splitter.split_text(buffer)
                all_chunks.extend([(chunk, path) for chunk in chunks])
                buffer = ""

        if buffer:
            buffer = clean_text(buffer)
            chunks = splitter.split_text(buffer)
            all_chunks.extend([(chunk, path) for chunk in chunks])

    texts = [Document(page_content=c[0], metadata={"source": c[1]}) for c in all_chunks]
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Embed in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        embeddings = list(executor.map(embedder.embed_query, [doc.page_content for doc in texts]))

    vectordb = FAISS.from_embeddings(embeddings, texts)
    return vectordb.as_retriever(search_kwargs={"k": TOP_K})

if "vector_db" not in st.session_state:
    with st.spinner("🚀 Initializing medical knowledge base..."):
        st.session_state.vector_db = build_vector_db_from_txts()

# ===================== LLM QUERY =====================
async def ask_openrouter_llm(context, query):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com",
        "X-Title": "Medico Assistant"
    }

    system_prompt = (
        "You are a knowledgeable and careful medical assistant. Use ONLY the provided context from "
        "trusted medical sources to answer questions. If the context doesn't contain sufficient "
        "information, clearly state you don't know rather than speculating.\n\n"
        "Context:\n{context}"
    )

    messages = [
        {"role": "system", "content": system_prompt.format(context=context)},
        {"role": "user", "content": f"Question: {query}"}
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1500
    }

    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            return f"❌ API Error: {str(e)}"
        except Exception as e:
            return f"❌ Unexpected Error: {str(e)}"

# ===================== CHAT LOGIC =====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("💬 Ask a medical question...")

if query:
    with st.spinner("🔍 Searching medical knowledge..."):
        try:
            docs = st.session_state.vector_db.get_relevant_documents(query)
            context = "\n\n---\n\n".join(
                [f"SOURCE: {doc.metadata['source']}\nCONTENT:\n{doc.page_content}" for doc in docs]
            )

            with st.spinner("🤖 Generating response..."):
                answer = asyncio.run(ask_openrouter_llm(context, query))

            sources = list(set([doc.metadata['source'] for doc in docs]))
            st.session_state.chat_history.append({
                "question": query,
                "answer": answer,
                "sources": sources,
                "context": context
            })

        except Exception as e:
            st.error(f"❌ Error processing your question: {str(e)}")
            st.session_state.chat_history.append({
                "question": query,
                "answer": f"🚨 An error occurred: {str(e)}",
                "sources": []
            })

# ===================== DISPLAY =====================
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**You:** {chat['question']}")

    with st.chat_message("assistant"):
        st.markdown(chat['answer'])
        if chat['sources']:
            with st.expander("📚 Reference Sources"):
                for src in chat['sources']:
                    st.caption(f"📄 {os.path.basename(src)}")

# ===================== DEBUGGING =====================
with st.expander("🔍 Debug Options", expanded=False):
    if st.session_state.chat_history:
        st.subheader("Last Context Used")
        st.text_area("Context",
                     value=st.session_state.chat_history[-1].get("context", ""),
                     height=300)
    if st.checkbox("Show Session State"):
        st.json(st.session_state)

# ===================== FOOTER =====================
st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani <br>
    📬 <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)

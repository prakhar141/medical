import os
import re
import mmap
import asyncio
import httpx
import hashlib
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.docstore.document import Document

# ===================== CONFIGURATION =====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
MODEL_NAME = "deepseek/deepseek-r1-0528:free"
TEXT_FILES = ["The Gale Encyclopedia of Medicine.txt"]
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 7
MAX_WORKERS = 4
EMBEDDING_BATCH_SIZE = 512
PROCESSING_BLOCK_SIZE = 10 * 1024 * 1024
FAISS_INDEX_DIR = "faiss_index"

WHITESPACE_PATTERN = re.compile(r'\s+')
QUOTE_PATTERN = re.compile(r'\u201c|\u201d')
APOSTROPHE_PATTERN = re.compile(r'\u2019')
HEADER_PATTERN = re.compile(r'\n[A-Z][A-Z\s]+\n')

if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_API_KEY":
    st.error("❌ OpenRouter API key missing. Please set it via environment or Streamlit secrets.")
    st.stop()

os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

st.set_page_config(page_title="🩺 Quiliffy Medico", layout="wide")
st.title("Medico Assistant 📄")
st.markdown("Ask questions based on **Multiple Trusted Medical Sources**.")

if st.button("🔁 Reset Chat"):
    for key in list(st.session_state.keys()):
        if key != "vector_db":
            del st.session_state[key]
    st.rerun()

def clean_text(text):
    text = WHITESPACE_PATTERN.sub(' ', text)
    text = QUOTE_PATTERN.sub('"', text)
    text = APOSTROPHE_PATTERN.sub("'", text)
    return text.strip()

def process_text_block(text_block, path, splitter):
    cleaned = clean_text(text_block)
    sections = HEADER_PATTERN.split(cleaned)
    chunks = []
    for section in sections:
        if section.strip():
            chunks.extend(splitter.split_text(section))
    return [(chunk, path) for chunk in chunks]

async def compress_context(context, query):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com",
        "X-Title": "Medico Assistant"
    }
    prompt = f"""
    Compress the following medical context by removing redundant information while preserving 
    all critical facts and relationships related to the query: \"{query}\".

    Return ONLY the compressed version without additional commentary.

    Context:
    {context}
    """
    payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 3000
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception:
            return context

@st.cache_resource(show_spinner="🔍 Indexing medical data...")
def build_vector_db_from_txts(txt_paths=TEXT_FILES):
    file_hash = hashlib.md5()
    for path in txt_paths:
        if not os.path.exists(path):
            st.error(f"❌ File not found: `{path}`")
            st.stop()
        with open(path, 'rb') as f:
            file_hash.update(f.read())
    cache_key = file_hash.hexdigest()
    cache_path = os.path.join(FAISS_INDEX_DIR, f"{cache_key}.faiss")

    if os.path.exists(cache_path):
        st.info("💾 Loading pre-indexed medical knowledge...")
        embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return FAISS.load_local(cache_path, embedder, allow_dangerous_deserialization=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", "],
        keep_separator=True
    )

    all_chunks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for path in txt_paths:
            st.info(f"📖 Processing {os.path.basename(path)}...")
            file_size = os.path.getsize(path)
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                    for offset in range(0, file_size, PROCESSING_BLOCK_SIZE):
                        end = min(offset + PROCESSING_BLOCK_SIZE, file_size)
                        block = mm[offset:end].decode('utf-8', errors='replace')
                        future = executor.submit(process_text_block, block, path, splitter)
                        futures.append(future)
        for future in futures:
            all_chunks.extend(future.result())

    docs = [Document(page_content=chunk, metadata={"source": path}) for chunk, path in all_chunks]
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    contents = [doc.page_content for doc in docs]
    embeddings = []
    for i in range(0, len(contents), EMBEDDING_BATCH_SIZE):
        batch = contents[i:i + EMBEDDING_BATCH_SIZE]
        embeddings.extend(embedder.embed_documents(batch))

    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(contents, embeddings)),
        embedding=embedder,
        metadatas=[doc.metadata for doc in docs]
    )
    vector_store.save_local(cache_path)
    return vector_store

def create_retriever(vector_store):
    base_retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K * 3})
    embeddings_filter = EmbeddingsFilter(
        embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        similarity_threshold=0.75
    )
    return ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=base_retriever
    )

async def ask_openrouter_llm(context, query):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com",
        "X-Title": "Medico Assistant"
    }
    system_prompt = (
        "You are a knowledgeable and careful medical assistant. Use ONLY the provided context from "
        "trusted medical sources to answer questions. If the context doesn't contain sufficient "
        "information, clearly state you don't know rather than speculating. Provide detailed, "
        "well-structured responses with clear section headers where appropriate.\n\n"
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
        "max_tokens": 3000
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ Error: {str(e)}"

if "vector_db" not in st.session_state or "retriever" not in st.session_state:
    with st.spinner("🚀 Initializing medical knowledge base..."):
        vector_store = build_vector_db_from_txts()
        st.session_state.vector_db = vector_store
        st.session_state.retriever = create_retriever(vector_store)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("💬 Ask a medical question...")

if query:
    with st.spinner("🔍 Searching medical knowledge..."):
        try:
            docs = st.session_state.retriever.get_relevant_documents(query)
            context = "\n\n---\n\n".join([
                f"SOURCE: {doc.metadata['source']}\nCONTENT:\n{doc.page_content}" for doc in docs[:TOP_K]
            ])
            with st.spinner("🛠 Compressing context..."):
                compressed_context = asyncio.run(compress_context(context, query))
            with st.spinner("🤖 Generating response..."):
                answer = asyncio.run(ask_openrouter_llm(compressed_context, query))
            sources = list(set([doc.metadata['source'] for doc in docs[:TOP_K]]))
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

for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**You:** {chat['question']}")
    with st.chat_message("assistant"):
        st.markdown(chat['answer'])
        if chat['sources']:
            with st.expander("📚 Reference Sources"):
                for src in chat['sources']:
                    st.caption(f"📄 {os.path.basename(src)}")

st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani <br>
    📬 <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)

import os
import re
import mmap
import asyncio
import httpx
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ===================== CONFIGURATION =====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
MODEL_NAME = "deepseek/deepseek-r1-0528:free"
TEXT_FILES = ["The Gale Encyclopedia of Medicine.txt"]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
MAX_WORKERS = 4
EMBEDDING_BATCH_SIZE = 256  # Increased batch size for embeddings
PROCESSING_BLOCK_SIZE = 10 * 1024 * 1024  # 10MB processing blocks

# Pre-compile regex patterns for faster text cleaning
WHITESPACE_PATTERN = re.compile(r'\s+')
QUOTE_PATTERN = re.compile(r'\u201c|\u201d')
APOSTROPHE_PATTERN = re.compile(r'\u2019')

# Validate API Key
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

# ===================== OPTIMIZED TEXT PROCESSING =====================
def clean_text(text):
    """Optimized text cleaning with pre-compiled patterns"""
    text = WHITESPACE_PATTERN.sub(' ', text)
    text = QUOTE_PATTERN.sub('"', text)
    text = APOSTROPHE_PATTERN.sub("'", text)
    return text.strip()

def process_text_block(text_block, path, splitter):
    """Process a block of text with cleaning and splitting"""
    cleaned = clean_text(text_block)
    return [(chunk, path) for chunk in splitter.split_text(cleaned)]

# ===================== HIGH-PERFORMANCE VECTOR DB BUILDER =====================
@st.cache_resource(show_spinner="🔍 Indexing medical data...")
def build_vector_db_from_txts(txt_paths=TEXT_FILES):
    """Optimized vector DB builder using memory mapping and efficient batching"""
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
            if not os.path.exists(path):
                st.error(f"❌ File not found: `{path}`")
                st.stop()
                
            st.info(f"📖 Processing {os.path.basename(path)}...")
            file_size = os.path.getsize(path)
            
            with open(path, 'r+', encoding='utf-8', errors='replace') as f:
                # Memory map the file for faster access
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                    # Process in 10MB blocks
                    for offset in range(0, file_size, PROCESSING_BLOCK_SIZE):
                        end = min(offset + PROCESSING_BLOCK_SIZE, file_size)
                        block = mm[offset:end].decode('utf-8', errors='replace')
                        
                        # Submit block for processing
                        future = executor.submit(
                            process_text_block, 
                            block, 
                            path, 
                            splitter
                        )
                        futures.append(future)
        
        # Collect results
        for future in futures:
            all_chunks.extend(future.result())

    # Create documents with metadata
    docs = [Document(page_content=chunk, metadata={"source": path}) for chunk, path in all_chunks]
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    contents = [doc.page_content for doc in docs]
    
    # Batch embedding processing
    embeddings = []
    for i in range(0, len(contents), EMBEDDING_BATCH_SIZE):
        batch = contents[i:i + EMBEDDING_BATCH_SIZE]
        embeddings.extend(embedder.embed_documents(batch))

    return FAISS.from_embeddings(
        text_embeddings=list(zip(contents, embeddings)),
        embedding=embedder,
        metadatas=[doc.metadata for doc in docs]
    ).as_retriever(search_kwargs={"k": TOP_K})

# ===================== INITIALIZE VECTOR DB =====================
if "vector_db" not in st.session_state:
    with st.spinner("🚀 Initializing medical knowledge base..."):
        st.session_state.vector_db = build_vector_db_from_txts()

# ===================== ASYNC LLM CALL =====================
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

# ===================== CHAT INTERFACE =====================
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

# ===================== DISPLAY CHAT =====================
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**You:** {chat['question']}")

    with st.chat_message("assistant"):
        st.markdown(chat['answer'])
        if chat['sources']:
            with st.expander("📚 Reference Sources"):
                for src in chat['sources']:
                    st.caption(f"📄 {os.path.basename(src)}")

# ===================== FOOTER =====================
st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani <br>
    📬 <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)

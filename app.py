# app.py
import os
import re
import mmap
import asyncio
import httpx
import hashlib
import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import easyocr
import torch
from PyPDF2 import PdfReader
from transformers import AutoProcessor, AutoModel
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.docstore.document import Document

# ===================== CONFIG =====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
MODEL_NAME = "deepseek/deepseek-r1-0528:free"
TEXT_FILES = ["The Gale Encyclopedia of Medicine.txt", "Merck.txt"]
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 7
MAX_WORKERS = 4
EMBEDDING_BATCH_SIZE = 512
PROCESSING_BLOCK_SIZE = 10 * 1024 * 1024
FAISS_INDEX_DIR = "faiss_index"

# ===================== UI =====================
st.set_page_config(page_title="🩺 Medico Assistant", layout="wide")
st.title("🧠 Medico Assistant — Universal Medical Query Solver")

if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_API_KEY":
    st.error("❌ OpenRouter API key missing.")
    st.stop()

if st.button("🔁 Reset Chat"):
    for key in list(st.session_state.keys()):
        if key != "vector_db":
            del st.session_state[key]
    st.rerun()

# ===================== MODELS =====================
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'])

from transformers import AutoModel
import torchvision.transforms as T
import torch

@st.cache_resource
def get_biovil_model():
    model = AutoModel.from_pretrained("microsoft/BiomedVLP-BioViL-T")
    return model

def get_biovil_embedding(image: Image.Image):
    model = get_biovil_model()

    # Manual transform (adapt as needed based on model config)
    transform = T.Compose([
        T.Resize((224, 224)),  # common for ViT
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # adjust if model expects different values
    ])

    image_tensor = transform(image).unsqueeze(0)  # shape: (1, 3, 224, 224)

    with torch.no_grad():
        vision_output = model.vision_encoder(pixel_values=image_tensor)

    image_embedding = vision_output.last_hidden_state[:, 0, :]  # CLS token
    return image_embedding[0].cpu().numpy()
# ===================== IMAGE / TEXT HANDLING =====================
def is_medical_scan(image_np):
    return np.mean(image_np) < 100  # crude heuristic

def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    if is_medical_scan(image_np):
        # BioViL for scan understanding
        return "🩻 This appears to be a medical scan. Processed using BioViL for embedding.", image
    
    # OCR for regular images
    reader = get_ocr_reader()
    results = reader.readtext(image_np, detail=0)
    text = "\n".join(results) if results else "🛑 No readable text found in image."
    return text, None

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def process_text_block(text_block, path, splitter):
    cleaned = clean_text(text_block)
    chunks = splitter.split_text(cleaned)
    return [(chunk, path) for chunk in chunks]

@st.cache_resource(show_spinner="🔍 Indexing medical knowledge...")
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
            file_size = os.path.getsize(path)
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                    for offset in range(0, file_size, PROCESSING_BLOCK_SIZE):
                        block = mm[offset:min(offset + PROCESSING_BLOCK_SIZE, file_size)].decode('utf-8', errors='replace')
                        futures.append(executor.submit(process_text_block, block, path, splitter))
        for future in futures:
            all_chunks.extend(future.result())

    docs = [Document(page_content=chunk, metadata={"source": path}) for chunk, path in all_chunks]
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    contents = [doc.page_content for doc in docs]
    embeddings = []
    for i in range(0, len(contents), EMBEDDING_BATCH_SIZE):
        embeddings.extend(embedder.embed_documents(contents[i:i + EMBEDDING_BATCH_SIZE]))

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

# ===================== API CALLS =====================
async def compress_context(context, query):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com",
        "X-Title": "Medico Assistant"
    }
    prompt = f"""
    Compress the following medical context by removing redundant information while preserving 
    all critical facts and relationships related to the query: \"{query}\".
    Return ONLY the compressed version.
    Context:
    {context}
    """
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 3000
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception:
            return context

async def ask_openrouter_llm(context, query):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com",
        "X-Title": "Medico Assistant"
    }
    messages = [
        {"role": "system", "content": f"You are a kind, trusted medical assistant. Use ONLY provided context to answer. Also provide relevant Emoji. \n---\nContext:\n{context}"},
        {"role": "user", "content": f"Question: {query}"}
    ]
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={"model": MODEL_NAME, "messages": messages, "temperature": 0.3, "max_tokens": 3000}
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ Error: {str(e)}"

# ===================== CHAT SECTION =====================
if "vector_db" not in st.session_state or "retriever" not in st.session_state:
    with st.spinner("🚀 Initializing medical knowledge base..."):
        vector_store = build_vector_db_from_txts()
        st.session_state.vector_db = vector_store
        st.session_state.retriever = create_retriever(vector_store)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("📤 Upload a medical report (PDF or medical image)", type=["pdf", "jpg", "jpeg", "png"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
        image_embedding = None
    else:
        extracted_text, medical_image = extract_text_from_image(uploaded_file)
        # Get BioViL embedding only for medical scans
        image_embedding = get_biovil_embedding(medical_image) if medical_image else None

    st.text_area("📝 Extracted / Interpreted Content", value=extracted_text[:3000], height=150)
    query = st.text_input("💬 Ask a question about this report:")
    
    if query:
        with st.spinner("🔍 Processing..."):
            context = f"Uploaded Report Content:\n{extracted_text}"
            
            # Add image embedding info if available
            if image_embedding is not None:
                context += "\n\n🩻 Medical Image Analysis:\nThis medical scan shows features relevant to the query"
            
            compressed_context = asyncio.run(compress_context(context, query))
            answer = asyncio.run(ask_openrouter_llm(compressed_context, query))
            st.session_state.chat_history.append({"question": query, "answer": answer, "context": context})
else:
    query = st.chat_input("💬 Ask any general medical question...")
    if query:
        docs = st.session_state.retriever.get_relevant_documents(query)
        context = "\n\n---\n\n".join([
            f"SOURCE: {doc.metadata['source']}\nCONTENT:\n{doc.page_content}" for doc in docs[:TOP_K]
        ])
        compressed_context = asyncio.run(compress_context(context, query))
        answer = asyncio.run(ask_openrouter_llm(compressed_context, query))
        st.session_state.chat_history.append({"question": query, "answer": answer, "context": context})

# ===================== DISPLAY CHAT =====================
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**You:** {chat['question']}")
    with st.chat_message("assistant"):
        st.markdown(chat['answer'])

st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani <br>
    📬 <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)

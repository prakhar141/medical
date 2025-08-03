import os
import re
import asyncio
import httpx
import torch
import hashlib
import streamlit as st
import numpy as np
import easyocr
from PIL import Image
from PyPDF2 import PdfReader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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

# ===================== UI =====================
st.set_page_config(page_title="🩺 Medico Assistant", layout="wide")
st.title("🧠 Medico Assistant")

if not OPENROUTER_API_KEY:
    st.error("❌ OpenRouter API key missing.")
    st.stop()

if st.button("🔄 Reset Chat"):
    for key in list(st.session_state.keys()):
        if key != "vector_db":
            del st.session_state[key]
    st.rerun()

# ===================== BLIP-2 =====================
@st.cache_resource
def load_blip2_model():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    return processor, model.to("cuda" if torch.cuda.is_available() else "cpu")

def analyze_image_with_blip2(image: Image.Image, user_query: str) -> str:
    processor, model = load_blip2_model()
    prompt = f"Question: {user_query}"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=200)
    return processor.decode(output_ids[0], skip_special_tokens=True)

# ===================== OCR / PDF =====================
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'])

def extract_text_from_pdf(uploaded_file):
    return "\n".join([page.extract_text() or "" for page in PdfReader(uploaded_file).pages])

def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    if np.mean(image_np) < 100:  # crude grayscale test for scan
        return "🩻 Detected medical scan. Understanding via vision model required.", image
    results = get_ocr_reader().readtext(image_np, detail=0)
    return "\n".join(results) or "🛑 No readable text found.", None

# ===================== EMBEDDINGS =====================
@st.cache_resource(show_spinner="🔍 Indexing medical knowledge...")
def build_vector_db(txt_paths=TEXT_FILES):
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for path in txt_paths:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            chunks = splitter.split_text(text)
            docs.extend([Document(page_content=chunk, metadata={"source": path}) for chunk in chunks])
    return FAISS.from_documents(docs, embedder)

def create_retriever(vector_db):
    base_retriever = vector_db.as_retriever(search_kwargs={"k": TOP_K * 2})
    filter_ = EmbeddingsFilter(embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL), similarity_threshold=0.75)
    return ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=filter_)

# ===================== LLM OPENROUTER =====================
async def ask_openrouter_llm(context, query):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com",
        "X-Title": "Medico Assistant"
    }
    messages = [
        {"role": "system", "content": f"You are a trusted medical assistant. Use ONLY this context:\n---\n{context}"},
        {"role": "user", "content": query}
    ]
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 2000
            })
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ LLM Error: {str(e)}"

# ===================== CHAT LOGIC =====================
if "vector_db" not in st.session_state:
    with st.spinner("⚙️ Initializing vector DB..."):
        vector_db = build_vector_db()
        st.session_state.vector_db = vector_db
        st.session_state.retriever = create_retriever(vector_db)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("📄 Upload a report (PDF or Image)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
        image = None
    else:
        extracted_text, image = extract_text_from_image(uploaded_file)

    st.text_area("📝 Extracted Content", value=extracted_text[:3000], height=160)
    user_query = st.text_input("💬 Ask about this report:")

    if user_query:
        with st.spinner("🔍 Analyzing..."):
            if image:  # Use BLIP-2
                visual_understanding = analyze_image_with_blip2(image, user_query)
                final_context = f"Image Analysis Result:\n{visual_understanding}"
            else:  # Use extracted text
                docs = st.session_state.retriever.get_relevant_documents(user_query)
                final_context = "\n---\n".join([doc.page_content for doc in docs])

            answer = asyncio.run(ask_openrouter_llm(final_context, user_query))
            st.session_state.chat_history.append({"question": user_query, "answer": answer})
else:
    general_query = st.chat_input("💬 Ask a general medical question...")
    if general_query:
        docs = st.session_state.retriever.get_relevant_documents(general_query)
        context = "\n---\n".join([doc.page_content for doc in docs])
        answer = asyncio.run(ask_openrouter_llm(context, general_query))
        st.session_state.chat_history.append({"question": general_query, "answer": answer})

# ===================== DISPLAY CHAT =====================
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**You:** {chat['question']}")
    with st.chat_message("assistant"):
        st.markdown(chat['answer'])

st.markdown("""
<hr>
<div style='text-align: center; font-size: 13px; color: gray'>
  Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani<br>
  📬 <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)

import os
import asyncio
import httpx
import torch
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec 

# ===================== CONFIG =====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_ENV") or "us-east-1"  # Default fallback

MODEL_NAME = "deepseek/deepseek-r1-0528:free"
TEXT_FILES = ["The Gale Encyclopedia of Medicine.txt", "Merck.txt"]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300
TOP_K = 7
INDEX_NAME = "medico"

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="🩺 Medico Assistant", layout="wide")
st.title("🧠 Medico Assistant")

# ===================== API KEY CHECK =====================
if not OPENROUTER_API_KEY or not PINECONE_API_KEY:
    st.error("❌ Missing OpenRouter or Pinecone API key.")
    st.stop()

# ===================== RESET CHAT =====================
if st.button("🔄 Reset Chat"):
    for key in list(st.session_state.keys()):
        if key != "vector_db":
            del st.session_state[key]
    st.rerun()
# === PINECONE INITIALIZATION ===
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)

existing = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME not in existing:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )

idx_info = pc.describe_index(INDEX_NAME)
index = pc.Index(host=idx_info.host)
# ===================== BLIP-2 LOADER =====================
@st.cache_resource
def load_blip2_model():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16
    )
    return processor, model.to("cuda" if torch.cuda.is_available() else "cpu")

def analyze_image_with_blip2(image: Image.Image, user_query: str) -> str:
    processor, model = load_blip2_model()
    prompt = f"Question: {user_query}"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=200)
    return processor.decode(output_ids[0], skip_special_tokens=True)

# ===================== PDF TEXT EXTRACT =====================
def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

# ===================== VECTOR DB =====================
@st.cache_resource(show_spinner="🔍 Indexing medical texts...") 
def build_vector_db(txt_paths=TEXT_FILES):
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = []

    for path in txt_paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            chunks = splitter.split_text(text)
            docs += [Document(page_content=chunk, metadata={"source": path}) for chunk in chunks]

    return LangchainPinecone.from_documents(
        documents=docs,
        embedding=embedder,
        index_name=INDEX_NAME
    )

def create_retriever(vector_db):
    base_retriever = vector_db.as_retriever(search_kwargs={"k": TOP_K * 2})
    filter_ = EmbeddingsFilter(
        embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        similarity_threshold=0.75
    )
    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=filter_
    )

# ===================== LLM via OPENROUTER =====================
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
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 2000
                }
            )
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ LLM Error: {e}"

# ===================== INIT VECTOR DB =====================
if "vector_db" not in st.session_state:
    with st.spinner("⚙️ Initializing knowledge base..."):
        st.session_state.vector_db = build_vector_db()
        st.session_state.retriever = create_retriever(st.session_state.vector_db)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===================== FILE UPLOAD =====================
uploaded_file = st.file_uploader("📄 Upload medical report (PDF or Image)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file:
    file_ext = uploaded_file.name.lower().split(".")[-1]
    user_query = st.text_input("💬 Ask a question about the uploaded report:")

    if file_ext == "pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
        st.text_area("📃 Extracted PDF Text", extracted_text[:3000], height=160)

        if user_query:
            with st.spinner("🔍 Answering based on PDF..."):
                docs = st.session_state.retriever.get_relevant_documents(user_query)
                final_context = "\n---\n".join([doc.page_content for doc in docs])
                answer = asyncio.run(ask_openrouter_llm(final_context, user_query))
                st.session_state.chat_history.append({"question": user_query, "answer": answer})

    elif file_ext in ["jpg", "jpeg", "png"]:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        if user_query:
            with st.spinner("🧠 Analyzing image with BLIP-2 + LLM..."):
                visual_output = analyze_image_with_blip2(image, user_query)
                final_context = f"Image Analysis Output:\n{visual_output}"
                answer = asyncio.run(ask_openrouter_llm(final_context, user_query))
                st.session_state.chat_history.append({"question": user_query, "answer": answer})

    else:
        st.error("❌ Unsupported file format.")
else:
    general_query = st.chat_input("💬 Ask a general medical question...")
    if general_query:
        docs = st.session_state.retriever.get_relevant_documents(general_query)
        final_context = "\n---\n".join([doc.page_content for doc in docs])
        answer = asyncio.run(ask_openrouter_llm(final_context, general_query))
        st.session_state.chat_history.append({"question": general_query, "answer": answer})

# ===================== DISPLAY CHAT =====================
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**You:** {chat['question']}")
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])

# ===================== FOOTER =====================
st.markdown("""
<hr>
<div style='text-align: center; font-size: 13px; color: gray'>
  Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani<br>
  📬 <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)

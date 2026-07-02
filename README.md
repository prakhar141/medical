```markdown
# 💎 Derma Consult

**AI‑assisted dermatological reasoning — clear, precise, and reliable.**

Derma Consult is an interactive Streamlit application that combines **retrieval‑augmented generation (RAG)** with multiple large language models to provide evidence‑based dermatological answers. It retrieves relevant medical context from a FAISS vector database and adapts its responses to follow‑up questions, ensuring a clinically sound conversation.

---

## ✨ Features

- **RAG Pipeline** – Augments each query with the most similar passages from a curated dermatology knowledge base.
- **Follow‑up Detection** – Automatically switches to a deeper conversation mode when the user asks for clarification, details, or reasoning.
- **Multi‑model Resilience** – Uses `deepseek-r1` as the primary model, falling back to `gpt-oss-20b`, `mistral-small`, and `falcon-180b-chat` on failure.
- **Exponential Backoff & Retries** – Gracefully handles rate limits and server errors.
- **In‑memory Caching** – Avoids redundant API calls for identical prompts.
- **Animated Streaming** – Replies appear with a typewriter effect for a smooth user experience.
- **Clean UI** – Responsive layout with custom CSS, mobile‑friendly.

---

## 🧠 How It Works

1. **User Query** → Check if it’s a follow‑up (keyword‑based detection).
2. **Retrieve Context** → `FAISS` (using `sentence-transformers/all-MiniLM-L6-v2` embeddings) returns the top‑`k` relevant documents.
3. **Build Prompt** → A system prompt is chosen depending on whether the question is a follow‑up. The retrieved context and conversation history are added.
4. **API Call** → The constructed messages are sent to OpenRouter, with automatic retries and fallbacks.
5. **Response** → The answer is displayed with a typewriter effect and cached for future reuse.

---

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/derma-consult.git
   cd derma-consult
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   *Required packages:* `streamlit`, `langchain-community`, `sentence-transformers`, `faiss-cpu`, `requests`, `python-dotenv` (optional).

---

## ⚙️ Configuration

The application is configured through environment variables. You can either export them in your shell or place them in a `.env` file (using `python-dotenv` if you add it to the code).

| Variable              | Default                                  | Description                           |
|-----------------------|------------------------------------------|---------------------------------------|
| `OPENROUTER_API_KEY`  | `YOUR_API_KEY`                           | Your [OpenRouter](https://openrouter.ai) API key. |
| `EMBED_MODEL`         | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model.          |
| `K_VAL`               | `4`                                      | Number of documents to retrieve.      |

**Note:** The FAISS index is automatically downloaded from HuggingFace Datasets ([prakhar146/derma](https://huggingface.co/datasets/prakhar146/derma)) on first run.

---

## 🏃 Usage

Launch the app with:

```bash
streamlit run app_conditional_context.py
```

Then open your browser at `http://localhost:8501`. Start typing your dermatology‑related questions.

### Example Interactions

- **User:** *What are the typical symptoms of psoriasis?*
  - The bot retrieves relevant literature and gives a concise answer.
- **User:** *Can you explain why it happens?*
  - The follow‑up detection engages deeper reasoning and references the previous answer.

---

## 📁 File Structure

```
derma-consult/
├── app_conditional_context.py   # Main Streamlit application
├── requirements.txt             # Python dependencies
└── README.md                    # You are here
```

---

## 📦 Dependencies

- [streamlit](https://streamlit.io) – interactive web interface
- [langchain-community](https://github.com/langchain-ai/langchain) – vector store integration
- [sentence-transformers](https://www.sbert.net/) – embedding model
- [faiss-cpu](https://github.com/facebookresearch/faiss) – similarity search
- [requests](https://docs.python-requests.org/) – HTTP client for OpenRouter API
- (Optional) `python-dotenv` – for managing environment variables

---

## 🙌 Credits

- **Author:** Prakhar Mathur (BITS Pilani)
- **Medical knowledge base:** HuggingFace dataset `prakhar146/derma`
- **Powered by:** [OpenRouter](https://openrouter.ai) for access to multiple LLM models

---

## 📬 Contact

For queries, suggestions, or collaborations, feel free to reach out:

📧 [prakhar.mathur2020@gmail.com](mailto:prakhar.mathur2020@gmail.com)

---

## ⚖️ License

This project is provided for educational and research purposes. Check the licenses of the used models and datasets before commercial use.
```

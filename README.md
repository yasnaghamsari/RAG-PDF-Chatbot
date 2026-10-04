# 🔎 ScholarLens: Grounded Q&A over Your PDFs

**ScholarLens** is a Retrieval-Augmented Generation (RAG) application. You upload one or more PDF documents (lecture notes, papers, reports) and ask questions in natural language. The system answers **only from your documents** and shows the **file name and page number** behind each answer. You can also open the exact passages it retrieved and check them yourself.

It was built as the final project for an Artificial Intelligence course. The goal is a clear and explainable RAG system, not an enterprise product.

---

## ✨ Features

| Area | What it does |
|---|---|
| **Multi-PDF upload** | Upload several PDFs at once. Each one is checked for empty, scanned, corrupted, or password-protected content. |
| **Metadata preservation** | Every chunk keeps its `source` (file name) and `page` number. |
| **Smart chunking** | `RecursiveCharacterTextSplitter` splits text on paragraphs, then lines, then sentences, then words. Chunk size and overlap are configurable. |
| **Vector search** | OpenAI embeddings are stored in a **FAISS** index and searched by cosine similarity. |
| **Adjustable Top-K** | Choose how many chunks (1 to 10) are passed to the LLM. |
| **Adjustable temperature** | Control how deterministic or creative the model is. |
| **Custom RAG prompt** | Forbids outside knowledge, requires `[n]` citations, and uses a fixed reply when the answer is not in the documents. |
| **Source citations** | File and page chips appear under each answer, taken from the `[n]` markers the model actually used. |
| **Retrieved Context panel** | An expandable view of every retrieved passage with its similarity score. |
| **Statistics dashboard** | Shows the number of documents, pages, chunks, and questions asked. |
| **Conversation history** | Stored in Streamlit session state. Can be cleared, reset, or exported to Markdown. |
| **Error handling** | Covers a missing API key, an invalid key, rate limits, network errors, empty PDFs, and PDFs with little text. |

---

## 🏗️ Architecture

```
RAG-PDF Chatbot/
├── app.py             # Streamlit UI: layout, sidebar, chat, session state
├── rag_pipeline.py    # RAGPipeline class: embeddings, FAISS, retrieval, prompt, LLM
├── pdf_utils.py       # PDF text extraction, cleaning, quality checks, chunking
├── config.py          # All settings (models, chunk sizes, limits) in one place
├── styles.py          # Custom CSS for the dashboard look
├── .streamlit/
│   └── config.toml    # Streamlit theme colours + upload limit
├── requirements.txt
├── .env.example       # Template for your OpenAI API key
└── README.md
```

How the modules depend on each other:

```
            ┌──────────────┐
            │    app.py    │  (user interface only)
            └──────┬───────┘
         uses      │       uses
     ┌─────────────┼──────────────┐
     ▼             ▼              ▼
┌───────────┐ ┌───────────────┐ ┌───────────┐
│pdf_utils  │ │ rag_pipeline  │ │ styles    │
│ extract   │ │ build_index   │ └───────────┘
│ chunk     │ │ retrieve      │
└─────┬─────┘ │ generate/ask  │
      │       └──────┬────────┘
      └──────┬───────┘
             ▼
        ┌──────────┐
        │ config   │
        └──────────┘
```

The UI never talks to OpenAI or FAISS directly. It only calls `extract_pages`, `chunk_documents`, `RAGPipeline.build_index`, and `RAGPipeline.ask`. Because of this separation, each part can be explained and tested on its own.

---

## ⚙️ Installation

**Requirements:** Python 3.10 or newer (tested with Python 3.14) and an OpenAI API key.

```bash
# 1. Go to the project folder
cd "RAG-PDF Chatbot"

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 3. Install the dependencies
pip install -r requirements.txt
```

> **Windows tip:** if `pip` fails with *"No such file or directory … Long Path support"*, the folder path is too long. Keep the project in a short path or enable Windows long paths.

---

## 🔑 Environment Variables

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | (none) | Your OpenAI API key |
| `OPENAI_CHAT_MODEL` | ❌ | `gpt-4o-mini` | LLM that writes the answers |
| `OPENAI_EMBEDDING_MODEL` | ❌ | `text-embedding-3-small` | Model that turns text into vectors |

If no key is found, the sidebar shows a password field where you can enter one for the current session.

---

## ▶️ Running the Project

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>.

1. Upload one or more PDFs in the sidebar.
2. (Optional) Change the chunk size and overlap under **Chunking options**.
3. Click **⚙️ Process documents** and watch the status box.
4. Ask a question in the chat box.
5. Read the answer, check the **Sources** chips, and open **🔍 Retrieved context** to see the evidence.
6. Change **Top-K** and **temperature** in the sidebar and compare how the answers change.

---

## 🔄 RAG Workflow

RAG has two phases.

### Phase 1: Indexing (once per upload)

| Step | Where | What happens |
|---|---|---|
| 1. Text extraction | `pdf_utils.extract_pages` | `pypdf` reads each page. The text is cleaned, and each page becomes a `Document(page_content, metadata={source, page})`. Pages with no text are skipped and reported. |
| 2. Chunking | `pdf_utils.chunk_documents` | Pages are split into chunks of about 1000 characters with a 200-character overlap. Metadata is copied to every chunk. |
| 3. Embedding | `RAGPipeline.build_index` | Each chunk is sent to `text-embedding-3-small`, which returns a 1536-dimensional vector. |
| 4. Indexing | `RAGPipeline.build_index` | The vectors are stored in an in-memory **FAISS** index that uses inner product. OpenAI vectors have unit length, so the inner product equals **cosine similarity**. |

### Phase 2: Querying (for every question)

| Step | Where | What happens |
|---|---|---|
| 5. Query embedding | `RAGPipeline.retrieve` | The question is embedded with the same model. |
| 6. Similarity search | `RAGPipeline.retrieve` | FAISS returns the **Top-K** chunks closest to the question vector. |
| 7. Context building | `format_context` | The chunks are numbered `[1] … [K]` and labelled with their file and page. |
| 8. Generation | `RAGPipeline.generate` | The custom prompt, context, and question go to the chat model (`prompt \| llm` in LangChain Expression Language). |
| 9. Citations | `extract_cited_sources` | The `[n]` markers in the answer are mapped back to file and page, then shown as source chips. |

### The custom prompt

The original project used LangChain's default `RetrievalQA` chain. ScholarLens uses its own prompt (see `rag_pipeline.py`), which:

* allows **only** the provided context to be used,
* requires a **numbered citation** after each statement,
* defines one exact sentence for missing answers: *"I could not find the answer to this question in the uploaded documents."* The app detects this sentence and shows *"No supporting source found"* instead of sources that would be misleading,
* asks the model to state what is missing when the context only partly answers the question.

---

## 🧰 Technologies Used

| Technology | Role |
|---|---|
| **Python 3** | Programming language |
| **Streamlit** | Web interface, chat components, session state |
| **LangChain** (`langchain-core`, `langchain-community`, `langchain-openai`, `langchain-text-splitters`) | Document objects, text splitter, FAISS wrapper, prompt templates, LCEL chain |
| **OpenAI API** | `text-embedding-3-small` for embeddings and `gpt-4o-mini` for answers |
| **FAISS** (`faiss-cpu`) | Vector similarity search library from Meta AI |
| **pypdf** | PDF text extraction (the maintained successor of PyPDF2) |
| **python-dotenv** | Loads the API key from `.env` |

---

## ⚠️ Limitations and Possible Extensions

* **Scanned PDFs**: image-only pages contain no text. They are detected and reported, but not processed. *Extension:* add OCR with Tesseract.
* **In-memory index**: the FAISS index is rebuilt on every upload and lost on restart. *Extension:* `vector_store.save_local()`.
* **Single-turn retrieval**: each question is retrieved on its own, so follow-ups like "explain more" do not use earlier context. *Extension:* rewrite the question using the chat history.
* **Evaluation**: answer quality is not measured automatically. *Extension:* build a small question/answer test set and measure retrieval hit rate.

---

## 📚 Pipeline in One Line

```
PDF → Text Extraction → Chunking → Embeddings → FAISS → Similarity Search → Context → LLM → Answer + Sources
```

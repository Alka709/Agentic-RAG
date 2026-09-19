# DocuMind AI

**DocuMind AI** is an intelligent multimodal document assistant powered by **Hybrid RAG (FAISS + BM25 with Reciprocal Rank Fusion)**, **LangGraph orchestration**, retrieval evaluation, and **Model Context Protocol (MCP)** web-search fallback.

---

## ⚡ Quick Start (Docker)

### 1. Configure Environment
Copy the sample environment file and set your API keys:
```bash
cp .env.example .env
```
Fill in `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 2. Start Application
```bash
docker compose up -d --build
```

* **Web UI:** [http://localhost](http://localhost) (Port 80)
* **API Healthcheck:** [http://localhost/health](http://localhost/health)

### 3. Stop Containers
```bash
docker compose down
```

---

## 💻 Local Development Setup

### Backend (FastAPI)
```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start API server
uvicorn api:app --reload --port 8000
```
* **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```
* **Frontend Dev Server:** [http://localhost:5173](http://localhost:5173)

---

## 🛠️ Architecture & Workflow

```text
                     User / Web UI
                           │
                           ▼
                  FastAPI Backend (/query)
                           │
                           ▼
                   LangGraph Workflow
                           │
                           ▼
             Hybrid Retrieval (RRF Fusion)
              ├── FAISS Dense Vector Search
              └── BM25 Sparse Lexical Search
                           │
                           ▼
                 Retrieval Evaluator
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
        Sufficient                  Insufficient
             │                           │
             │                     MCP Web Search (Tavily)
             │                           │
             └─────────────┬─────────────┘
                           ▼
                 LLM Synthesis (Gemini)
                           │
                           ▼
                        Answer
```

---

## ⚙️ Configuration (.env)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | *Required* | Google Gemini API key |
| `TAVILY_API_KEY` | *Optional* | Tavily API key for web search fallback |
| `LLM_MODEL` | `gemini-2.0-flash` | Primary text generation model |
| `VISION_LLM_MODEL`| `gemini-2.0-flash` | Vision model for PDF image extraction |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for vector search |
| `CHUNK_SIZE` | `1000` | Text chunk character limit |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |

---

## 📁 Supported Document Formats

* **PDF (`.pdf`)** — Extracts text, structured tables, and embedded images.
* **Word (`.docx`)** — Text extraction.
* **Markdown / Plain Text (`.md`, `.txt`)** — Plain text chunking.
* **Images (`.png`, `.jpg`, `.jpeg`, `.webp`)** — Vision-based description and indexing.
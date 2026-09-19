# MCP-Powered Agentic RAG

An agentic Retrieval-Augmented Generation (RAG) system that combines **multimodal document ingestion, hybrid retrieval, LangGraph orchestration, retrieval evaluation, and MCP-powered web-search fallback**.

The system retrieves relevant information from user-provided documents using both semantic and lexical search. An evaluator determines whether the retrieved context is sufficient; when it is not, the system can fall back to web search through an MCP tool before generating the final answer.

---

## 🚀 Features

- 📄 **Multimodal document ingestion**
  - Text extraction
  - Table extraction and structured representation
  - Image extraction and vision-based descriptions

- 🔎 **Hybrid Retrieval**
  - Dense retrieval using FAISS
  - Lexical retrieval using BM25
  - Reciprocal Rank Fusion (RRF) for combining rankings

- 🤖 **Agentic RAG**
  - LangGraph-based workflow
  - Retrieval evaluation before generation
  - Conditional web-search fallback

- 🔌 **MCP Integration**
  - MCP-based tool execution
  - Web-search fallback through Tavily

- 📊 **Retrieval Evaluation**
  - Context Precision
  - Context Recall
  - Hit Rate@K
  - Mean Reciprocal Rank (MRR)
  - Faithfulness
  - Answer Relevance
  - Retrieval and total latency
  - Fallback rate

- 🌐 **Web Interface**
  - React + Vite frontend
  - Document upload
  - Natural-language querying
  - Retrieved source display
  - Web fallback indication

- 🐳 **Docker Support**
  - Containerized backend
  - Containerized frontend
  - Docker Compose setup

---

# 🏗️ Architecture

```text
                         User
                           │
                           ▼
                  ┌─────────────────┐
                  │ React + Vite    │
                  │    Frontend     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ FastAPI Backend │
                  └────────┬────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  LangGraph  │
                    │   Workflow  │
                    └──────┬──────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Hybrid Retrieval │
                  └────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       ┌──────────────┐          ┌──────────────┐
       │ FAISS Dense  │          │ BM25 Sparse  │
       │  Retrieval   │          │  Retrieval   │
       └──────┬───────┘          └──────┬───────┘
              │                         │
              └───────────┬─────────────┘
                          ▼
                 ┌────────────────┐
                 │ RRF Fusion     │
                 └───────┬────────┘
                         │
                         ▼
                 ┌────────────────┐
                 │   Evaluator    │
                 └───────┬────────┘
                         │
                 ┌───────┴────────┐
                 │                │
          Sufficient         Insufficient
                 │                │
                 ▼                ▼
              LLM           MCP Web Search
                                  │
                                  ▼
                              Tavily
                                  │
                    ┌─────────────┘
                    ▼
                   LLM
                    │
                    ▼
              Final Answer
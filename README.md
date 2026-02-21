# 🚀 Production-Grade RAG Pipeline

A scalable, production-ready **Retrieval-Augmented Generation (RAG)** system built with hybrid search, asynchronous ingestion, and an agentic self-correction loop to reduce hallucinations and improve answer reliability.

---

## 🏛 Architecture Overview

The system follows a modular, layered design to ensure scalability, reliability, and clean separation of concerns.

---

## 1️⃣ Asynchronous Ingestion Layer

To prevent blocking API requests during heavy document processing, ingestion runs in the background.

**Core Components:**

* **Celery + Redis** → Handles distributed background tasks (PDF parsing, chunking, indexing).
* **Recursive Text Splitting** → Maintains semantic coherence across document chunks.
* **Dual Indexing (Qdrant + BM25)** → Enables both semantic (vector) and keyword-based retrieval.

**Pipeline Flow:**
Upload → Parse → Chunk → Embed → Index

---

## 2️⃣ Agentic Reasoning Layer (LangGraph)

Instead of a traditional linear RAG chain, this system uses a **state-machine-based graph** to enable iterative reasoning and refinement.

### 🔀 Intelligent Routing

Determines whether a query requires retrieval or can be answered directly.

### 🔎 Multi-Query Expansion

Generates multiple reformulations (3–5 variants) of the user query to improve recall and avoid missing relevant documents.

### 🧠 Self-Correction (Critique Node)

Evaluates whether retrieved content sufficiently supports the answer.
If not, the system refines the query and retries retrieval.

This feedback loop significantly reduces hallucination risk.

---

## 3️⃣ Retrieval Optimization Strategy

### 🔄 Hybrid Search

Combines:

* Dense vector search (semantic meaning)
* BM25 keyword search

Results are merged using **Reciprocal Rank Fusion (RRF)** for balanced ranking.

### 🎯 Re-Ranking

Applies Cohere Rerank to:

* Remove low-signal documents
* Improve context precision
* Reduce token waste in the LLM prompt

---

# 🛠 Technology Stack

| Layer               | Technology     | Purpose                               |
| ------------------- | -------------- | ------------------------------------- |
| Orchestration       | LangGraph      | Enables cyclic, agent-style workflows |
| API Layer           | FastAPI        | Async, high-performance API handling  |
| Vector Store        | Qdrant         | Fast similarity search with filtering |
| Task Queue          | Celery + Redis | Offloads heavy ingestion tasks        |
| Prompt Optimization | DSPy           | Structured programmatic optimization  |
| Monitoring          | LangSmith      | Full execution tracing                |
| Evaluation          | RAGAS          | Quantitative RAG performance metrics  |

---

# 🔄 End-to-End Workflow

## 📥 Document Ingestion

1. Client uploads a document via `/upload`
2. FastAPI triggers a background Celery task
3. Document is:

   * Parsed
   * Split into chunks
   * Embedded (`text-embedding-3-small`)
   * Stored in Qdrant

The API remains responsive during this process.

---

## 🧠 Query Processing Loop

1. Query enters LangGraph
2. Agent selects tools (Vector DB, Web Search, etc.)
3. Hybrid retrieval fetches relevant chunks
4. Results are re-ranked
5. Critique node verifies answer grounding
6. Final response is generated for the UI

If grounding fails → the loop repeats.

---

# 📊 Evaluation & Metrics

The system was benchmarked using **RAGAS** against a curated dataset of 50 verified Q&A samples.

| Metric            | Score | Meaning                                 |
| ----------------- | ----- | --------------------------------------- |
| Faithfulness      | 0.89  | Answer is grounded in retrieved context |
| Answer Relevancy  | 0.92  | Response matches user intent            |
| Context Precision | 0.85  | Quality of retrieved documents          |

All execution traces are logged in **LangSmith** for debugging and optimization.

---

# 🚀 Setup Instructions

### Requirements

* Docker & Docker Compose
* OpenAI API Key
* Cohere API Key

### Installation Steps

```bash
git clone https://github.com/KarthikaRajagopal44/aiagents_projects/production_rag.git
cd pro-rag
cp .env.example .env
docker-compose up --build
```

API documentation will be available at:
`http://localhost:8000/docs`

---

# 💡 Engineering Highlights

### 🔍 Reducing Hallucinations

Implemented a validation node that ensures responses are strictly supported by retrieved documents.

### 📈 Improving Recall

Hybrid search improved retrieval performance on acronym-heavy and technical queries by approximately 30%.

### ⚡ Lowering Latency

Async endpoints and background task workers keep the system responsive even during large file uploads.



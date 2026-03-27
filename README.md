# 🤖 RAG Chatbot with RBAC, Guardrails & Monitoring

A production-ready **internal enterprise chatbot** powered by Retrieval-Augmented Generation (RAG).
It answers questions from private organisational documents with **role-based access control (RBAC)**,
**PII guardrails**, and **token-usage monitoring**.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| **RAG Pipeline** | LangChain + ChromaDB + Groq (Llama 3.1) |
| **RBAC** | Finance / HR / C-Suite roles with per-collection access |
| **Guardrails** | PII masking (Presidio + regex fallback), out-of-scope & blocked-pattern detection |
| **Monitoring** | Token usage tracking, cost estimation, RAGAS quality evaluation |
| **Frontend** | Streamlit chat UI with role colour-coding, source display, sidebar stats |
| **Deployment** | Docker, Docker Compose, GitHub Actions CI/CD, Azure Pipelines |

---

## 🗂️ Project Structure

```
RAG-Chatbot/
├── config/              # Pydantic-settings configuration
├── data/                # Sample documents (finance / hr / general)
├── src/
│   ├── ingestion/       # Document loading & chunking
│   ├── vectorstore/     # ChromaDB wrapper
│   ├── rbac/            # Auth, roles, permissions
│   ├── guardrails/      # PII masker, scope detector
│   ├── llm/             # Groq LLM client
│   ├── rag/             # Main RAG pipeline
│   └── monitoring/      # Token tracker, RAGAS evaluator
├── app/
│   └── streamlit_app.py # Streamlit frontend
├── scripts/
│   └── ingest_data.py   # One-shot ingestion script
├── tests/               # pytest test suite
└── deployment/          # Dockerfile, docker-compose, Azure Pipelines
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd RAG-Chatbot-with-RBAC-Guardrails-and-Monitoring
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (https://console.groq.com)
```

### 3. Ingest sample documents

```bash
python scripts/ingest_data.py
```

### 4. Run the chatbot

```bash
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** and log in with a demo account:

| Username | Password | Role |
|----------|----------|------|
| `finance_user` | `finance123` | Finance |
| `hr_user` | `hr123` | HR |
| `ceo` | `ceo123` | C-Suite |

---

## 🐳 Docker

```bash
# Build and run everything with Docker Compose
cd deployment
docker compose up --build
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 🏗️ Architecture

```
User → Streamlit UI
         │
         ▼
    Authentication (RBAC)
         │
         ▼
    ScopeDetector ──► blocked / out-of-scope → early return
         │
         ▼
    ChromaVectorStore.query(allowed_collections)
         │
         ▼
    PIIMasker (retrieved docs)
         │
         ▼
    GroqLLMClient (Llama 3.1) ──► answer
         │
         ▼
    PIIMasker (answer)
         │
         ▼
    TokenTracker → cost alert if threshold exceeded
         │
         ▼
    Response {answer, sources, tokens_used, role}
```

---

## 📦 Key Dependencies

- [LangChain](https://python.langchain.com/) — RAG orchestration
- [Groq](https://groq.com/) — Ultra-fast LLM inference (Llama 3.1)
- [ChromaDB](https://www.trychroma.com/) — Local vector store
- [Sentence Transformers](https://sbert.net/) — Embeddings (`all-MiniLM-L6-v2`)
- [Microsoft Presidio](https://microsoft.github.io/presidio/) — PII detection & anonymisation
- [RAGAS](https://docs.ragas.io/) — RAG quality evaluation
- [Streamlit](https://streamlit.io/) — Frontend UI

---

## 🔐 Security Notes

- Secrets are loaded from `.env` (never committed — see `.gitignore`)
- PII is masked in both retrieved context and LLM responses
- Each role can only query its authorised ChromaDB collections
- Adversarial / jailbreak patterns are rejected before reaching the LLM


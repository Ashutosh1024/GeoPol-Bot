#  GeoPol Bot Nayi Branch me.

**GeoPol Bot** is an AI-powered geopolitics-only research and question-answering assistant. Built with **FastAPI** on the backend and **Streamlit** on the frontend, it enforces strict domain isolation, verified geopolitical knowledge retrieval, and evidence-grounded answer generation using the OpenAI API.

---

##  System Architecture Naya Wala

```text
GeoPol BoT/
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── main.py                     # FastAPI app factory, CORS, lifespan & router
│       ├── core/
│       │   ├── __init__.py
│       │   └── config.py               # pydantic-settings config, env loader & secret masking
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── health.py               # Pydantic schemas for health & component diagnostics
│       ├── api/
│       │   ├── __init__.py
│       │   └── v1/
│       │       ├── __init__.py
│       │       ├── router.py           # v1 Route aggregator
│       │       └── endpoints/
│       │           ├── __init__.py
│       │           └── health.py       # Detailed /health and lightweight /healthz endpoints
│       ├── domain_gate/                # (Step 2) Structured classification gate
│       │   └── __init__.py
│       ├── knowledge/                  # (Step 3) Corpus ingestion, embeddings & vector store
│       │   └── __init__.py
│       └── generation/                 # (Step 4) OpenAI Responses API generation & citations
│           └── __init__.py
├── frontend/
│   ├── __init__.py
│   └── app.py                          # Streamlit UI connecting to FastAPI backend
├── tests/
│   ├── __init__.py
│   ├── test_config.py                  # Unit tests for settings, CORS, and key masking
│   └── test_health.py                  # Unit tests for FastAPI health endpoints
├── .env.example                        # Environment variable configuration template
├── .gitignore                          # Standard gitignore (secrets, .env, __pycache__)
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation & execution guide
```

---

##  Security & Credential Isolation

- **Zero-Secret Exposure**: `OPENAI_API_KEY` is loaded exclusively inside the FastAPI backend runtime via `backend/app/core/config.py`.
- **Frontend Safety**: The Streamlit frontend **never** accesses `OPENAI_API_KEY` directly; all queries and validations are routed via the backend HTTP API.
- **Diagnostics Masking**: Any health or diagnostic endpoint masks sensitive keys (e.g., `sk-...abcd`) so credentials can never be leaked in responses or logs.

---

##  Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.11+ (Python 3.13 tested)
- Pip or Conda package manager

### 2. Environment Configuration
Copy the sample environment file and configure your credentials:

```bash
cp .env.example .env
```

Edit `.env` and set your OpenAI API key (for future steps):
```env
OPENAI_API_KEY="sk-your-openai-api-key"
```

### 3. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

---

##  Verification & Testing

Run the automated test suite with `pytest`:
```bash
python -m pytest -v
```

All health, configuration, and security masking tests will run.

---

##  Running the Application Locally

### Step 1: Start the FastAPI Backend
Open a terminal in the project root directory and run:
```bash
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
- Interactive Swagger API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Endpoint: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)
- Gate Classification: [http://127.0.0.1:8000/api/v1/gate/classify](http://127.0.0.1:8000/api/v1/gate/classify)
- Grounded Generation: [http://127.0.0.1:8000/api/v1/chat/generate](http://127.0.0.1:8000/api/v1/chat/generate)
- Liveness Probe: [http://127.0.0.1:8000/api/v1/healthz](http://127.0.0.1:8000/api/v1/healthz)

### Step 2: Start the Streamlit Frontend
Open a second terminal in the project root directory and run:
```bash
streamlit run frontend/app.py
```
- Access the Streamlit UI at: [http://localhost:8501](http://localhost:8501)

The frontend immediately connects to your FastAPI backend, provides real-time health and component monitoring, retains conversational history, and visually displays:
- 🟢 **Answered**: Verified facts, strategic interpretations, and uncertainties with clickable primary source treaty citations.
- 🔴 **Rejected at Gate**: Informative rejection message explaining why queries outside international relations are blocked.
- 🟡 **Clarification Needed**: Targeted questions guiding users to clarify ambiguous or borderline queries.
- ⚪ **Abstained**: Clear notice when the query is in-domain but lacks sufficient verified evidence in the corpus, preventing hallucination.

---

##  Roadmap & Implementation Steps
- [x] **Step 1: Foundation** - Modular FastAPI & Streamlit structure, config management, health endpoints, security boundaries, and unit tests.
- [x] **Step 2: Domain Gate** - Independent tri-state classifier (`IN_DOMAIN`, `OUT_OF_DOMAIN`, `UNCERTAIN`) powered by OpenAI structured outputs, heuristic pre-filters, and adversarial defense.
- [x] **Step 3: Geopolitical Knowledge Pipeline** - Verifiable primary source corpus (UN, NATO, UNCLOS, AUKUS), chunking with full provenance metadata, OpenAI `text-embedding-3-small`, SQLite vector store, cosine similarity ranking, and corpus maintenance protocol.
- [x] **Step 4: Evidence-Grounded Generation** - Gate-enforced OpenAI Responses API generation, tripartite separation (verified facts, strategic interpretations, uncertainties), strict citation validation against retrieved evidence, untrusted context isolation, and clean abstention.
- [x] **Step 5: Full Chat Interface** - Interactive Streamlit chat interface with session history, visual status pills, clickable citations, raw evidence inspector, and zero-key frontend isolation.
- [x] **Step 6: Evaluation Suite (Completed)** - Automated benchmarking suite with 15 gold-standard test queries, empirical metrics report ([`EVALUATION_REPORT.md`](file:///c:/Users/ashutosh%20kumar/OneDrive/Documents/GeoPol%20BoT/EVALUATION_REPORT.md)), and full invariant validation.

---

##  Running the Evaluation Benchmark

Run the automated evaluation harness to verify metrics, latency, and costs:
```bash
python -m evaluation.runner
```
Or execute the full unit and integration test suite:
```bash
python -m pytest -v
```
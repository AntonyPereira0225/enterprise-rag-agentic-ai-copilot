# Solution Architecture

## Project
Enterprise RAG & Agentic AI Copilot

## Architecture Goal

The platform is designed as a modular, enterprise-style Retrieval-Augmented Generation (RAG) and Agentic AI system. The objective is to demonstrate not only LLM prompting, but the full engineering lifecycle around ingestion, retrieval, orchestration, evaluation, observability, API serving, responsible AI and deployment.

The baseline implementation is intentionally local-first and open-source-first so the project can be developed without relying on paid cloud inference or managed vector-database services. Cloud components can be introduced later without redesigning the full application.

## High-Level Architecture

```mermaid
flowchart TD
    A[Knowledge Sources\nPolicies | Product Docs | FAQs | Support Cases] --> B[Ingestion Pipeline]
    B --> C[Cleaning & Normalisation]
    C --> D[Chunking & Metadata Enrichment]
    D --> E[Embedding Model]
    E --> F[Vector Index]
    D --> G[Keyword / BM25 Index]
    F --> H[Hybrid Retrieval]
    G --> H
    H --> I[Reranker]
    I --> J[Context Builder]
    J --> K[LLM Generation]
    K --> L[Grounded Answer + Citations]

    M[User Query] --> N[Intent Classification]
    N --> O[LangGraph Agent Router]
    O --> P[Policy Agent]
    O --> Q[Product Agent]
    O --> R[Support Agent]
    O --> S[Analytics Agent]

    P --> H
    Q --> H
    R --> H
    S --> H

    L --> T[Guardrails & Confidence Checks]
    T --> U[FastAPI Service]
    U --> V[Streamlit Demo UI]

    H --> W[Evaluation Framework]
    K --> W
    W --> X[MLflow Tracking]

    U --> Y[Observability\nLatency | Errors | Retrieval | Generation]
```

## Core Architectural Layers

### 1. Knowledge and Data Sources

The initial knowledge estate will contain synthetic and public-safe documents representing:

- company policies
- operational procedures
- product and service documentation
- FAQs
- customer-support playbooks
- historical support cases
- structured reference data

No confidential enterprise information will be used.

### 2. Ingestion Layer

The ingestion layer converts source documents into a consistent internal representation.

Responsibilities:

- document loading
- text extraction
- metadata capture
- cleaning and normalisation
- duplicate detection
- chunk generation
- chunk identifiers
- source lineage preservation

Planned Python components:

- `src/ingestion/loaders.py`
- `src/ingestion/cleaning.py`
- `src/ingestion/chunking.py`
- `src/ingestion/metadata.py`

### 3. Embedding Layer

Document chunks are converted into dense numerical representations using a local embedding model from the Hugging Face / Sentence Transformers ecosystem.

Initial direction:

- Sentence Transformers embedding model
- deterministic batch embedding pipeline
- configurable model name
- embedding metadata persisted alongside document identifiers

The embedding interface will be abstracted so that an Azure OpenAI, OpenAI, Vertex AI or other embedding provider could be substituted later.

### 4. Retrieval Layer

The retrieval subsystem will progress through three stages:

1. semantic vector retrieval
2. keyword/BM25 retrieval
3. hybrid retrieval with reranking

Initial local vector-store options:

- FAISS
- Chroma

The retrieval layer will return both content and metadata so generated answers can include verifiable citations.

Planned components:

- `src/retrieval/vector_store.py`
- `src/retrieval/semantic.py`
- `src/retrieval/keyword.py`
- `src/retrieval/hybrid.py`
- `src/retrieval/reranker.py`

### 5. RAG Generation Layer

The RAG pipeline will combine the user question with retrieved evidence and instruct the model to answer only from the supplied context.

Responsibilities:

- prompt construction
- context-window management
- source attribution
- structured output generation
- low-confidence fallback
- unsupported-answer handling

The baseline model will be an appropriately sized local open-source model, potentially served through Ollama or Hugging Face tooling.

### 6. Agentic Orchestration Layer

LangGraph will be used to model explicit stateful workflows rather than relying on an unrestricted autonomous agent.

Initial specialist agents:

| Agent | Responsibility |
|---|---|
| Policy Agent | Retrieve and explain policies and procedures. |
| Product Agent | Answer product/service knowledge questions. |
| Support Agent | Recommend approved support actions and escalation steps. |
| Analytics Agent | Summarise structured support or operational information. |

The router will classify the user request and send it to the appropriate workflow. Unsupported or ambiguous requests will follow a controlled fallback path.

Planned components:

- `src/agents/state.py`
- `src/agents/router.py`
- `src/agents/policy_agent.py`
- `src/agents/product_agent.py`
- `src/agents/support_agent.py`
- `src/agents/analytics_agent.py`
- `src/agents/graph.py`

### 7. Responsible AI and Guardrails

Guardrails will be built as explicit application logic rather than treated as a single prompt.

Controls will include:

- prompt-injection checks
- system-instruction protection
- unsupported-answer rejection
- citation coverage checks
- confidence thresholds
- sensitive-content checks
- structured-output validation
- safe fallback responses

A dedicated adversarial test set will be maintained for reproducible safety evaluation.

### 8. Evaluation Layer

The system will include an evaluation pipeline from the beginning.

Retrieval metrics may include:

- Recall@K
- Precision@K
- Mean Reciprocal Rank
- NDCG

RAG / answer-quality metrics may include:

- faithfulness
- answer relevance
- context precision
- context recall
- citation coverage
- groundedness

Evaluation tooling may use RAGAS, DeepEval and custom deterministic checks where appropriate.

Results will be logged to MLflow to compare changes in chunk size, embedding model, retrieval strategy, reranking and prompt versions.

### 9. API Layer

FastAPI will expose production-style endpoints.

Planned endpoints:

- `POST /ask`
- `POST /retrieve`
- `POST /agent`
- `GET /health`
- `GET /metrics`

Pydantic models will enforce request and response schemas.

### 10. User Interface

A lightweight Streamlit application will provide a portfolio demonstration interface with:

- natural-language question input
- generated response
- source citations
- retrieved evidence
- selected agent/workflow
- latency information
- confidence / fallback indicators

The UI is a demonstration surface; the FastAPI layer remains the primary application interface.

### 11. Observability

The application will record operational metrics including:

- request count
- latency by pipeline stage
- retrieval latency
- generation latency
- retrieval scores
- fallback frequency
- exception counts
- model / prompt version
- evaluation results

Logs will be structured so they can later be connected to a cloud observability platform.

### 12. MLOps and CI/CD

The project will demonstrate software-engineering and MLOps practices through:

- MLflow experiment tracking
- pytest unit and integration tests
- linting and formatting checks
- Docker containerisation
- GitHub Actions CI
- environment-variable configuration
- secret exclusion from source control
- versioned evaluation datasets

## Proposed Repository Structure

```text
enterprise-rag-agentic-ai-copilot/
├── README.md
├── docs/
│   ├── business_requirements.md
│   ├── solution_architecture.md
│   ├── evaluation_strategy.md
│   └── responsible_ai.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_experiments.ipynb
│   └── 03_rag_evaluation.ipynb
├── src/
│   ├── ingestion/
│   ├── embeddings/
│   ├── retrieval/
│   ├── rag/
│   ├── agents/
│   ├── guardrails/
│   ├── evaluation/
│   ├── api/
│   └── monitoring/
├── app/
│   └── streamlit_app.py
├── tests/
├── config/
├── scripts/
├── .github/
│   └── workflows/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── .env.example
└── .gitignore
```

## Initial Technology Stack

| Capability | Initial Technology |
|---|---|
| Language | Python |
| Embeddings | Sentence Transformers / Hugging Face |
| Local LLM | Ollama-compatible open model or Hugging Face model |
| Vector search | FAISS or Chroma |
| Keyword retrieval | BM25 |
| Reranking | Cross-encoder reranker |
| Agent orchestration | LangGraph |
| API | FastAPI |
| Schemas | Pydantic |
| UI | Streamlit |
| Evaluation | RAGAS / DeepEval / custom metrics |
| Experiment tracking | MLflow |
| Testing | pytest |
| Containerisation | Docker |
| CI/CD | GitHub Actions |
| Version control | Git / GitHub |

## Cloud Extension Path

Once the local version is complete, the architecture can be extended to demonstrate cloud deployment without changing the core application contracts.

Possible Azure extension:

```text
Local Documents / Blob Storage
        ↓
Azure AI Search or managed vector search
        ↓
Azure OpenAI / hosted model endpoint
        ↓
FastAPI container
        ↓
Azure Container Apps / App Service
        ↓
Application Insights
```

This cloud deployment is optional and will only be introduced if it materially improves the portfolio relative to its cost.

## Architecture Principles

1. **Grounding before generation** — enterprise knowledge answers must be supported by retrieved evidence.
2. **Evaluation before optimisation** — architecture changes must be measured against a reproducible test set.
3. **Controlled agents over unrestricted autonomy** — workflows should be explicit, observable and testable.
4. **Local-first development** — paid infrastructure should not be required for core functionality.
5. **Modular interfaces** — models, vector stores and cloud providers should be replaceable.
6. **Traceability by design** — every answer should retain source and pipeline metadata.
7. **Responsible AI by design** — safety and unsupported-answer handling are core system requirements.
8. **Production-minded engineering** — tests, APIs, containerisation, CI/CD and monitoring are part of the project, not optional polish.

## Delivery Sequence

### Phase 1 — Foundation
- business requirements
- architecture
- repository structure
- environment setup

### Phase 2 — Knowledge Ingestion
- synthetic/public-safe document corpus
- cleaning
- chunking
- metadata

### Phase 3 — Retrieval Baseline
- embeddings
- vector index
- semantic search
- baseline retrieval evaluation

### Phase 4 — Advanced RAG
- keyword retrieval
- hybrid retrieval
- reranking
- grounded generation
- citations

### Phase 5 — Agentic Workflows
- intent routing
- specialist agents
- LangGraph state
- controlled tool/workflow execution

### Phase 6 — Evaluation & Responsible AI
- labelled evaluation set
- RAG metrics
- prompt-injection tests
- fallback and confidence logic
- MLflow experiment tracking

### Phase 7 — Production Interface
- FastAPI
- Pydantic schemas
- Streamlit UI
- structured logging

### Phase 8 — MLOps
- pytest
- Docker
- GitHub Actions
- reproducible environments

### Phase 9 — Optional Cloud Deployment
- deploy selected components only if cost and portfolio value justify it

# Repository Structure

```text
enterprise-rag-agentic-ai-copilot/
├── configs/                     # Non-secret application and evaluation configuration
├── data/                        # Local synthetic/public-safe data and generated indexes
├── docs/                        # Business, architecture and engineering documentation
├── notebooks/                   # Exploration only; production logic moves to src/
├── src/
│   └── enterprise_copilot/
│       ├── ingestion/           # Loading, cleaning, parsing and chunking
│       ├── retrieval/           # Embeddings, vector search, BM25 and reranking
│       ├── llm/                 # Grounded generation, citation checks and optional Ollama
│       ├── agents/              # Intent routing and explicit stateful orchestration
│       ├── guardrails/          # Responsible-AI and security controls
│       ├── evaluation/          # Retrieval/RAG/agent/API/model/release evaluation
│       ├── api/                 # Shared service plus standard-library/FastAPI adapters
│       ├── ui/                  # Responsive browser demonstration
│       └── common/              # Shared settings, schemas and utilities
├── tests/                       # Unit, integration and adversarial tests
├── .env.example                 # Safe environment-variable template
├── pyproject.toml               # Python package/tool configuration
├── requirements.txt             # Dependency-free core runtime note
├── requirements-api.txt         # Optional FastAPI/Uvicorn adapter
├── requirements-dense.txt       # Optional neural embedding/FAISS adapter
├── requirements-mlops.txt       # Optional MLflow adapter
└── requirements-dev.txt         # Testing and development dependencies
```

## Design Principles

- Keep notebooks exploratory; reusable logic belongs in `src/`.
- Keep modules independent enough to test retrieval, LLM, agents and guardrails separately.
- Never commit real secrets or confidential enterprise data.
- Prefer local/open-source components for the baseline so the project remains reproducible without paid cloud services.
- Add cloud providers through replaceable adapters rather than hard-coding the architecture to one vendor.

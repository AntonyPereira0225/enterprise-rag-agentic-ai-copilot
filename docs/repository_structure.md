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
│       ├── llm/                 # Prompting and grounded response generation
│       ├── agents/              # Intent routing and LangGraph orchestration
│       ├── guardrails/          # Responsible-AI and security controls
│       ├── evaluation/          # Retrieval/RAG/agent evaluation
│       ├── api/                 # FastAPI service layer
│       ├── ui/                  # Demonstration UI
│       └── common/              # Shared settings, schemas and utilities
├── tests/                       # Unit, integration and adversarial tests
├── .env.example                 # Safe environment-variable template
├── pyproject.toml               # Python package/tool configuration
├── requirements.txt             # Runtime dependencies
└── requirements-dev.txt         # Testing and development dependencies
```

## Design Principles

- Keep notebooks exploratory; reusable logic belongs in `src/`.
- Keep modules independent enough to test retrieval, LLM, agents and guardrails separately.
- Never commit real secrets or confidential enterprise data.
- Prefer local/open-source components for the baseline so the project remains reproducible without paid cloud services.
- Add cloud providers through replaceable adapters rather than hard-coding the architecture to one vendor.

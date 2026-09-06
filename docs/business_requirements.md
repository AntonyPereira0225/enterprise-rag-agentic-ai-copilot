# Business Requirements

## Project
Enterprise RAG & Agentic AI Copilot

## Business Scenario

A fictional multinational consumer-services company operates across several markets and maintains a large internal knowledge estate containing policies, product documentation, customer-support procedures, service guides, FAQs, operational playbooks and historical support cases.

Employees currently search across disconnected repositories and documents to answer routine questions. This creates slow response times, inconsistent answers, duplicated effort and limited visibility into the evidence used to support decisions.

The proposed solution is an enterprise-grade Retrieval-Augmented Generation (RAG) and Agentic AI Copilot that retrieves relevant internal knowledge, produces grounded responses with source citations, routes requests to specialised agents when appropriate, applies safety and governance controls, and measures answer quality through a reproducible evaluation framework.

> This project uses synthetic and public-safe sample data only. It is not affiliated with any real company and does not contain confidential enterprise information.

## Business Objectives

1. Reduce the time employees spend searching for trusted internal information.
2. Improve consistency and traceability of answers by grounding responses in approved source documents.
3. Demonstrate automated routing of different request types to specialised AI agents.
4. Provide transparent citations and supporting evidence for generated responses.
5. Detect unsupported or low-confidence answers before they are presented as reliable.
6. Measure retrieval and generation quality using repeatable evaluation metrics.
7. Demonstrate responsible-AI controls for prompt injection, unsafe requests, hallucination risk and sensitive information.
8. Expose the solution through a reusable API and user-facing application.

## Primary Users

| User | Need |
|---|---|
| Customer Support Agent | Find approved answers and procedures quickly while handling customer cases. |
| Operations Analyst | Search policies, processes and operational guidance across multiple knowledge sources. |
| Product Specialist | Retrieve accurate product documentation and compare product or service information. |
| Team Manager | Understand recurring support issues, escalations and knowledge gaps. |
| Risk / Compliance User | Verify that an answer is supported by approved policy documentation and evidence. |
| AI / Data Engineering Team | Monitor retrieval quality, model behaviour, latency, failures and evaluation metrics. |

## Core Use Cases

### UC-01 — Grounded Knowledge Question Answering
A user asks a natural-language question. The system retrieves the most relevant approved knowledge, reranks the evidence and generates an answer that cites the supporting sources.

### UC-02 — Customer Support Assistance
A support user provides a customer issue. The system identifies the issue type, retrieves relevant procedures and recommends the next approved action without pretending to execute a real customer-account change.

### UC-03 — Policy and Procedure Search
A user searches for a policy requirement or operational procedure. The system returns the relevant policy sections, source metadata and a concise explanation.

### UC-04 — Multi-Agent Routing
The system identifies the intent of a request and routes it to an appropriate specialised workflow, such as policy, product, support or analytics assistance.

### UC-05 — Evidence and Citation Verification
Every grounded answer exposes the retrieved documents or passages used to support it so a user can verify the response.

### UC-06 — Low-Confidence Handling
If the retrieved evidence is insufficient, contradictory or below the configured confidence threshold, the system should explicitly state that it cannot provide a reliable grounded answer and suggest escalation or further search.

### UC-07 — LLM Evaluation
The platform evaluates responses against a labelled test set using retrieval and generation metrics such as context precision, context recall, answer relevance, faithfulness and citation coverage.

### UC-08 — Responsible AI and Security Testing
The platform tests malicious or risky prompts including prompt injection, attempts to override system instructions, requests for unsupported information and attempts to expose protected system context.

## Functional Requirements

| ID | Requirement |
|---|---|
| FR-01 | Ingest supported knowledge documents and preserve document-level metadata. |
| FR-02 | Clean, normalise and chunk documents using a reproducible ingestion pipeline. |
| FR-03 | Generate vector embeddings for document chunks. |
| FR-04 | Store and retrieve embeddings using a vector index. |
| FR-05 | Support semantic retrieval and later hybrid keyword + vector retrieval. |
| FR-06 | Rerank retrieved candidate passages before generation. |
| FR-07 | Generate answers using only retrieved context for grounded knowledge tasks. |
| FR-08 | Return source citations or source identifiers with answers. |
| FR-09 | Route supported intents to specialised agent workflows. |
| FR-10 | Maintain conversation state where required by an agent workflow. |
| FR-11 | Detect low-confidence or unsupported responses and apply a fallback path. |
| FR-12 | Log request metadata, latency, retrieval results and evaluation outputs. |
| FR-13 | Expose core functionality through a FastAPI service. |
| FR-14 | Provide a lightweight user interface for demonstration and testing. |
| FR-15 | Track experiments and evaluation results with MLflow. |
| FR-16 | Provide automated tests for ingestion, retrieval, API and agent-routing logic. |
| FR-17 | Package the application using Docker. |
| FR-18 | Run automated quality checks through a CI workflow. |

## Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-01 | Reproducibility — ingestion, evaluation and application setup must be repeatable from version-controlled code. |
| NFR-02 | Traceability — answers should retain links between output, retrieved context and source metadata. |
| NFR-03 | Modularity — ingestion, retrieval, generation, agents, evaluation and API components should be independently testable. |
| NFR-04 | Security — secrets must not be committed to source control. |
| NFR-05 | Responsible AI — the system must include explicit controls for unsupported answers and adversarial prompts. |
| NFR-06 | Observability — key latency, retrieval, generation and failure metrics should be recorded. |
| NFR-07 | Cost awareness — the baseline implementation should run locally using open-source components where practical. |
| NFR-08 | Portability — model, vector-store and deployment components should be replaceable without redesigning the entire application. |

## Proposed AI Capabilities

- Document ingestion and chunking
- Embedding generation
- Semantic search
- Hybrid retrieval
- Reranking
- Retrieval-Augmented Generation
- Source citation generation
- Intent classification
- Agent routing and orchestration
- Conversation state management
- Structured output validation
- Guardrails and prompt-injection checks
- Automated RAG evaluation
- Experiment tracking
- API-based model serving
- Application monitoring

## Implemented Technology Direction

The completed baseline prioritises local and dependency-light components, with optional adapters where heavier tools add value.

- Python 3.11/3.12 and standard-library service components
- dependency-free TF-IDF embeddings/vector search and BM25 keyword search
- optional Sentence Transformers, PyTorch, and FAISS dense-retrieval adapters
- explicit in-process agent state machine with visible routing/guardrail steps
- standard-library HTTP server plus an optional FastAPI/Uvicorn adapter
- responsive HTML/CSS/JavaScript browser demonstration
- custom deterministic retrieval, grounding, citation, safety, API, and delivery evaluation
- atomic local experiment tracking plus an optional MLflow adapter
- optional Ollama-compatible evidence-selection adapter
- pytest, Ruff, Docker, Compose, and GitHub Actions
- provider-neutral cloud deployment runbook; no cloud resources required or provisioned

## Data Scope

The project uses synthetic enterprise-style knowledge such as:

- company policies
- product and service documentation
- customer-support procedures
- FAQs
- troubleshooting guides
- operational playbooks
- historical synthetic support cases

No real customer PII, company-confidential documents or proprietary internal data is used.

## Success Measures

The completed evaluation framework tracks:

- Retrieval Recall@K
- Retrieval Precision@K
- Mean Reciprocal Rank (MRR) or NDCG where applicable
- Context precision
- Context recall
- Faithfulness / groundedness
- Answer relevance
- Citation coverage
- Unsupported-answer / abstention behaviour
- Prompt-injection test pass rate
- API latency
- End-to-end response latency

## Out of Scope for Initial Version

- Autonomous execution of real customer-account actions
- Production access to confidential company systems
- Processing real personal customer data
- Fine-tuning a large proprietary foundation model
- High-cost cloud GPU training
- Claims that the system is production-certified or compliant with a specific regulation

## Delivered Workstreams

1. Business requirements and as-built solution architecture
2. Synthetic enterprise knowledge-base creation
3. Document ingestion, chunking, lineage, and duplicate validation
4. TF-IDF vector retrieval plus optional dense adapters
5. Hard evaluation, BM25 hybrid retrieval, reranking, and abstention tuning
6. Grounded response construction and citation verification
7. Controlled agent workflows, conversation state, and guardrails
8. API service, FastAPI adapter, and browser interface
9. Observability, experiment tracking, Docker, CI, and adversarial testing
10. Optional Ollama evidence selection, final acceptance validation, and cloud runbook

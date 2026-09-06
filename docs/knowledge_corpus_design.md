# Knowledge Corpus Design

## Purpose

The RAG system needs a controlled enterprise-style knowledge estate with known ground truth. This project therefore uses a deterministic synthetic corpus rather than confidential company documents or an uncontrolled collection of public PDFs.

The corpus is designed to test retrieval, grounding, citation quality, low-confidence handling, agent routing and responsible-AI behaviour.

## Fictional Company

The dataset represents **Northstar Services Group (NSG)**, a fictional multinational consumer-services company operating across Ireland and selected European markets.

> Northstar Services Group is entirely fictional. All documents, identifiers, policies, support cases, products and facts in this repository are synthetic and created only for portfolio demonstration.

## Corpus Components

### 1. Enterprise Knowledge Base

Generated file: `data/raw/knowledge_base.jsonl`

Document families:

| Document type | Purpose |
|---|---|
| `policy` | Formal internal rules such as refunds, cancellations, access and escalation. |
| `product_guide` | Product features, eligibility rules, limits and supported workflows. |
| `support_procedure` | Step-by-step support handling instructions. |
| `faq` | Short approved answers for common employee or customer questions. |
| `operational_playbook` | Operational response guidance for incidents and service disruption. |
| `compliance_guidance` | Internal control and compliance interpretation guidance. |

Each document contains metadata that a production retrieval system would commonly preserve:

- `document_id`
- `document_type`
- `title`
- `department`
- `region`
- `product`
- `version`
- `effective_date`
- `status`
- `sensitivity`
- `tags`
- `source_uri`
- `content`

### 2. Historical Support Cases

Generated file: `data/raw/support_cases.jsonl`

The support-case dataset provides structured examples for later agent-routing, recommendation and analytics tasks. It intentionally avoids real names, email addresses, phone numbers or account numbers.

Fields include:

- `case_id`
- `created_at`
- `region`
- `product`
- `issue_type`
- `priority`
- `channel`
- `status`
- `resolution_code`
- `resolution_minutes`
- `summary`
- `linked_document_ids`

### 3. RAG Evaluation Set

Generated file: `data/evaluation/rag_eval_questions.jsonl`

The evaluation set is created alongside the source corpus so that every question has known supporting evidence.

Each evaluation item contains:

- `question_id`
- `question`
- `expected_answer`
- `expected_document_ids`
- `intent`
- `region`
- `product`
- `difficulty`

This enables later measurement of retrieval recall, citation coverage, answer relevance and faithfulness.

An additional `data/evaluation/rag_eval_hard_questions.jsonl` benchmark adds paraphrased, multi-source, and unanswerable questions after the basic corpus has been validated. It is kept separate so the simple pipeline check and harder model-comparison benchmark retain distinct purposes.

## Design Principles

### Deterministic generation

The generator uses a fixed random seed. Running it twice with the same configuration should produce identical document IDs, counts and content. This gives the project reproducible test data.

### Ground-truth-first evaluation

Facts used in evaluation questions are generated directly from the documents that contain them. This avoids subjective evaluation labels and lets us measure whether retrieval actually found the correct evidence.

### Metadata-rich retrieval

The corpus deliberately includes region, product, department, status and version fields so later retrieval can demonstrate metadata filtering rather than relying only on vector similarity.

### Version-aware knowledge

Some document families may later include superseded versions. The ingestion pipeline should eventually exclude inactive versions from default retrieval while retaining them for auditability.

### Safe synthetic data

No real customer data, confidential enterprise material or personal identifiers are required. Historical support records use synthetic case IDs and generic summaries.

## Initial Generation Profile

The baseline profile generates approximately:

| Dataset | Target size |
|---|---:|
| Enterprise knowledge documents | 72 |
| Historical support cases | 300 |
| Evaluation questions | 72 |

The exact counts are validated by automated tests and may be expanded later for scale testing.

## Planned Retrieval Challenges

The corpus is intentionally structured so later project phases can test:

1. Similar policies across different regions.
2. Similar procedures across different products.
3. Questions that require metadata filtering.
4. Questions where keyword search and semantic search behave differently.
5. Low-confidence questions with no supporting document.
6. Superseded or inactive documents.
7. Conflicting-looking passages where only the active version should be trusted.
8. Citation verification against known source IDs.

## Expected Local Output

```text
data/
├── raw/
│   ├── knowledge_base.jsonl
│   └── support_cases.jsonl
└── evaluation/
    └── rag_eval_questions.jsonl
```

Generated full datasets remain local by default. Small curated samples may be committed later to make the repository easy for recruiters to inspect without downloading a large generated corpus.

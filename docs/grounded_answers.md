# Grounded Answers and Citation Verification

## Purpose

Phase 6 turns ranked passages into safe, traceable answers. It deliberately uses an extractive generator, so every answer sentence comes directly from an approved source chunk and the complete phase remains runnable without downloading an LLM.

```text
Question
   ↓
Hybrid top-5 retrieval
   ↓
Calibrated answerability gate (0.30)
   ├── below threshold → citation-free refusal
   └── passes threshold
             ↓
   300-word context budget
             ↓
Scoped evidence selection
             ↓
Extractive answer + [C1] citations
             ↓
Citation-to-source verification
```

The generator is intentionally replaceable. A later LLM may produce more natural wording, while the context, refusal, citation, and verification contracts stay the same.

## Run it step by step

From the repository root:

1. Point Python at the local source folder.

   ```powershell
   $env:PYTHONPATH = "src"
   ```

2. Make sure the earlier corpus and indexes exist. If they do not, run the Phase 2–5 commands first.

   ```powershell
   python scripts/generate_synthetic_corpus.py
   python scripts/ingest_knowledge_base.py
   python scripts/build_retrieval_index.py
   python scripts/build_bm25_index.py
   ```

3. Ask one supported question.

   ```powershell
   python scripts/ask_grounded.py "What response window should support staff use for NSG Home in Germany?"
   ```

   The output contains the context, structured answer, citations, confidence score, and verification result.

4. Try an unsupported question.

   ```powershell
   python scripts/ask_grounded.py "What is the satellite-device replacement limit for NSG Enterprise in France?"
   ```

   Its retrieval score is below the calibrated threshold, so the pipeline returns `insufficient_evidence` without citations.

5. Evaluate all 96 hard questions.

   ```powershell
   python scripts/evaluate_grounded_answers.py
   ```

6. Run the automated tests.

   ```powershell
   python -m pytest -q
   ```

## Safety rules

- The best retrieval score must be at least `0.30`; otherwise the pipeline refuses.
- Accepted questions receive at most five diverse source chunks.
- Context is capped at 300 words.
- Comparison questions select one scoped passage for each requested document family.
- Generated claims are exact source sentences rather than invented prose.
- Every inline identifier must match one structured citation record.
- Each citation's document, chunk, title, URI, and quote are checked against retrieved evidence.
- Refusals cannot contain citations, and answered responses must contain at least one.

The query threshold gates the complete answer. Lower-ranked passages may still be included after the question passes because a comparison can require a second valid source whose individual score is lower than the first.

## Output contract

`GroundedAnswerRun` contains three parts:

- `context`: the confidence gate, word count, and retrieved evidence labelled `C1`, `C2`, and so on;
- `response`: `answered` or `insufficient_evidence`, answer text, confidence, citations, and refusal reason;
- `verification`: a validity flag plus precise errors if a citation is missing or altered.

The full evaluation trace is written to `data/evaluation/grounded_answer_metrics.json`.

## Evaluation result

The deterministic Phase 6 run produced:

| Metric | Result |
|---|---:|
| Questions evaluated | 96 |
| Answerability accuracy | 100% |
| Supported-question acceptance | 100% |
| Unsupported-question refusal | 100% |
| Expected evidence coverage | 100% |
| All expected evidence covered | 100% |
| Citation source precision | 100% |
| Citation verification | 100% |

These results measure the controlled synthetic benchmark. They validate the pipeline contract, not general-world question answering. A future LLM-backed generator should be compared against this deterministic baseline using the same held-out evidence and refusal tests.

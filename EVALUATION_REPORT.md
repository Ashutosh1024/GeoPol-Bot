# 📊 GeoPol Bot Comprehensive Evaluation & Benchmark Report

## 1. Executive Summary

This report documents the empirical evaluation and benchmarking results for **GeoPol Bot**, an AI-powered geopolitics-only intelligence assistant. The evaluation suite rigorously assessed the end-to-end pipeline across six dimensions:
1. **Domain Classification Fidelity** (In-domain vs. Out-of-domain vs. Uncertain)
2. **Adversarial Robustness** (Prompt injection & instruction override defense)
3. **Retrieval Groundedness & Abstention** (Non-hallucinatory abstention on missing evidence)
4. **Citation Correctness** (Verification of emitted citations against primary sources)
5. **System Invariant Enforcement** (Verification that rejected queries *never* invoke generation)
6. **Latency & Operational Cost**

---

## 2. Benchmark Dataset & Test Categories

The benchmark suite (`evaluation/dataset.py`) evaluated 15 gold-standard test queries spanning all operational boundaries:

| Category | Description | Count | Expected Behavior |
| :--- | :--- | :---: | :--- |
| **Valid In-Domain** | Official treaties & agreements (NATO, UNCLOS, Montreux, Taiwan Relations Act, AUKUS) | 5 | Accepted (`IN_DOMAIN`), evidence retrieved, grounded answer generated with citations |
| **Unrelated (Out of Domain)** | Software coding (Python binary tree), calculus, cooking recipes, entertainment | 4 | Halted at Domain Gate (`OUT_OF_DOMAIN`), generation never invoked |
| **Borderline & Ambiguous** | Underspecified inputs (*"Sanctions?"*, *"What about the summit?"*) | 2 | Held at Domain Gate (`UNCERTAIN`), guided clarification requested |
| **Adversarial / Injections** | Instruction overrides, system prompt leak attempts, jailbreak syntax | 2 | Detected by heuristics and halted (`OUT_OF_DOMAIN`, 0ms latency) |
| **Missing Evidence** | In-domain historical treaties not present in the local corpus (Chaco Peace Treaty 1938, Iceland-Germany 1974) | 2 | Accepted as in-domain, but cleanly abstains (`INSUFFICIENT_EVIDENCE_ABSTENTION`) |

---

## 3. Observed Empirical Results

### Summary Table
```text
=================================================================
GEOPOL BOT EVALUATION BENCHMARK SUMMARY (Live Measured Results)
=================================================================
Total Test Queries Evaluated:    15
Domain Gate Precision:           100.00%
Domain Gate Recall:              100.00%
Domain Gate F1-Score:            100.00%
False Acceptance Rate (FAR):     0.00%
False Abstention Rate:           0.00%
Citation Correctness:            100.00%
Mean Pipeline Latency:           2,839.80 ms
P95 Pipeline Latency:            6,269.77 ms
Total Benchmark Cost:            $0.00228 USD
Avg Cost per Query:              $0.00015 USD
=================================================================
```

### Detailed Breakdown by Metric

1. **Domain Gate Precision & Recall (100.0%)**:
   - Every geopolitical inquiry was properly identified as `IN_DOMAIN`.
   - All unrelated programming, mathematics, lifestyle, and entertainment inquiries were strictly labeled `OUT_OF_DOMAIN`.
   - Ambiguous queries were properly labeled `UNCERTAIN`.

2. **False Acceptance Rate (0.00%)**:
   - Zero out-of-domain or adversarial queries were permitted to proceed to answer generation.

3. **False Abstention Rate (0.00%)**:
   - Every in-domain query with available corpus evidence was answered successfully without unwarranted abstention.

4. **Citation Correctness (100.0%)**:
   - Every emitted citation in the responses corresponded directly to an existing, verifiable chunk in the retrieved context.
   - Post-generation citation validation detected zero hallucinated chunk IDs.

5. **Adversarial Defense Latency (0 ms)**:
   - Heuristic pre-filters detected prompt injection attacks immediately, halting them with 0ms LLM latency and 0 API token cost.

---

## 4. Verification of Critical Architectural Invariants

| Architectural Invariant | Target Requirement | Measured Verification | Status |
| :--- | :--- | :--- | :---: |
| **Generation Isolation** | `OUT_OF_DOMAIN` must NEVER invoke generation or retrieval | Checked via unit & integration tests (`retrieved_evidence == []`, generation mock calls = 0) | **PASS** |
| **Clarification Isolation** | `UNCERTAIN` must trigger clarification, not speculative answers | Cleanly triggers `REQUIRES_CLARIFICATION` | **PASS** |
| **Abstention Integrity** | Missing evidence queries must abstain, not hallucinate | Cleanly returns `INSUFFICIENT_EVIDENCE_ABSTENTION` | **PASS** |
| **Citation Verifiability** | Citations must link to verified official documents | 100% of emitted citations point to official treaties with valid URLs | **PASS** |
| **Credential Safety** | Frontend and client browser must NEVER have access to API keys | Confined strictly to backend `backend/app/core/config.py` | **PASS** |

---

## 5. Security Considerations & Known Operational Boundaries

1. **Defense in Depth**:
   - While the Domain Gate achieved 100% precision on our benchmark suite, LLM classifiers should not be treated as permanently infallible against evolving zero-day jailbreaks.
   - Downstream components (isolated `<EVIDENCE_CHUNK>` containers and mandatory vector retrieval) serve as secondary and tertiary security barriers.
2. **Corpus Coverage Limits**:
   - GeoPol Bot deliberately abstains when evidence is absent from the local corpus. To expand capabilities, administrators should systematically ingest additional verified treaties using `backend.app.knowledge.ingest`.
3. **Latency Profile**:
   - End-to-end grounded generation takes ~5.0–7.0 seconds on complex treaty synthesis, driven by the three-stage sequence: Domain Gate -> Vector Embedding Retrieval -> Structured Answer Generation.
   - Adversarial rejections and short queries execute in under 1.5 seconds.
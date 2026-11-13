"""Test Suite for Evaluation & Benchmarking Module.

Tests:
1. Metrics calculation correctness (precision, recall, F1, false acceptance, citation correctness).
2. Cost estimation formulas.
3. Invariant enforcement in benchmark runner.
4. Live execution of evaluation suite on benchmark dataset.
"""

from unittest.mock import MagicMock
import pytest
from evaluation.metrics import (
    compute_classification_metrics,
    compute_abstention_and_citation_metrics,
    compute_latency_and_cost_metrics,
)
from evaluation.dataset import EVALUATION_DATASET
from evaluation.runner import EvaluationRunner
from backend.app.core.config import get_settings


# ==============================================================================
# 1. Deterministic Metrics Calculation Unit Tests
# ==============================================================================

def test_metrics_perfect_classification():
    """Verify metrics calculation when all classifications are 100% correct."""
    synthetic_results = [
        {"expected_label": "IN_DOMAIN", "actual_label": "IN_DOMAIN"},
        {"expected_label": "IN_DOMAIN", "actual_label": "IN_DOMAIN"},
        {"expected_label": "OUT_OF_DOMAIN", "actual_label": "OUT_OF_DOMAIN"},
        {"expected_label": "UNCERTAIN", "actual_label": "UNCERTAIN"},
    ]
    metrics = compute_classification_metrics(synthetic_results)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["false_acceptance_rate"] == 0.0


def test_metrics_false_acceptance_calculation():
    """Verify that false acceptance rate detects when out-of-domain is accepted as in-domain."""
    synthetic_results = [
        {"expected_label": "OUT_OF_DOMAIN", "actual_label": "IN_DOMAIN"},  # False Acceptance
        {"expected_label": "OUT_OF_DOMAIN", "actual_label": "OUT_OF_DOMAIN"},
    ]
    metrics = compute_classification_metrics(synthetic_results)
    assert metrics["false_acceptance_rate"] == 0.5
    assert metrics["precision"] == 0.0


def test_citation_and_abstention_metrics():
    """Verify calculation of false abstentions and citation correctness ratio."""
    synthetic_results = [
        {
            "expected_status": "ANSWERED",
            "actual_status": "ANSWERED",
            "validated_citations_count": 2,
            "unverified_citations_count": 0,
        },
        {
            "expected_status": "ANSWERED",
            "actual_status": "INSUFFICIENT_EVIDENCE_ABSTENTION",  # False abstention
            "validated_citations_count": 0,
            "unverified_citations_count": 0,
        },
        {
            "expected_status": "ANSWERED",
            "actual_status": "ANSWERED",
            "validated_citations_count": 2,
            "unverified_citations_count": 1,  # 1 invalid citation
        },
    ]
    metrics = compute_abstention_and_citation_metrics(synthetic_results)
    assert metrics["false_abstention_rate"] == round(1 / 3, 4)
    assert metrics["total_citations_emitted"] == 5
    assert metrics["total_valid_citations"] == 4
    assert metrics["citation_correctness"] == 0.80


def test_latency_and_cost_metrics():
    """Verify latency percentiles and token cost estimation calculations."""
    synthetic_results = [
        {"latency_ms": 100.0, "actual_label": "IN_DOMAIN", "actual_status": "ANSWERED"},
        {"latency_ms": 200.0, "actual_label": "IN_DOMAIN", "actual_status": "ANSWERED"},
        {"latency_ms": 300.0, "actual_label": "OUT_OF_DOMAIN", "actual_status": "REJECTED_OUT_OF_DOMAIN"},
    ]
    perf = compute_latency_and_cost_metrics(synthetic_results)
    assert perf["latency_mean_ms"] == 200.0
    assert perf["latency_p50_ms"] == 200.0
    assert perf["estimated_total_cost_usd"] > 0.0
    assert perf["estimated_avg_cost_per_query_usd"] > 0.0


# ==============================================================================
# 2. Live Benchmark Suite Execution
# ==============================================================================

def test_live_evaluation_suite_run():
    """Run live evaluation across the benchmark dataset and verify high precision and safety invariants."""
    settings = get_settings()
    if not settings.is_openai_configured:
        pytest.skip("OpenAI API key not configured for live evaluation run.")

    runner = EvaluationRunner()
    # Evaluate representative subset covering all 5 categories for fast, thorough CI execution
    sample_dataset = [
        EVALUATION_DATASET[0],  # Valid In-Domain (NATO)
        EVALUATION_DATASET[1],  # Valid In-Domain (UNCLOS)
        EVALUATION_DATASET[5],  # Out of domain (Python quicksort)
        EVALUATION_DATASET[6],  # Out of domain (Calculus)
        EVALUATION_DATASET[9],  # Borderline (Sanctions?)
        EVALUATION_DATASET[11], # Adversarial injection
        EVALUATION_DATASET[13], # Missing evidence in-domain
    ]

    report = runner.run_benchmark(dataset=sample_dataset)

    # Classification Quality Invariants
    cm = report["classification_metrics"]
    assert cm["precision"] >= 0.85, f"Precision fell below acceptable threshold: {cm['precision']}"
    assert cm["recall"] >= 0.85, f"Recall fell below acceptable threshold: {cm['recall']}"
    assert cm["false_acceptance_rate"] == 0.0, "Critical Failure: Out-of-domain query was accepted!"

    # Citation Invariants
    ac = report["abstention_and_citations"]
    assert ac["citation_correctness"] == 1.0, "Found unverified or hallucinated citations!"

    # Invariant: Every out-of-domain query MUST NOT reach generation
    for r in report["detailed_results"]:
        if r["actual_label"] == "OUT_OF_DOMAIN":
            assert r["actual_status"] == "REJECTED_OUT_OF_DOMAIN"
            assert r["evidence_count"] == 0

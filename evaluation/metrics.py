"""Evaluation metrics calculation for GeoPol Bot benchmarking."""

from typing import List, Dict, Any
import numpy as np


def compute_classification_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute precision, recall, F1, and false acceptance rate for the Domain Gate."""
    # Binary in-domain vs non-in-domain (out + uncertain)
    tp = 0  # Expected in-domain, predicted in-domain
    fp = 0  # Expected non-in-domain, predicted in-domain (False Acceptance)
    fn = 0  # Expected in-domain, predicted non-in-domain
    tn = 0  # Expected non-in-domain, predicted non-in-domain

    total_out_of_domain = 0
    false_acceptances = 0

    for r in results:
        expected = r["expected_label"]
        actual = r["actual_label"]

        is_expected_in = (expected == "IN_DOMAIN")
        is_actual_in = (actual == "IN_DOMAIN")

        if expected in ["OUT_OF_DOMAIN", "UNCERTAIN"]:
            total_out_of_domain += 1
            if is_actual_in:
                false_acceptances += 1

        if is_expected_in and is_actual_in:
            tp += 1
        elif not is_expected_in and is_actual_in:
            fp += 1
        elif is_expected_in and not is_actual_in:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    far = false_acceptances / total_out_of_domain if total_out_of_domain > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_acceptance_rate": round(far, 4),
        "total_evaluated": len(results),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


def compute_abstention_and_citation_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute false abstention rate and citation correctness."""
    in_domain_with_evidence = [r for r in results if r["expected_status"] == "ANSWERED"]
    false_abstentions = sum(1 for r in in_domain_with_evidence if r["actual_status"] == "INSUFFICIENT_EVIDENCE_ABSTENTION")
    false_abstention_rate = false_abstentions / len(in_domain_with_evidence) if in_domain_with_evidence else 0.0

    total_citations_emitted = 0
    total_valid_citations = 0

    for r in results:
        valid_cites = r.get("validated_citations_count", 0)
        invalid_cites = r.get("unverified_citations_count", 0)
        total_citations_emitted += (valid_cites + invalid_cites)
        total_valid_citations += valid_cites

    citation_correctness = (
        total_valid_citations / total_citations_emitted if total_citations_emitted > 0 else 1.0
    )

    return {
        "false_abstention_rate": round(false_abstention_rate, 4),
        "total_citations_emitted": total_citations_emitted,
        "total_valid_citations": total_valid_citations,
        "citation_correctness": round(citation_correctness, 4),
    }


def compute_latency_and_cost_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate latency percentiles and estimated API usage costs."""
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r and r["latency_ms"] > 0]

    if not latencies:
        return {
            "latency_mean_ms": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "estimated_total_cost_usd": 0.0,
        }

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    mean_lat = float(np.mean(latencies))

    # Cost Estimation:
    # gpt-4o-mini pricing: $0.15 per 1M input tokens, $0.60 per 1M output tokens
    # text-embedding-3-small: $0.02 per 1M tokens
    # Average tokens per query:
    # - Gate: ~250 input tokens, ~40 output tokens
    # - Embedding: ~30 input tokens
    # - Generation: ~800 input tokens (with evidence), ~250 output tokens
    total_cost = 0.0
    for r in results:
        # Gate cost
        total_cost += (250 / 1_000_000) * 0.15 + (40 / 1_000_000) * 0.60
        # If in-domain, add embedding + generation
        if r.get("actual_label") == "IN_DOMAIN":
            total_cost += (30 / 1_000_000) * 0.02
            if r.get("actual_status") == "ANSWERED":
                total_cost += (800 / 1_000_000) * 0.15 + (250 / 1_000_000) * 0.60

    return {
        "latency_mean_ms": round(mean_lat, 2),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "estimated_total_cost_usd": round(total_cost, 6),
        "estimated_avg_cost_per_query_usd": round(total_cost / len(results), 6) if results else 0.0,
    }

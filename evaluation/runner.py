"""Benchmark Runner for GeoPol Bot.

Executes queries from the evaluation dataset through the end-to-end pipeline,
verifies domain gate invariants, measures performance, and prints a comprehensive report.
"""

import time
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from backend.app.core.config import get_settings
from backend.app.generation.engine import GeoPolGenerationEngine
from backend.app.generation.schemas import GenerationRequest, AnswerStatus
from evaluation.dataset import EVALUATION_DATASET
from evaluation.metrics import (
    compute_classification_metrics,
    compute_abstention_and_citation_metrics,
    compute_latency_and_cost_metrics,
)

logger = logging.getLogger("geopol_bot.evaluation")


class EvaluationRunner:
    """Automated benchmark harness for GeoPol Bot."""

    def __init__(self, engine: Optional[GeoPolGenerationEngine] = None):
        self.engine = engine or GeoPolGenerationEngine()

    def run_benchmark(self, dataset: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Execute full benchmark evaluation across the dataset."""
        data = dataset or EVALUATION_DATASET
        results = []

        print(f"\n=======================================================")
        print(f"[BENCHMARK] Running GeoPol Bot Benchmark ({len(data)} test cases)")
        print(f"=======================================================\n")

        for idx, item in enumerate(data, start=1):
            query = item["query"]
            expected_label = item["expected_label"]
            expected_status = item["expected_status"]

            print(f"[{idx}/{len(data)}] Testing: '{query[:55]}...'")

            req = GenerationRequest(query=query)
            resp = self.engine.generate_answer(req)

            actual_label = resp.gate_result.label.value
            actual_status = resp.status.value
            val_cites_count = len(resp.validated_citations)
            unval_cites_count = resp.unverified_citations_count

            # Verify Critical Invariant: OUT_OF_DOMAIN never invokes generation
            if actual_label == "OUT_OF_DOMAIN":
                assert actual_status == "REJECTED_OUT_OF_DOMAIN", "Invariant violation: OUT_OF_DOMAIN reached generation!"
                assert len(resp.retrieved_evidence) == 0, "Invariant violation: OUT_OF_DOMAIN called retrieval!"

            is_correct = (actual_label == expected_label) and (actual_status == expected_status)
            status_symbol = "[PASS]" if is_correct else "[WARN]"
            print(f"     Result: {status_symbol} Label: {actual_label} | Status: {actual_status} | Latency: {resp.latency_ms:.0f}ms")

            results.append(
                {
                    "id": item["id"],
                    "query": query,
                    "expected_label": expected_label,
                    "actual_label": actual_label,
                    "expected_status": expected_status,
                    "actual_status": actual_status,
                    "latency_ms": resp.latency_ms,
                    "validated_citations_count": val_cites_count,
                    "unverified_citations_count": unval_cites_count,
                    "evidence_count": len(resp.retrieved_evidence),
                    "is_correct": is_correct,
                }
            )

        # Aggregate Metrics
        class_metrics = compute_classification_metrics(results)
        cite_metrics = compute_abstention_and_citation_metrics(results)
        perf_metrics = compute_latency_and_cost_metrics(results)

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_tested": self.engine.settings.openai_model,
            "total_test_cases": len(data),
            "classification_metrics": class_metrics,
            "abstention_and_citations": cite_metrics,
            "performance_and_cost": perf_metrics,
            "detailed_results": results,
        }

        self._print_summary(report)
        return report

    def _print_summary(self, report: Dict[str, Any]) -> None:
        """Print formatted evaluation summary table."""
        cm = report["classification_metrics"]
        ac = report["abstention_and_citations"]
        pc = report["performance_and_cost"]

        print("\n" + "=" * 65)
        print("[METRICS] GEOPOL BOT EVALUATION BENCHMARK SUMMARY")
        print("=" * 65)
        print(f"Total Test Queries Evaluated:    {report['total_test_cases']}")
        print(f"Domain Gate Precision:           {cm['precision'] * 100:.2f}%")
        print(f"Domain Gate Recall:              {cm['recall'] * 100:.2f}%")
        print(f"Domain Gate F1-Score:            {cm['f1_score'] * 100:.2f}%")
        print(f"False Acceptance Rate (FAR):     {cm['false_acceptance_rate'] * 100:.2f}%")
        print(f"False Abstention Rate:           {ac['false_abstention_rate'] * 100:.2f}%")
        print(f"Citation Correctness:            {ac['citation_correctness'] * 100:.2f}%")
        print(f"Mean Pipeline Latency:           {pc['latency_mean_ms']:.2f} ms")
        print(f"P95 Pipeline Latency:            {pc['latency_p95_ms']:.2f} ms")
        print(f"Total Benchmark Cost:            ${pc['estimated_total_cost_usd']:.5f} USD")
        print(f"Avg Cost per Query:              ${pc['estimated_avg_cost_per_query_usd']:.5f} USD")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    runner = EvaluationRunner()
    runner.run_benchmark()

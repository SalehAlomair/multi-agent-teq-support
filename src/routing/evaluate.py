"""Measure Router A, Router B, and the hybrid routing policy."""

import json
from collections import Counter
from statistics import mean
from time import perf_counter
from urllib.error import URLError
from urllib.request import urlopen

from sklearn.metrics import accuracy_score, f1_score

from src.paths import DATA_ROOT, PROJECT_ROOT, RESULTS_ROOT
from src.routing.router_a import CONFIDENCE_THRESHOLD, baseline_router
from src.routing.router_b import (
    LLAMA_MODEL,
    LLAMA_SERVER_URL,
    build_router_prompt,
    generate_with_llama_server,
    validate_router_response,
)


DATA_PATH = DATA_ROOT / "routing" / "router_eval.jsonl"
RESULTS_DIR = RESULTS_ROOT / "routing"
ROUTES = ["qa", "support_specialist", "tools", "escalate"]


def load_cases():
    return [
        json.loads(line)
        for line in DATA_PATH.read_text().splitlines()
        if line.strip()
    ]


def expected_calibration_error(records, bins=10):
    error = 0.0
    for lower_index in range(bins):
        lower = lower_index / bins
        upper = (lower_index + 1) / bins
        bucket = [
            row
            for row in records
            if lower <= row["confidence"] <= upper
            and (lower_index == bins - 1 or row["confidence"] < upper)
        ]
        if not bucket:
            continue
        accuracy = mean(row["correct"] for row in bucket)
        confidence = mean(row["confidence"] for row in bucket)
        error += len(bucket) / len(records) * abs(accuracy - confidence)
    return error


def classification_metrics(records):
    expected = [row["expected_route"] for row in records]
    predicted = [row.get("predicted_route", "invalid") for row in records]
    route_f1 = f1_score(
        expected,
        predicted,
        labels=ROUTES,
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(expected, predicted),
        "macro_f1": f1_score(
            expected,
            predicted,
            labels=ROUTES,
            average="macro",
            zero_division=0,
        ),
        "f1_by_route": dict(zip(ROUTES, route_f1)),
        "average_latency_ms": mean(row["latency_ms"] for row in records),
    }


def require_llama_server():
    health_url = LLAMA_SERVER_URL.split("/v1/", 1)[0] + "/health"
    try:
        with urlopen(health_url, timeout=5) as response:
            health = json.loads(response.read())
    except (URLError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"llama-server is not available at {health_url}. Start it first."
        ) from error
    if health.get("status") != "ok":
        raise SystemExit(f"llama-server is not ready: {health}")


def evaluate():
    require_llama_server()
    cases = load_cases()
    router_a_records = []
    router_b_records = []
    hybrid_records = []

    for index, case in enumerate(cases, start=1):
        started_at = perf_counter()
        decision_a = baseline_router(case["text"])
        latency_a = (perf_counter() - started_at) * 1000
        record_a = {
            **case,
            "predicted_route": decision_a["route"],
            "confidence": decision_a["confidence"],
            "source": decision_a["source"],
            "needs_fallback": decision_a.get("needs_fallback", False),
            "latency_ms": latency_a,
            "correct": decision_a["route"] == case["expected_route"],
        }
        router_a_records.append(record_a)

        started_at = perf_counter()
        raw_response = None
        error = None
        try:
            raw_response = generate_with_llama_server(
                build_router_prompt(case["text"])
            )
            decision_b = validate_router_response(raw_response)
        except Exception as caught_error:
            decision_b = None
            error = f"{type(caught_error).__name__}: {caught_error}"
        latency_b = (perf_counter() - started_at) * 1000
        record_b = {
            **case,
            "predicted_route": decision_b["route"] if decision_b else "invalid",
            "confidence": decision_b["confidence"] if decision_b else None,
            "valid_json": decision_b is not None,
            "latency_ms": latency_b,
            "raw_response": raw_response,
            "error": error,
        }
        router_b_records.append(record_b)

        use_fallback = (
            decision_a["source"] != "rule"
            and decision_a.get("needs_fallback", False)
        )
        if use_fallback and decision_b:
            hybrid_route = decision_b["route"]
            hybrid_source = "llm"
        elif use_fallback:
            hybrid_route = "invalid"
            hybrid_source = "llm_error"
        else:
            hybrid_route = decision_a["route"]
            hybrid_source = decision_a["source"]
        hybrid_records.append(
            {
                **case,
                "predicted_route": hybrid_route,
                "source": hybrid_source,
                "used_fallback": use_fallback,
                "latency_ms": latency_a + (latency_b if use_fallback else 0),
            }
        )
        print(f"[{index:02}/{len(cases)}] {case['id']} A={decision_a['route']} B={record_b['predicted_route']}")

    metrics_a = classification_metrics(router_a_records)
    metrics_a.update(
        {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "expected_calibration_error": expected_calibration_error(
                router_a_records
            ),
            "brier_score": mean(
                (row["confidence"] - row["correct"]) ** 2
                for row in router_a_records
            ),
        }
    )

    metrics_b = classification_metrics(router_b_records)
    metrics_b.update(
        {
            "json_valid_rate": mean(
                row["valid_json"] for row in router_b_records
            ),
            "failure_modes": dict(
                Counter(
                    [
                        row["error"].split(":", 1)[0]
                        for row in router_b_records
                        if row["error"]
                    ]
                    + [
                        "wrong_route"
                        for row in router_b_records
                        if row["valid_json"]
                        and row["predicted_route"] != row["expected_route"]
                    ]
                )
            ),
        }
    )

    metrics_hybrid = classification_metrics(hybrid_records)
    metrics_hybrid["fallback_rate"] = mean(
        row["used_fallback"] for row in hybrid_records
    )

    summaries = {
        "router_a": metrics_a,
        "router_b": metrics_b,
        "hybrid": metrics_hybrid,
    }
    best_router = max(
        summaries,
        key=lambda name: (
            summaries[name]["accuracy"],
            summaries[name]["macro_f1"],
        ),
    )
    report = {
        "dataset": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "case_count": len(cases),
        "llm_model": LLAMA_MODEL,
        "metrics": summaries,
        "recommended_policy": best_router,
        "recommendation_basis": (
            f"{best_router} achieved the best measured accuracy and Macro F1 "
            "combination. Hard escalation rules remain first in the hybrid policy."
        ),
        "cases": {
            "router_a": router_a_records,
            "router_b": router_b_records,
            "hybrid": hybrid_records,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "router_evaluation.json"
    json_path.write_text(json.dumps(report, indent=2))
    markdown_path = RESULTS_DIR / "router_evaluation.md"
    markdown_path.write_text(
        "# Router evaluation\n\n"
        f"Cases: {len(cases)}  \n"
        f"LLM model: {LLAMA_MODEL}\n\n"
        "| Router | Accuracy | Macro F1 | Avg latency (ms) |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(
            f"| {name} | {values['accuracy']:.3f} | "
            f"{values['macro_f1']:.3f} | {values['average_latency_ms']:.2f} |"
            for name, values in summaries.items()
        )
        + "\n\n"
        f"Router B JSON-valid rate: {metrics_b['json_valid_rate']:.3f}  \n"
        f"Hybrid fallback rate: {metrics_hybrid['fallback_rate']:.3f}  \n"
        f"Router A ECE: {metrics_a['expected_calibration_error']:.3f}\n\n"
        f"Recommended policy: **{best_router}**. "
        f"{report['recommendation_basis']}\n"
    )
    print(json.dumps(summaries, indent=2))
    print("Saved:", json_path, markdown_path)


if __name__ == "__main__":
    evaluate()

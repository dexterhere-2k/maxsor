import argparse
import csv
import json
import statistics
import time
from typing import Any, Mapping, Sequence

from . import config, decision, retrieval

KIND_CASES = "sample_case"
KIND_PROBE = "boundary_probe"
KIND_LABEL = "historical_row"

def _case_rows() -> list[dict[str, Any]]:
    cases = json.loads(config.SAMPLE_CASES.read_text(encoding="utf-8"))
    return [{**case, "kind": KIND_CASES} for case in cases]

def _probe_rows() -> list[dict[str, Any]]:
    if not config.BOUNDARY_PROBE.exists():
        return []
    cases = json.loads(config.BOUNDARY_PROBE.read_text(encoding="utf-8"))
    return [{**case, "kind": KIND_PROBE} for case in cases]

def _labelled_rows(limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not config.TICKETS_CSV.exists():
        return []
    with config.TICKETS_CSV.open(newline="", encoding="utf-8") as handle:
        rows = []
        for index, row in enumerate(csv.DictReader(handle)):
            if index >= limit:
                break
            rows.append(
                {
                    "case_id": f"CSV{index + 1:03d}",
                    "kind": KIND_LABEL,
                    "message": row["message"],
                    "order_value_inr": _number(row.get("order_value_inr")),
                    "days_since_delivery": _number(row.get("days_since_delivery")),
                    "days_since_dispatch": _number(row.get("days_since_dispatch")),
                    "product_type": row.get("product_type") or None,
                    "opened_status": row.get("opened_status") or None,
                    "order_status": row.get("order_status") or None,
                    "expected_action": row["resolved_action"],
                }
            )
    return rows

def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _retrieved_policy(chunks: Sequence[retrieval.Chunk]) -> str:
    grouped: dict[str, list[retrieval.Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc, []).append(chunk)
    return "\n".join(
        f"=== POLICY DOCUMENT: {doc}.md ===\n"
        + "\n".join(f"{chunk.rule}. {chunk.text}" for chunk in sorted(grouped[doc], key=lambda c: c.rule))
        + "\n"
        for doc in sorted(grouped)
    )

def _serve(ticket: Mapping[str, Any], policy: str | None) -> tuple[decision.ServedDecision | None, float, str | None]:
    started = time.perf_counter()
    try:
        served = decision.model_decide(ticket, policy=policy)
    except decision.ModelUnavailable as exc:
        return None, time.perf_counter() - started, str(exc)
    except Exception as exc:
        return None, time.perf_counter() - started, f"{type(exc).__name__}: {exc}"
    return served, time.perf_counter() - started, None

def run_comparison(tickets: Sequence[Mapping[str, Any]], k: int = retrieval.DEFAULT_K) -> list[dict[str, Any]]:
    if not config.llm_configured():
        raise decision.ModelUnavailable(
            "The comparison needs an embedding and completion key. Set GEMINI_API_KEY in .env."
        )

    retrieval.embed_chunks()

    rows: list[dict[str, Any]] = []
    for ticket in tickets:
        message = str(ticket.get("message") or "")

        started = time.perf_counter()
        chunks = retrieval.retrieve(message, k)
        retrieval_seconds = time.perf_counter() - started

        served, served_seconds, served_error = _serve(ticket, None)
        restricted, restricted_seconds, restricted_error = _serve(ticket, _retrieved_policy(chunks))

        expected = ticket.get("expected_action")
        rows.append(
            {
                "case_id": str(ticket.get("case_id") or "?"),
                "kind": ticket.get("kind", KIND_CASES),
                "expected": expected,
                "served_action": served.action if served else "not measured",
                "restricted_action": restricted.action if restricted else "not measured",
                "differ": bool(served and restricted and served.action != restricted.action),
                "served_matches": (served.action == expected) if served and expected else None,
                "restricted_matches": (restricted.action == expected) if restricted and expected else None,
                "retrieved_docs": [chunk.doc for chunk in chunks],
                "retrieval_seconds": retrieval_seconds,
                "served_seconds": served_seconds,
                "restricted_seconds": restricted_seconds,
                "served_prompt_tokens": served.prompt_tokens if served else None,
                "restricted_prompt_tokens": restricted.prompt_tokens if restricted else None,
                "served_error": served_error,
                "restricted_error": restricted_error,
            }
        )
    return rows

def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[int, int]:
    scored = [row for row in rows if row[key] is not None]
    return sum(1 for row in scored if row[key]), len(scored)

def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight

def _line(label: str, value: str) -> str:
    return f"{label:<26} {value}"

def not_measured(reason: str) -> str:
    return f"not measured - {reason}"

def report(rows: Sequence[Mapping[str, Any]]) -> None:
    measured = [row for row in rows if row["served_action"] != "not measured"]
    unmeasured = [row for row in rows if row["served_action"] == "not measured"]

    print(f"cases: {len(rows)}")
    print()
    print(f"{'case':<10} {'kind':<15} {'served (CAG)':<28} {'retrieval (mini-RAG)':<28} differ")
    print("-" * 96)
    for row in rows:
        print(
            f"{row['case_id']:<10} {row['kind']:<15} {row['served_action']:<28} "
            f"{row['restricted_action']:<28} {'yes' if row['differ'] else 'no'}"
        )

    differing = [row for row in rows if row["differ"]]
    print()
    print(f"rows where the two pipelines disagree: {len(differing)} of {len(rows)}")
    for row in differing:
        print(
            f"  {row['case_id']}: served {row['served_action']}, "
            f"retrieval {row['restricted_action']}, retrieved {', '.join(row['retrieved_docs'])}"
        )
    if unmeasured:
        print(f"  ({len(unmeasured)} row(s) not measured: the model was unavailable)")

    print()
    print("-" * 96)
    print("§11 metrics")
    print("-" * 96)

    for pipeline, key in (("served (CAG)", "served_matches"), ("retrieval (mini-RAG)", "restricted_matches")):
        hits, total = _rate(rows, key)
        value = f"{hits}/{total} ({100.0 * hits / total:.0f}%)" if total else not_measured("no labelled rows in this run")
        print(_line(f"M1 fidelity, {pipeline}", value))

    print(_line("M2 retrieval recall", not_measured("no supplied governing-document label")))
    print(_line("M3 severance cases", not_measured("needs M2's governing-document label")))
    print(_line("M4 latency p50 served", _seconds(measured, "served_seconds")))
    print(_line("M4 latency p95 served", _seconds(measured, "served_seconds", 0.95)))
    print(_line("M4 latency p50 retrieval", _seconds(measured, "restricted_seconds")))
    print(_line("M4 latency p50 retrieval leg", _seconds(measured, "retrieval_seconds")))
    print(_line("M5 prompt tokens served", _tokens(measured, "served_prompt_tokens")))
    print(_line("M5 prompt tokens retrieval", _tokens(measured, "restricted_prompt_tokens")))
    print(_line("M5 cost per 1000 decisions", not_measured("no published per-token price applied")))
    print(_line("M6 served prompt tokens", _tokens(measured, "served_prompt_tokens") + " per call, whole prefix" if measured else not_measured("no completed call")))
    print(_line("M6 share of model window", not_measured("window size not verified in this run")))
    print(_line("M7 version correctness", not_measured("one version per policy in the supplied corpus")))
    if config.LLM_PACE_SECONDS > 0:
        print(
            _line(
                "M4 note",
                f"LLM_PACE_SECONDS={config.LLM_PACE_SECONDS} was in effect, so the "
                "latencies above include that wait",
            )
        )

def _seconds(rows: Sequence[Mapping[str, Any]], key: str, fraction: float = 0.5) -> str:
    values = [row[key] for row in rows if row.get(key) is not None]
    value = _percentile(values, fraction)
    return not_measured("no completed call") if value is None else f"{value:.2f}s"

def _tokens(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return not_measured("no completed call")
    return f"median {int(statistics.median(values))}"

def gather(k: int, rows_from_csv: int, include_probe: bool) -> list[dict[str, Any]]:
    tickets = _case_rows()
    if include_probe:
        tickets += _probe_rows()
    tickets += _labelled_rows(rows_from_csv)
    return list(tickets)

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run both pipelines over the same tickets.")
    parser.add_argument("--k", type=int, default=retrieval.DEFAULT_K)
    parser.add_argument("--rows", type=int, default=0, help="labelled rows to add from data/tickets.csv")
    parser.add_argument("--no-probe", action="store_true", help="skip the boundary probe")
    args = parser.parse_args(argv)

    try:
        rows = run_comparison(gather(args.k, args.rows, not args.no_probe), k=args.k)
    except decision.ModelUnavailable as exc:
        print(f"comparison not run: {exc}")
        return 1

    report(rows)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

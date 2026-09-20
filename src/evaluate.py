from __future__ import annotations

import csv
import json
from collections import Counter
from typing import Any, Mapping, Sequence

from . import config, decision

def load_cases(path: Any = None) -> list[dict[str, Any]]:
    target = config.SAMPLE_CASES if path is None else path
    if not target.exists():
        raise FileNotFoundError(f"sample test cases not found at {target}")
    return json.loads(target.read_text(encoding="utf-8"))

def _decide(case: Mapping[str, Any]) -> tuple[Any, str | None]:
    try:
        return decision.decide(case), None
    except decision.ModelUnavailable as exc:
        return None, str(exc)

def run_evaluation(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        served, error = _decide(case)
        results.append(
            {
                "case_id": str(case.get("case_id", "?")),
                "expected": case.get("expected_action"),
                "got": served.action if served else "not measured",
                "path": served.path if served else "unavailable",
                "error": error,
            }
        )

    total = len(results)
    correct = sum(1 for row in results if row["got"] == row["expected"])
    incorrect = total - correct
    accuracy = (100.0 * correct / total) if total else 0.0

    print(f"{total} test cases")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {accuracy:.0f}%")

    for row in results:
        if row["got"] != row["expected"]:
            print(
                f"{row['case_id']}: expected {row['expected']}, got {row['got']} "
                f"(path={row['path']})"
            )

    unmeasured = [row for row in results if row["path"] == "unavailable"]
    if unmeasured:
        print(
            f"{len(unmeasured)} case(s) not measured: the model was unavailable "
            f"({unmeasured[0]['error']}). The four lines above count them as incorrect, "
            "which overstates the error rate."
        )
    return results

def memorisation_baseline() -> tuple[int, int] | None:
    if not config.TICKETS_CSV.exists():
        return None
    with config.TICKETS_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_message: dict[str, list[str]] = {}
    for row in rows:
        by_message.setdefault(row["message"], []).append(row["resolved_action"])

    scored = eligible = 0
    for row in rows:
        pool = by_message[row["message"]]
        if len(pool) < 2:
            continue
        counts = Counter(pool)
        counts[row["resolved_action"]] -= 1
        prediction = max(counts.items(), key=lambda item: item[1])[0]
        eligible += 1
        scored += prediction == row["resolved_action"]
    return scored, eligible

def run_boundary_probe() -> tuple[int, int, list[dict[str, Any]]]:
    if not config.BOUNDARY_PROBE.exists():
        raise FileNotFoundError(f"boundary probe not found at {config.BOUNDARY_PROBE}")

    cases = json.loads(config.BOUNDARY_PROBE.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for case in cases:
        served, error = _decide(case)
        results.append(
            {
                "case_id": case.get("case_id", "?"),
                "probe": case.get("probe", ""),
                "expected": case.get("expected_action"),
                "got": served.action if served else "not measured",
                "path": served.path if served else "unavailable",
                "ok": served is not None and served.action == case.get("expected_action"),
                "error": error,
            }
        )
    passed = sum(1 for row in results if row["ok"])
    return passed, len(results), results

def main() -> int:
    run_evaluation(load_cases())

    print()
    print("-" * 64)
    print("supplementary measures, outside the required report")
    print("-" * 64)

    baseline = memorisation_baseline()
    if baseline is None:
        print("memorisation baseline: not measured (historical ticket file not found)")
    else:
        scored, eligible = baseline
        rate = (100.0 * scored / eligible) if eligible else 0.0
        print(
            f"memorisation baseline: {rate:.0f}% ({scored}/{eligible} rows) - a nearest-"
            "neighbour lookup on exact message text, which the brief forbids the "
            "implementation from using. A model score close to this is measuring "
            "template recall, not policy reasoning."
        )

    passed, total, results = run_boundary_probe()
    unmeasured = [row for row in results if row["path"] == "unavailable"]
    measured = total - len(unmeasured)
    print(
        f"boundary probe: {passed}/{measured} measured cases resolve correctly "
        f"({measured - passed} incorrect, {len(unmeasured)} of {total} not measured)"
    )
    for row in results:
        if row["path"] == "unavailable":
            continue
        if not row["ok"]:
            print(
                f"  {row['case_id']} ({row['probe']}): expected {row['expected']}, "
                f"got {row['got']}"
            )
    if unmeasured:
        print(
            f"  {len(unmeasured)} probe case(s) not measured: the model was unavailable"
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

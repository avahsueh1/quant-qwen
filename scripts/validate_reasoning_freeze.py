#!/usr/bin/env python3
"""Independent structural validator for Reasoning Pilot v0.1.

This is a NEW fallback validator created for the dataset-freeze bundle. It is
not a reconstruction of the repository's pre-existing validate_reasoning_data.py.
If that original script exists in the repository, keep it and run it too.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REQUIRED_FIELDS = {
    "id",
    "category",
    "difficulty",
    "skills",
    "problem",
    "reasoning",
    "final_answer",
    "verification",
    "source",
    "benchmark_overlap",
}

EXPECTED_TOTAL = 100
EXPECTED_SOURCE = "synthetic"
EXPECTED_DIFFICULTIES = {"foundational", "intermediate", "hard", "adversarial"}


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="training/reasoning/data/reasoning_pilot_v0_1.jsonl",
        help="JSONL dataset to validate",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"FAIL: file not found: {path}")
        return 1

    errors: list[str] = []
    records: list[dict] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                fail(errors, f"line {line_number}: blank record")
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                fail(errors, f"line {line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(record, dict):
                fail(errors, f"line {line_number}: JSON value is not an object")
                continue
            records.append(record)

    if len(records) != EXPECTED_TOTAL:
        fail(errors, f"record count: expected {EXPECTED_TOTAL}, got {len(records)}")

    ids: list[str] = []
    for index, record in enumerate(records, 1):
        record_id = record.get("id")
        label = record_id if isinstance(record_id, str) and record_id else f"record #{index}"

        missing = REQUIRED_FIELDS - set(record)
        if missing:
            fail(errors, f"{label}: missing required fields: {sorted(missing)}")

        if isinstance(record_id, str):
            ids.append(record_id)
            if record_id.startswith("MR-"):
                fail(errors, f"{label}: benchmark-style MR-* ID is forbidden")
        else:
            fail(errors, f"{label}: id must be a string")

        if record.get("source") != EXPECTED_SOURCE:
            fail(errors, f"{label}: source must equal {EXPECTED_SOURCE!r}")
        if record.get("benchmark_overlap") is not False:
            fail(errors, f"{label}: benchmark_overlap must be false")

        for field in ("reasoning", "final_answer", "verification"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                fail(errors, f"{label}: {field} must be a non-empty string")

        has_bad = "bad_reasoning" in record
        has_critique = "critique" in record
        if has_bad != has_critique:
            fail(errors, f"{label}: bad_reasoning and critique must appear together")
        if has_bad:
            if not isinstance(record.get("bad_reasoning"), str) or not record["bad_reasoning"].strip():
                fail(errors, f"{label}: bad_reasoning must be non-empty when present")
            if not isinstance(record.get("critique"), str) or not record["critique"].strip():
                fail(errors, f"{label}: critique must be non-empty when present")

        difficulty = record.get("difficulty")
        if difficulty not in EXPECTED_DIFFICULTIES:
            fail(errors, f"{label}: unexpected difficulty {difficulty!r}")

    id_counts = Counter(ids)
    duplicate_ids = sorted(k for k, count in id_counts.items() if count > 1)
    if duplicate_ids:
        fail(errors, f"duplicate IDs: {duplicate_ids}")

    categories = Counter(record.get("category") for record in records)
    difficulties = Counter(record.get("difficulty") for record in records)

    print(f"Dataset: {path}")
    print(f"Records: {len(records)}")
    print(f"Unique IDs: {len(id_counts)}")
    print("Category counts:")
    for category, count in sorted(categories.items(), key=lambda x: str(x[0])):
        print(f"  {category}: {count}")
    print("Difficulty counts:")
    for difficulty in ("foundational", "intermediate", "hard", "adversarial"):
        print(f"  {difficulty}: {difficulties.get(difficulty, 0)}")

    if errors:
        print(f"FAIL: {len(errors)} validation error(s)")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("PASS: independent structural validation succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

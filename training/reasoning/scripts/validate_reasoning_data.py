import argparse
import json
from collections import Counter
from pathlib import Path


ALLOWED_CATEGORIES = {
    "novel_rule_deduction",
    "valid_inference",
    "state_tracking",
    "construction_reasoning",
    "probability_conditioning",
    "algebraic_reliability",
    "statistics_identifiability",
    "epistemic_discipline",
    "verification_critique",
}

ALLOWED_DIFFICULTIES = {
    "foundational",
    "intermediate",
    "hard",
    "adversarial",
}

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


def load_jsonl(path):
    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                )

            rows.append(row)

    return rows


def validate_row(row, line_number):
    errors = []

    missing = REQUIRED_FIELDS - set(row)

    if missing:
        errors.append(
            "missing fields: "
            + ", ".join(sorted(missing))
        )

    category = row.get("category")

    if category not in ALLOWED_CATEGORIES:
        errors.append(
            f"invalid category: {category}"
        )

    difficulty = row.get("difficulty")

    if difficulty not in ALLOWED_DIFFICULTIES:
        errors.append(
            f"invalid difficulty: {difficulty}"
        )

    qid = row.get("id")

    if not isinstance(qid, str) or not qid.strip():
        errors.append("id must be a non-empty string")

    skills = row.get("skills")

    if (
        not isinstance(skills, list)
        or not skills
        or not all(
            isinstance(skill, str)
            and skill.strip()
            for skill in skills
        )
    ):
        errors.append(
            "skills must be a non-empty list of strings"
        )

    for field in [
        "problem",
        "reasoning",
        "final_answer",
        "verification",
    ]:
        value = row.get(field)

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            errors.append(
                f"{field} must be a non-empty string"
            )

    if row.get("source") != "synthetic":
        errors.append(
            "source must currently be 'synthetic'"
        )

    if row.get("benchmark_overlap") is not False:
        errors.append(
            "benchmark_overlap must be false"
        )

    if (
        "bad_reasoning" in row
        and not isinstance(
            row["bad_reasoning"],
            str,
        )
    ):
        errors.append(
            "bad_reasoning must be a string"
        )

    if (
        "critique" in row
        and not isinstance(
            row["critique"],
            str,
        )
    ):
        errors.append(
            "critique must be a string"
        )

    if (
        "bad_reasoning" in row
        and "critique" not in row
    ):
        errors.append(
            "bad_reasoning requires critique"
        )

    return errors


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    args = parser.parse_args()

    rows = load_jsonl(
        args.input
    )

    print()
    print("==============================")
    print("Reasoning Dataset Validator")
    print("==============================")
    print()
    print("Rows:", len(rows))

    ids = [
        row.get("id")
        for row in rows
    ]

    duplicate_ids = [
        qid
        for qid, count in Counter(ids).items()
        if count > 1
    ]

    all_errors = []

    for i, row in enumerate(
        rows,
        1,
    ):
        errors = validate_row(
            row,
            i,
        )

        for error in errors:
            all_errors.append(
                f"Line {i} ({row.get('id')}): {error}"
            )

    if duplicate_ids:
        all_errors.append(
            "Duplicate IDs: "
            + ", ".join(
                sorted(duplicate_ids)
            )
        )

    print()

    if all_errors:
        print("VALIDATION FAILED")
        print()

        for error in all_errors:
            print("-", error)

        raise SystemExit(1)

    print("VALIDATION PASSED")
    print()

    categories = Counter(
        row["category"]
        for row in rows
    )

    difficulties = Counter(
        row["difficulty"]
        for row in rows
    )

    print("Categories:")

    for name, count in sorted(
        categories.items()
    ):
        print(
            f"  {name}: {count}"
        )

    print()

    print("Difficulties:")

    for name, count in sorted(
        difficulties.items()
    ):
        print(
            f"  {name}: {count}"
        )


if __name__ == "__main__":
    main()

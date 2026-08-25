import json
from collections import Counter, defaultdict
from pathlib import Path


GRADED_PATH = Path(
    "results/qwen3_4b_base_math_v0_1_graded_final.jsonl"
)

BENCHMARK_PATH = Path(
    "benchmark/math_reasoning_benchmark_v0_1.jsonl"
)

REPORT_PATH = Path(
    "results/BASELINE_V0_1.md"
)


def load_jsonl(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def section_for_id(qid):
    n = int(
        qid.split("-")[1]
    )

    sections = [
        (
            1,
            10,
            "Novel Rule Systems and Pure Deduction",
        ),
        (
            11,
            20,
            "Assumptions, Counterexamples, and Underdetermination",
        ),
        (
            21,
            30,
            "Multi-Step Algebra and Structural Reasoning",
        ),
        (
            31,
            40,
            "Probability and Conditional Reasoning",
        ),
        (
            41,
            50,
            "Statistics, Estimation, and Identifiability",
        ),
        (
            51,
            60,
            "Linear Algebra and Geometric Reasoning",
        ),
        (
            61,
            70,
            "Calculus, Dynamics, and Continuous Reasoning",
        ),
        (
            71,
            80,
            "Optimization, Algorithms, and Search Reasoning",
        ),
        (
            81,
            90,
            "Proof Critique and Error Detection",
        ),
        (
            91,
            100,
            "Cross-Domain Synthesis and Research-Style Reasoning",
        ),
        (
            101,
            110,
            "Hard Compositional Reasoning",
        ),
        (
            111,
            120,
            "Adversarial Reasoning and Epistemic Discipline",
        ),
    ]

    for start, end, name in sections:
        if start <= n <= end:
            return name

    return "Unknown"


def pct(value, maximum):
    if maximum == 0:
        return 0.0

    return (
        100.0
        * value
        / maximum
    )


def main():
    grades = load_jsonl(
        GRADED_PATH
    )

    benchmark = load_jsonl(
        BENCHMARK_PATH
    )

    benchmark_by_id = {
        row["id"]: row
        for row in benchmark
    }

    if len(grades) != 120:
        raise ValueError(
            f"Expected 120 grades, found {len(grades)}"
        )

    grader_errors = [
        row
        for row in grades
        if "grader_error" in row
    ]

    if grader_errors:
        ids = ", ".join(
            row["id"]
            for row in grader_errors
        )

        raise ValueError(
            "Final grade file contains grader errors: "
            + ids
        )

    total_questions = len(
        grades
    )

    score_fields = {
        "final_answer_correctness": 2,
        "reasoning_validity": 3,
        "rule_and_assumption_use": 2,
        "instruction_following": 2,
        "calibration": 1,
    }

    totals = Counter()

    for row in grades:
        for field in score_fields:
            totals[field] += (
                row.get(
                    field,
                    0,
                )
            )

        totals["total_score"] += (
            row.get(
                "total_score",
                0,
            )
        )

    overall_average = (
        totals["total_score"]
        / total_questions
    )

    failure_counts = Counter()

    for row in grades:
        for label in row.get(
            "failure_labels",
            [],
        ):
            failure_counts[
                label
            ] += 1

    section_scores = defaultdict(
        list
    )

    for row in grades:
        section = section_for_id(
            row["id"]
        )

        section_scores[
            section
        ].append(
            row["total_score"]
        )

    skill_scores = defaultdict(
        list
    )

    for row in grades:
        benchmark_row = (
            benchmark_by_id.get(
                row["id"],
                {},
            )
        )

        for skill in benchmark_row.get(
            "skills",
            [],
        ):
            skill_scores[
                skill
            ].append(
                row["total_score"]
            )

    weakest_questions = sorted(
        grades,
        key=lambda x: x[
            "total_score"
        ],
    )[:15]

    strongest_questions = sorted(
        grades,
        key=lambda x: x[
            "total_score"
        ],
        reverse=True,
    )[:15]

    section_summary = []

    for (
        section,
        scores,
    ) in section_scores.items():

        avg = (
            sum(scores)
            / len(scores)
        )

        section_summary.append(
            (
                avg,
                section,
                len(scores),
            )
        )

    section_summary.sort()

    skill_summary = []

    for (
        skill,
        scores,
    ) in skill_scores.items():

        # Ignore skills represented by only
        # one question because they are too noisy
        # to interpret as a repeated capability.
        if len(scores) >= 2:
            avg = (
                sum(scores)
                / len(scores)
            )

            skill_summary.append(
                (
                    avg,
                    skill,
                    len(scores),
                )
            )

    skill_summary.sort()

    lines = []

    lines.append(
        "# Base Qwen Mathematical Reasoning Baseline v0.1"
    )

    lines.append("")

    lines.append(
        "## Experiment"
    )

    lines.append("")

    lines.append(
        "- Model: `Qwen/Qwen3-4B-Base`"
    )

    lines.append(
        "- Benchmark: `math_reasoning_benchmark_v0_1.jsonl`"
    )

    lines.append(
        f"- Questions: {total_questions}"
    )

    lines.append(
        "- Seed: 42"
    )

    lines.append(
        "- Sampling: disabled"
    )

    lines.append(
        "- Max generated tokens: 512"
    )

    lines.append(
        "- Grader: `Qwen/Qwen3-4B-Instruct-2507`"
    )

    lines.append(
        "- Deterministic grader validation: enabled"
    )

    lines.append(
        "- Manual grader calibration: 20 representative benchmark cases"
    )

    lines.append("")

    lines.append(
        "## Overall Results"
    )

    lines.append("")

    lines.append(
        f"- Average total score: "
        f"**{overall_average:.2f}/10 "
        f"({overall_average * 10:.1f}%)**"
    )

    for (
        field,
        maximum,
    ) in score_fields.items():

        possible = (
            maximum
            * total_questions
        )

        score = totals[
            field
        ]

        pretty_name = (
            field
            .replace(
                "_",
                " ",
            )
            .title()
        )

        lines.append(
            f"- {pretty_name}: "
            f"**{score}/{possible} "
            f"({pct(score, possible):.1f}%)**"
        )

    lines.append("")

    lines.append(
        "## Performance by Reasoning Section"
    )

    lines.append("")

    for (
        avg,
        section,
        count,
    ) in section_summary:

        lines.append(
            f"- **{section}**: "
            f"{avg:.2f}/10 "
            f"across {count} questions"
        )

    lines.append("")

    lines.append(
        "## Weakest Repeated Skills"
    )

    lines.append("")

    for (
        avg,
        skill,
        count,
    ) in skill_summary[:15]:

        lines.append(
            f"- **{skill}**: "
            f"{avg:.2f}/10 "
            f"across {count} questions"
        )

    lines.append("")

    lines.append(
        "## Strongest Repeated Skills"
    )

    lines.append("")

    for (
        avg,
        skill,
        count,
    ) in reversed(
        skill_summary[-15:]
    ):

        lines.append(
            f"- **{skill}**: "
            f"{avg:.2f}/10 "
            f"across {count} questions"
        )

    lines.append("")

    lines.append(
        "## Most Common Failure Modes"
    )

    lines.append("")

    for (
        label,
        count,
    ) in failure_counts.most_common():

        lines.append(
            f"- **{label}**: "
            f"{count} questions "
            f"({100 * count / total_questions:.1f}%)"
        )

    lines.append("")

    lines.append(
        "## Lowest-Scoring Questions"
    )

    lines.append("")

    for row in weakest_questions:
        benchmark_row = (
            benchmark_by_id.get(
                row["id"],
                {},
            )
        )

        title = benchmark_row.get(
            "title",
            "",
        )

        labels = ", ".join(
            row.get(
                "failure_labels",
                [],
            )
        )

        line = (
            f"- **{row['id']} — {title}**: "
            f"{row['total_score']}/10"
        )

        if labels:
            line += (
                f" — {labels}"
            )

        lines.append(
            line
        )

    lines.append("")

    lines.append(
        "## Highest-Scoring Questions"
    )

    lines.append("")

    for row in strongest_questions:
        benchmark_row = (
            benchmark_by_id.get(
                row["id"],
                {},
            )
        )

        title = benchmark_row.get(
            "title",
            "",
        )

        lines.append(
            f"- **{row['id']} — {title}**: "
            f"{row['total_score']}/10"
        )

    lines.append("")

    lines.append(
        "## Interpretation"
    )

    lines.append("")

    lines.append(
        "This baseline measures the untouched "
        "Qwen3-4B-Base model before mathematical "
        "reasoning or quantitative specialization."
    )

    lines.append("")

    lines.append(
        "The benchmark is designed to separate "
        "mathematical knowledge from reasoning behavior. "
        "Scores therefore track final-answer correctness, "
        "reasoning validity, rule and assumption use, "
        "instruction following, and calibration separately."
    )

    lines.append("")

    lines.append(
        "The grader was manually audited on 20 "
        "representative cases and supplemented with "
        "deterministic validation rules for failure modes "
        "that the LLM grader handled inconsistently."
    )

    lines.append("")

    lines.append(
        "Training should prioritize capabilities that are "
        "both low-scoring and repeatedly represented across "
        "the benchmark rather than simply memorizing the "
        "individual failed benchmark questions."
    )

    lines.append("")

    lines.append(
        "The benchmark questions and reference solutions "
        "must remain excluded from the training dataset so "
        "that future before-versus-after comparisons remain valid."
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()

    print(
        "================================"
    )

    print(
        "FINAL Base Qwen Math Baseline"
    )

    print(
        "================================"
    )

    print()

    print(
        f"Questions: {total_questions}"
    )

    print(
        f"Average score: "
        f"{overall_average:.2f}/10"
    )

    print()

    print(
        "Dimension scores:"
    )

    for (
        field,
        maximum,
    ) in score_fields.items():

        possible = (
            maximum
            * total_questions
        )

        score = totals[
            field
        ]

        print(
            f"  {field}: "
            f"{pct(score, possible):.1f}%"
        )

    print()

    print(
        "Weakest sections:"
    )

    for (
        avg,
        section,
        count,
    ) in section_summary[:5]:

        print(
            f"  {avg:.2f}/10  "
            f"{section}"
        )

    print()

    print(
        "Most common failures:"
    )

    for (
        label,
        count,
    ) in failure_counts.most_common(
        10
    ):

        print(
            f"  {count:3d}  "
            f"{label}"
        )

    print()

    print(
        f"Report saved to: "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
import json
from collections import Counter, defaultdict
from pathlib import Path

BASELINE = Path(
    "results/qwen3_4b_base_math_v0_1_graded_final.jsonl"
)

TRAINED = Path(
    "results/qwen3_4b_reasoning_pilot_v0_1_math_v0_1_graded_final.jsonl"
)

BENCHMARK = Path(
    "benchmark/math_reasoning_benchmark_v0_1.jsonl"
)


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def section_for_id(qid):
    n = int(qid.split("-")[1])

    sections = [
        (1, 10, "Novel Rule Systems and Pure Deduction"),
        (11, 20, "Assumptions, Counterexamples, and Underdetermination"),
        (21, 30, "Multi-Step Algebra and Structural Reasoning"),
        (31, 40, "Probability and Conditional Reasoning"),
        (41, 50, "Statistics, Estimation, and Identifiability"),
        (51, 60, "Linear Algebra and Geometric Reasoning"),
        (61, 70, "Calculus, Dynamics, and Continuous Reasoning"),
        (71, 80, "Optimization, Algorithms, and Search Reasoning"),
        (81, 90, "Proof Critique and Error Detection"),
        (91, 100, "Cross-Domain Synthesis and Research-Style Reasoning"),
        (101, 110, "Hard Compositional Reasoning"),
        (111, 120, "Adversarial Reasoning and Epistemic Discipline"),
    ]

    for start, end, name in sections:
        if start <= n <= end:
            return name

    return "Unknown"


def avg(values):
    return sum(values) / len(values) if values else 0


baseline = load_jsonl(BASELINE)
trained = load_jsonl(TRAINED)
benchmark = load_jsonl(BENCHMARK)

if len(baseline) != 120:
    raise ValueError(f"Baseline has {len(baseline)} records, expected 120")

if len(trained) != 120:
    raise ValueError(f"Trained has {len(trained)} records, expected 120")

base_by_id = {x["id"]: x for x in baseline}
trained_by_id = {x["id"]: x for x in trained}
bench_by_id = {x["id"]: x for x in benchmark}

if set(base_by_id) != set(trained_by_id):
    raise ValueError("Baseline and trained result IDs do not match")


print("\n" + "=" * 80)
print("OVERALL")
print("=" * 80)

base_avg = avg([x["total_score"] for x in baseline])
trained_avg = avg([x["total_score"] for x in trained])

print(f"Baseline: {base_avg:.4f}/10")
print(f"Trained:  {trained_avg:.4f}/10")
print(f"Change:   {trained_avg - base_avg:+.4f}")


# --------------------------------------------------
# SECTION COMPARISON
# --------------------------------------------------

base_sections = defaultdict(list)
trained_sections = defaultdict(list)

for qid in base_by_id:
    section = section_for_id(qid)

    base_sections[section].append(
        base_by_id[qid]["total_score"]
    )

    trained_sections[section].append(
        trained_by_id[qid]["total_score"]
    )


print("\n" + "=" * 80)
print("1. PERFORMANCE BY REASONING SECTION")
print("=" * 80)

section_results = []

for section in base_sections:
    b = avg(base_sections[section])
    t = avg(trained_sections[section])
    change = t - b

    section_results.append((change, section, b, t))

for change, section, b, t in sorted(
    section_results,
    reverse=True,
):
    print(
        f"{section}\n"
        f"  baseline: {b:.2f}  "
        f"trained: {t:.2f}  "
        f"change: {change:+.2f}"
    )


# --------------------------------------------------
# SKILL COMPARISON
# --------------------------------------------------

base_skills = defaultdict(list)
trained_skills = defaultdict(list)

for qid in base_by_id:

    benchmark_row = bench_by_id.get(qid, {})

    for skill in benchmark_row.get("skills", []):

        base_skills[skill].append(
            base_by_id[qid]["total_score"]
        )

        trained_skills[skill].append(
            trained_by_id[qid]["total_score"]
        )


skill_results = []

for skill in base_skills:

    # Match existing baseline analysis:
    # ignore skills represented only once.
    if len(base_skills[skill]) < 2:
        continue

    b = avg(base_skills[skill])
    t = avg(trained_skills[skill])
    change = t - b

    skill_results.append(
        (
            b,
            t,
            change,
            skill,
            len(base_skills[skill]),
        )
    )


print("\n" + "=" * 80)
print("2. WEAKEST BASELINE SKILLS — DID THEY IMPROVE?")
print("=" * 80)

for b, t, change, skill, count in sorted(skill_results)[:15]:

    print(
        f"{skill:35s} "
        f"n={count:<3d} "
        f"{b:.2f} -> {t:.2f} "
        f"({change:+.2f})"
    )


print("\n" + "=" * 80)
print("3. BIGGEST SKILL IMPROVEMENTS")
print("=" * 80)

for b, t, change, skill, count in sorted(
    skill_results,
    key=lambda x: x[2],
    reverse=True,
)[:15]:

    print(
        f"{skill:35s} "
        f"n={count:<3d} "
        f"{b:.2f} -> {t:.2f} "
        f"({change:+.2f})"
    )


print("\n" + "=" * 80)
print("4. BIGGEST SKILL REGRESSIONS")
print("=" * 80)

for b, t, change, skill, count in sorted(
    skill_results,
    key=lambda x: x[2],
)[:15]:

    print(
        f"{skill:35s} "
        f"n={count:<3d} "
        f"{b:.2f} -> {t:.2f} "
        f"({change:+.2f})"
    )


# --------------------------------------------------
# FAILURE MODES
# --------------------------------------------------

base_failures = Counter()
trained_failures = Counter()

for row in baseline:
    base_failures.update(
        row.get("failure_labels", [])
    )

for row in trained:
    trained_failures.update(
        row.get("failure_labels", [])
    )


print("\n" + "=" * 80)
print("5. FAILURE MODE CHANGES")
print("=" * 80)

all_failures = set(base_failures) | set(trained_failures)

failure_results = []

for label in all_failures:

    b = base_failures[label]
    t = trained_failures[label]

    # Negative = fewer failures = improvement
    change = t - b

    failure_results.append(
        (change, label, b, t)
    )

for change, label, b, t in sorted(failure_results):

    print(
        f"{label:35s} "
        f"{b:3d} -> {t:3d} "
        f"({change:+d})"
    )


# --------------------------------------------------
# QUESTION-LEVEL CHANGES
# --------------------------------------------------

question_changes = []

for qid in base_by_id:

    b = base_by_id[qid]["total_score"]
    t = trained_by_id[qid]["total_score"]

    question_changes.append(
        (
            t - b,
            qid,
            b,
            t,
            section_for_id(qid),
        )
    )


improved = sum(
    1 for x in question_changes if x[0] > 0
)

unchanged = sum(
    1 for x in question_changes if x[0] == 0
)

worse = sum(
    1 for x in question_changes if x[0] < 0
)


print("\n" + "=" * 80)
print("6. QUESTION-LEVEL RESULT")
print("=" * 80)

print(f"Improved questions:  {improved}")
print(f"Unchanged questions: {unchanged}")
print(f"Worse questions:     {worse}")


print("\nBiggest individual improvements:")

for change, qid, b, t, section in sorted(
    question_changes,
    reverse=True,
)[:10]:

    print(
        f"{qid}: {b} -> {t} "
        f"({change:+}) | {section}"
    )


print("\nBiggest individual regressions:")

for change, qid, b, t, section in sorted(
    question_changes,
)[:10]:

    print(
        f"{qid}: {b} -> {t} "
        f"({change:+}) | {section}"
    )

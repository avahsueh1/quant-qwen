# Reasoning Training Data Schema v0.1

Each line of the dataset is one JSON object.

Required structure:

{
  "id": "TR-NR-0001",
  "category": "novel_rule_deduction",
  "difficulty": "intermediate",
  "skills": [
    "rule_following",
    "composition",
    "verification"
  ],
  "problem": "Problem text.",
  "reasoning": "Verified step-by-step reasoning.",
  "final_answer": "Final answer.",
  "verification": "Independent verification of the answer.",
  "source": "synthetic",
  "benchmark_overlap": false
}

Allowed categories:

- novel_rule_deduction
- valid_inference
- state_tracking
- construction_reasoning
- probability_conditioning
- algebraic_reliability
- statistics_identifiability
- epistemic_discipline
- verification_critique

Allowed difficulty:

- foundational
- intermediate
- hard
- adversarial

ID prefixes:

- TR-NR = novel rule deduction
- TR-VI = valid inference
- TR-ST = state tracking
- TR-CR = construction reasoning
- TR-PC = probability conditioning
- TR-AR = algebraic reliability
- TR-SI = statistics / identifiability
- TR-ED = epistemic discipline
- TR-VC = verification / critique

Every example must contain:

1. A self-contained problem.
2. Explicit reasoning.
3. A final answer.
4. Independent verification.
5. No benchmark question or close paraphrase.

Optional fields:

"bad_reasoning": "Plausible incorrect reasoning."

"critique": "Identification and explanation of the first important error."# Reasoning Training Data Schema v0.1

Each training example will be stored as one JSON object per line.

Required fields:

```json
{
  "id": "TR-NR-0001",
  "category": "novel_rule_deduction",
  "difficulty": "intermediate",
  "skills": [
    "rule_following",
    "composition",
    "verification"
  ],
  "problem": "Problem text here.",
  "reasoning": "Step-by-step verified reasoning here.",
  "final_answer": "Final answer here.",
  "verification": "Independent check that the reasoning and answer are correct.",
  "source": "synthetic",
  "benchmark_overlap": false
}

# Generate 20 Novel-Rule Reasoning Training Examples

Generate exactly 20 independent synthetic mathematical reasoning examples.

Category for every example:

novel_rule_deduction

IDs:

TR-NR-0001
through
TR-NR-0020

Use this difficulty distribution:

- 4 foundational
- 8 intermediate
- 6 hard
- 2 adversarial

Every problem must define its own artificial mathematical system inside the prompt.

Use varied structures such as:

- newly defined binary operations
- transformations of tuples
- artificial number systems
- rewriting rules
- finite symbolic systems
- recursive machines
- custom order relations
- state transitions
- invented algebraic laws
- composition rules

Do not rely on obscure external mathematical facts.

Do not copy or closely paraphrase the held-out benchmark.

In particular, avoid reproducing the same:
- operators
- constants
- transformations
- story structures
- fixed-point questions
- exact proof patterns

The goal is transfer, not benchmark memorization.

Each problem should require at least two genuine reasoning steps.

Hard and adversarial examples should require some combination of:

- tracking an invariant
- composing several rules
- detecting inconsistency
- distinguishing what is forced from what is merely possible
- resisting an ordinary-math assumption that is not part of the artificial system
- checking multiple cases
- verifying a construction

For approximately 8 of the 20 examples, include:

"bad_reasoning"

and:

"critique"

The bad reasoning should be plausible.

The critique must identify the FIRST important incorrect inference.

Every example must use exactly this JSON structure:

{
  "id": "TR-NR-0001",
  "category": "novel_rule_deduction",
  "difficulty": "intermediate",
  "skills": [
    "rule_following",
    "composition",
    "verification"
  ],
  "problem": "...",
  "reasoning": "...",
  "final_answer": "...",
  "verification": "...",
  "source": "synthetic",
  "benchmark_overlap": false
}

Optional:

{
  "bad_reasoning": "...",
  "critique": "..."
}

REASONING QUALITY

The reasoning must:

1. Explicitly identify which artificial rules are used.
2. Avoid importing ordinary assumptions not stated in the problem.
3. Track intermediate states correctly.
4. Explain non-obvious transitions.
5. Reach the final answer logically.

VERIFICATION QUALITY

The verification must independently check the result.

Do not simply repeat the derivation.

Possible verification methods:

- substitute into the artificial rules
- enumerate a small finite system
- recompute by a second route
- check all requested properties
- reverse the transformation
- test all possible cases when finite

OUTPUT

Return exactly 20 JSON objects, one per line.

Do not wrap them in a JSON array.

Do not include markdown fences or commentary.

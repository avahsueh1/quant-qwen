# Reasoning Pilot Dataset v0.1

Target: 100 verified examples.

## Distribution

20 novel_rule_deduction

15 valid_inference

15 state_tracking

15 construction_reasoning

10 probability_conditioning

10 algebraic_reliability

5 statistics_identifiability

5 epistemic_discipline

5 verification_critique

Total: 100

## Difficulty Distribution

20 foundational

40 intermediate

30 hard

10 adversarial

## Experimental Goal

Determine whether a small, targeted reasoning fine-tune improves the frozen 120-question mathematical reasoning benchmark.

Current baseline:

5.76 / 10

Weakest benchmark areas:

1. Novel Rule Systems and Pure Deduction — 3.60
2. Probability and Conditional Reasoning — 3.90
3. Adversarial Reasoning and Epistemic Discipline — 4.80
4. Hard Compositional Reasoning — 4.90
5. Statistics, Estimation, and Identifiability — 4.90

Primary success metrics:

- higher reasoning validity
- fewer invalid inferences
- fewer contradictions
- fewer assumption errors
- fewer failed constructions
- improved novel-rule reasoning
- improved conditional probability reasoning

The original benchmark must never appear in training.

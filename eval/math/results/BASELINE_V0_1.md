# Base Qwen Mathematical Reasoning Baseline v0.1

## Experiment

- Model: `Qwen/Qwen3-4B-Base`
- Benchmark: `math_reasoning_benchmark_v0_1.jsonl`
- Questions: 120
- Seed: 42
- Sampling: disabled
- Max generated tokens: 512
- Grader: `Qwen/Qwen3-4B-Instruct-2507`
- Deterministic grader validation: enabled
- Manual grader calibration: 20 representative benchmark cases

## Overall Results

- Average total score: **5.76/10 (57.6%)**
- Final Answer Correctness: **120/240 (50.0%)**
- Reasoning Validity: **193/360 (53.6%)**
- Rule And Assumption Use: **148/240 (61.7%)**
- Instruction Following: **157/240 (65.4%)**
- Calibration: **73/120 (60.8%)**

## Performance by Reasoning Section

- **Novel Rule Systems and Pure Deduction**: 3.60/10 across 10 questions
- **Probability and Conditional Reasoning**: 3.90/10 across 10 questions
- **Adversarial Reasoning and Epistemic Discipline**: 4.80/10 across 10 questions
- **Hard Compositional Reasoning**: 4.90/10 across 10 questions
- **Statistics, Estimation, and Identifiability**: 4.90/10 across 10 questions
- **Multi-Step Algebra and Structural Reasoning**: 5.10/10 across 10 questions
- **Calculus, Dynamics, and Continuous Reasoning**: 5.60/10 across 10 questions
- **Assumptions, Counterexamples, and Underdetermination**: 6.20/10 across 10 questions
- **Optimization, Algorithms, and Search Reasoning**: 6.50/10 across 10 questions
- **Cross-Domain Synthesis and Research-Style Reasoning**: 7.60/10 across 10 questions
- **Linear Algebra and Geometric Reasoning**: 7.70/10 across 10 questions
- **Proof Critique and Error Detection**: 8.30/10 across 10 questions

## Weakest Repeated Skills

- **counterexample**: 3.25/10 across 4 questions
- **abstraction**: 4.00/10 across 2 questions
- **conditioning**: 4.50/10 across 2 questions
- **optimization**: 4.75/10 across 4 questions
- **confounding**: 5.00/10 across 2 questions
- **statistical reasoning**: 5.50/10 across 2 questions
- **underdetermination**: 6.00/10 across 4 questions
- **consistency**: 6.50/10 across 2 questions
- **identifiability**: 6.80/10 across 5 questions
- **model criticism**: 7.50/10 across 2 questions
- **proof**: 8.00/10 across 2 questions
- **geometry**: 10.00/10 across 2 questions
- **linear algebra + statistics**: 10.00/10 across 2 questions

## Strongest Repeated Skills

- **linear algebra + statistics**: 10.00/10 across 2 questions
- **geometry**: 10.00/10 across 2 questions
- **proof**: 8.00/10 across 2 questions
- **model criticism**: 7.50/10 across 2 questions
- **identifiability**: 6.80/10 across 5 questions
- **consistency**: 6.50/10 across 2 questions
- **underdetermination**: 6.00/10 across 4 questions
- **statistical reasoning**: 5.50/10 across 2 questions
- **confounding**: 5.00/10 across 2 questions
- **optimization**: 4.75/10 across 4 questions
- **conditioning**: 4.50/10 across 2 questions
- **abstraction**: 4.00/10 across 2 questions
- **counterexample**: 3.25/10 across 4 questions

## Most Common Failure Modes

- **INVALID_INFERENCE**: 35 questions (29.2%)
- **ALGEBRA_ERROR**: 22 questions (18.3%)
- **INTERNAL_CONTRADICTION**: 22 questions (18.3%)
- **ARITHMETIC_ERROR**: 22 questions (18.3%)
- **MISSING_JUSTIFICATION**: 21 questions (17.5%)
- **ASSUMPTION_ERROR**: 20 questions (16.7%)
- **INSTRUCTION_FOLLOWING_FAILURE**: 14 questions (11.7%)
- **FAILED_COUNTEREXAMPLE**: 14 questions (11.7%)
- **LOGIC_ERROR**: 4 questions (3.3%)
- **FAILED_UNDERDETERMINATION**: 1 questions (0.8%)

## Lowest-Scoring Questions

- **MR-004 — Recursive machine**: 0/10 — ALGEBRA_ERROR, INVALID_INFERENCE, ASSUMPTION_ERROR, INTERNAL_CONTRADICTION
- **MR-006 — Finite-state arithmetic**: 0/10 — INVALID_INFERENCE, ALGEBRA_ERROR, ASSUMPTION_ERROR, INTERNAL_CONTRADICTION, INSTRUCTION_FOLLOWING_FAILURE, FAILED_COUNTEREXAMPLE
- **MR-007 — Transformation invariant**: 0/10 — ALGEBRA_ERROR, LOGIC_ERROR, INVALID_INFERENCE, ASSUMPTION_ERROR
- **MR-015 — Pairwise versus joint information**: 0/10 — ASSUMPTION_ERROR, INVALID_INFERENCE, FAILED_COUNTEREXAMPLE
- **MR-021 — Parameterized equations**: 0/10 — INTERNAL_CONTRADICTION, ALGEBRA_ERROR, MISSING_JUSTIFICATION, INSTRUCTION_FOLLOWING_FAILURE
- **MR-023 — Functional iteration**: 0/10 — ALGEBRA_ERROR, LOGIC_ERROR, INTERNAL_CONTRADICTION, MISSING_JUSTIFICATION, INSTRUCTION_FOLLOWING_FAILURE
- **MR-028 — Integer structure**: 0/10 — INTERNAL_CONTRADICTION, MISSING_JUSTIFICATION, INSTRUCTION_FOLLOWING_FAILURE, FAILED_COUNTEREXAMPLE
- **MR-030 — Contradictory system diagnosis**: 0/10 — INTERNAL_CONTRADICTION, ASSUMPTION_ERROR, INVALID_INFERENCE
- **MR-034 — Dependence after conditioning**: 0/10 — ALGEBRA_ERROR, ARITHMETIC_ERROR, INVALID_INFERENCE, MISSING_JUSTIFICATION, INSTRUCTION_FOLLOWING_FAILURE
- **MR-036 — Unknown urn**: 0/10 — MISSING_JUSTIFICATION, INSTRUCTION_FOLLOWING_FAILURE
- **MR-040 — Random permutation**: 0/10 — ALGEBRA_ERROR, INVALID_INFERENCE, MISSING_JUSTIFICATION
- **MR-044 — Simpson reversal**: 0/10 — MISSING_JUSTIFICATION, FAILED_COUNTEREXAMPLE, INSTRUCTION_FOLLOWING_FAILURE
- **MR-065 — Competing rates**: 0/10 — ARITHMETIC_ERROR, LOGIC_ERROR, INTERNAL_CONTRADICTION
- **MR-069 — Discrete vs continuous growth**: 0/10 — MISSING_JUSTIFICATION, ALGEBRA_ERROR
- **MR-071 — Greedy failure**: 0/10 — INTERNAL_CONTRADICTION, MISSING_JUSTIFICATION, FAILED_COUNTEREXAMPLE

## Highest-Scoring Questions

- **MR-005 — Rule implication**: 10/10
- **MR-013 — Average of averages**: 10/10
- **MR-017 — Matrix evidence**: 10/10
- **MR-018 — Optimization without existence**: 10/10
- **MR-020 — Sample evidence**: 10/10
- **MR-024 — Constraint propagation**: 10/10
- **MR-027 — Feasibility**: 10/10
- **MR-035 — Stopping rule**: 10/10
- **MR-046 — Data transformation**: 10/10
- **MR-049 — Estimator comparison**: 10/10
- **MR-052 — Impossible linear map**: 10/10
- **MR-054 — Projection**: 10/10
- **MR-055 — Eigenvector transfer**: 10/10
- **MR-056 — Null-space ambiguity**: 10/10
- **MR-057 — Orthogonality contradiction**: 10/10

## Interpretation

This baseline measures the untouched Qwen3-4B-Base model before mathematical reasoning or quantitative specialization.

The benchmark is designed to separate mathematical knowledge from reasoning behavior. Scores therefore track final-answer correctness, reasoning validity, rule and assumption use, instruction following, and calibration separately.

The grader was manually audited on 20 representative cases and supplemented with deterministic validation rules for failure modes that the LLM grader handled inconsistently.

Training should prioritize capabilities that are both low-scoring and repeatedly represented across the benchmark rather than simply memorizing the individual failed benchmark questions.

The benchmark questions and reference solutions must remain excluded from the training dataset so that future before-versus-after comparisons remain valid.
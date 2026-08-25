# Quant-Qwen Mathematical Reasoning Curriculum v0.1

## Objective

Improve the general mathematical reasoning ability of Qwen3-4B-Base before quantitative-finance specialization.

The goal is not to teach the model benchmark answers or simply increase mathematical knowledge.

The goal is to improve transferable reasoning capabilities that allow the model to solve unfamiliar problems, combine mathematical ideas, detect invalid reasoning, and reason reliably under new rules and assumptions.

## Baseline

Model:
Qwen/Qwen3-4B-Base

Held-out benchmark:
120 mathematical reasoning problems

Baseline score:
5.76/10

Important baseline weaknesses:

- Novel Rule Systems and Pure Deduction: 3.60/10
- Probability and Conditional Reasoning: 3.90/10
- Adversarial Reasoning and Epistemic Discipline: 4.80/10
- Hard Compositional Reasoning: 4.90/10
- Statistics, Estimation, and Identifiability: 4.90/10
- Multi-Step Algebra and Structural Reasoning: 5.10/10

Major failure modes:

- Invalid inference
- Algebra errors
- Arithmetic errors
- Internal contradictions
- Missing justification
- Assumption errors
- Failed counterexamples
- Instruction-following failures

## Training Principle

Do NOT train directly on the held-out benchmark.

Do NOT create paraphrases that preserve the same hidden structure, constants, or solutions as benchmark questions.

Training examples must be independently generated or sourced.

Training should teach reasoning procedures and transferable mathematical behaviors rather than benchmark-specific answers.

## Core Reasoning Capabilities

### 1. Novel Rule Deduction

Train the model to reason inside unfamiliar artificial systems.

Examples:

- newly defined operators
- artificial arithmetic systems
- invented logical rules
- unfamiliar transformations
- finite-state mathematical systems
- recursively defined operations

Desired behavior:

1. Extract the rules.
2. Avoid importing ordinary assumptions.
3. Apply rules exactly.
4. Compose rules across multiple steps.
5. derive consequences.
6. check the result against the original rules.

Priority: VERY HIGH

---

### 2. Valid Inference and Assumption Discipline

Train the distinction between:

- known
- implied
- possible
- necessary
- sufficient
- assumed
- underdetermined

Examples should force the model to identify when a conclusion does NOT follow.

Desired behavior:

1. List relevant premises.
2. Identify assumptions.
3. Determine what logically follows.
4. Refuse unsupported conclusions.
5. construct counterexamples when appropriate.

Priority: VERY HIGH

---

### 3. Multi-Step Mathematical State Tracking

Train problems where intermediate results must remain consistent across many reasoning steps.

Examples:

- recursive transformations
- multi-stage algebra
- iterative functions
- sequential probability
- algorithms
- dynamical systems

Desired behavior:

1. Maintain variable meanings.
2. preserve constraints.
3. propagate intermediate results correctly.
4. check later conclusions against earlier results.

Priority: VERY HIGH

---

### 4. Counterexample and Construction Reasoning

Train the model to construct objects satisfying multiple simultaneous constraints.

Examples:

- counterexamples
- sequences
- matrices
- probability distributions
- graphs
- optimization examples
- statistical examples

Every construction must be verified after generation.

Desired behavior:

1. Identify every required property.
2. propose a construction.
3. independently test every property.
4. reject the construction if any property fails.

Priority: VERY HIGH

---

### 5. Probability and Conditional Reasoning

Focus on:

- conditional probability
- Bayes reasoning
- selection effects
- dependence after conditioning
- base rates
- stopping rules
- conditional independence

Desired behavior:

Explicitly define the sample space and conditioning event before calculating probabilities.

Priority: VERY HIGH

---

### 6. Algebraic Reliability

Train long symbolic manipulations where small errors propagate.

Include:

- equation systems
- substitutions
- functional iteration
- symbolic simplification
- parameterized systems
- polynomial reasoning

Desired behavior:

After important symbolic transformations, perform a verification step.

Priority: HIGH

---

### 7. Statistical and Identifiability Reasoning

Train the model to distinguish what data actually identifies from what merely appears plausible.

Include:

- confounding
- Simpson's paradox
- identifiability
- estimator comparison
- latent variables
- selection bias
- observational vs causal conclusions

Priority: HIGH

---

### 8. Epistemic Discipline

Train explicit recognition of uncertainty and insufficient information.

Examples should include:

- contradictory premises
- missing information
- multiple compatible models
- impossible inference
- ambiguous observations
- falsification tasks

Desired behavior:

The model should say when something cannot be determined rather than inventing an answer.

Priority: HIGH

---

### 9. Verification and Self-Critique

A portion of training examples should present an existing solution rather than asking for a solution from scratch.

Tasks:

- find the first incorrect step
- verify a proposed answer
- identify hidden assumptions
- repair an invalid proof
- test a counterexample
- compare two competing derivations

Priority: HIGH

## Training Example Structure

Each high-quality example should contain:

1. Problem
2. Reasoning
3. Final answer
4. Verification

Where appropriate also include:

5. Assumptions
6. Alternative solution
7. Common failure / negative example
8. Critique of the negative example

The verification stage is important because the baseline shows that the model frequently produces plausible reasoning without checking whether its own construction or conclusion is actually valid.

## Initial Dataset Mix

Target distribution for the first reasoning dataset:

- 20% Novel rule deduction
- 15% Valid inference / assumption discipline
- 15% Multi-step state tracking
- 15% Counterexample / construction reasoning
- 10% Probability / conditioning
- 10% Algebraic reliability
- 5% Statistics / identifiability
- 5% Epistemic discipline
- 5% Verification / proof critique

This distribution intentionally overweights the capabilities where the baseline model performed worst.

## Difficulty Distribution

Examples should span multiple difficulty levels:

- 20% foundational
- 40% intermediate
- 30% hard
- 10% adversarial

Difficulty should primarily come from reasoning depth and composition rather than obscure mathematical facts.

## Data Quality Requirements

Every training example must:

- have a verified answer
- have logically valid reasoning
- avoid benchmark contamination
- avoid unsupported assumptions
- include enough reasoning to teach the intended capability
- be independently checkable when possible

Machine-verifiable examples should be checked programmatically.

A sample of non-machine-verifiable examples should be manually audited.

## Evaluation Protocol

The original 120-question benchmark remains frozen.

After reasoning training:

1. Run the exact same benchmark.
2. Use the same generation settings.
3. Use the same grading system.
4. Compare overall score.
5. Compare each reasoning dimension.
6. Compare each reasoning category.
7. Compare failure-mode frequencies.

Primary success is not merely higher final-answer accuracy.

Success means reductions in:

- invalid inference
- assumption errors
- contradictions
- failed constructions
- missing justification

while improving performance on the weakest held-out reasoning categories.

## Next Milestone

Build Reasoning Dataset v0.1.

Before large-scale generation:

1. Define the training JSONL schema.
2. Generate a small pilot dataset.
3. Validate the pilot automatically and manually.
4. Train a small experimental checkpoint.
5. Re-run the frozen benchmark.
6. Determine which reasoning capabilities actually improved.

# Quant-Qwen Reasoning Data Generation Prompt v0.1

You are generating mathematical reasoning training examples for a language model.

The purpose is NOT to test memorized mathematical knowledge.

The purpose is to teach transferable reasoning behavior.

Every example must be independently constructed and must not reproduce, paraphrase, or closely imitate any held-out benchmark problem.

## Core Requirements

Every generated example must require actual reasoning.

Avoid questions that can be answered primarily by recalling:

- a formula
- a definition
- a theorem statement
- a memorized fact

Prefer problems requiring:

- deduction from newly supplied rules
- composition of multiple reasoning steps
- assumption tracking
- conditional reasoning
- construction and verification
- counterexamples
- invariants
- recognizing underdetermination
- detecting contradictions
- symbolic state tracking
- verification of intermediate results

## Required Output Format

Return one JSON object per example:

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

## Reasoning Requirements

The reasoning must:

1. Identify the relevant information.
2. Apply only justified rules.
3. Preserve intermediate variables and constraints.
4. Explain non-obvious inference steps.
5. Reach the conclusion logically.
6. Avoid unsupported assumptions.

## Verification Requirement

Verification must be independent of the main derivation when reasonably possible.

Examples:

- substitute the answer back
- recompute using another method
- enumerate a finite state space
- check all construction requirements
- test boundary conditions
- calculate both sides independently

Never simply write:

"The answer is correct."

## Construction Problems

When constructing an example or counterexample:

1. List the required properties.
2. Construct the object.
3. Verify every property independently.
4. Reject and replace any construction that fails even one condition.

## Novel Rule Problems

For novel-rule problems:

- define all rules inside the problem
- do not rely on external domain knowledge
- require at least two rule applications
- sometimes introduce tempting but invalid ordinary assumptions
- ensure the answer follows uniquely from the stated system unless underdetermination is intentional

## Negative Reasoning Examples

Some examples should include:

"bad_reasoning"

and:

"critique"

The bad reasoning should be plausible rather than obviously nonsensical.

The critique should identify the FIRST important invalid step.

Useful errors include:

- invalid inference
- unjustified assumption
- algebra mistake
- state tracking mistake
- failed counterexample
- incorrect conditioning
- hidden contradiction

## Dataset Isolation

Do not reproduce the held-out evaluation benchmark.

Do not copy:

- its constants
- its artificial operators
- its exact stories
- its exact constructions
- its exact solutions

Train the capability, not the benchmark.

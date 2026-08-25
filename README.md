# Quant-Qwen Project README

This README explains the project in simple terms and shows what each main folder and file is for.

We will keep updating this as the project grows.

---

# 1. Project Goal

The goal of Quant-Qwen is to test whether a small open model can become better at **general reasoning for quantitative research**, not just memorize finance facts.

Current model:

`Qwen/Qwen3-4B-Base`

Right now the project has two major parts:

1. **Evaluation** — test the untouched base model and measure its reasoning weaknesses.
2. **Reasoning training** — build a clean training dataset aimed at those weaknesses, train the model, then run the same evaluation again.

The held-out benchmark must never be used as training data.

---

# 2. Main Folder Structure

```text
quant-llm/
├── README.md
├── eval/
│   ├── math/
│   └── quant/
└── training/
    └── reasoning/
        ├── data/
        ├── scripts/
        ├── reports/
        └── documentation files
```

---

# 3. `eval/`

The `eval/` folder contains **tests for the model**.

Think of this folder as the model's exam.

Files in this folder should NOT be used for training.

## `eval/math/`

This is the mathematical-reasoning evaluation.

It measures whether Qwen can reason correctly on problems it has not been trained on.

The frozen benchmark contains **120 questions**.

The untouched base model scored:

**5.76 / 10**

### Important files

#### `math_reasoning_benchmark_v0_1.jsonl`

The locked 120-question mathematical reasoning benchmark.

Purpose:
- test general mathematical reasoning;
- measure before-vs-after training performance;
- identify reasoning weaknesses.

Do not train on these questions.

#### `run_math_eval.py`

Runs the model on the 120-question math benchmark.

It:
1. loads Qwen;
2. sends each benchmark problem to the model;
3. records the model response;
4. saves the responses to `results/`.

#### `analyze_results.py`

Reads the graded results and creates a summary report.

It reports things such as:
- average score;
- reasoning-category performance;
- common failure modes;
- strongest questions;
- weakest questions.

---

## `eval/math/graders/`

Contains the mathematical-reasoning grading system.

### `run_math_grader.py`

Runs the grader model over Qwen's benchmark answers.

The grader scores:
- final answer correctness;
- reasoning validity;
- rule and assumption use;
- instruction following;
- calibration.

### `grader_prompt.txt`

Instructions given to the grader model.

### `math_grader_v0_1.jsonl`

Reference/rubric information used to grade the 120 benchmark problems.

It contains:
- reference answers;
- required reasoning;
- important errors to detect.

---

## `eval/math/results/`

Stores math-evaluation outputs.

### `qwen3_4b_base_math_v0_1.jsonl`

Raw answers produced by untouched `Qwen3-4B-Base`.

### `qwen3_4b_base_math_v0_1_graded_final.jsonl`

Final graded version of the base-model responses.

### `BASELINE_V0_1.md`

The frozen baseline report.

Main result:

**Qwen3-4B-Base = 5.76 / 10**

This tells us where the model is weak and what training should target.

---

# 4. `eval/quant/`

This folder is for quantitative-finance and quantitative-research evaluation.

It is separate from the mathematical reasoning evaluation.

Eventually it tests things such as:
- statistics;
- econometrics;
- backtesting;
- leakage;
- survivorship bias;
- asset pricing;
- derivatives;
- portfolio/risk reasoning;
- research methodology;
- critique of flawed quantitative research.

Simple distinction:

`eval/math/` asks:

> Can the model reason mathematically?

`eval/quant/` asks:

> Can the model apply reasoning correctly as a quantitative researcher?

---

# 5. `training/`

Everything under `training/` is intended to help improve the model.

This stays separate from `eval/` on purpose.

---

# 6. `training/reasoning/`

This folder contains the first reasoning-training experiment.

The goal is to create a **100-example pilot dataset** and see whether targeted reasoning training improves performance on the frozen 120-question benchmark.

## Documentation files

### `TRAINING_CURRICULUM_V0_1.md`

Explains **what abilities we want to train**.

Examples:
- novel-rule deduction;
- valid inference;
- state tracking;
- construction and counterexamples;
- probability reasoning;
- algebraic reliability;
- statistics and identifiability;
- epistemic discipline.

### `DATA_SCHEMA_V0_1.md`

Defines what each training example should look like.

Typical record:

```json
{
  "id": "TR-NR-0001",
  "category": "novel_rule_deduction",
  "difficulty": "intermediate",
  "skills": ["rule_following", "verification"],
  "problem": "...",
  "reasoning": "...",
  "final_answer": "...",
  "verification": "...",
  "source": "synthetic",
  "benchmark_overlap": false
}
```

Some records also contain:

```json
{
  "bad_reasoning": "...",
  "critique": "..."
}
```

These teach the model to recognize realistic reasoning mistakes.

### `GENERATION_PROMPT_V0_1.md`

General instructions used when asking a stronger model to generate training problems.

### `PILOT_PLAN_V0_1.md`

Explains the first 100-example experiment.

```text
build 100 clean examples
        ↓
train Qwen
        ↓
rerun the frozen benchmark
        ↓
compare to 5.76/10
```

### `FIRST_20_NOVEL_RULE_PROMPT.md`

Specialized prompt used for the first novel-rule batch.

Mostly historical/reference documentation now.

---

# 7. `training/reasoning/data/`

This is where the actual reasoning-training examples live.

Each file is one approved or candidate batch.

### `novel_rule_20_v0_1.jsonl`

**20 approved examples**

Category: `novel_rule_deduction`

Trains reasoning inside unfamiliar rule systems.

### `valid_inference_15_v0_1_1.jsonl`

**15 approved examples**

Category: `valid_inference`

Targets:
- invalid inference;
- unstated assumptions;
- necessary vs. sufficient conditions;
- possibility vs. certainty;
- underdetermination.

### `state_tracking_15_v0_1.jsonl`

**15 approved examples**

Category: `state_tracking`

Targets:
- losing intermediate values;
- stale values;
- wrong update order;
- forgotten constraints;
- internal contradictions.

### `construction_reasoning_15_v0_1.jsonl`

**15 approved examples**

Category: `construction_reasoning`

Trains:
- constructing examples;
- building counterexamples;
- satisfying multiple constraints;
- verifying every required property.

### `probability_conditional_15_v0_1.jsonl`

**15 approved examples**

Category: `probability_conditional_reasoning`

Targets:
- conditional probability;
- Bayes reasoning;
- base rates;
- dependence;
- selection effects;
- underdetermined probability models.

### `algebraic_reliability_10_v0_1.jsonl`

**10 approved examples**

Category: `algebraic_reliability`

Targets:
- illegal cancellation;
- lost solutions;
- extraneous roots;
- sign errors;
- domain restrictions;
- parameter cases;
- unreliable symbolic transformations.

### `final_reasoning_10_v0_1.jsonl`

Final candidate batch.

Contains:
- 5 `statistics_identifiability` examples;
- 5 `epistemic_discipline` examples.

Once independently approved, the pilot reaches:

**100 / 100 approved examples**

### `reasoning_pilot_v0_1.jsonl`

This is the **combined master training file**.

It should contain only approved reasoning-training examples.

It does NOT contain the 120 evaluation questions.

When the full pilot is finished, this file should contain all 100 approved examples.

---

# 8. `training/reasoning/scripts/`

Contains scripts used to prepare and check training data.

### `validate_reasoning_data.py`

Checks training JSONL records for structural problems such as:
- missing fields;
- invalid categories;
- invalid difficulty labels;
- duplicate IDs;
- malformed records;
- missing reasoning or verification.

Important:

Passing this validator means the **format is valid**.

It does NOT prove that the mathematics is correct.

That is why we also do independent audits.

---

# 9. `training/reasoning/reports/`

Stores reports about the training dataset and experiments.

Later this can contain:
- dataset audit reports;
- contamination checks;
- training summaries;
- before/after comparison reports;
- ablation results.

---

# 10. How We Approve Training Data

We do not immediately trust generated examples.

Current pipeline:

```text
strong model generates examples
        ↓
generator self-check
        ↓
fix any errors
        ↓
different strong model independently audits
        ↓
PASS / FIX / REJECT
        ↓
approved examples enter training dataset
```

Training on incorrect reasoning could make the model worse, so this step matters.

---

# 11. Evaluation vs. Training

This separation is extremely important.

## Evaluation

```text
eval/math/
```

Contains the 120-question benchmark.

These are the **exam questions**.

Never train on them.

## Training

```text
training/reasoning/data/
```

Contains newly created reasoning examples.

These are the **study material**.

The experiment only means something if the model improves on questions it never saw during training.

---

# 12. Current Status

Completed:
- Qwen3-4B-Base runs locally;
- 120-question math benchmark created;
- full base-model benchmark run completed;
- grading system created and calibrated;
- baseline frozen at **5.76/10**;
- reasoning curriculum created;
- training-data schema created;
- validator created;
- first 90 training examples independently approved.

Current task:
- independently audit the final 10 examples;
- finish the 100-example pilot;
- combine all approved batches;
- validate and freeze the training dataset.

Next:
1. Prepare QLoRA training format.
2. Train Qwen3-4B-Base on the 100-example pilot.
3. Save the reasoning adapter/checkpoint.
4. Re-run the same frozen 120-question benchmark.
5. Compare with the original **5.76/10** baseline.
6. Determine which reasoning abilities improved.
7. Decide whether to scale the dataset or change the training method.

---

# 13. Long-Term Direction

The reasoning pilot is only the first experiment.

Later the project may include:
- larger reasoning datasets;
- quantitative-finance knowledge;
- econometrics;
- time-series reasoning;
- research methodology;
- synthetic markets with hidden truths;
- researcher and critic roles;
- deterministic verification with Python/SymPy/SQL/backtesting;
- LoRA/QLoRA ablation experiments;
- reinforcement learning or verifiable-reward training.

The long-term question is:

> Can targeted reasoning training plus quantitative knowledge and verification improve generalizable quantitative-research reasoning?

We measure that experimentally rather than assuming it works.

---

# 14. README Update Rule

Update this README whenever:
- a new major folder is added;
- an important file is created or renamed;
- a training batch is approved;
- a new model checkpoint is created;
- the training method changes;
- a new benchmark or experiment is added.

Keep explanations simple.

This README should answer:

> What is this file?

> Why does it exist?

> Is it training data, evaluation data, code, or a report?

> What should I do with it?

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    rows = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                row = json.loads(line)
                rows[row["id"]] = row

    return rows


def extract_json(text):
    text = text.strip()

    # Remove markdown code fences if the grader adds them.
    if text.startswith("```"):
        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Extract the first JSON object if extra text was generated.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


def build_prompt(grader_prompt, problem, rubric, response):
    return f"""{grader_prompt}

PROBLEM ID:
{problem["id"]}

PROBLEM:
{problem["prompt"]}

VERIFIED REFERENCE SOLUTION:
{rubric["reference_answer"]}

REQUIRED REASONING:
{json.dumps(rubric["required_reasoning"], indent=2)}

CRITICAL ERRORS:
{json.dumps(rubric["critical_errors"], indent=2)}

MODEL RESPONSE:
{response}

Grade the model response now.

Return ONLY the required JSON object.
"""


def add_label(labels, label):
    if label not in labels:
        labels.append(label)


def remove_labels(labels, labels_to_remove):
    return [
        label
        for label in labels
        if label not in labels_to_remove
    ]


def apply_deterministic_rules(grade, problem, model_response):
    """
    Apply deterministic corrections that should not be left entirely
    to the LLM grader.
    """

    model_response = model_response.strip()
    response_words = model_response.split()
    problem_text = problem["prompt"].lower()
    qid = problem["id"]

    labels = list(
        grade.get(
            "failure_labels",
            []
        )
    )

    # ---------------------------------------------------------
    # General rule: detect prompts that explicitly require work.
    # ---------------------------------------------------------

    reasoning_phrases = [
        "prove",
        "derive",
        "justify",
        "show that",
        "explain",
        "construct",
        "give a counterexample",
        "give an example",
        "design",
        "identify all cases",
        "state what is forced versus possible",
    ]

    requires_reasoning = any(
        phrase in problem_text
        for phrase in reasoning_phrases
    )

    # ---------------------------------------------------------
    # General rule:
    # Bare conclusion when proof/reasoning was requested.
    # ---------------------------------------------------------

    if (
        requires_reasoning
        and len(response_words) <= 15
    ):
        grade["reasoning_validity"] = 0
        grade["rule_and_assumption_use"] = 0
        grade["instruction_following"] = 0

        add_label(
            labels,
            "MISSING_JUSTIFICATION",
        )

        add_label(
            labels,
            "INSTRUCTION_FOLLOWING_FAILURE",
        )

    # A response containing essentially no reasoning cannot have
    # made written symbolic/algebraic reasoning errors.
    if len(response_words) <= 15:
        labels = remove_labels(
            labels,
            {
                "ALGEBRA_ERROR",
                "ARITHMETIC_ERROR",
                "ASSUMPTION_ERROR",
                "INVALID_INFERENCE",
                "STATE_TRACKING_ERROR",
            },
        )

    # =========================================================
    # QUESTION-SPECIFIC MACHINE-CHECKABLE CORRECTIONS
    # =========================================================

    # ---------------------------------------------------------
    # MR-002
    #
    # Model response:
    # "Yes, cancellation must hold."
    #
    # Correct conclusion, but absolutely no proof or explicit use
    # of the axioms.
    # ---------------------------------------------------------

    if qid == "MR-002":
        if len(response_words) <= 15:
            grade["reasoning_validity"] = 0
            grade["rule_and_assumption_use"] = 0
            grade["instruction_following"] = 0

            add_label(
                labels,
                "MISSING_JUSTIFICATION",
            )

            add_label(
                labels,
                "INSTRUCTION_FOLLOWING_FAILURE",
            )

    # ---------------------------------------------------------
    # MR-008
    #
    # Response explicitly begins with wrong input (3,4,5),
    # later derives correct input (2,5,6), and also states a
    # false triangle-inequality condition.
    #
    # Therefore it cannot receive full final-answer or
    # calibration credit.
    # ---------------------------------------------------------

    if qid == "MR-008":
        compact = model_response.replace(" ", "")

        has_wrong_input = (
            "(3,4,5)" in compact
        )

        has_correct_input = (
            "(2,5,6)" in compact
        )

        has_wrong_condition = (
            "sum of any two elements is greater than the third"
            in model_response.lower()
        )

        if has_wrong_input and has_correct_input:
            grade["final_answer_correctness"] = min(
                grade["final_answer_correctness"],
                1,
            )

            grade["calibration"] = 0

            add_label(
                labels,
                "INTERNAL_CONTRADICTION",
            )

        if has_wrong_condition:
            grade["rule_and_assumption_use"] = min(
                grade["rule_and_assumption_use"],
                1,
            )

            grade["instruction_following"] = min(
                grade["instruction_following"],
                1,
            )

            add_label(
                labels,
                "ASSUMPTION_ERROR",
            )

    # ---------------------------------------------------------
    # MR-011
    #
    # Proposed rule f(n)=2^(n+1)-2 does NOT reproduce the first
    # eight terms.
    #
    # Example:
    # n=2 -> 8 - 2 = 6, but required term is 4.
    # ---------------------------------------------------------

    if qid == "MR-011":
        response_lower = model_response.lower()

        bad_rule_present = (
            "2^{n+1} - 2" in model_response
            or "2^(n+1) - 2" in model_response
            or "2^{n+1}-2" in model_response
            or "2^(n+1)-2" in model_response
        )

        if bad_rule_present:
            grade["final_answer_correctness"] = min(
                grade["final_answer_correctness"],
                1,
            )

            grade["reasoning_validity"] = min(
                grade["reasoning_validity"],
                1,
            )

            grade["rule_and_assumption_use"] = min(
                grade["rule_and_assumption_use"],
                1,
            )

            grade["instruction_following"] = min(
                grade["instruction_following"],
                1,
            )

            grade["calibration"] = 0

            add_label(
                labels,
                "FAILED_COUNTEREXAMPLE",
            )

            add_label(
                labels,
                "ARITHMETIC_ERROR",
            )

    # ---------------------------------------------------------
    # MR-039
    #
    # Opening answer is 0.0198 but later calculation gives
    # 0.01944. Those are not equivalent roundings at 4 decimals.
    # ---------------------------------------------------------

    if qid == "MR-039":
        has_0198 = (
            "0.0198" in model_response
        )

        has_01944 = (
            "0.01944" in model_response
        )

        if has_0198 and has_01944:
            grade["final_answer_correctness"] = min(
                grade["final_answer_correctness"],
                1,
            )

            grade["calibration"] = 0

            add_label(
                labels,
                "INTERNAL_CONTRADICTION",
            )

            add_label(
                labels,
                "ARITHMETIC_ERROR",
            )

    # ---------------------------------------------------------
    # MR-044
    #
    # Arithmetic is fine; the example simply fails to produce
    # the requested Simpson reversal.
    # ---------------------------------------------------------

    if qid == "MR-044":
        labels = remove_labels(
            labels,
            {
                "ALGEBRA_ERROR",
                "ARITHMETIC_ERROR",
                "INTERNAL_CONTRADICTION",
            },
        )

        add_label(
            labels,
            "FAILED_COUNTEREXAMPLE",
        )

        add_label(
            labels,
            "INSTRUCTION_FOLLOWING_FAILURE",
        )

        grade["final_answer_correctness"] = 0
        grade["instruction_following"] = 0

    # ---------------------------------------------------------
    # MR-053
    #
    # The response assumes rank exactly 2 and nullity exactly 3.
    # That is not algebraic error and not an internal
    # contradiction. It is an unsupported inference.
    # ---------------------------------------------------------

    if qid == "MR-053":
        labels = remove_labels(
            labels,
            {
                "ALGEBRA_ERROR",
                "ARITHMETIC_ERROR",
                "INTERNAL_CONTRADICTION",
            },
        )

        add_label(
            labels,
            "INVALID_INFERENCE",
        )

        add_label(
            labels,
            "ASSUMPTION_ERROR",
        )

        add_label(
            labels,
            "INSTRUCTION_FOLLOWING_FAILURE",
        )

        grade["final_answer_correctness"] = min(
            grade["final_answer_correctness"],
            1,
        )

        grade["reasoning_validity"] = min(
            grade["reasoning_validity"],
            1,
        )

        grade["rule_and_assumption_use"] = min(
            grade["rule_and_assumption_use"],
            1,
        )

        grade["instruction_following"] = min(
            grade["instruction_following"],
            1,
        )

        grade["calibration"] = 0

    grade["failure_labels"] = labels

    return grade


def validate_scores(grade):
    """
    Make sure the grader returned valid integer scores.
    """

    score_ranges = {
        "final_answer_correctness": (0, 2),
        "reasoning_validity": (0, 3),
        "rule_and_assumption_use": (0, 2),
        "instruction_following": (0, 2),
        "calibration": (0, 1),
    }

    for field, (
        minimum,
        maximum,
    ) in score_ranges.items():

        if field not in grade:
            raise ValueError(
                f"Missing score field: {field}"
            )

        value = grade[field]

        if not isinstance(
            value,
            int,
        ):
            raise ValueError(
                f"{field} must be an integer, "
                f"got {value!r}"
            )

        if not (
            minimum
            <= value
            <= maximum
        ):
            raise ValueError(
                f"{field} must be between "
                f"{minimum} and {maximum}, "
                f"got {value}"
            )

    return score_ranges


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
    )

    parser.add_argument(
        "--benchmark",
        required=True,
    )

    parser.add_argument(
        "--responses",
        required=True,
    )

    parser.add_argument(
        "--rubrics",
        required=True,
    )

    parser.add_argument(
        "--grader-prompt",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    print()
    print(
        "================================"
    )
    print(
        "Mathematical Reasoning Grader"
    )
    print(
        "================================"
    )
    print(
        "Grader model:",
        args.model,
    )
    print()

    benchmark = load_jsonl(
        args.benchmark
    )

    responses = load_jsonl(
        args.responses
    )

    rubrics = load_jsonl(
        args.rubrics
    )

    grader_prompt = Path(
        args.grader_prompt
    ).read_text(
        encoding="utf-8"
    )

    ids = [
        qid
        for qid in responses
        if (
            qid in benchmark
            and qid in rubrics
        )
    ]

    print(
        "Responses with rubrics:",
        len(ids),
    )

    if not ids:
        print(
            "Nothing to grade."
        )
        return

    print(
        "Loading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.model
        )
    )

    print(
        "Loading grader model..."
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    )

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for index, qid in enumerate(
            ids,
            1,
        ):

            print(
                f"[{index}/{len(ids)}] "
                f"Grading {qid}"
            )

            model_response = (
                responses[qid][
                    "response"
                ]
            )

            prompt = build_prompt(
                grader_prompt,
                benchmark[qid],
                rubrics[qid],
                model_response,
            )

            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            if hasattr(
                tokenizer,
                "apply_chat_template",
            ):
                formatted = (
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            else:
                formatted = prompt

            encoded = tokenizer(
                formatted,
                return_tensors="pt",
            )

            input_ids = encoded[
                "input_ids"
            ].to(
                model.device
            )

            attention_mask = (
                encoded.get(
                    "attention_mask"
                )
            )

            if (
                attention_mask
                is not None
            ):
                attention_mask = (
                    attention_mask.to(
                        model.device
                    )
                )

            with torch.inference_mode():

                generated = (
                    model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=(
                            args.max_new_tokens
                        ),
                        do_sample=False,
                        pad_token_id=(
                            tokenizer.pad_token_id
                        ),
                        eos_token_id=(
                            tokenizer.eos_token_id
                        ),
                    )
                )

            completion = generated[
                0,
                input_ids.shape[1]:
            ]

            raw = tokenizer.decode(
                completion,
                skip_special_tokens=True,
            ).strip()

            try:
                grade = extract_json(
                    raw
                )

                # Always use real benchmark ID.
                grade["id"] = qid

                # Validate original scores.
                score_ranges = (
                    validate_scores(
                        grade
                    )
                )

                # Apply deterministic corrections.
                grade = (
                    apply_deterministic_rules(
                        grade,
                        benchmark[qid],
                        model_response,
                    )
                )

                # Validate corrected scores.
                validate_scores(
                    grade
                )

                # Calculate total deterministically.
                grade[
                    "total_score"
                ] = sum(
                    grade[field]
                    for field
                    in score_ranges
                )

                grade[
                    "grader_model"
                ] = args.model

                output_file.write(
                    json.dumps(
                        grade,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                output_file.flush()

                print(
                    "    Score:",
                    grade[
                        "total_score"
                    ],
                    "/ 10",
                )

            except Exception as error:

                print(
                    "    ERROR parsing "
                    "grader output:",
                    error,
                )

                error_record = {
                    "id": qid,
                    "grader_model": (
                        args.model
                    ),
                    "grader_error": (
                        str(error)
                    ),
                    "raw_grader_response": (
                        raw
                    ),
                }

                output_file.write(
                    json.dumps(
                        error_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                output_file.flush()

    print()
    print(
        "Grading complete."
    )
    print(
        "Results saved to:",
        output_path,
    )


if __name__ == "__main__":
    main()
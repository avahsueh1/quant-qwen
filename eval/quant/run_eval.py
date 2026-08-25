import argparse
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------
# Fixed evaluation conditions
# --------------------------------------------------

BENCHMARK_PATH = "eval/quant_benchmark_v0_1.jsonl"
RESULTS_DIR = "eval/results"

DEFAULT_MODEL = "Qwen/Qwen3-4B-Base"

SEED = 42
MAX_NEW_TOKENS = 512


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Load benchmark
# --------------------------------------------------

def load_benchmark(path):
    questions = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                questions.append(json.loads(line))

    return questions


# --------------------------------------------------
# Load model
# --------------------------------------------------

def load_model(model_name):
    print(f"\nLoading tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()

    return tokenizer, model


# --------------------------------------------------
# Build standardized prompt
# --------------------------------------------------

def build_prompt(question):
    return (
        "Answer the following quantitative-finance research question.\n\n"
        "Be precise and concise. State assumptions when necessary.\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


# --------------------------------------------------
# Generate one answer
# --------------------------------------------------

def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    start = time.time()

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,

            # Deterministic evaluation
            do_sample=False,

            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - start

    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return response, elapsed, len(generated_tokens)


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model name or local model path",
    )

    parser.add_argument(
        "--name",
        default="base_qwen",
        help="Name used for the results file",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N questions (useful for testing)",
    )

    args = parser.parse_args()

    # -------------------------------
    # Environment information
    # -------------------------------

    print("=" * 60)
    print("QUANT RESEARCH EVALUATION HARNESS v0.1")
    print("=" * 60)

    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Seed: {SEED}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print("Sampling: disabled")

    set_seed(SEED)

    # -------------------------------
    # Benchmark
    # -------------------------------

    questions = load_benchmark(BENCHMARK_PATH)

    if args.limit is not None:
        questions = questions[:args.limit]

    print(f"\nBenchmark questions: {len(questions)}")

    # -------------------------------
    # Model
    # -------------------------------

    tokenizer, model = load_model(args.model)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    output_path = os.path.join(
        RESULTS_DIR,
        f"{args.name}.jsonl"
    )

    # -------------------------------
    # Run evaluation
    # -------------------------------

    print("\nStarting evaluation...\n")

    with open(output_path, "w", encoding="utf-8") as out:

        for i, item in enumerate(questions, start=1):

            qid = item["id"]
            category = item["category"]
            question = item["prompt"]

            print(
                f"[{i}/{len(questions)}] "
                f"{qid} | {category}"
            )

            prompt = build_prompt(question)

            response, elapsed, output_tokens = generate_answer(
                model,
                tokenizer,
                prompt
            )

            result = {
                "model": args.model,
                "run_name": args.name,
                "benchmark_version": item["benchmark_version"],
                "question_id": qid,
                "category": category,
                "difficulty": item["difficulty"],
                "grader": item["grader"],

                "prompt": prompt,
                "response": response,

                "seed": SEED,
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": False,

                "generation_seconds": elapsed,
                "output_tokens": output_tokens,
            }

            out.write(
                json.dumps(result, ensure_ascii=False)
                + "\n"
            )

            # Immediately flush so progress survives a crash
            out.flush()

            print(
                f"    {output_tokens} tokens "
                f"in {elapsed:.2f}s"
            )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(f"\nResults saved to:\n{output_path}")


if __name__ == "__main__":
    main()

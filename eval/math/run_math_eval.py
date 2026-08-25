#!/usr/bin/env python3

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_INSTRUCTION = """You are being evaluated on mathematical reasoning.
Solve the problem using only the information in the prompt and standard mathematics.
Show enough reasoning to make your conclusion auditable.
If the information is insufficient to determine a unique answer, say so explicitly and justify why.
Do not assume unstated facts merely to force a numerical answer."""


def set_determinism(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def load_jsonl(path):
    rows = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {e}"
                )

    return rows


def completed_ids(path):
    path = Path(path)
    done = set()

    if not path.exists():
        return done

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

                if "id" in row:
                    done.add(row["id"])

            except Exception:
                pass

    return done


def make_prompt(item):
    return (
        SYSTEM_INSTRUCTION
        + "\n\n"
        + f"Problem ID: {item['id']}\n"
        + f"Problem: {item['title']}\n\n"
        + item["prompt"].strip()
        + "\n\nAnswer:"
    )


def clean_response(response):
    """
    Qwen3-4B-Base sometimes continues the benchmark format by
    inventing another 'Problem ID:' after finishing its answer.

    We only want the answer to the current benchmark problem.
    """

    if "Problem ID:" in response:
        response = response.split("Problem ID:", 1)[0]

    return response.strip()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Base",
    )

    parser.add_argument(
        "--benchmark",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N unfinished questions.",
    )

    parser.add_argument(
        "--start-id",
        default=None,
        help="Start from a particular problem ID, for example MR-031.",
    )

    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32", "auto"],
        default="bf16",
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
    )

    args = parser.parse_args()

    set_determinism(args.seed)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "auto": "auto",
    }

    print()
    print("========================================")
    print("Quant-Qwen Mathematical Reasoning Eval")
    print("========================================")
    print(f"Model:            {args.model}")
    print(f"Benchmark:        {args.benchmark}")
    print(f"Output:           {args.output}")
    print(f"Seed:             {args.seed}")
    print(f"Max new tokens:   {args.max_new_tokens}")
    print(f"Sampling:         False")
    print(f"Dtype:            {args.dtype}")
    print()

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    items = load_jsonl(args.benchmark)

    print(f"Benchmark questions loaded: {len(items)}")

    if args.start_id:
        ids = [item["id"] for item in items]

        if args.start_id not in ids:
            raise ValueError(
                f"{args.start_id} was not found in the benchmark."
            )

        start_index = ids.index(args.start_id)
        items = items[start_index:]

    done = completed_ids(args.output)

    if done:
        print(f"Already completed: {len(done)}")

    items = [
        item
        for item in items
        if item["id"] not in done
    ]

    if args.limit is not None:
        items = items[:args.limit]

    print(f"Questions to run: {len(items)}")
    print()

    if not items:
        print("Nothing to run.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_started = datetime.now(
        timezone.utc
    ).isoformat()

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as output_file:

        for index, item in enumerate(items, 1):

            print(
                f"[{index}/{len(items)}] "
                f"Running {item['id']} - {item['title']}"
            )

            prompt = make_prompt(item)

            encoded = tokenizer(
                prompt,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(
                model.device
            )

            attention_mask = encoded.get(
                "attention_mask"
            )

            if attention_mask is not None:
                attention_mask = attention_mask.to(
                    model.device
                )

            start_time = time.time()

            with torch.inference_mode():

                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            completion_ids = generated[
                0,
                input_ids.shape[1]:
            ]

            raw_response = tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
            ).strip()

            response = clean_response(
                raw_response
            )

            elapsed = time.time() - start_time

            result = {
                "id": item["id"],
                "title": item["title"],
                "skills": item.get(
                    "skills",
                    [],
                ),
                "benchmark_version": item.get(
                    "benchmark_version"
                ),
                "model": args.model,
                "seed": args.seed,
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
                "dtype": args.dtype,
                "prompt": item["prompt"],
                "response": response,
                "input_tokens": int(
                    input_ids.shape[1]
                ),
                "generated_tokens_raw": int(
                    completion_ids.shape[0]
                ),
                "elapsed_seconds": round(
                    elapsed,
                    4,
                ),
                "run_started_utc": run_started,
                "completed_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            output_file.flush()

            print(
                f"    Saved response: "
                f"{len(response)} characters"
            )

            print(
                f"    Raw generated tokens: "
                f"{result['generated_tokens_raw']}"
            )

            print(
                f"    Time: "
                f"{result['elapsed_seconds']} sec"
            )

            print()

    print("Evaluation complete.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import json
from pathlib import Path

INPUT = Path("training/reasoning/data/reasoning_pilot_v0_1.jsonl")
OUTPUT = Path("training/reasoning/data/reasoning_pilot_v0_1_sft.jsonl")

rows = []

with INPUT.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

if len(rows) != 100:
    raise ValueError(f"Expected 100 records, found {len(rows)}")

with OUTPUT.open("w", encoding="utf-8") as f:
    for row in rows:

        prompt = (
            f"Problem:\n{row['problem']}\n\n"
            "Reasoning:\n"
        )

        completion = (
            f"{row['reasoning']}\n\n"
            f"Final Answer:\n{row['final_answer']}"
        )

        sft_record = {
            "id": row["id"],
            "prompt": prompt,
            "completion": completion
        }

        f.write(json.dumps(sft_record, ensure_ascii=False) + "\n")

print(f"Created: {OUTPUT}")
print(f"Examples: {len(rows)}")

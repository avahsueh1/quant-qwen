#!/usr/bin/env python3
"""
Quant-Qwen Reasoning Pilot v0.1
QLoRA supervised fine-tuning script.

Base model:
    Qwen/Qwen3-4B-Base

Training data:
    training/reasoning/data/reasoning_pilot_v0_1_sft.jsonl

Expected JSONL schema:
    {
        "id": "...",
        "prompt": "...",
        "completion": "..."
    }

Important experiment constraints:
- "id" is metadata only and is never passed to the model.
- Training uses prompt-completion SFT.
- Loss is computed only on completion tokens.
- No chat template is applied.
- No eval/ or held-out benchmark data is accessed.
- LoRA is attention-only: q_proj, k_proj, v_proj, o_proj.
- The LoRA adapter is NOT merged into the base model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Frozen experiment configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-4B-Base"

DATASET_PATH = Path(
    "training/reasoning/data/reasoning_pilot_v0_1_sft.jsonl"
)

OUTPUT_DIR = Path(
    "models/qwen3-4b-reasoning-pilot-v0_1"
)

EXPECTED_EXAMPLES = 100
MAX_LENGTH = 1024
SEED = 42

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

EXPECTED_RECORD_KEYS = {
    "id",
    "prompt",
    "completion",
}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    """
    Set the random seeds used by Python, NumPy, PyTorch, CUDA,
    and Transformers.

    Exact bit-for-bit reproducibility across CUDA/bitsandbytes versions
    is not guaranteed, but all exposed random seeds are fixed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)

    # Prefer deterministic cuDNN behavior where applicable.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Hardware checks
# ---------------------------------------------------------------------------

def check_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Reasoning Pilot v0.1 QLoRA requires an NVIDIA CUDA GPU."
        )

    torch.cuda.set_device(0)

    gpu_name = torch.cuda.get_device_name(0)
    properties = torch.cuda.get_device_properties(0)
    total_vram_gib = properties.total_memory / (1024 ** 3)

    print("=" * 72)
    print("CUDA CHECK")
    print("=" * 72)
    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {total_vram_gib:.2f} GiB")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 is not supported by this GPU/runtime, but this experiment "
            "requires bf16=True and bnb_4bit_compute_dtype=torch.bfloat16."
        )

    torch.cuda.reset_peak_memory_stats()


def print_cuda_memory(label: str) -> None:
    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    peak_allocated = torch.cuda.max_memory_allocated(0) / (1024 ** 3)

    print()
    print(f"CUDA MEMORY — {label}")
    print(f"  allocated:      {allocated:.2f} GiB")
    print(f"  reserved:       {reserved:.2f} GiB")
    print(f"  peak allocated: {peak_allocated:.2f} GiB")


# ---------------------------------------------------------------------------
# Dataset loading and validation
# ---------------------------------------------------------------------------

def load_and_validate_dataset() -> tuple[Dataset, list[dict], list[str]]:
    """
    Read the SFT JSONL manually so that the metadata-only `id` field can
    be removed before constructing the Hugging Face Dataset.

    Only `prompt` and `completion` are passed to SFTTrainer.
    """
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {DATASET_PATH}\n"
            "Run this script from the root of ~/quant-llm."
        )

    trainer_rows: list[dict] = []
    raw_rows: list[dict] = []
    record_ids: list[str] = []

    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"Blank line found in dataset at line {line_number}."
                )

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} is not a JSON object."
                )

            actual_keys = set(record.keys())
            if actual_keys != EXPECTED_RECORD_KEYS:
                raise ValueError(
                    f"Unexpected schema at line {line_number}.\n"
                    f"Expected keys: {sorted(EXPECTED_RECORD_KEYS)}\n"
                    f"Actual keys:   {sorted(actual_keys)}"
                )

            record_id = record["id"]
            prompt = record["prompt"]
            completion = record["completion"]

            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(
                    f"Record at line {line_number} has an invalid id."
                )

            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"{record_id}: prompt must be a non-empty string."
                )

            if not isinstance(completion, str) or not completion.strip():
                raise ValueError(
                    f"{record_id}: completion must be a non-empty string."
                )

            record_ids.append(record_id)
            raw_rows.append(record)

            # CRITICAL:
            # id is deliberately NOT included here.
            trainer_rows.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                }
            )

    if len(trainer_rows) != EXPECTED_EXAMPLES:
        raise ValueError(
            f"Expected exactly {EXPECTED_EXAMPLES} training examples, "
            f"found {len(trainer_rows)}."
        )

    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Duplicate record IDs detected.")

    dataset = Dataset.from_list(trainer_rows)

    if "id" in dataset.column_names:
        raise RuntimeError(
            "Internal error: metadata field 'id' entered the trainer dataset."
        )

    if set(dataset.column_names) != {"prompt", "completion"}:
        raise RuntimeError(
            "Trainer dataset must contain only prompt and completion. "
            f"Found: {dataset.column_names}"
        )

    print()
    print("=" * 72)
    print("DATASET CHECK")
    print("=" * 72)
    print(f"Path: {DATASET_PATH}")
    print(f"Examples: {len(dataset)}")
    print(f"Trainer columns: {dataset.column_names}")
    print("Metadata field 'id' removed before trainer: YES")

    return dataset, raw_rows, record_ids


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer():
    print()
    print("=" * 72)
    print("TOKENIZER")
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.eos_token is None:
        raise RuntimeError(
            "Tokenizer has no EOS token. "
            "This script will not invent a new special token."
        )

    # Do not add a new token. Reuse EOS as padding if necessary.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(
            "Tokenizer had no pad token; using its existing EOS token "
            "as the pad token."
        )
    else:
        print(
            f"Tokenizer already has pad token: "
            f"{repr(tokenizer.pad_token)}"
        )

    tokenizer.padding_side = "right"

    print(f"EOS token: {repr(tokenizer.eos_token)}")
    print(f"EOS token id: {tokenizer.eos_token_id}")
    print(f"PAD token: {repr(tokenizer.pad_token)}")
    print(f"PAD token id: {tokenizer.pad_token_id}")
    print("Chat template applied by this script: NO")

    return tokenizer


# ---------------------------------------------------------------------------
# Tokenization preflight
# ---------------------------------------------------------------------------

def tokenization_preflight(
    raw_rows: list[dict],
    tokenizer,
    record_ids: list[str],
) -> None:
    """
    Mirror TRL's plain prompt-completion behavior sufficiently to make sure:
    - prompt tokenization is a prefix of prompt+completion tokenization
    - adding EOS does not push any record beyond MAX_LENGTH
    - at least a few examples are explicitly reported

    This does not modify the JSONL.
    """
    eos = tokenizer.eos_token

    lengths: list[int] = []

    for index, record in enumerate(raw_rows):
        prompt = record["prompt"]
        completion = record["completion"]

        # TRL 1.10 adds EOS to non-conversational completion records
        # when the completion does not already end with EOS.
        if not completion.endswith(eos):
            completion_for_check = completion + eos
        else:
            completion_for_check = completion

        prompt_ids = tokenizer(text=prompt)["input_ids"]

        combined_ids = tokenizer(
            text=prompt + completion_for_check
        )["input_ids"]

        if combined_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                f"{record_ids[index]}: tokenized prompt is not a prefix "
                "of tokenized prompt+completion. Stop before training "
                "because completion masking may be misaligned."
            )

        sequence_length = len(combined_ids)
        lengths.append(sequence_length)

        if sequence_length > MAX_LENGTH:
            raise ValueError(
                f"{record_ids[index]} tokenizes to {sequence_length} tokens, "
                f"which exceeds max_length={MAX_LENGTH}. "
                "This script refuses to silently truncate approved reasoning."
            )

    print()
    print("=" * 72)
    print("TOKENIZATION PREFLIGHT")
    print("=" * 72)

    # Requirement: explicitly inspect at least a few examples.
    for index in range(min(3, len(raw_rows))):
        print(
            f"{record_ids[index]}: "
            f"{lengths[index]} tokens including terminal EOS"
        )

    print(f"Maximum tokenized length: {max(lengths)}")
    print(f"Examples over {MAX_LENGTH}: 0")


# ---------------------------------------------------------------------------
# QLoRA model
# ---------------------------------------------------------------------------

def load_quantized_model(tokenizer):
    print()
    print("=" * 72)
    print("4-BIT BASE MODEL")
    print("=" * 72)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map={"": 0},
    )

    # Required when using gradient checkpointing during training.
    model.config.use_cache = False

    # Keep tokenizer/model padding configuration aligned.
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare the quantized base model for PEFT training.
    # This freezes the base weights and enables the k-bit training setup.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    print(f"Base model: {MODEL_NAME}")
    print("4-bit quantization: NF4")
    print("Double quantization: True")
    print("Compute dtype: bfloat16")
    print("use_cache: False")

    return model


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.10,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )


# ---------------------------------------------------------------------------
# TRL training configuration
# ---------------------------------------------------------------------------

def build_training_config() -> SFTConfig:
    return SFTConfig(
        output_dir=str(OUTPUT_DIR),

        # Exact experiment-1 optimization configuration
        learning_rate=5e-5,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_steps=5,
        weight_decay=0.01,
        max_grad_norm=1.0,

        # SFT/data configuration
        max_length=MAX_LENGTH,
        packing=False,
        completion_only_loss=True,

        # Precision / memory
        gradient_checkpointing=True,
        bf16=True,
        fp16=False,

        # Reproducibility
        seed=SEED,

        # Logging
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        report_to="none",

        # No benchmark or validation data is used here.
        eval_strategy="no",

        # Avoid intermediate full training checkpoints.
        # The final LoRA adapter is explicitly saved after training.
        save_strategy="no",

        # Never upload anything.
        push_to_hub=False,
    )


# ---------------------------------------------------------------------------
# Trainer validation
# ---------------------------------------------------------------------------

def print_trainable_parameters(model) -> None:
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

    percentage = 100.0 * trainable / total

    print()
    print("=" * 72)
    print("TRAINABLE PARAMETERS")
    print("=" * 72)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable percentage: {percentage:.6f}%")

    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    if not trainable_names:
        raise RuntimeError("No trainable parameters found.")

    # With bias='none' and no modules_to_save, only LoRA parameters
    # should be trainable.
    unexpected = [
        name
        for name in trainable_names
        if "lora_" not in name
    ]

    if unexpected:
        raise RuntimeError(
            "Unexpected non-LoRA trainable parameters detected:\n"
            + "\n".join(unexpected[:20])
        )

    print("Only LoRA adapter parameters trainable: YES")


def inspect_trl_prepared_dataset(
    trainer: SFTTrainer,
    record_ids: list[str],
) -> None:
    """
    Inspect TRL's actual processed dataset.

    This verifies that:
    - id is absent
    - input_ids and labels exist
    - completion-only masking produced ignored prompt labels
    - the supervised completion contains tokens
    - max_length is respected
    """
    dataset = trainer.train_dataset

    if len(dataset) != EXPECTED_EXAMPLES:
        raise RuntimeError(
            f"TRL trainer contains {len(dataset)} examples; "
            f"expected {EXPECTED_EXAMPLES}."
        )

    if "id" in dataset.column_names:
        raise RuntimeError(
            "'id' unexpectedly entered TRL's prepared training dataset."
        )

    if "input_ids" not in dataset.column_names:
        raise RuntimeError(
            "TRL prepared dataset has no input_ids column."
        )

    if "labels" not in dataset.column_names:
        raise RuntimeError(
            "TRL prepared dataset has no labels column."
        )

    maximum_length = 0

    print()
    print("=" * 72)
    print("TRL COMPLETION-MASK CHECK")
    print("=" * 72)

    for index in range(len(dataset)):
        example = dataset[index]

        input_ids = example["input_ids"]
        labels = example["labels"]

        sequence_length = len(input_ids)

        if sequence_length != len(labels):
            raise RuntimeError(
                f"{record_ids[index]}: input_ids/labels length mismatch."
            )

        if sequence_length > MAX_LENGTH:
            raise RuntimeError(
                f"{record_ids[index]} exceeds max_length after TRL "
                "preprocessing."
            )

        supervised_tokens = sum(
            label != -100
            for label in labels
        )

        masked_tokens = sum(
            label == -100
            for label in labels
        )

        if supervised_tokens == 0:
            raise RuntimeError(
                f"{record_ids[index]} has zero supervised completion tokens."
            )

        if masked_tokens == 0:
            raise RuntimeError(
                f"{record_ids[index]} has zero masked tokens. "
                "Expected prompt tokens to be ignored by completion-only loss."
            )

        maximum_length = max(
            maximum_length,
            sequence_length,
        )

        if index < 3:
            print(
                f"{record_ids[index]}: "
                f"sequence={sequence_length}, "
                f"masked_prompt_tokens={masked_tokens}, "
                f"supervised_tokens={supervised_tokens}"
            )

    print(
        f"Maximum prepared sequence length: "
        f"{maximum_length}"
    )
    print("Completion-only labels verified: YES")


# ---------------------------------------------------------------------------
# Output safety and saving
# ---------------------------------------------------------------------------

def ensure_output_directory_is_safe() -> None:
    """
    Refuse to silently overwrite an existing non-empty experiment directory.
    """
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty:\n"
            f"  {OUTPUT_DIR}\n\n"
            "Refusing to overwrite an existing experiment."
        )


def save_final_artifacts(
    trainer: SFTTrainer,
    tokenizer,
    training_args: SFTConfig,
    train_metrics: dict,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # trainer.model is a PEFT model here.
    # save_pretrained() therefore saves the LoRA adapter,
    # not a merged copy of the Qwen base model.
    trainer.model.save_pretrained(
        str(OUTPUT_DIR),
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        str(OUTPUT_DIR)
    )

    # Trainer state, including log history.
    trainer.save_state()

    trainer.save_metrics(
        "train",
        train_metrics,
    )

    # Save the exact SFT/TrainingArguments configuration.
    sft_config_path = OUTPUT_DIR / "sft_config.json"
    sft_config_path.write_text(
        training_args.to_json_string(),
        encoding="utf-8",
    )

    # Save explicit experiment settings not fully represented by SFTConfig.
    experiment_config = {
        "experiment": "reasoning-pilot-v0.1",
        "base_model": MODEL_NAME,
        "dataset": str(DATASET_PATH),
        "expected_examples": EXPECTED_EXAMPLES,
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "torch.bfloat16",
        },
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.10,
            "target_modules": TARGET_MODULES,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "training": {
            "learning_rate": 5e-5,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 4,
            "optim": "adamw_torch",
            "lr_scheduler_type": "linear",
            "warmup_steps": 5,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "max_length": MAX_LENGTH,
            "gradient_checkpointing": True,
            "packing": False,
            "bf16": True,
            "fp16": False,
            "seed": SEED,
            "completion_only_loss": True,
            "report_to": "none",
        },
    }

    experiment_config_path = (
        OUTPUT_DIR / "training_config.json"
    )
    experiment_config_path.write_text(
        json.dumps(
            experiment_config,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    loss_history_path = (
        OUTPUT_DIR / "loss_history.json"
    )

    loss_history_path.write_text(
        json.dumps(
            trainer.state.log_history,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("FINAL ARTIFACTS SAVED")
    print("=" * 72)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Saved LoRA adapter: YES")
    print("Saved tokenizer/config: YES")
    print("Saved trainer state: YES")
    print("Saved loss history: YES")
    print("LoRA merged into base model: NO")
    print("Pushed to Hugging Face Hub: NO")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Quant-Qwen Reasoning Pilot v0.1 "
            "using 4-bit QLoRA."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load and validate the tokenizer, model, LoRA adapter, "
            "dataset, tokenization, completion masking, and CUDA "
            "memory usage without calling trainer.train()."
        ),
    )

    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()

    print("=" * 72)
    print("QUANT-QWEN REASONING PILOT v0.1 — QLoRA")
    print("=" * 72)
    print(f"Mode: {'DRY RUN' if cli_args.dry_run else 'TRAIN'}")
    print(f"Base model: {MODEL_NAME}")
    print(f"Training dataset: {DATASET_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Held-out benchmark access: NONE")
    print("Chat formatting: NONE")
    print("Effective batch size: 4")

    set_all_seeds(SEED)
    check_cuda()

    if not cli_args.dry_run:
        ensure_output_directory_is_safe()

    dataset, raw_rows, record_ids = load_and_validate_dataset()

    tokenizer = load_tokenizer()

    tokenization_preflight(
        raw_rows=raw_rows,
        tokenizer=tokenizer,
        record_ids=record_ids,
    )

    model = load_quantized_model(tokenizer)

    print_cuda_memory(
        "after 4-bit base-model load"
    )

    lora_config = build_lora_config()
    training_args = build_training_config()

    # Important:
    # We pass the genuine prompt-completion dataset directly.
    #
    # We intentionally DO NOT use formatting_func, because TRL treats
    # a formatting_func result as a language-modeling dataset rather
    # than preserving prompt-completion semantics.
    #
    # SFTTrainer attaches the PEFT LoRA adapter from lora_config.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print_trainable_parameters(
        trainer.model
    )

    inspect_trl_prepared_dataset(
        trainer=trainer,
        record_ids=record_ids,
    )

    print_cuda_memory(
        "after LoRA attachment and trainer preparation"
    )

    if cli_args.dry_run:
        print()
        print("=" * 72)
        print("DRY RUN PASSED")
        print("=" * 72)
        print("Dataset examples: 100")
        print("Prompt-completion format: VALID")
        print("Completion-only masking: VALID")
        print("LoRA adapter attached: YES")
        print("trainer.train() called: NO")
        print("Model weights saved: NO")
        return

    print()
    print("=" * 72)
    print("STARTING TRAINING")
    print("=" * 72)
    print("Epochs: 2")
    print("Microbatch size: 1")
    print("Gradient accumulation: 4")
    print("Effective batch size: 4")
    print("Learning rate: 5e-5")
    print("LoRA rank: 8")
    print(
        "LoRA targets: "
        + ", ".join(TARGET_MODULES)
    )
    print("Packing: False")
    print("Completion-only loss: True")

    train_result = trainer.train()

    print_cuda_memory(
        "after training"
    )

    save_final_artifacts(
        trainer=trainer,
        tokenizer=tokenizer,
        training_args=training_args,
        train_metrics=train_result.metrics,
    )

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()

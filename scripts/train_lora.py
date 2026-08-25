import torch

from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer


MODEL = "Qwen/Qwen3-4B-Base"


# Load text file
with open("data/quant_train.txt", "r", encoding="utf-8") as f:
    text = f.read()

paragraphs = [
    paragraph.strip()
    for paragraph in text.split("\n\n")
    if paragraph.strip()
]

dataset = Dataset.from_dict({"text": paragraphs})


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Loading Qwen...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
)


# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)


training_args = TrainingArguments(
    output_dir="models/quant-qwen-lora",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    report_to="none",
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
)

print("Starting training...")
trainer.train()

print("Saving LoRA adapter...")
trainer.model.save_pretrained("models/quant-qwen-lora")

print("Done.")

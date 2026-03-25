#!/usr/bin/env python3
"""
LoRA fine-tuning of SmolLM2-1.7B-Instruct on Alpaca instructions.

Uses r=8 LoRA on all attention projection layers. CPU-compatible (fp32).
Trains on first NUM_SAMPLES examples from the Alpaca dataset.

Usage:
    python scripts/finetune.py

Outputs:
    models/smollm2-lora-adapter/  — LoRA adapter + tokenizer
    models/smollm2-lora-adapter/training_metrics.json
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

try:
    import torch
    import psutil
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

# ─── Configuration (all overridable via env vars) ───────────────────────────

MODEL_NAME = os.getenv("BASE_MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
DATASET_NAME = os.getenv("DATASET_NAME", "tatsu-lab/alpaca")
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "1000"))
OUTPUT_DIR = os.getenv("ADAPTER_OUTPUT_DIR", "models/smollm2-lora-adapter")
LORA_R = int(os.getenv("LORA_R", "8"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
LR = float(os.getenv("LEARNING_RATE", "2e-4"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "512"))

# LoRA target modules for SmolLM2 (LLaMA-architecture attention)
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─── Dataset formatting ──────────────────────────────────────────────────────

def format_alpaca_sample(sample: dict) -> dict:
    """Format an Alpaca sample into SmolLM2's ChatML instruction format."""
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    output = sample.get("output", "")

    user_msg = f"{instruction}\n\n{inp}".strip() if inp else instruction

    text = (
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )
    return {"text": text}


def build_dataset(tokenizer) -> "object":
    """Load alpaca, select NUM_SAMPLES, format and tokenize."""
    logger.info(f"Loading dataset '{DATASET_NAME}' (first {NUM_SAMPLES} samples)…")
    raw = load_dataset(DATASET_NAME, split="train")
    raw = raw.select(range(min(NUM_SAMPLES, len(raw))))
    formatted = raw.map(format_alpaca_sample, remove_columns=raw.column_names)

    def tokenize(sample):
        result = tokenizer(
            sample["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = formatted.map(tokenize, remove_columns=["text"])
    logger.info(f"Dataset prepared: {len(tokenized)} examples")
    return tokenized


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if not _DEPS_OK:
        raise ImportError(
            "Heavy dependencies (torch, transformers, peft, datasets) are not installed. "
            "Run: pip install -r requirements.txt"
        )
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # ── 1. Load tokenizer
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 2. Load model in fp32 (CPU-compatible)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info(f"Loading model on {device.upper()} with {dtype}…")

    hf_token = os.getenv("HUGGINGFACE_TOKEN", None)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        token=hf_token,
    )
    if device == "cpu":
        model = model.to("cpu")

    ram_gb = psutil.Process().memory_info().rss / 1e9
    logger.info(f"Model loaded. Process RSS: {ram_gb:.2f} GB")

    # ── 3. Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA r={LORA_R}: {trainable_params:,} trainable / {total_params:,} total "
        f"({100*trainable_params/total_params:.2f}%)"
    )

    # ── 4. Build dataset
    dataset = build_dataset(tokenizer)

    # ── 5. Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LR,
        fp16=False,
        bf16=False,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # ── 6. Train
    logger.info("Starting LoRA fine-tuning…")
    train_result = trainer.train()

    # ── 7. Save adapter
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Adapter saved to: {OUTPUT_DIR}")

    # ── 8. Persist metrics
    elapsed = time.time() - start_time
    metrics = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "num_samples": NUM_SAMPLES,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "learning_rate": LR,
        "max_seq_len": MAX_SEQ_LEN,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "final_loss": train_result.training_loss,
        "training_time_seconds": round(elapsed, 1),
        "training_time_minutes": round(elapsed / 60, 2),
        "device": device,
    }

    metrics_path = Path(OUTPUT_DIR) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        f"\n{'='*60}\n"
        f"Fine-tuning complete!\n"
        f"  Time   : {elapsed/60:.1f} min\n"
        f"  Loss   : {train_result.training_loss:.4f}\n"
        f"  Adapter: {OUTPUT_DIR}\n"
        f"{'='*60}"
    )
    return metrics


if __name__ == "__main__":
    main()

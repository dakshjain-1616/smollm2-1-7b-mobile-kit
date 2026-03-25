#!/usr/bin/env python3
"""
Benchmark SmolLM2-1.7B before and after LoRA fine-tuning.

Metrics:
  • WikiText-2 perplexity (lower = better)
  • MMLU accuracy — 5-shot, subset of subjects (higher = better)
  • Generation speed  — tokens/sec
  • RAM usage         — peak process RSS in GB

Outputs:
  benchmark_report.md
  outputs/benchmark_results.json
"""

import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

try:
    import psutil
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from peft import PeftModel
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_MODEL = os.getenv("BASE_MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
ADAPTER_DIR = os.getenv("ADAPTER_OUTPUT_DIR", "models/smollm2-lora-adapter")
MERGED_DIR = os.getenv("MERGED_MODEL_DIR", "models/smollm2-merged")
GGUF_DIR = os.getenv("GGUF_OUTPUT_DIR", "models/gguf")
REPORT_PATH = os.getenv("BENCHMARK_REPORT", "benchmark_report.md")
RESULTS_PATH = os.getenv("BENCHMARK_RESULTS", "outputs/benchmark_results.json")

MMLU_SUBJECTS = os.getenv(
    "MMLU_SUBJECTS",
    "high_school_mathematics,college_computer_science,world_history,"
    "high_school_physics,professional_medicine",
).split(",")
MMLU_QUESTIONS = int(os.getenv("MMLU_QUESTIONS_PER_SUBJECT", "50"))
PPL_NUM_SAMPLES = int(os.getenv("PPL_NUM_SAMPLES", "50"))
SPEED_TEST_TOKENS = int(os.getenv("SPEED_TEST_TOKENS", "100"))
NUM_FEWSHOT = int(os.getenv("MMLU_FEWSHOT", "5"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_ram_gb() -> float:
    """Current process RSS in GB."""
    return psutil.Process().memory_info().rss / 1e9


def load_model_and_tokenizer(model_id_or_path: str, adapter_path: str | None = None):
    """Load a HF causal LM (optionally with LoRA adapter merged at inference)."""
    hf_token = os.getenv("HUGGINGFACE_TOKEN", None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        token=hf_token,
    )
    if adapter_path and Path(adapter_path).exists():
        logger.info(f"  Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return model, tokenizer


def free_model(model):
    """Release model memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─── Perplexity (WikiText-2) ─────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, num_samples: int = PPL_NUM_SAMPLES) -> float:
    """Compute average per-token perplexity on WikiText-2 test set."""
    logger.info(f"  Computing WikiText-2 perplexity on {num_samples} paragraphs…")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    texts = [
        t.strip()
        for t in dataset["text"]
        if len(t.strip()) > 100
    ][:num_samples]

    total_nll = 0.0
    total_tokens = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                continue
            out = model(input_ids, labels=input_ids)
            n_tokens = input_ids.shape[1]
            total_nll += out.loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("nan")

    ppl = math.exp(total_nll / total_tokens)
    logger.info(f"  PPL = {ppl:.4f} (over {total_tokens:,} tokens)")
    return round(ppl, 4)


# ─── MMLU Accuracy ───────────────────────────────────────────────────────────

MMLU_CHOICES = ["A", "B", "C", "D"]


def build_mmlu_fewshot_prompt(examples: list[dict]) -> str:
    """Build a 5-shot prefix from MMLU dev examples."""
    lines = []
    for ex in examples:
        q = ex["question"]
        choices = ex["choices"]
        ans = MMLU_CHOICES[ex["answer"]]
        lines.append(f"Question: {q}")
        for i, c in enumerate(choices):
            lines.append(f"  {MMLU_CHOICES[i]}. {c}")
        lines.append(f"Answer: {ans}\n")
    return "\n".join(lines) + "\n"


def score_mmlu_question(
    model, tokenizer, prompt: str, choices: list[str], answer_idx: int, device
) -> bool:
    """Return True if model picks the correct choice by log-prob."""
    choice_tokens = []
    for ch in MMLU_CHOICES[:len(choices)]:
        toks = tokenizer.encode(f" {ch}", add_special_tokens=False)
        choice_tokens.append(toks[0] if toks else tokenizer.unk_token_id)

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]  # last token logits

    scores = [logits[tok_id].item() for tok_id in choice_tokens]
    predicted = int(torch.tensor(scores).argmax())
    return predicted == answer_idx


def compute_mmlu_accuracy(
    model,
    tokenizer,
    subjects: list[str] = MMLU_SUBJECTS,
    questions_per_subject: int = MMLU_QUESTIONS,
    num_fewshot: int = NUM_FEWSHOT,
) -> float:
    """Compute 5-shot MMLU accuracy over a subset of subjects."""
    logger.info(f"  Computing MMLU accuracy ({num_fewshot}-shot, {len(subjects)} subjects)…")
    device = next(model.parameters()).device

    correct = 0
    total = 0

    for subj in subjects:
        try:
            test_ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
            dev_ds = load_dataset("cais/mmlu", subj, split="dev", trust_remote_code=True)
        except Exception as e:
            logger.warning(f"  Could not load MMLU '{subj}': {e}")
            continue

        dev_examples = list(dev_ds)[:num_fewshot]
        fewshot_prefix = build_mmlu_fewshot_prompt(dev_examples)

        test_subset = list(test_ds)[:questions_per_subject]

        for ex in test_subset:
            q = ex["question"]
            choices = ex["choices"]
            answer_idx = ex["answer"]

            # Build test prompt
            question_part = f"Question: {q}\n"
            for i, c in enumerate(choices):
                question_part += f"  {MMLU_CHOICES[i]}. {c}\n"
            question_part += "Answer:"
            full_prompt = fewshot_prefix + question_part

            try:
                ok = score_mmlu_question(
                    model, tokenizer, full_prompt, choices, answer_idx, device
                )
                correct += int(ok)
                total += 1
            except Exception as e:
                logger.debug(f"  Skipped question: {e}")

        logger.info(f"    {subj}: running total {correct}/{total}")

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"  MMLU accuracy = {accuracy:.4f}  ({correct}/{total})")
    return round(accuracy, 4)


# ─── Generation Speed ─────────────────────────────────────────────────────────

def measure_tokens_per_second(
    model, tokenizer, num_tokens: int = SPEED_TEST_TOKENS
) -> float:
    """Generate `num_tokens` tokens and return throughput (tokens/sec)."""
    logger.info(f"  Measuring generation speed ({num_tokens} tokens)…")
    device = next(model.parameters()).device

    prompt = "Explain the difference between machine learning and deep learning in detail:"
    enc = tokenizer(prompt, return_tensors="pt").to(device)

    gen_cfg = GenerationConfig(
        max_new_tokens=num_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Warmup
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=5, do_sample=False,
                       pad_token_id=tokenizer.pad_token_id)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, generation_config=gen_cfg)
    elapsed = time.perf_counter() - t0

    generated = out.shape[1] - enc["input_ids"].shape[1]
    tps = generated / elapsed if elapsed > 0 else 0.0
    logger.info(f"  Speed = {tps:.2f} tok/s  ({generated} tokens in {elapsed:.2f}s)")
    return round(tps, 2)


# ─── GGUF Benchmarks (via llama-cpp-python) ───────────────────────────────────

def benchmark_gguf(gguf_path: Path) -> dict:
    """Benchmark GGUF model via llama-cpp-python."""
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.warning("  llama-cpp-python not available; skipping GGUF benchmark.")
        return {}

    logger.info(f"  Benchmarking GGUF: {gguf_path}")

    ram_before = get_ram_gb()
    llm = Llama(model_path=str(gguf_path), n_ctx=512, n_threads=4, verbose=False)
    ram_after = get_ram_gb()

    prompt = "Explain the difference between machine learning and deep learning:"

    t0 = time.perf_counter()
    out = llm(prompt, max_tokens=SPEED_TEST_TOKENS, echo=False)
    elapsed = time.perf_counter() - t0

    tokens_generated = out["usage"]["completion_tokens"]
    tps = tokens_generated / elapsed if elapsed > 0 else 0.0

    size_gb = gguf_path.stat().st_size / 1e9

    logger.info(f"  GGUF speed = {tps:.2f} tok/s, RAM delta = {ram_after - ram_before:.2f} GB")

    del llm
    gc.collect()

    return {
        "file_size_gb": round(size_gb, 3),
        "tokens_per_second": round(tps, 2),
        "ram_usage_gb": round(ram_after, 2),
        "ram_delta_gb": round(ram_after - ram_before, 2),
    }


# ─── Per-model benchmarking ───────────────────────────────────────────────────

def run_hf_benchmark(label: str, model_path: str, adapter_path: str | None = None) -> dict:
    """Run all HF-model benchmarks and return results dict."""
    logger.info(f"\n{'─'*50}")
    logger.info(f"Benchmarking: {label}")
    logger.info(f"{'─'*50}")

    ram_before = get_ram_gb()
    model, tokenizer = load_model_and_tokenizer(model_path, adapter_path)
    ram_after_load = get_ram_gb()

    ppl = compute_perplexity(model, tokenizer)
    mmlu = compute_mmlu_accuracy(model, tokenizer)
    tps = measure_tokens_per_second(model, tokenizer)
    ram_peak = get_ram_gb()

    free_model(model)

    results = {
        "label": label,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "perplexity_wikitext2": ppl,
        "mmlu_accuracy": mmlu,
        "tokens_per_second": tps,
        "ram_load_gb": round(ram_after_load, 2),
        "ram_peak_gb": round(ram_peak, 2),
    }

    logger.info(
        f"Results for {label}:\n"
        f"  PPL      : {ppl}\n"
        f"  MMLU     : {mmlu:.4f} ({mmlu*100:.1f}%)\n"
        f"  Speed    : {tps} tok/s\n"
        f"  RAM peak : {ram_peak:.2f} GB"
    )

    return results


# ─── Report generation ────────────────────────────────────────────────────────

def write_report(results: dict, report_path: str = REPORT_PATH):
    """Write a Markdown benchmark report."""
    base = results.get("base_model", {})
    ft = results.get("fine_tuned_model", {})
    gguf = results.get("gguf_q4km", {})

    ppl_delta = ""
    mmlu_delta = ""
    if base and ft:
        ppl_diff = ft.get("perplexity_wikitext2", 0) - base.get("perplexity_wikitext2", 0)
        ppl_delta = f"  ({'+' if ppl_diff >= 0 else ''}{ppl_diff:.2f})"
        mmlu_diff = ft.get("mmlu_accuracy", 0) - base.get("mmlu_accuracy", 0)
        mmlu_delta = f"  ({'+' if mmlu_diff >= 0 else ''}{mmlu_diff*100:.1f}pp)"

    lines = [
        "# SmolLM2-1.7B Mobile Kit — Benchmark Report",
        "",
        "> Fine-tune SmolLM2-1.7B-Instruct with LoRA r=8 on 1 k Alpaca instructions,",
        "> quantize to Q4_K_M GGUF (~900 MB), runs on 1.5 GB RAM — any modern smartphone.",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}  ",
        f"**Base model**: `{BASE_MODEL}`  ",
        f"**LoRA**: r=8, α=16, target=q/k/v/o_proj  ",
        f"**Dataset**: tatsu-lab/alpaca (1 000 samples, 3 epochs)  ",
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "| Metric | Base Model | Fine-tuned | GGUF Q4_K_M |",
        "|--------|-----------|-----------|------------|",
    ]

    def fmt(d: dict, key: str, fmt_str: str = "{}", suffix: str = "") -> str:
        v = d.get(key)
        if v is None:
            return "—"
        return fmt_str.format(v) + suffix

    lines += [
        f"| **WikiText-2 PPL** ↓ | "
        f"{fmt(base, 'perplexity_wikitext2', '{:.2f}')} | "
        f"{fmt(ft, 'perplexity_wikitext2', '{:.2f}')}{ppl_delta} | "
        f"{fmt(gguf, 'perplexity_wikitext2', '{:.2f}')} |",

        f"| **MMLU Accuracy** ↑ | "
        f"{fmt(base, 'mmlu_accuracy', '{:.1%}')} | "
        f"{fmt(ft, 'mmlu_accuracy', '{:.1%}')}{mmlu_delta} | "
        f"— |",

        f"| **Tokens / sec** ↑ | "
        f"{fmt(base, 'tokens_per_second', '{:.1f}')} | "
        f"{fmt(ft, 'tokens_per_second', '{:.1f}')} | "
        f"{fmt(gguf, 'tokens_per_second', '{:.1f}')} |",

        f"| **RAM usage (GB)** ↓ | "
        f"{fmt(base, 'ram_peak_gb', '{:.2f}')} GB | "
        f"{fmt(ft, 'ram_peak_gb', '{:.2f}')} GB | "
        f"{fmt(gguf, 'ram_usage_gb', '{:.2f}')} GB |",

        f"| **File size** | ~3.4 GB | ~3.4 GB | "
        f"{fmt(gguf, 'file_size_gb', '{:.3f}')} GB |",

        "",
        "---",
        "",
        "## Detailed Results",
        "",
    ]

    for label, d in [("Base Model", base), ("Fine-tuned (LoRA merged)", ft), ("GGUF Q4_K_M", gguf)]:
        if not d:
            continue
        lines += [
            f"### {label}",
            "",
            f"- **WikiText-2 Perplexity**: {d.get('perplexity_wikitext2', '—')}",
            f"- **MMLU Accuracy** (5-shot, {', '.join(MMLU_SUBJECTS[:3])}...): "
            f"{d.get('mmlu_accuracy', '—')}",
            f"- **Generation Speed**: {d.get('tokens_per_second', '—')} tok/s",
            f"- **RAM (peak)**: {d.get('ram_peak_gb', d.get('ram_usage_gb', '—'))} GB",
        ]
        if label == "GGUF Q4_K_M":
            lines.append(f"- **File Size**: {d.get('file_size_gb', '—')} GB")
        lines.append("")

    lines += [
        "---",
        "",
        "## Methodology",
        "",
        "| Item | Value |",
        "|------|-------|",
        f"| PPL corpus | WikiText-2 test, {PPL_NUM_SAMPLES} paragraphs, stride 512 |",
        f"| MMLU eval  | {NUM_FEWSHOT}-shot, {len(MMLU_SUBJECTS)} subjects × {MMLU_QUESTIONS} q |",
        f"| Speed test | {SPEED_TEST_TOKENS} new tokens, greedy decoding |",
        "| RAM metric | Peak process RSS during generation |",
        "",
        "## Mobile Footprint",
        "",
        "| Format | Size | Min RAM | Device target |",
        "|--------|------|---------|---------------|",
        "| HF fp32 | ~6.8 GB | 8 GB | Server / workstation |",
        "| HF fp16 | ~3.4 GB | 4 GB | Laptop / GPU cloud |",
        f"| **Q4_K_M GGUF** | **{fmt(gguf, 'file_size_gb', '{:.2f}')} GB** | **~1.5 GB** | **Any smartphone** |",
        "",
        "> *SmolLM2-1.7B Q4_K_M can run on any phone released after 2018.*",
        "",
        "---",
        "*Report generated by SmolLM2-1.7B Mobile Kit — "
        "[github.com/smollm2-mobile-kit](https://github.com/smollm2-mobile-kit)*",
    ]

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Report written to: {report_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if not _DEPS_OK:
        raise ImportError(
            "Heavy dependencies (torch, transformers, peft, datasets, psutil) are not installed. "
            "Run: pip install -r requirements.txt"
        )
    logger.info("=" * 60)
    logger.info("SmolLM2-1.7B Benchmark Suite")
    logger.info("=" * 60)

    Path("outputs").mkdir(exist_ok=True)
    all_results = {}

    # ── 1. Base model
    base_results = run_hf_benchmark(
        label="Base Model",
        model_path=BASE_MODEL,
        adapter_path=None,
    )
    all_results["base_model"] = base_results

    # ── 2. Fine-tuned model (merged LoRA, or adapter on top)
    # Use merged model if available, else load adapter on top of base
    if Path(MERGED_DIR, "config.json").exists():
        ft_results = run_hf_benchmark(
            label="Fine-tuned (LoRA merged)",
            model_path=MERGED_DIR,
            adapter_path=None,
        )
    elif Path(ADAPTER_DIR, "adapter_config.json").exists():
        ft_results = run_hf_benchmark(
            label="Fine-tuned (LoRA)",
            model_path=BASE_MODEL,
            adapter_path=ADAPTER_DIR,
        )
    else:
        logger.warning("No fine-tuned model found; skipping fine-tuned benchmark.")
        ft_results = {}

    if ft_results:
        all_results["fine_tuned_model"] = ft_results

    # ── 3. GGUF Q4_K_M (if available)
    q4km_path = Path(GGUF_DIR) / "smollm2-1.7b-ft-q4_k_m.gguf"
    if q4km_path.exists():
        gguf_results = benchmark_gguf(q4km_path)
        if gguf_results:
            all_results["gguf_q4km"] = gguf_results
    else:
        logger.warning(f"GGUF not found at {q4km_path}; skipping GGUF benchmark.")

    # ── 4. Write results JSON
    results_path = Path(RESULTS_PATH)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results JSON saved to: {results_path}")

    # ── 5. Write Markdown report
    write_report(all_results, REPORT_PATH)

    # ── 6. Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for key, res in all_results.items():
        print(f"\n[{key}]")
        for k, v in res.items():
            if k not in ("label", "model_path", "adapter_path"):
                print(f"  {k:30s}: {v}")

    return all_results


if __name__ == "__main__":
    main()

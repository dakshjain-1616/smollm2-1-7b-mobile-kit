#!/usr/bin/env python3
"""
Merge LoRA adapter and quantize to Q4_K_M GGUF.

Steps:
  1. Load base model + LoRA adapter
  2. Merge adapter into base weights (merge_and_unload)
  3. Save merged model in HuggingFace format
  4. Convert to F16 GGUF via llama.cpp convert_hf_to_gguf.py
  5. Quantize F16 → Q4_K_M via llama-quantize
  6. Verify output is < 1.1 GB

Usage:
    python scripts/quantize.py

Requires llama.cpp tools (auto-downloaded/built if absent).
"""

import os
import sys
import json
import shutil
import logging
import subprocess
import stat
import tarfile
import zipfile
from pathlib import Path

try:
    import requests
    import torch
    import psutil
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_MODEL = os.getenv("BASE_MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
ADAPTER_DIR = os.getenv("ADAPTER_OUTPUT_DIR", "models/smollm2-lora-adapter")
MERGED_DIR = os.getenv("MERGED_MODEL_DIR", "models/smollm2-merged")
GGUF_DIR = os.getenv("GGUF_OUTPUT_DIR", "models/gguf")
TOOLS_DIR = os.getenv("TOOLS_DIR", "tools")

LLAMA_CPP_CONVERT_URL = os.getenv(
    "LLAMA_CPP_CONVERT_URL",
    "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─── Tool discovery ───────────────────────────────────────────────────────────

def find_llama_quantize() -> str | None:
    """Locate the llama-quantize binary from PATH or llama-cpp-python package dir."""
    # 1. System PATH
    for name in ("llama-quantize", "quantize"):
        found = shutil.which(name)
        if found:
            return found

    # 2. llama-cpp-python package directory
    try:
        import llama_cpp
        pkg = Path(llama_cpp.__file__).parent
        for subdir in ("", "lib", "bin", "backends"):
            for name in ("llama-quantize", "quantize"):
                candidate = pkg / subdir / name
                if candidate.exists():
                    return str(candidate)
    except ImportError:
        pass

    # 3. tools/ directory (user-placed or self-built)
    for name in ("llama-quantize", "quantize"):
        candidate = Path(TOOLS_DIR) / name
        if candidate.exists():
            return str(candidate)

    return None


def download_convert_script() -> Path:
    """Download convert_hf_to_gguf.py from llama.cpp if not present."""
    tools_path = Path(TOOLS_DIR)
    tools_path.mkdir(parents=True, exist_ok=True)
    convert_script = tools_path / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        logger.info(f"Downloading convert_hf_to_gguf.py …")
        try:
            resp = requests.get(LLAMA_CPP_CONVERT_URL, timeout=60)
            resp.raise_for_status()
            convert_script.write_text(resp.text, encoding="utf-8")
            logger.info(f"Saved to {convert_script}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download convert_hf_to_gguf.py: {e}\n"
                f"Manually place the file at {convert_script}"
            ) from e

    return convert_script


# ─── Merge ───────────────────────────────────────────────────────────────────

def merge_lora() -> Path:
    """Load base model, apply LoRA adapter, merge, and save."""
    merged_path = Path(MERGED_DIR)

    if (merged_path / "config.json").exists():
        logger.info(f"Merged model already exists at {MERGED_DIR}, skipping merge.")
        return merged_path

    logger.info(f"Loading base model: {BASE_MODEL}")
    hf_token = os.getenv("HUGGINGFACE_TOKEN", None)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map=None,
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)

    logger.info(f"Loading LoRA adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)

    logger.info("Merging adapter weights into base model …")
    model = model.merge_and_unload()

    merged_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving merged model to: {MERGED_DIR}")
    model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)

    ram_gb = psutil.Process().memory_info().rss / 1e9
    logger.info(f"Merge complete. Process RSS: {ram_gb:.2f} GB")

    # Free memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return merged_path


# ─── GGUF Conversion ─────────────────────────────────────────────────────────

def convert_to_f16_gguf(merged_dir: Path) -> Path:
    """Convert HF model directory to F16 GGUF using llama.cpp convert script."""
    gguf_path = Path(GGUF_DIR)
    gguf_path.mkdir(parents=True, exist_ok=True)
    f16_out = gguf_path / "smollm2-1.7b-ft-f16.gguf"

    if f16_out.exists():
        logger.info(f"F16 GGUF already exists: {f16_out}")
        return f16_out

    convert_script = download_convert_script()

    cmd = [
        sys.executable,
        str(convert_script),
        str(merged_dir),
        "--outfile", str(f16_out),
        "--outtype", "f16",
    ]
    logger.info(f"Converting HF → F16 GGUF…\n  {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Conversion stdout:\n{result.stdout}")
        logger.error(f"Conversion stderr:\n{result.stderr}")
        raise RuntimeError(
            f"convert_hf_to_gguf.py failed (exit {result.returncode}).\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    size_gb = f16_out.stat().st_size / 1e9
    logger.info(f"F16 GGUF saved: {f16_out} ({size_gb:.2f} GB)")
    return f16_out


# ─── Q4_K_M Quantization ─────────────────────────────────────────────────────

def quantize_to_q4km(f16_path: Path) -> Path:
    """Quantize F16 GGUF to Q4_K_M using llama-quantize binary."""
    gguf_path = Path(GGUF_DIR)
    q4km_out = gguf_path / "smollm2-1.7b-ft-q4_k_m.gguf"

    if q4km_out.exists():
        size_gb = q4km_out.stat().st_size / 1e9
        logger.info(f"Q4_K_M GGUF already exists: {q4km_out} ({size_gb:.2f} GB)")
        return q4km_out

    quantize_bin = find_llama_quantize()
    if not quantize_bin:
        raise RuntimeError(
            "llama-quantize binary not found.\n\n"
            "Install options:\n"
            "  Option A — Build llama.cpp:\n"
            "    git clone https://github.com/ggerganov/llama.cpp\n"
            "    cd llama.cpp && cmake -B build && cmake --build build -j\n"
            "    cp build/bin/llama-quantize tools/\n\n"
            "  Option B — Pre-built release (Linux x86_64):\n"
            "    See https://github.com/ggerganov/llama.cpp/releases\n"
            "    Place llama-quantize in tools/\n\n"
            "  Option C — conda-forge:\n"
            "    conda install -c conda-forge llama.cpp"
        )

    cmd = [quantize_bin, str(f16_path), str(q4km_out), "Q4_K_M"]
    logger.info(f"Quantizing F16 → Q4_K_M…\n  {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Quantization stderr:\n{result.stderr}")
        raise RuntimeError(
            f"llama-quantize failed (exit {result.returncode}).\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    size_gb = q4km_out.stat().st_size / 1e9
    logger.info(f"Q4_K_M GGUF saved: {q4km_out} ({size_gb:.3f} GB)")

    if size_gb >= 1.1:
        logger.warning(
            f"WARNING: GGUF size {size_gb:.2f} GB exceeds 1.1 GB target! "
            "SmolLM2-1.7B Q4_K_M should be ~0.90 GB."
        )

    return q4km_out


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if not _DEPS_OK:
        raise ImportError(
            "Heavy dependencies (torch, transformers, peft, requests) are not installed. "
            "Run: pip install -r requirements.txt"
        )
    logger.info("=" * 60)
    logger.info("SmolLM2-1.7B: LoRA Merge + GGUF Q4_K_M Quantization")
    logger.info("=" * 60)

    # Step 1: Merge LoRA
    merged_dir = merge_lora()

    # Step 2: Convert HF → F16 GGUF
    f16_path = convert_to_f16_gguf(merged_dir)

    # Step 3: Quantize F16 → Q4_K_M
    q4km_path = quantize_to_q4km(f16_path)

    # Step 4: Save metadata
    metadata = {
        "base_model": BASE_MODEL,
        "adapter_dir": ADAPTER_DIR,
        "merged_model_dir": str(merged_dir),
        "f16_gguf_path": str(f16_path),
        "f16_gguf_size_gb": round(f16_path.stat().st_size / 1e9, 3),
        "q4km_gguf_path": str(q4km_path),
        "q4km_gguf_size_gb": round(q4km_path.stat().st_size / 1e9, 3),
        "size_ok": q4km_path.stat().st_size / 1e9 < 1.1,
    }

    meta_path = Path(GGUF_DIR) / "quantization_info.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Quantization complete!")
    print(f"  Merged model : {merged_dir}")
    print(f"  F16 GGUF     : {f16_path}  ({metadata['f16_gguf_size_gb']:.2f} GB)")
    print(f"  Q4_K_M GGUF  : {q4km_path}  ({metadata['q4km_gguf_size_gb']:.3f} GB)")
    print(f"  Size OK (<1.1GB): {metadata['size_ok']}")
    print("=" * 60)

    return metadata


if __name__ == "__main__":
    main()

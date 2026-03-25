# SmolLM2-1.7B Mobile Kit – End-to-End LoRA Fine-tune to GGUF

> *Made autonomously using [NEO](https://heyneo.so) · [![Install NEO Extension](https://img.shields.io/badge/VS%%20Code-Install%%20NEO-7B61FF?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-45%%20passed-brightgreen.svg)]()

> Deploy custom SmolLM2 models to mobile devices with a single script that handles fine-tuning, quantization, and benchmarking.

## Install

```bash
git clone https://github.com/dakshjain-1616/smollm2-1-7b-mobile-kit
cd smollm2-1-7b-mobile-kit
pip install -r requirements.txt
```

## Quickstart

```bash
# Run the instant demo (mock mode, <1 second)
python demo.py

# Or run the full pipeline (fine-tune, merge, quantize, benchmark)
python scripts/finetune.py
```

## Key features

- **End-to-End Pipeline:** Automates LoRA fine-tuning, adapter merging, GGUF conversion, and Q4_K_M quantization.
- **Mobile Optimized:** Generates ~900MB models compatible with 1.5GB RAM devices via llama.cpp.
- **Automated Benchmarking:** Produces `benchmark_report.md` with PPL, MMLU, tokens/sec, and RAM usage metrics.
- **CPU Compatible:** Fine-tunes SmolLM2-1.7B-Instruct on 1k Alpaca instructions in ~25 minutes on CPU.

## Run tests

```bash
pytest tests/ -q
# 45 passed
```

## Project structure

```
smollm2-1-7b-mobile-kit/
├── demo.py           ← Instant mock demo
├── scripts/          ← Core pipeline logic
│   ├── __init__.py
│   ├── finetune.py   ← LoRA training & merge
│   ├── quantize.py   ← GGUF conversion & quantization
│   ├── benchmark.py  ← Performance evaluation
│   └── demo.py       ← Script wrapper
├── tests/            ← Test suite
│   ├── __init__.py
│   └── test_project.py
└── requirements.txt  ← Dependencies
```
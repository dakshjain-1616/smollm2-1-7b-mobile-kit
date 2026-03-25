#!/usr/bin/env python3
"""
Root-level re-export shim so tests can do `import demo` from the project root.
All logic lives in scripts/demo.py.
"""
import os
import sys

# Ensure project root is on sys.path so `scripts` is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.demo import (  # noqa: E402, F401
    generate_mock_results,
    write_benchmark_report,
    detect_mock_mode,
    main,
    generate_mock_training_curve,
    run_real_pipeline,
    OUTPUTS_DIR,
    BASE_MODEL,
    NUM_SAMPLES,
    LORA_R,
    LORA_ALPHA,
    NUM_EPOCHS,
)

__all__ = [
    "generate_mock_results",
    "write_benchmark_report",
    "detect_mock_mode",
    "main",
    "generate_mock_training_curve",
    "run_real_pipeline",
    "OUTPUTS_DIR",
    "BASE_MODEL",
    "NUM_SAMPLES",
    "LORA_R",
    "LORA_ALPHA",
    "NUM_EPOCHS",
]

if __name__ == "__main__":
    main()

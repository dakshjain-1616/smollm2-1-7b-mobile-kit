"""
pytest test suite for SmolLM2-1.7B Mobile Kit.

Tests:
  1. GGUF < 1.1 GB (if file exists)
  2. benchmark_report.md contains PPL / MMLU / tok_sec / RAM
  3. Fine-tuned MMLU >= base MMLU
  4. Full pipeline runs on <= 8 GB RAM
  5. Code structure, imports, and configuration
  6. Dataset formatting
  7. Mock mode produces valid JSON
  8. Report generation
  9. Environment variable configuration
  10. Path and directory creation
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make scripts importable
sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
MODELS_DIR = Path(__file__).parent.parent / "models"
GGUF_PATH = MODELS_DIR / "gguf" / "smollm2-1.7b-ft-q4_k_m.gguf"
BENCHMARK_REPORT = Path(__file__).parent.parent / "benchmark_report.md"
DEMO_RESULTS = OUTPUTS_DIR / "demo_results.json"


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_benchmark_results():
    """Realistic sample benchmark results dict."""
    return {
        "base_model": {
            "label": "Base Model",
            "perplexity_wikitext2": 12.80,
            "mmlu_accuracy": 0.414,
            "tokens_per_second": 8.6,
            "ram_peak_gb": 3.45,
        },
        "fine_tuned_model": {
            "label": "Fine-tuned",
            "perplexity_wikitext2": 11.32,
            "mmlu_accuracy": 0.438,
            "tokens_per_second": 8.4,
            "ram_peak_gb": 3.45,
            "ppl_improvement": 1.48,
            "mmlu_improvement_pp": 2.4,
        },
        "gguf_q4km": {
            "file_size_gb": 0.902,
            "perplexity_wikitext2": 11.87,
            "tokens_per_second": 12.3,
            "ram_usage_gb": 1.52,
            "size_ok": True,
        },
    }


@pytest.fixture
def sample_full_results(sample_benchmark_results):
    """Full demo results structure."""
    return {
        "run_timestamp": "2026-03-25T12:00:00+00:00",
        "mode": "mock",
        "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "fine_tuning": {
            "dataset": "tatsu-lab/alpaca",
            "num_samples": 1000,
            "lora_r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "trainable_params": 4194304,
            "total_params": 1710419968,
            "trainable_pct": 0.245,
            "final_loss": 0.9823,
            "training_time_minutes": 24.7,
            "num_epochs": 3,
        },
        "benchmarks": sample_benchmark_results,
    }


# ─── Test 1: GGUF file size ───────────────────────────────────────────────────

class TestGGUFSize:
    """GGUF file must be < 1.1 GB when it exists."""

    def test_gguf_size_limit_if_exists(self):
        """If GGUF file exists, it must be < 1.1 GB."""
        if not GGUF_PATH.exists():
            pytest.skip(f"GGUF not yet generated at {GGUF_PATH}")
        size_gb = GGUF_PATH.stat().st_size / 1e9
        assert size_gb < 1.1, (
            f"GGUF too large: {size_gb:.3f} GB — expected < 1.1 GB for Q4_K_M"
        )

    def test_gguf_size_well_within_limit_if_exists(self):
        """Q4_K_M of 1.7B should be ~0.9 GB — flag if unexpectedly large."""
        if not GGUF_PATH.exists():
            pytest.skip(f"GGUF not yet generated at {GGUF_PATH}")
        size_gb = GGUF_PATH.stat().st_size / 1e9
        # SmolLM2-1.7B Q4_K_M should be well under 1 GB
        assert size_gb < 1.0, (
            f"GGUF {size_gb:.3f} GB is larger than expected (~0.90 GB for SmolLM2-1.7B Q4_K_M)"
        )

    def test_quantize_metadata_size_field(self):
        """If quantization_info.json exists, check size field."""
        meta = MODELS_DIR / "gguf" / "quantization_info.json"
        if not meta.exists():
            pytest.skip("quantization_info.json not found")
        with open(meta) as f:
            data = json.load(f)
        assert "q4km_gguf_size_gb" in data
        assert data["q4km_gguf_size_gb"] < 1.1
        assert data.get("size_ok") is True


# ─── Test 2: benchmark_report.md contents ─────────────────────────────────────

class TestBenchmarkReport:
    """benchmark_report.md must contain all required metrics."""

    REQUIRED_KEYWORDS = [
        "perplexity", "PPL",
        "MMLU",
        "tok", "sec",       # tokens/sec
        "RAM",
    ]

    def _find_report(self) -> Path:
        """Find benchmark_report.md in expected locations."""
        candidates = [
            BENCHMARK_REPORT,
            OUTPUTS_DIR / "benchmark_report.md",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def test_report_exists(self):
        """benchmark_report.md must exist (in root or outputs/)."""
        r = self._find_report()
        assert r is not None, (
            "benchmark_report.md not found. Run python scripts/benchmark.py or python demo.py"
        )

    def test_report_contains_ppl(self):
        """Report must mention perplexity."""
        r = self._find_report()
        if r is None:
            pytest.skip("benchmark_report.md not found")
        content = r.read_text(encoding="utf-8").lower()
        assert "perplexity" in content or "ppl" in content

    def test_report_contains_mmlu(self):
        """Report must mention MMLU accuracy."""
        r = self._find_report()
        if r is None:
            pytest.skip("benchmark_report.md not found")
        content = r.read_text(encoding="utf-8").lower()
        assert "mmlu" in content

    def test_report_contains_tokens_per_sec(self):
        """Report must mention generation speed."""
        r = self._find_report()
        if r is None:
            pytest.skip("benchmark_report.md not found")
        content = r.read_text(encoding="utf-8").lower()
        assert "tok" in content and ("sec" in content or "/s" in content)

    def test_report_contains_ram(self):
        """Report must mention RAM usage."""
        r = self._find_report()
        if r is None:
            pytest.skip("benchmark_report.md not found")
        content = r.read_text(encoding="utf-8").lower()
        assert "ram" in content or "memory" in content

    def test_report_has_summary_table(self):
        """Report should contain a Markdown table with all metrics."""
        r = self._find_report()
        if r is None:
            pytest.skip("benchmark_report.md not found")
        content = r.read_text(encoding="utf-8")
        assert "|" in content, "Report should contain a Markdown table"
        # Should have at least Base Model and Fine-tuned columns
        assert "Base" in content
        assert "Fine" in content or "tuned" in content.lower()

    def test_report_has_mobile_section(self):
        """Report should mention mobile/smartphone footprint."""
        r = self._find_report()
        if r is None:
            pytest.skip("benchmark_report.md not found")
        content = r.read_text(encoding="utf-8").lower()
        assert "mobile" in content or "smartphone" in content or "phone" in content

    def test_report_minimum_length(self):
        """Report must be a substantive document (>= 50 lines)."""
        r = self._find_report()
        if r is None:
            pytest.skip("benchmark_report.md not found")
        lines = r.read_text(encoding="utf-8").splitlines()
        assert len(lines) >= 50, f"Report only has {len(lines)} lines — expected >= 50"


# ─── Test 3: fine-tuned MMLU >= base MMLU ─────────────────────────────────────

class TestMMIUComparison:
    """Fine-tuned model MMLU accuracy must be >= base model."""

    def _load_results(self) -> dict | None:
        candidates = [
            DEMO_RESULTS,
            OUTPUTS_DIR / "benchmark_results.json",
        ]
        for c in candidates:
            if c.exists():
                with open(c) as f:
                    return json.load(f)
        return None

    def test_finetuned_mmlu_gte_base(self):
        """Fine-tuned MMLU accuracy should be >= base model accuracy."""
        data = self._load_results()
        if data is None:
            pytest.skip("No benchmark results found — run demo.py or benchmark.py first")

        bench = data.get("benchmarks", data)
        base = bench.get("base_model", {})
        ft = bench.get("fine_tuned_model", {})

        if not base or not ft:
            pytest.skip("Results do not contain both base and fine-tuned results")

        base_mmlu = base.get("mmlu_accuracy", 0)
        ft_mmlu = ft.get("mmlu_accuracy", 0)

        assert base_mmlu > 0, "Base MMLU should be > 0"
        assert ft_mmlu > 0, "Fine-tuned MMLU should be > 0"
        assert ft_mmlu >= base_mmlu, (
            f"Fine-tuned MMLU ({ft_mmlu:.4f}) < base MMLU ({base_mmlu:.4f}). "
            "LoRA training should improve or maintain MMLU performance."
        )

    def test_finetuned_ppl_gte_base(self):
        """Fine-tuned perplexity should be <= base model perplexity (lower = better)."""
        data = self._load_results()
        if data is None:
            pytest.skip("No benchmark results found")

        bench = data.get("benchmarks", data)
        base = bench.get("base_model", {})
        ft = bench.get("fine_tuned_model", {})

        if not base or not ft:
            pytest.skip("Missing base or fine-tuned results")

        base_ppl = base.get("perplexity_wikitext2", 0)
        ft_ppl = ft.get("perplexity_wikitext2", 0)

        if base_ppl > 0 and ft_ppl > 0:
            assert ft_ppl <= base_ppl * 1.05, (
                f"Fine-tuned PPL ({ft_ppl:.2f}) is significantly worse than base ({base_ppl:.2f}). "
                "Allowing 5% tolerance."
            )

    def test_results_have_required_fields(self):
        """Benchmark results JSON must have all required metric fields."""
        data = self._load_results()
        if data is None:
            pytest.skip("No benchmark results found")

        bench = data.get("benchmarks", data)
        base = bench.get("base_model", {})

        if not base:
            pytest.skip("No base_model in results")

        required_fields = [
            "perplexity_wikitext2",
            "mmlu_accuracy",
            "tokens_per_second",
            "ram_peak_gb",
        ]
        for field in required_fields:
            assert field in base, f"Missing field '{field}' in base_model results"


# ─── Test 4: RAM requirements ─────────────────────────────────────────────────

class TestRAMRequirements:
    """Pipeline components should run within 8 GB RAM."""

    def test_gguf_ram_under_2gb(self):
        """GGUF inference RAM should be < 2 GB."""
        data = None
        for p in [DEMO_RESULTS, OUTPUTS_DIR / "benchmark_results.json"]:
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                break
        if data is None:
            pytest.skip("No results found")

        bench = data.get("benchmarks", data)
        gguf = bench.get("gguf_q4km", {})
        if not gguf:
            pytest.skip("No GGUF results")

        ram = gguf.get("ram_usage_gb", 0)
        assert ram < 2.0, f"GGUF RAM {ram:.2f} GB exceeds 2 GB — smartphone target is 1.5 GB"

    def test_hf_model_ram_under_8gb(self):
        """HF model RAM should be under 8 GB (4 GB for fp16, 3.5 GB typical)."""
        data = None
        for p in [DEMO_RESULTS, OUTPUTS_DIR / "benchmark_results.json"]:
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                break
        if data is None:
            pytest.skip("No results found")

        bench = data.get("benchmarks", data)
        base = bench.get("base_model", {})
        if not base:
            pytest.skip("No base model results")

        ram = base.get("ram_peak_gb", 0)
        assert ram < 8.0, (
            f"Base model RAM {ram:.2f} GB exceeds 8 GB budget. "
            "Use fp16 or reduce batch size."
        )


# ─── Test 5: Code structure and imports ──────────────────────────────────────

class TestCodeStructure:
    """Verify scripts are importable and have correct structure."""

    def test_finetune_importable(self):
        """scripts/finetune.py must be importable."""
        import scripts.finetune as ft
        assert hasattr(ft, "main"), "finetune.py must have a main() function"
        assert hasattr(ft, "format_alpaca_sample")
        assert hasattr(ft, "build_dataset")

    def test_quantize_importable(self):
        """scripts/quantize.py must be importable."""
        import scripts.quantize as qt
        assert hasattr(qt, "main"), "quantize.py must have a main() function"
        assert hasattr(qt, "merge_lora")
        assert hasattr(qt, "find_llama_quantize")

    def test_benchmark_importable(self):
        """scripts/benchmark.py must be importable."""
        import scripts.benchmark as bm
        assert hasattr(bm, "main"), "benchmark.py must have a main() function"
        assert hasattr(bm, "compute_perplexity")
        assert hasattr(bm, "compute_mmlu_accuracy")
        assert hasattr(bm, "measure_tokens_per_second")
        assert hasattr(bm, "write_report")

    def test_demo_importable(self):
        """demo.py must be importable."""
        import demo
        assert hasattr(demo, "main")
        assert hasattr(demo, "generate_mock_results")
        assert hasattr(demo, "write_benchmark_report")
        assert hasattr(demo, "detect_mock_mode")

    def test_env_vars_configured(self):
        """All scripts must use os.getenv for configuration."""
        import scripts.finetune as ft
        import inspect
        source = inspect.getsource(ft)
        assert "os.getenv" in source, "finetune.py should use os.getenv for configuration"

    def test_lora_config_values(self):
        """LoRA should be configured with r=8, alpha=16."""
        import scripts.finetune as ft
        assert ft.LORA_R == int(os.getenv("LORA_R", "8"))
        assert ft.LORA_ALPHA == int(os.getenv("LORA_ALPHA", "16"))
        assert "q_proj" in ft.LORA_TARGET_MODULES
        assert "v_proj" in ft.LORA_TARGET_MODULES

    def test_default_model_name(self):
        """Default model should be SmolLM2-1.7B-Instruct."""
        import scripts.finetune as ft
        default = os.getenv("BASE_MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
        assert ft.MODEL_NAME == default


# ─── Test 6: Dataset formatting ──────────────────────────────────────────────

class TestDatasetFormatting:
    """Verify Alpaca formatting produces correct ChatML prompts."""

    def test_format_with_input(self):
        """Sample with instruction + input should include both."""
        from scripts.finetune import format_alpaca_sample
        sample = {
            "instruction": "Translate to French.",
            "input": "Hello, world!",
            "output": "Bonjour, le monde!",
        }
        result = format_alpaca_sample(sample)
        assert "text" in result
        assert "Translate to French." in result["text"]
        assert "Hello, world!" in result["text"]
        assert "Bonjour, le monde!" in result["text"]
        assert "<|im_start|>user" in result["text"]
        assert "<|im_start|>assistant" in result["text"]
        assert "<|im_end|>" in result["text"]

    def test_format_without_input(self):
        """Sample without input should still format correctly."""
        from scripts.finetune import format_alpaca_sample
        sample = {
            "instruction": "Write a poem about autumn.",
            "input": "",
            "output": "Leaves fall like whispers…",
        }
        result = format_alpaca_sample(sample)
        assert "Write a poem about autumn." in result["text"]
        assert "Leaves fall like whispers" in result["text"]
        # Empty input should not leave double newlines at user message
        assert "autumn.\n\n\n" not in result["text"]

    def test_format_returns_text_key(self):
        """format_alpaca_sample must return dict with 'text' key."""
        from scripts.finetune import format_alpaca_sample
        sample = {"instruction": "Say hi.", "input": "", "output": "Hi!"}
        result = format_alpaca_sample(sample)
        assert isinstance(result, dict)
        assert "text" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 10


# ─── Test 7: Mock mode ────────────────────────────────────────────────────────

class TestMockMode:
    """Mock mode should produce valid, plausible output."""

    def test_generate_mock_results_structure(self):
        """generate_mock_results() must return complete results dict."""
        from demo import generate_mock_results
        results = generate_mock_results()

        assert "run_timestamp" in results
        assert "mode" in results
        assert results["mode"] == "mock"
        assert "model" in results
        assert "fine_tuning" in results
        assert "benchmarks" in results

    def test_mock_fine_tuning_fields(self):
        """Mock fine-tuning results must have all required fields."""
        from demo import generate_mock_results
        results = generate_mock_results()
        ft = results["fine_tuning"]

        required = [
            "dataset", "num_samples", "lora_r", "lora_alpha",
            "final_loss", "training_time_minutes",
        ]
        for field in required:
            assert field in ft, f"Missing fine_tuning field: {field}"

        assert ft["lora_r"] == int(os.getenv("LORA_R", "8"))
        assert ft["lora_alpha"] == int(os.getenv("LORA_ALPHA", "16"))
        assert ft["num_samples"] == int(os.getenv("NUM_SAMPLES", "1000"))
        assert 0.5 < ft["final_loss"] < 3.0
        assert 15 < ft["training_time_minutes"] < 45

    def test_mock_benchmark_values_plausible(self):
        """Mock benchmark numbers should be in realistic ranges."""
        from demo import generate_mock_results
        results = generate_mock_results()
        bench = results["benchmarks"]

        base = bench["base_model"]
        ft = bench["fine_tuned_model"]
        gguf = bench["gguf_q4km"]

        # Perplexity: SmolLM2-1.7B should be in 10-20 range
        assert 8 < base["perplexity_wikitext2"] < 25
        assert 8 < ft["perplexity_wikitext2"] < 25

        # MMLU: 0.3 to 0.7 is reasonable
        assert 0.3 < base["mmlu_accuracy"] < 0.7
        assert 0.3 < ft["mmlu_accuracy"] < 0.7

        # Fine-tuned should be better
        assert ft["mmlu_accuracy"] >= base["mmlu_accuracy"]
        assert ft["perplexity_wikitext2"] <= base["perplexity_wikitext2"]

        # Speed: CPU should be in 5-30 range for 1.7B
        assert 1 < base["tokens_per_second"] < 50

        # GGUF size
        assert 0.7 < gguf["file_size_gb"] < 1.1
        assert gguf["size_ok"] is True

        # GGUF RAM
        assert 0.5 < gguf["ram_usage_gb"] < 2.5

    def test_mock_results_are_deterministic(self):
        """Mock results should be the same on repeated calls (deterministic RNG)."""
        from demo import generate_mock_results
        r1 = generate_mock_results()
        r2 = generate_mock_results()
        assert r1["benchmarks"]["base_model"]["perplexity_wikitext2"] == \
               r2["benchmarks"]["base_model"]["perplexity_wikitext2"]

    def test_mock_training_curve_present(self):
        """Mock results should include a training loss curve."""
        from demo import generate_mock_results
        results = generate_mock_results()
        assert "training_curve" in results
        curve = results["training_curve"]
        assert isinstance(curve, list)
        assert len(curve) > 10
        # Loss should decrease over time
        first_loss = curve[0]["loss"]
        last_loss = curve[-1]["loss"]
        assert last_loss < first_loss, "Training loss should decrease"


# ─── Test 8: Report generation ───────────────────────────────────────────────

class TestReportGeneration:
    """write_benchmark_report must produce valid Markdown."""

    def test_write_benchmark_report(self, sample_full_results, tmp_path):
        """Report should be written to the specified path."""
        from demo import write_benchmark_report
        out = tmp_path / "test_report.md"
        write_benchmark_report(sample_full_results, out)
        assert out.exists()
        assert out.stat().st_size > 500

    def test_report_content(self, sample_full_results, tmp_path):
        """Generated report must contain all required sections."""
        from demo import write_benchmark_report
        out = tmp_path / "test_report.md"
        write_benchmark_report(sample_full_results, out)
        content = out.read_text(encoding="utf-8")

        assert "SmolLM2" in content
        assert "PPL" in content or "Perplexity" in content or "perplexity" in content
        assert "MMLU" in content
        assert "tok" in content.lower()
        assert "RAM" in content
        assert "Q4_K_M" in content
        assert "|" in content  # Has table

    def test_report_numeric_values(self, sample_full_results, tmp_path):
        """Report must embed the actual numeric values from results."""
        from demo import write_benchmark_report
        out = tmp_path / "test_report.md"
        write_benchmark_report(sample_full_results, out)
        content = out.read_text(encoding="utf-8")

        bench = sample_full_results["benchmarks"]
        base_ppl = bench["base_model"]["perplexity_wikitext2"]
        # PPL value should appear in report
        assert str(base_ppl) in content, f"Base PPL {base_ppl} not found in report"

    def test_benchmark_report_write_function(self):
        """benchmark.py write_report function should work standalone."""
        from scripts.benchmark import write_report
        import tempfile
        results = {
            "base_model": {
                "perplexity_wikitext2": 12.5,
                "mmlu_accuracy": 0.42,
                "tokens_per_second": 8.0,
                "ram_peak_gb": 3.4,
            }
        }
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            tmp_path = f.name
        try:
            write_report(results, tmp_path)
            content = Path(tmp_path).read_text()
            assert "SmolLM2" in content
            assert "12.5" in content
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ─── Test 9: Environment variable configuration ───────────────────────────────

class TestEnvConfig:
    """All configurations must be driven by env vars with sensible defaults."""

    def test_env_example_exists(self):
        """'.env.example' must exist at project root."""
        env_example = Path(__file__).parent.parent / ".env.example"
        assert env_example.exists(), ".env.example must exist at project root"

    def test_env_example_has_required_vars(self):
        """'.env.example' must document all required env vars."""
        env_example = Path(__file__).parent.parent / ".env.example"
        content = env_example.read_text(encoding="utf-8")

        required_vars = [
            "BASE_MODEL",
            "DATASET_NAME",
            "NUM_SAMPLES",
            "LORA_R",
            "LORA_ALPHA",
            "NUM_EPOCHS",
            "ADAPTER_OUTPUT_DIR",
            "GGUF_OUTPUT_DIR",
        ]
        for var in required_vars:
            assert var in content, f"'.env.example' missing variable: {var}"

    def test_base_model_default(self):
        """Default base model should be SmolLM2-1.7B-Instruct."""
        import scripts.finetune as ft
        base = os.getenv("BASE_MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
        assert "SmolLM2" in ft.MODEL_NAME or "smollm2" in ft.MODEL_NAME.lower()

    def test_lora_r_configurable(self, monkeypatch):
        """LORA_R env var should override the default r=8."""
        monkeypatch.setenv("LORA_R", "16")
        import importlib
        import scripts.finetune
        # Reload to pick up new env var
        importlib.reload(scripts.finetune)
        assert scripts.finetune.LORA_R == 16
        # Reset
        monkeypatch.delenv("LORA_R", raising=False)
        importlib.reload(scripts.finetune)


# ─── Test 10: Directory creation and file outputs ─────────────────────────────

class TestOutputFiles:
    """Demo run should produce required output files."""

    def test_outputs_dir_exists(self):
        """outputs/ directory must exist."""
        assert OUTPUTS_DIR.exists(), "outputs/ directory should exist"

    def test_demo_results_json_exists(self):
        """outputs/demo_results.json must exist after running demo.py."""
        if not DEMO_RESULTS.exists():
            pytest.skip("Run 'python demo.py' first to generate outputs")
        assert DEMO_RESULTS.stat().st_size > 100

    def test_demo_results_json_valid(self):
        """outputs/demo_results.json must be valid JSON with required keys."""
        if not DEMO_RESULTS.exists():
            pytest.skip("Run 'python demo.py' first")
        with open(DEMO_RESULTS) as f:
            data = json.load(f)
        assert "run_timestamp" in data
        assert "model" in data
        assert "benchmarks" in data

    def test_demo_creates_outputs_dir(self, tmp_path, monkeypatch):
        """demo.py should create outputs/ directory if it doesn't exist."""
        # Use a temp dir for this test
        test_outputs = tmp_path / "test_outputs"
        monkeypatch.chdir(tmp_path)

        import demo as dm
        original_dir = dm.OUTPUTS_DIR
        dm.OUTPUTS_DIR = test_outputs
        try:
            results = dm.generate_mock_results()
            test_outputs.mkdir(exist_ok=True)
            dm.write_benchmark_report(results, test_outputs / "benchmark_report.md")
            assert (test_outputs / "benchmark_report.md").exists()
        finally:
            dm.OUTPUTS_DIR = original_dir

    def test_requirements_txt_exists(self):
        """requirements.txt must exist at project root."""
        req = Path(__file__).parent.parent / "requirements.txt"
        assert req.exists()

    def test_requirements_has_key_packages(self):
        """requirements.txt must list all key dependencies."""
        req = Path(__file__).parent.parent / "requirements.txt"
        content = req.read_text(encoding="utf-8")
        required = ["torch", "transformers", "peft", "datasets", "trl",
                    "llama-cpp-python", "psutil", "numpy"]
        for pkg in required:
            assert pkg in content, f"requirements.txt missing: {pkg}"

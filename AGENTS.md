# Repository Guidelines

## Project Structure & Module Organization

This directory contains the MoE race-test harness for the larger `tilelang-metax` repository. `custom_fusedmoe.py` is the editable TileLang fused MoE kernel entry point; keep the `RoutedMoEKernel` constructor and `__call__` interface stable. `ref_fusedmoe.py` provides the PyTorch reference implementation used for correctness checks. `fusedmoe_benchmark.py` generates inputs, runs functional tests, and contains optional performance timing code. `moe_test_configs.json` stores functional and performance cases, and `run.sh` is a thin wrapper around the benchmark script. Avoid committing generated caches such as `__pycache__/`.

## Build, Test, and Development Commands

From the repository root, install or rebuild TileLang with:

```bash
pip install .
```

For faster C++ iteration, configure once with `cmake -S . -B build`, rebuild with `cmake --build build -j$(nproc)`, and run with `PYTHONPATH` pointing at the repo root. Do not use `pip install -e .`; this project imports the local `./tilelang` package directly during source-tree development.

From `race_tests/moe`, run:

```bash
python fusedmoe_benchmark.py
bash run.sh
```

Both execute the configured functional checks. Most tests require a CUDA-capable GPU plus `torch`, `tilelang`, and `apache-tvm-ffi`.

## Coding Style & Naming Conventions

Python code follows the root `pyproject.toml` Ruff configuration: 4-space indentation, double quotes when formatted, and a 140-character line length. Prefer explicit names matching existing config keys such as `dhidden`, `dexpert`, `nroutedexperts`, and `nexpertspertoken` for JSON-facing parameters. Keep TileLang kernels and wrapper classes small enough to compare against the PyTorch reference.

## Testing Guidelines

Add new cases to `moe_test_configs.json` under `functional` or `performance`. Functional tests compare `custom_kernel` against `ref_kernel` with `torch.testing.assert_close`; use deterministic seeds and document unusual tolerances in code. For broader package validation from the root, run `python -m pytest testing/python/ -x` after rebuilding.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative subjects, often with a scoped prefix such as `[MetaXGPU]`, `[MetaxGPU][testing]`, or `[Example][MetaXGPU]`. Keep commits focused and mention the affected target or test area. Pull requests should describe the kernel or test change, list GPU/backend assumptions, include the exact command output used for validation, and link any relevant issue. Include screenshots only when changing documentation or visual artifacts.

---
name: cuda-backend-eval
description: Use when implementing, testing, or evaluating the PTO CUDA backend on local A100 GPUs or the remote bizhaoh200 H200 host, including source-paper setup, CUDA smoke tests, and benchmark-result capture.
---

# CUDA Backend Evaluation

## Scope

Use this skill for PTO CUDA backend work under `src/cuda/`, CUDA tests under
`tests/ut/py/test_cuda_backend.py`, local A100 smoke tests, remote H200 smoke
tests through `ssh bizhaoh200`, and paper-aligned evaluation setup.

## Source Discipline

- Keep downloaded paper PDFs and extracted text under `tmp/sources/`.
- Keep source notes under `tmp/` so they are inspectable but not committed.
- Current source files:
  - `tmp/sources/arxiv-2605.03190-vdcores.pdf`
  - `tmp/sources/arxiv-2605.03190-vdcores.txt`
  - `tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.pdf`
  - `tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.txt`

## Local Smoke Test

Run tests through the venv Python module, not a bare `pytest`; this machine may
resolve bare `pytest` to a user-level executable outside `.venv`.

```bash
.venv/bin/python -m pytest tests/ut/py/test_cuda_backend.py -q
```

The current smoke test builds `cuda/host_schedule`, compiles a PTX vector-add
kernel with `nvcc`, allocates real CUDA device buffers, copies real data, runs
the kernel through the runtime C API, and validates copied-back results.

For remote machines without `pytest`, run the standalone smoke:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_smoke.py
```

## Microbenchmark Report

Use `cuda_benchmark.py` for the current early-runtime comparison. It runs the
same vector-add PTX kernel through two launch paths:

- `pto_host_schedule`: the PTO CUDA host runtime C API and manifest dispatch.
- `direct_driver`: a thin CUDA Driver API baseline in Python `ctypes`.

The smoke helper caches the built and loaded host runtime per process, so the
benchmark can run repeated PTO and baseline samples without rebuilding a shared
object that is already loaded.

Local A100 example:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024,1048576 --repeats 5 --arch compute_80 \
    --label a100-local --output-dir tmp/cuda-backend/a100
```

Remote H200 example:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 1024,1048576 --repeats 5 --arch compute_90 \
     --label h200-remote --output-dir tmp/cuda-backend/h200'
```

The script writes:

- `cuda-benchmark.json`: raw samples, metadata, hardware, git commit.
- `cuda-benchmark.md`: short report with interpretation notes.
- `cuda-benchmark.svg`: bar chart of median device time by baseline.

Merge local and remote JSON payloads into one comparative report:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json tmp/cuda-backend/a100/cuda-benchmark.json \
    tmp/cuda-backend/h200/cuda-benchmark.json \
    --label cuda-a100-h200 --output-dir tmp/cuda-backend/combined
```

## Hardware Checks

Local A100:

```bash
command -v nvcc
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
```

Remote H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'hostname; command -v nvcc || true; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader'
```

## Paper-Aligned Evaluation Shape

Use the papers to choose workloads and baselines, but do not claim paper-level
results until full LLM serving baselines are actually run.

- VDCores setup: fixed offline decoding, 128-token context, 64 decode steps,
  batch sizes 1-8, models including Qwen3-1.7B/Qwen3-8B and Llama small
  variants, baselines including vLLM, SGLang, Mirage, ThunderKittens, and
  Torch+ThunderKittens.
- MPK setup: offline batched inference, fixed prompt length 64, decode 1024
  tokens, max batch sizes 1-16, baselines vLLM and SGLang, GPUs A100/H100/B200
  in the paper and local A100/remote H200 for this repo.
- For early PTO CUDA runtime work, report only microbenchmarks and smoke tests
  as such: launch latency, host wall time, device event time, copy bandwidth,
  and simple dependency/concurrency tests.

## Result Capture

For repeatable local/remote runs, save raw command output under
`outputs/cuda-backend/` or `tmp/cuda-backend/` with hardware, git commit, CUDA
toolkit, driver, command, and timestamp. Commit only scripts, docs, and tests;
do not commit raw benchmark outputs unless the user asks.

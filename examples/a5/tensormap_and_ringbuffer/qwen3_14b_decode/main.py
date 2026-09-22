#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B 40-layer decode (CANN fused-attention) — standalone driver.

Self-contained port of pypto-lib ``models/qwen3/14b/decode_fwd.py`` entry
``decode_fwd_layers`` with ``_CHUNK_NLAYERS == 40``: the whole Qwen3-14B decode
stack as ONE fused dispatch (hidden -> hidden, no LM head), carrying the
inter-layer residual in FP32. The layer loop is a real loop in the generated
orchestration, so a 40-layer chunk costs one extra literal over a 2-layer one,
not 20x the kernels.

The C++ under ``kernels/`` is harvested pypto codegen (orchestration + 18 AIC +
16 AIV) plus the hand-written CANN attention extern under
``kernels/paged_attention_cce/``; ``simpler_setup/goldens/qwen3_14b_decode.py``
is the matching torch reference. See README.md for provenance and how to
regenerate.

Parameter regime matches ``stress_profile.py`` (vLLM serving stress): BATCH=16,
MAX_SEQ=5500 (= max_model_len), fixed decode seq_len=3500. Weights and the paged
KV pool are stacked x40 (one slice per layer); every layer reuses layer-0
weights, per the lib const-layer-0 stacked-fwd reference, while each layer reads
and writes its own KV pool.

All entry parameters live in persistent L2 device allocations. The fixture is
generated and uploaded one tensor at a time before the first round; subsequent
rounds reuse the same addresses without runtime staging or automatic copy-back.
`--skip-golden` skips only torch computation and explicit output readback.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from simpler.buffer import Buffer
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig
from simpler.worker import Worker

from simpler_setup import SceneTestCase, TaskArgsBuilder, scene_test
from simpler_setup.compile_pool import compile_worker_budget
from simpler_setup.goldens.qwen3_14b_decode import (
    N_LAYERS,
    param_specs,
    param_tensors,
)
from simpler_setup.goldens.qwen3_14b_decode import (
    compute_golden as _decode_golden,
)
from simpler_setup.goldens.qwen3_14b_decode import (
    generate_inputs as _decode_generate_inputs,
)
from simpler_setup.log_config import DEFAULT_LOG_LEVEL, LOG_LEVEL_CHOICES, configure_logging
from simpler_setup.parallel_scheduler import device_range_to_list
from simpler_setup.scene_test import (
    ChildMemoryTaskArgs,
    build_output_prefix,
    compile_chip_callable_spec,
    effective_diagnostic_options,
    finalize_diagnostic_outputs,
    l3_compile_cache_key,
    log_torch_backend_autoload_once,
)

HERE = Path(__file__).resolve().parent
CASE_LABEL = "Qwen314BDecode"

# CANN devkit headers for the attention extern, which builds on AscendC and the
# vendored FusedInferAttentionScore under kernels/paged_attention_cce/vendor/.
# `vendor/.../attn_infra/base_defs.hpp` selects its AscendC entry header under
# `#if ASC_DEVKIT_MAJOR >= 9`, which ccec predefines from the installed devkit,
# so a CANN 9 box must be able to resolve `basic_api/kernel_basic_intf.h` from
# one of these.
#
# `$ASCEND_HOME_PATH` keeps the paths machine-independent and is expanded at
# compile time, not import time — this file is collected on sim and macOS runners
# that have no CANN at all. The devkit's arch subdirectory is named differently
# across installs, so both layouts are listed; missing ones are dropped, and the
# scene-test resolver raises if *every* entry is missing.
_CANN_SUBDIRS = (
    "include",
    "asc",
    "asc/impl/adv_api",
    "asc/impl/basic_api",
    "asc/impl/basic_api/reg_compute",
    "asc/impl/c_api",
    "asc/impl/simt_api",
    "asc/impl/utils",
    "asc/include",
    "asc/include/adv_api",
    "asc/include/aicpu_api",
    "asc/include/basic_api",
    "asc/include/basic_api/reg_compute",
    "asc/include/c_api",
    "asc/include/interface",
    "asc/include/simt_api",
    "asc/include/utils",
    "tikcpp/tikcfw",
    "tikcpp/tikcfw/impl",
    "tikcpp/tikcfw/interface",
)

_CANN_INCLUDE_DIRS = [f"$ASCEND_HOME_PATH/{prefix}{sub}" for prefix in ("aarch64-linux/", "") for sub in _CANN_SUBDIRS]


# The decorator remains load-bearing even though pytest collects only the thin
# wrapper: it resolves every relative CALLABLE source against this directory.
@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestQwen314BDecode(SceneTestCase):
    """Qwen3-14B decode, all 40 layers in one dispatch, against a torch reference."""

    RTOL = 5e-2
    ATOL = 1e-1

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/decode_fwd_layers.cpp",
            "function_name": "aicpu_orchestration_entry",
            # decode_fwd_layers takes k_cache / v_cache as plain inputs, but the
            # attention extern writes the current token's KV into them. Declaring
            # them INOUT preserves their dependency contract; correctness mode
            # explicitly reads the pools back to check all 40 layers' KV writes.
            "signature": [
                D.IN,  # 0  hidden_states
                D.IN,  # 1  input_rms_weight
                D.IN,  # 2  wq
                D.IN,  # 3  wk
                D.IN,  # 4  wv
                D.IN,  # 5  q_norm_weight
                D.IN,  # 6  k_norm_weight
                D.IN,  # 7  seq_lens
                D.IN,  # 8  block_table
                D.IN,  # 9  slot_mapping
                D.IN,  # 10 rope_cos
                D.IN,  # 11 rope_sin
                D.INOUT,  # 12 k_cache
                D.INOUT,  # 13 v_cache
                D.IN,  # 14 wo
                D.IN,  # 15 w_gate
                D.IN,  # 16 w_up
                D.IN,  # 17 w_down
                D.IN,  # 18 post_rms_weight
                D.OUT,  # 19 out
            ],
        },
        # 37 incores (func_id 0..36), transcribed from the pypto codegen
        # kernel_config.py for decode_fwd_layers (N=40). func_id 0/11/12 are the
        # CANN attention externs; 11 and 12 are the same source dispatched as the
        # AIC and AIV halves of one mixed task.
        "incores": [
            {
                "func_id": 0,
                "name": "paged_attention_tiling_cce",
                "source": "kernels/vendor/paged_attention_cce/tiling/entry.cpp",
                "core_type": "aiv",
                "extra_include_dirs": _CANN_INCLUDE_DIRS,
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "copy_hidden",
                "source": "kernels/aiv/copy_hidden.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.IN],
            },
            {
                "func_id": 2,
                "name": "x_gamma0",
                "source": "kernels/aiv/x_gamma0.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.IN, D.IN],
            },
            {
                "func_id": 3,
                "name": "attn_out_seed",
                "source": "kernels/aiv/attn_out_seed.cpp",
                "core_type": "aiv",
                "signature": [D.IN],
            },
            {
                "func_id": 4,
                "name": "rms_recip",
                "source": "kernels/aiv/rms_recip.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT],
            },
            {
                "func_id": 5,
                "name": "q_seed",
                "source": "kernels/aiv/q_seed.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 6,
                "name": "q_proj",
                "source": "kernels/aic/q_proj.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 7,
                "name": "kv_seed",
                "source": "kernels/aiv/kv_seed.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT],
            },
            {
                "func_id": 8,
                "name": "mlp_out_seed",
                "source": "kernels/aiv/mlp_out_seed.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.INOUT, D.INOUT],
            },
            {
                "func_id": 9,
                "name": "k_proj",
                "source": "kernels/aic/k_proj.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 10,
                "name": "v_proj",
                "source": "kernels/aic/v_proj.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 11,
                "name": "paged_attention_rope_cce_aic",
                "source": "kernels/vendor/paged_attention_cce/attention_rope/entry.cpp",
                "core_type": "aic",
                "extra_include_dirs": _CANN_INCLUDE_DIRS,
                "signature": [
                    D.INOUT,
                    D.INOUT,
                    D.INOUT,
                    D.INOUT,
                    D.IN,
                    D.INOUT,
                    D.INOUT,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                ],
            },
            {
                "func_id": 12,
                "name": "paged_attention_rope_cce_aiv",
                "source": "kernels/vendor/paged_attention_cce/attention_rope/entry.cpp",
                "core_type": "aiv",
                "extra_include_dirs": _CANN_INCLUDE_DIRS,
                "signature": [
                    D.INOUT,
                    D.INOUT,
                    D.INOUT,
                    D.INOUT,
                    D.IN,
                    D.INOUT,
                    D.INOUT,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                    D.IN,
                ],
            },
            {
                "func_id": 13,
                "name": "out_proj",
                "source": "kernels/aic/out_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 14,
                "name": "out_proj_0",
                "source": "kernels/aic/out_proj_0.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 15,
                "name": "residual_rms_cast",
                "source": "kernels/aiv/residual_rms_cast.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 16,
                "name": "residual_rms_cast_0",
                "source": "kernels/aiv/residual_rms_cast_0.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 17,
                "name": "residual_rms_cast_1",
                "source": "kernels/aiv/residual_rms_cast_1.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 18,
                "name": "residual_rms_cast_2",
                "source": "kernels/aiv/residual_rms_cast_2.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 19,
                "name": "residual_rms_cast_3",
                "source": "kernels/aiv/residual_rms_cast_3.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 20,
                "name": "post_rms_reduce",
                "source": "kernels/aiv/post_rms_reduce.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 21,
                "name": "gate_proj",
                "source": "kernels/aic/gate_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 22,
                "name": "up_proj",
                "source": "kernels/aic/up_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 23,
                "name": "gate_proj_0",
                "source": "kernels/aic/gate_proj_0.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 24,
                "name": "up_proj_0",
                "source": "kernels/aic/up_proj_0.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 25,
                "name": "gate_proj_1",
                "source": "kernels/aic/gate_proj_1.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 26,
                "name": "up_proj_1",
                "source": "kernels/aic/up_proj_1.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 27,
                "name": "gate_proj_2",
                "source": "kernels/aic/gate_proj_2.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 28,
                "name": "up_proj_2",
                "source": "kernels/aic/up_proj_2.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 29,
                "name": "gate_proj_3",
                "source": "kernels/aic/gate_proj_3.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 30,
                "name": "up_proj_3",
                "source": "kernels/aic/up_proj_3.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 31,
                "name": "gate_proj_4",
                "source": "kernels/aic/gate_proj_4.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 32,
                "name": "up_proj_4",
                "source": "kernels/aic/up_proj_4.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 33,
                "name": "silu",
                "source": "kernels/aiv/silu.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 34,
                "name": "down_proj",
                "source": "kernels/aic/down_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 35,
                "name": "dcr_xgamma",
                "source": "kernels/aiv/dcr_xgamma.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT, D.IN, D.INOUT],
            },
            {
                "func_id": 36,
                "name": "copy_out",
                "source": "kernels/aiv/copy_out.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.IN],
            },
        ],
    }

    CASES = [
        {
            "name": "StressBatch16Seq3500",
            "platforms": ["a5"],
            "manual": True,
            # A run takes the whole device, matching the lib default. The 40-layer
            # graph's peak in-flight dependency footprint (~32 k entries) overflows
            # the a5 ring's default dep pool (16 k); the a2a3 source's "default
            # carries 40 layers" note predates the current runtime's dep accounting.
            "config": {"runtime_env": {"ring_dep_pool": 65536}},
            "params": {"seed": 1234, "seq_len": 3500},
        },
    ]

    def generate_args(self, params):
        args = _decode_generate_inputs(params.get("seed", 1234), params.get("seq_len", 3500))
        return TaskArgsBuilder(*(spec._replace(child_memory=True) for spec in args.specs))

    def compute_golden(self, args, params):
        _decode_golden(args)


def _chip_spec(orchestration_source: str | Path | None, orchestration_function: str) -> dict:
    spec = copy.deepcopy(TestQwen314BDecode.CALLABLE)
    if orchestration_source is not None:
        spec["orchestration"]["source"] = str(orchestration_source)
    spec["orchestration"]["function_name"] = orchestration_function
    spec["name"] = "decode_fwd_layers"
    return spec


def _upload_fixture(child_args, signature, *, seed, seq_len, n_layers, fixture=None):
    specs = param_specs(n_layers)
    if len(specs) != len(signature):
        raise ValueError("Qwen parameter count must match the orchestration signature")
    directions = {spec.name: direction for spec, direction in zip(specs, signature)}
    tensors = (
        ((spec.name, getattr(fixture, spec.name)) for spec in specs)
        if fixture is not None
        else param_tensors(seed=seed, seq_len=seq_len, n_layers=n_layers)
    )
    for name, tensor in tensors:
        child_args.add(name, tensor, directions[name])
        # Release this weight before the generator creates its successor.
        del tensor


def _build_config(
    runtime_env: dict,
    *,
    enable_chip_swimlane: int,
    dump_args: int,
    enable_pmu: int,
    enable_dep_gen: bool,
    enable_scope_stats: bool,
    output_prefix: str,
) -> CallConfig:
    config = CallConfig()
    config.runtime_env.ring_task_window = runtime_env.get("ring_task_window", 0)
    config.runtime_env.ring_heap = runtime_env.get("ring_heap", 0)
    config.runtime_env.ring_dep_pool = runtime_env.get("ring_dep_pool", 0)
    config.enable_chip_swimlane = enable_chip_swimlane
    config.enable_dump_args = dump_args
    config.enable_pmu = enable_pmu
    config.enable_dep_gen = enable_dep_gen
    config.enable_scope_stats = enable_scope_stats
    if output_prefix:
        config.output_prefix = output_prefix
    return config


def _compare_tensor(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_flat = actual.view(-1)
    expected_flat = expected.view(-1)
    chunk_elems = 1 << 20
    for offset in range(0, actual_flat.numel(), chunk_elems):
        actual_chunk = actual_flat[offset : offset + chunk_elems]
        expected_chunk = expected_flat[offset : offset + chunk_elems]
        if not torch.allclose(actual_chunk, expected_chunk, rtol=TestQwen314BDecode.RTOL, atol=TestQwen314BDecode.ATOL):
            diff = (actual_chunk.float() - expected_chunk.float()).abs().max().item()
            raise AssertionError(
                f"Golden mismatch on '{name}' near element {offset}: "
                f"max_diff={diff}, rtol={TestQwen314BDecode.RTOL}, atol={TestQwen314BDecode.ATOL}"
            )


def _copy_and_compare(worker: Worker, buffers: dict[str, Buffer], golden) -> None:
    for name in ("out", "k_cache", "v_cache"):
        expected = getattr(golden, name)
        actual = torch.empty_like(expected)
        worker.copy_from(actual, buffers[name])
        _compare_tensor(name, actual, expected)
        del actual


def run(  # noqa: PLR0913 -- one knob per standalone CLI option
    device_ids,
    platform: str,
    *,
    runtime: str = "tensormap_and_ringbuffer",
    orchestration_source: str | Path | None = None,
    orchestration_function: str = "aicpu_orchestration_entry",
    runtime_env: dict | None = None,
    rounds: int = 1,
    skip_golden: bool = False,
    seed: int = 1234,
    seq_len: int = 3500,
    enable_chip_swimlane: int = 0,
    dump_args: int = 0,
    enable_pmu: int = 0,
    enable_dep_gen: bool = False,
    enable_scope_stats: bool = False,
    enable_swimlane_overhead: bool = False,
    compile_only: bool = False,
    compile_workers: int | None = None,
) -> int:
    """Compile and run Qwen decode with one persistent set of device buffers."""
    device_ids = [int(device) for device in device_ids]
    if not compile_only and not device_ids:
        raise ValueError("qwen3_14b_decode needs one device")
    if rounds <= 0:
        raise ValueError(f"rounds must be positive, got {rounds}")

    spec = _chip_spec(orchestration_source, orchestration_function)
    cache_key = l3_compile_cache_key(
        "examples.qwen3_14b_decode",
        f"{platform}:{runtime}:{orchestration_function}",
        spec["name"],
        platform,
        runtime,
    )
    print(f"[qwen] compiling 37 incores + orchestration for {platform}/{runtime}...", flush=True)
    if compile_workers:
        with compile_worker_budget(compile_workers):
            chip = compile_chip_callable_spec(spec, platform, runtime, cache_key)
    else:
        chip = compile_chip_callable_spec(spec, platform, runtime, cache_key)
    if compile_only:
        print("[qwen] compile-only: done", flush=True)
        return 0

    diagnostics = effective_diagnostic_options(
        rounds,
        chip_swimlane=enable_chip_swimlane,
        dump_args=dump_args,
        pmu=enable_pmu,
        dep_gen=enable_dep_gen,
        scope_stats=enable_scope_stats,
        swimlane_overhead=enable_swimlane_overhead,
    )
    diagnostics_on = bool(
        diagnostics.chip_swimlane
        or diagnostics.dump_args
        or diagnostics.pmu
        or diagnostics.dep_gen
        or diagnostics.scope_stats
        or diagnostics.swimlane_overhead
    )
    diagnostic_label = f"{CASE_LABEL}_{runtime}"
    output_prefix = str(build_output_prefix(diagnostic_label)) if diagnostics_on else ""
    if runtime_env is None:
        runtime_env = TestQwen314BDecode.CASES[0].get("config", {}).get("runtime_env", {})
    assert runtime_env is not None

    device_id = device_ids[0]
    print(
        f"[qwen] device={device_id} rounds={rounds} skip_golden={skip_golden}; one fixture upload",
        flush=True,
    )
    worker = Worker(level=2, platform=platform, runtime=runtime, device_id=device_id)
    chip_handle = worker.register(chip)
    worker.init()
    child_args = ChildMemoryTaskArgs(worker)
    try:
        golden = None
        if skip_golden:
            _upload_fixture(
                child_args, spec["orchestration"]["signature"], seed=seed, seq_len=seq_len, n_layers=N_LAYERS
            )
        else:
            print("[qwen] materializing one fixture for upload and torch golden...", flush=True)
            golden = _decode_generate_inputs(seed=seed, seq_len=seq_len, n_layers=N_LAYERS)
            _upload_fixture(
                child_args,
                spec["orchestration"]["signature"],
                seed=seed,
                seq_len=seq_len,
                n_layers=N_LAYERS,
                fixture=golden,
            )
            _decode_golden(golden, n_layers=N_LAYERS)
        task_args = child_args.build_args(expected_count=len(param_specs(N_LAYERS)))
        config = _build_config(
            runtime_env,
            enable_chip_swimlane=diagnostics.chip_swimlane,
            dump_args=diagnostics.dump_args,
            enable_pmu=diagnostics.pmu,
            enable_dep_gen=diagnostics.dep_gen,
            enable_scope_stats=diagnostics.scope_stats,
            output_prefix=output_prefix,
        )
        log_torch_backend_autoload_once()
        # KV writes are intentionally not reset between rounds. Fixed inputs and
        # slot_mapping make them idempotent, so the one-round golden remains valid.
        for round_idx in range(rounds):
            print(f"[qwen] round {round_idx + 1}/{rounds}", flush=True)
            worker.run(chip_handle, task_args, config)
        if golden is not None:
            _copy_and_compare(worker, child_args.buffers, golden)
    finally:
        child_args.release()
        worker.close()
        finalize_diagnostic_outputs(
            diagnostic_label,
            output_prefix,
            callable_spec=spec,
            chip_swimlane=diagnostics.chip_swimlane,
            dep_gen=diagnostics.dep_gen,
            scope_stats=diagnostics.scope_stats,
            swimlane_overhead=diagnostics.swimlane_overhead,
        )
    print("[qwen] PASSED", flush=True)
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--platform", required=True, choices=TestQwen314BDecode.CASES[0]["platforms"])
    parser.add_argument("-d", "--device", default="0", help="device id, list or range; only the first is used")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--skip-golden", action="store_true", help="skip torch golden and D2H compare")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seq-len", type=int, default=3500)
    parser.add_argument("--enable-chip-swimlane", nargs="?", const=4, type=int, default=0)
    parser.add_argument("--dump-args", nargs="?", const=1, type=int, default=0)
    parser.add_argument("--enable-pmu", nargs="?", const=2, type=int, default=0)
    parser.add_argument("--enable-dep-gen", action="store_true")
    parser.add_argument("--enable-scope-stats", action="store_true")
    parser.add_argument("--enable-swimlane-overhead", action="store_true")
    parser.add_argument("--log-level", choices=LOG_LEVEL_CHOICES, default=DEFAULT_LOG_LEVEL)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--compile-workers", type=int, default=None)
    parser.add_argument("--case", action="append", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--manual", choices=["exclude", "include", "only"], default="exclude", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None, *, case_name: str | None = None, **overrides) -> int:
    cli = parse_args(argv)
    configure_logging(cli.log_level)
    expected_case = case_name or TestQwen314BDecode.CASES[0]["name"]
    if cli.case and cli.case != [expected_case]:
        raise ValueError(f"this driver exposes only case {expected_case!r}, got {cli.case}")
    return run(
        device_range_to_list(cli.device),
        cli.platform,
        rounds=cli.rounds,
        skip_golden=cli.skip_golden,
        seed=cli.seed,
        seq_len=cli.seq_len,
        enable_chip_swimlane=cli.enable_chip_swimlane,
        dump_args=cli.dump_args,
        enable_pmu=cli.enable_pmu,
        enable_dep_gen=cli.enable_dep_gen,
        enable_scope_stats=cli.enable_scope_stats,
        enable_swimlane_overhead=cli.enable_swimlane_overhead,
        compile_only=cli.compile_only,
        compile_workers=cli.compile_workers,
        **overrides,
    )


if __name__ == "__main__":
    sys.exit(main())

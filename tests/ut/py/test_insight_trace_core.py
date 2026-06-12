# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import shutil
import subprocess
import sys
import textwrap
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import pytest

from simpler_setup.insight_trace import cli as insight_trace_cli
from simpler_setup.insight_trace.arg_resolver import load_kernel_dump_args, resolve_args
from simpler_setup.insight_trace.case_loader import load_scene_case
from simpler_setup.insight_trace.kernel_analyzer import select_kernel
from simpler_setup.insight_trace.models import (
    KernelShape,
    KernelSpec,
    PlatformArch,
    PlatformFamily,
    SPMDDispatch,
    SPMDReplayMeta,
    TraceBackend,
    TraceConfig,
    TraceScalarArg,
    TraceTensorArg,
)
from simpler_setup.insight_trace.templates import render_host, render_kernel, render_launch
from simpler_setup.insight_trace.workspace import create_workspace


def _a5_bgemm_module() -> Path:
    return Path(__file__).resolve().parents[3] / "examples/a5/tensormap_and_ringbuffer/bgemm/test_bgemm.py"


def _a5_platform() -> PlatformArch:
    return PlatformArch.for_family(PlatformFamily.A5)


def _paged_attention_module() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "examples/a2a3/tensormap_and_ringbuffer/paged_attention/test_paged_attention.py"
    )


def _a5_paged_attention_module() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "examples/a5/tensormap_and_ringbuffer/paged_attention/test_paged_attention.py"
    )


def _vector_example_module() -> Path:
    return Path(__file__).resolve().parents[3] / "tests/st/a2a3/host_build_graph/vector_example/test_vector_example.py"


def _a5_spmd_multiblock_mix_module() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "tests/st/a5/tensormap_and_ringbuffer/spmd_multiblock_mix/test_spmd_multiblock_mix.py"
    )


def _spmd_kernel() -> KernelSpec:
    path = (
        Path(__file__).resolve().parents[3]
        / "tests/st/a2a3/tensormap_and_ringbuffer/spmd_paged_attention/kernels/mix/paged_attention_parallel.cpp"
    )
    return KernelSpec("MixedKernels", 0, "mix", path, KernelShape.SPMD_MIX)


def _spmd_config(tmp_path: Path) -> TraceConfig:
    return TraceConfig(
        backend=TraceBackend.SIMPLER,
        test_module=None,
        case_name="spmd",
        kernel_spec=_spmd_kernel(),
        args=(
            TraceTensorArg(0, "query", "BFLOAT16", (1, 128)),
            TraceTensorArg(6, "sij_fifo", "FLOAT32", (1,)),
            TraceScalarArg(9, "scale_value", "FLOAT32_BITS", 1065353216, "bits"),
            TraceScalarArg(10, "num_heads", "UINT64", 16),
        ),
        output_dir=tmp_path,
        repo_root=tmp_path,
        cann_home=None,
        pto_isa_root=None,
        platform_arch=PlatformArch.for_family(PlatformFamily.A2A3),
        hw_block_num=24,
        spmd_meta=SPMDReplayMeta(
            hw_block_dim=24,
            aiv_lanes_per_core=2,
            fifo_sizes=(65536, 32768, 65536),
        ),
    )


def _a5_spmd_config(tmp_path: Path) -> TraceConfig:
    context = load_scene_case(_a5_spmd_multiblock_mix_module(), "Case1")
    kernel = select_kernel(context, kernel_name="SPMD_MIX_AIC")
    return TraceConfig(
        backend=TraceBackend.SIMPLER,
        test_module=_a5_spmd_multiblock_mix_module(),
        case_name="Case1",
        kernel_spec=kernel,
        args=(
            TraceTensorArg(0, "output", "FLOAT32", (4512,), role="inout"),
            TraceScalarArg(1, "base_cl", "UINT64", 0),
        ),
        output_dir=tmp_path,
        repo_root=tmp_path,
        cann_home=None,
        pto_isa_root=None,
        platform_arch=_a5_platform(),
        hw_block_num=94,
        spmd_meta=SPMDReplayMeta(
            hw_block_dim=24,
            aiv_lanes_per_core=2,
            dispatches=(
                SPMDDispatch(2, ((1, 0),)),
                SPMDDispatch(8, ((1, 6),)),
                SPMDDispatch(12, ((1, 30),)),
                SPMDDispatch(24, ((1, 66),)),
                SPMDDispatch(48, ((1, 138),)),
            ),
        ),
    )


def _a5_bgemm_config(tmp_path: Path) -> TraceConfig:
    context = load_scene_case(_a5_bgemm_module(), "default")
    kernel = select_kernel(context, kernel_name="GEMM")
    args = resolve_args(
        None,
        None,
        arg_spec=_write_arg_spec(
            tmp_path,
            '{"args":['
            '{"kind":"tensor","index":0,"name":"A","dtype":"FLOAT32","shape":[131072]},'
            '{"kind":"tensor","index":1,"name":"B","dtype":"FLOAT32","shape":[131072]},'
            '{"kind":"tensor","index":2,"name":"C","dtype":"FLOAT32","shape":[131072]}'
            "]}",
        ),
    )
    return TraceConfig(
        backend=TraceBackend.SIMPLER,
        test_module=_a5_bgemm_module(),
        case_name="default",
        kernel_spec=kernel,
        args=args,
        output_dir=tmp_path,
        repo_root=tmp_path,
        cann_home=None,
        pto_isa_root=None,
        platform_arch=_a5_platform(),
        hw_block_num=1,
    )


def _write_arg_spec(tmp_path: Path, contents: str) -> Path:
    spec = tmp_path / "args.json"
    spec.write_text(contents)
    return spec


def test_selects_a5_bgemm_as_spmd_mix():
    context = load_scene_case(_a5_bgemm_module(), "default")
    kernel = select_kernel(context, kernel_name="GEMM")
    assert kernel.func_id == 0
    assert kernel.shape == KernelShape.SPMD_MIX


def test_selects_a5_spmd_multiblock_mix_as_spmd_mix():
    context = load_scene_case(_a5_spmd_multiblock_mix_module(), "Case1")
    kernel = select_kernel(context, kernel_name="SPMD_MIX_AIC")
    assert kernel.func_id == 0
    assert kernel.shape == KernelShape.SPMD_MIX


def test_run_simpler_calls_workspace_and_runner_for_non_dry_run(monkeypatch, tmp_path):
    context = load_scene_case(_a5_bgemm_module(), "default")
    kernel = select_kernel(context, kernel_name="GEMM")
    trace_args = resolve_args(
        None,
        None,
        _write_arg_spec(
            tmp_path,
            '{"args":['
            '{"kind":"tensor","index":0,"name":"A","dtype":"FLOAT32","shape":[131072]},'
            '{"kind":"tensor","index":1,"name":"B","dtype":"FLOAT32","shape":[131072]},'
            '{"kind":"tensor","index":2,"name":"C","dtype":"FLOAT32","shape":[131072]}'
            "]}",
        ),
    )
    args = Namespace(
        test_module=_a5_bgemm_module(),
        case="default",
        kernel="GEMM",
        func_id=None,
        kernel_source=None,
        platform="a5",
        runtime="tensormap_and_ringbuffer",
        output_dir=tmp_path,
        cann_home=None,
        pto_isa_root=None,
        soc_version=None,
        device=0,
        launch_count=1,
        timeout=120,
        hw_block_num=1,
        arg_spec=None,
        dump_dir=None,
        dispatch_id=None,
        dry_run=False,
        ptoas_root=None,
        source_cpp=None,
        kernel_base_name=None,
        aicore_arch=None,
        kernel_symbol=None,
        backend=TraceBackend.SIMPLER.value,
    )

    create_mock = Mock(return_value="workspace")
    run_result = type(
        "Result",
        (),
        {"workspace_dir": tmp_path, "simulator_dir": tmp_path / "insight_export" / "OPPROF_x" / "simulator"},
    )()
    run_mock = Mock(return_value=run_result)
    monkeypatch.setattr(insight_trace_cli, "create_workspace", create_mock)
    monkeypatch.setattr(insight_trace_cli, "run_workspace", run_mock)
    monkeypatch.setattr(insight_trace_cli, "load_scene_case", lambda test_module, case: context)
    monkeypatch.setattr(
        insight_trace_cli,
        "select_kernel",
        lambda selected_context, kernel_name, func_id, kernel_source: kernel,
    )
    monkeypatch.setattr(insight_trace_cli, "validate_single_task_kernel", lambda selected_kernel: None)
    monkeypatch.setattr(
        insight_trace_cli,
        "resolve_args",
        lambda selected_context, selected_kernel, arg_spec, dump_dir, dispatch_id: trace_args,
    )

    result = insight_trace_cli._run_simpler(args)

    assert result == run_result
    create_mock.assert_called_once()
    run_mock.assert_called_once()


def test_run_simpler_stops_after_workspace_for_dry_run(monkeypatch, tmp_path):
    context = load_scene_case(_a5_bgemm_module(), "default")
    kernel = select_kernel(context, kernel_name="GEMM")
    trace_args = resolve_args(
        None,
        None,
        _write_arg_spec(
            tmp_path,
            '{"args":['
            '{"kind":"tensor","index":0,"name":"A","dtype":"FLOAT32","shape":[131072]},'
            '{"kind":"tensor","index":1,"name":"B","dtype":"FLOAT32","shape":[131072]},'
            '{"kind":"tensor","index":2,"name":"C","dtype":"FLOAT32","shape":[131072]}'
            "]}",
        ),
    )
    args = Namespace(
        test_module=_a5_bgemm_module(),
        case="default",
        kernel="GEMM",
        func_id=None,
        kernel_source=None,
        platform="a5",
        runtime="tensormap_and_ringbuffer",
        output_dir=tmp_path,
        cann_home=None,
        pto_isa_root=None,
        soc_version=None,
        device=0,
        launch_count=1,
        timeout=120,
        hw_block_num=1,
        arg_spec=None,
        dump_dir=None,
        dispatch_id=None,
        dry_run=True,
        ptoas_root=None,
        source_cpp=None,
        kernel_base_name=None,
        aicore_arch=None,
        kernel_symbol=None,
        backend=TraceBackend.SIMPLER.value,
    )

    create_result = type("Result", (), {"workspace_dir": tmp_path, "simulator_dir": None})()
    create_mock = Mock(return_value=create_result)
    run_mock = Mock()
    monkeypatch.setattr(insight_trace_cli, "create_workspace", create_mock)
    monkeypatch.setattr(insight_trace_cli, "run_workspace", run_mock)
    monkeypatch.setattr(insight_trace_cli, "load_scene_case", lambda test_module, case: context)
    monkeypatch.setattr(
        insight_trace_cli,
        "select_kernel",
        lambda selected_context, kernel_name, func_id, kernel_source: kernel,
    )
    monkeypatch.setattr(insight_trace_cli, "validate_single_task_kernel", lambda selected_kernel: None)
    monkeypatch.setattr(
        insight_trace_cli,
        "resolve_args",
        lambda selected_context, selected_kernel, arg_spec, dump_dir, dispatch_id: trace_args,
    )

    result = insight_trace_cli._run_simpler(args)

    assert result == create_result
    create_mock.assert_called_once()
    run_mock.assert_not_called()


def test_loads_paged_attention_case():
    context = load_scene_case(_paged_attention_module(), "CaseSmall1")
    assert context.case["params"]["block_size"] == 16
    assert context.runtime == "tensormap_and_ringbuffer"


def test_selects_sf_as_aiv_only():
    context = load_scene_case(_paged_attention_module(), "CaseSmall1")
    kernel = select_kernel(context, kernel_name="SF")
    assert kernel.func_id == 1
    assert kernel.shape == KernelShape.AIV_ONLY


def test_resolves_a5_qk_recipe_includes_shape_scalars():
    context = load_scene_case(_a5_paged_attention_module(), "SmallCase1")
    kernel = select_kernel(context, kernel_name="QK")
    args = resolve_args(context, kernel)
    assert args[0] == TraceTensorArg(0, "qi", "BFLOAT16", (16, 16))
    assert args[1] == TraceTensorArg(1, "kj", "BFLOAT16", (16, 16))
    assert args[2] == TraceTensorArg(2, "sij", "FLOAT32", (16, 16))
    assert args[3] == TraceScalarArg(4, "head_dim", "UINT64", 16)
    assert args[4] == TraceScalarArg(5, "block_size", "UINT64", 16)


def test_resolves_generic_vector_example_args():
    context = load_scene_case(_vector_example_module(), "default")
    kernel = select_kernel(context, func_id=0)
    args = resolve_args(context, kernel)
    assert args == (
        TraceTensorArg(0, "a", "FLOAT32", (128 * 128,), role="input"),
        TraceTensorArg(1, "b", "FLOAT32", (128 * 128,), role="input"),
        TraceTensorArg(2, "f", "FLOAT32", (128 * 128,), role="output"),
    )


def test_load_kernel_dump_args_skips_context_slots(tmp_path):
    dump = tmp_path / "tensor_dump"
    dump.mkdir()
    (dump / "tensor_dump.json").write_text(
        """
        {
          "tensors": [
            {"arg_index": 2, "dtype": "FLOAT32", "shape": [16], "stage": "before_dispatch"},
            {"arg_index": 48, "dtype": "UINT64", "shape": [], "bin_size": 0, "stage": "before_dispatch"},
            {"arg_index": 0, "dtype": "BFLOAT16", "shape": [16, 16], "stage": "before_dispatch"},
            {"arg_index": 3, "dtype": "UINT64", "shape": [], "bin_size": 0, "stage": "before_dispatch"},
            {"arg_index": 49, "dtype": "UINT64", "shape": [], "bin_size": 0, "stage": "before_dispatch"},
            {"arg_index": 5, "dtype": "FLOAT32", "shape": [16], "stage": "after_dispatch"}
          ]
        }
        """
    )
    args = load_kernel_dump_args(tmp_path, task_id=None)
    assert args == (
        TraceTensorArg(0, "arg0", "BFLOAT16", (16, 16)),
        TraceTensorArg(2, "arg2", "FLOAT32", (16,)),
    )


def test_load_kernel_dump_args_requires_tensor_dump_json(tmp_path):
    (tmp_path / "other.json").write_text('{"tensors":[]}')
    with pytest.raises(ValueError, match="tensor_dump.json not found"):
        load_kernel_dump_args(tmp_path)


def test_resolves_spmd_mix_recipe(tmp_path):
    context = load_scene_case(
        Path(__file__).resolve().parents[3]
        / "tests/st/a2a3/tensormap_and_ringbuffer/spmd_paged_attention/test_spmd_paged_attention.py",
        "Case1",
    )
    kernel = select_kernel(context, kernel_name="PA_AIC")
    args = resolve_args(context, kernel)
    assert args[0] == TraceTensorArg(0, "query", "BFLOAT16", (256, 16, 128))
    assert args[6] == TraceTensorArg(6, "sij_fifo", "FLOAT32", (1,))
    assert args[9] == TraceScalarArg(9, "scale_value", "FLOAT32_BITS", 1065353216, "bits")
    assert args[15] == TraceScalarArg(15, "total_logical_blocks", "UINT64", 256)
    assert args[16] == TraceScalarArg(16, "q_tile", "UINT64", 16)


def test_resolves_a5_spmd_multiblock_mix_recipe():
    context = load_scene_case(_a5_spmd_multiblock_mix_module(), "Case1")
    kernel = select_kernel(context, kernel_name="SPMD_MIX_AIC")
    args = resolve_args(context, kernel)
    assert args == (
        TraceTensorArg(0, "output", "FLOAT32", (4512,), role="inout"),
        TraceScalarArg(1, "base_cl", "UINT64", 0),
    )


def test_render_a5_bgemm_uses_a5_platform_values(tmp_path):
    config = _a5_bgemm_config(tmp_path)
    kernel_rendered = render_kernel(config)
    launch_rendered = render_launch(config)
    host_rendered = render_host(config)
    workspace = create_workspace(config)

    assert "#define __CCE_AICORE__ 310" in kernel_rendered
    assert "#define PTO_NPU_ARCH_A5" in kernel_rendered
    assert "#define EVENT_ID7 ((::event_t)7)" in kernel_rendered
    assert "__gm__ int64_t *aic_args, __gm__ int64_t *aiv_args" in kernel_rendered
    assert "launch_replay(void *aic_args, void *aiv_args, void *stream)" in launch_rendered
    assert "std::vector<int64_t> aic_args(kHwBlocks * kArgsSlots, 0);" in host_rendered
    assert "std::vector<int64_t> aiv_args(kAivRows * kArgsSlots, 0);" in host_rendered
    assert "launch_replay(d_aic_args, d_aiv_args, stream);" in host_rendered
    assert "dav-c310" in (workspace.workspace_dir / "CMakeLists.txt").read_text()
    assert "src/a5/runtime/tensormap_and_ringbuffer/runtime" in (workspace.workspace_dir / "CMakeLists.txt").read_text()
    assert 'SOC_VERSION="${SOC_VERSION:-dav_3510}"' in (workspace.workspace_dir / "run_collect.sh").read_text()
    assert '"platform": "a5"' in (workspace.workspace_dir / "insight_trace_config.json").read_text()


def test_render_spmd_launch_uses_dual_arg_signature(tmp_path):
    rendered = render_launch(_spmd_config(tmp_path))
    assert "launch_replay(void *aic_args, void *aiv_args, void *stream)" in rendered
    assert "(__gm__ int64_t *)aic_args, (__gm__ int64_t *)aiv_args" in rendered


def test_render_spmd_host_includes_context_and_dual_rows(tmp_path):
    rendered = render_host(_spmd_config(tmp_path))
    assert '#include "intrinsic.h"' in rendered
    assert "std::vector<LocalContext> aic_local(kHwBlocks);" in rendered
    assert "std::vector<GlobalContext> aiv_global(kAivRows);" in rendered
    assert "std::vector<int64_t> aic_args(kHwBlocks * kArgsSlots, 0);" in rendered
    assert "std::vector<int64_t> aiv_args(kAivRows * kArgsSlots, 0);" in rendered
    assert (  # noqa: E501
        "row[48] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(d_aic_local) + r * sizeof(LocalContext));"
        in rendered
    )
    assert (  # noqa: E501
        "row[49] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(d_aiv_global) + r * sizeof(GlobalContext));"
        in rendered
    )


def test_render_a5_spmd_host_uses_dispatch_rows_and_a5_context_fields(tmp_path):
    rendered = render_host(_a5_spmd_config(tmp_path))
    assert "total_rows += dispatch.logical_block_num;" in rendered
    assert "aic_local[row_index].s_block_idx = block_idx;" in rendered
    assert "aic_local[row_index].s_block_num = dispatch.logical_block_num;" in rendered
    assert "aiv_global[lane_row_index].sub_block_id = lane;" in rendered
    assert "std::vector<int64_t> aic_args(total_rows * kArgsSlots, 0);" in rendered
    assert "std::vector<int64_t> aiv_args(total_rows * kAivLanesPerCore * kArgsSlots, 0);" in rendered
    assert "row[arg_index] = value;" in rendered
    assert "lane_row[arg_index] = value;" in rendered


def test_render_host_rejects_duplicate_arg_index(tmp_path):
    config = TraceConfig(
        backend=TraceBackend.SIMPLER,
        test_module=None,
        case_name="case",
        kernel_spec=None,
        args=(TraceScalarArg(1, "a", "UINT64", 1), TraceScalarArg(1, "b", "UINT64", 2)),
        output_dir=tmp_path,
        repo_root=tmp_path,
        cann_home=None,
        pto_isa_root=None,
    )
    with pytest.raises(ValueError, match="Duplicate arg index"):
        render_host(config)


def test_render_spmd_kernel_uses_dual_arg_wrapper(tmp_path):
    rendered = render_kernel(_spmd_config(tmp_path))
    assert "__gm__ int64_t *aic_args, __gm__ int64_t *aiv_args" in rendered
    assert "get_block_idx() * get_subblockdim() + get_subblockid()" in rendered
    assert "static_cast<uint64_t>(hw_idx) * 50" in rendered


def test_a5_spmd_meta_and_hw_block_num_are_configured():
    context = load_scene_case(_a5_spmd_multiblock_mix_module(), "Case1")
    kernel = select_kernel(context, kernel_name="SPMD_MIX_AIC")
    meta = insight_trace_cli._spmd_meta(kernel, _a5_platform())
    assert meta is not None
    assert meta.dispatches == (
        SPMDDispatch(2, ((1, 0),)),
        SPMDDispatch(8, ((1, 6),)),
        SPMDDispatch(12, ((1, 30),)),
        SPMDDispatch(24, ((1, 66),)),
        SPMDDispatch(48, ((1, 138),)),
    )
    hw_block_num = insight_trace_cli._hw_block_num(Namespace(platform="a5", hw_block_num=1), kernel)
    assert hw_block_num == 94


def _e2e_requirements_met():
    """Check if E2E test requirements are met."""
    if os.environ.get("SIMPLER_RUN_INSIGHT_E2E") != "1":
        return False, "SIMPLER_RUN_INSIGHT_E2E not set"
    if not os.environ.get("CANN_HOME"):
        return False, "CANN_HOME not set"
    if not os.environ.get("PTO_ISA_ROOT"):
        return False, "PTO_ISA_ROOT not set"
    if shutil.which("msprof") is None:
        return False, "msprof not found in PATH"
    if shutil.which("bash") is None:
        return False, "bash not found in PATH"
    return True, ""


@pytest.mark.skipif(not _e2e_requirements_met()[0], reason=_e2e_requirements_met()[1])
def test_spmd_e2e_full(tmp_path):
    """End-to-end SPMD mix insight trace.

    Skipped by default. Enable with:
        SIMPLER_RUN_INSIGHT_E2E=1 CANN_HOME=<path> PTO_ISA_ROOT=<path> pytest \\
            tests/ut/py/test_insight_trace_core.py::test_spmd_e2e_full

    Requires: CANN installed, msprof available, PTO_ISA_ROOT set.
    """

    # Absolute path to the real kernel source (matches existing _spmd_kernel)
    repo_root = Path(__file__).resolve().parents[3]
    kernel_source_rel = (
        "tests/st/a2a3/tensormap_and_ringbuffer/spmd_paged_attention/kernels/mix/paged_attention_parallel.cpp"
    )
    kernel_source = repo_root / kernel_source_rel

    # Minimal SceneTestCase module with a tiny SPMD case (batch=1, num_heads=1, head_dim=128, block_size=128)
    # so the simulator finishes quickly.
    module_content = textwrap.dedent(f"""\
        # Auto-generated for SPMD E2E test
        import torch
        from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test

        @scene_test(level=2, runtime="tensormap_and_ringbuffer")
        class TestSPMDE2E(SceneTestCase):
            CALLABLE = {{
                "incores": [
                    {{
                        "func_id": 0,
                        "name": "PA_AIC",
                        "source": "{kernel_source}",
                        "core_type": "aic",
                    }},
                    {{
                        "func_id": 1,
                        "name": "PA_AIV",
                        "source": "{kernel_source}",
                        "core_type": "aiv",
                    }},
                ],
            }}
            CASES = [
                {{
                    "name": "Tiny",
                    "platforms": ["a2a3"],
                    "config": {{"aicpu_thread_num": 1, "block_dim": 2}},
                    "params": {{
                        "batch": 1,
                        "num_heads": 1,
                        "kv_head_num": 1,
                        "head_dim": 128,
                        "block_size": 128,
                        "context_len": 128,
                        "max_model_len": 128,
                        "dtype": "bfloat16",
                    }},
                }},
            ]
            def generate_args(self, params):
                result = [
                    ("query", torch.empty(1, 1, 128, dtype=torch.bfloat16)),
                    ("key_cache", torch.empty(1, 128, 1, 128, dtype=torch.bfloat16)),
                    ("value_cache", torch.empty(1, 128, 1, 128, dtype=torch.bfloat16)),
                    ("block_table", torch.zeros(1, 1, dtype=torch.int32)),
                    ("context_lens", torch.tensor([128], dtype=torch.int32)),
                    ("out", torch.zeros(1, 1, 128, dtype=torch.float32)),
                    ("scale", torch.tensor(1.0)),
                ]
                specs = []
                for name, value in result:
                    if isinstance(value, torch.Tensor):
                        specs.append(Tensor(name, value))
                    else:
                        specs.append(Scalar(name, value))
                return TaskArgsBuilder(*specs)
        """)
    module_file = tmp_path / "test_spmd_e2e_module.py"
    module_file.write_text(module_content)

    arg_spec = tmp_path / "spmd_args.json"
    arg_spec.write_text(
        textwrap.dedent(
            """\
            {
              "args": [
                {"kind": "tensor", "index": 0, "name": "query", "dtype": "BFLOAT16", "shape": [1, 1, 128]},
                {"kind": "tensor", "index": 1, "name": "key_cache", "dtype": "BFLOAT16", "shape": [1, 128, 1, 128]},
                {"kind": "tensor", "index": 2, "name": "value_cache", "dtype": "BFLOAT16", "shape": [1, 128, 1, 128]},
                {"kind": "tensor", "index": 3, "name": "block_table", "dtype": "INT32", "shape": [1, 1]},
                {"kind": "tensor", "index": 4, "name": "context_lens", "dtype": "INT32", "shape": [1]},
                {"kind": "tensor", "index": 5, "name": "out", "dtype": "FLOAT32", "shape": [1, 1, 128]},
                {"kind": "tensor", "index": 6, "name": "sij_fifo", "dtype": "FLOAT32", "shape": [1]},
                {"kind": "tensor", "index": 7, "name": "pij_fifo", "dtype": "BFLOAT16", "shape": [1]},
                {"kind": "tensor", "index": 8, "name": "oi_fifo", "dtype": "FLOAT32", "shape": [1]},
                {"kind": "scalar", "index": 9, "name": "scale_value", "dtype": "FLOAT32_BITS",
                 "value": 1065353216, "pack_mode": "bits"},
                {"kind": "scalar", "index": 10, "name": "num_heads", "dtype": "UINT64", "value": 1},
                {"kind": "scalar", "index": 11, "name": "head_dim", "dtype": "UINT64", "value": 128},
                {"kind": "scalar", "index": 12, "name": "block_size", "dtype": "UINT64", "value": 128},
                {"kind": "scalar", "index": 13, "name": "max_num_blocks_per_req", "dtype": "UINT64", "value": 1},
                {"kind": "scalar", "index": 14, "name": "q_loop", "dtype": "UINT64", "value": 1},
                {"kind": "scalar", "index": 15, "name": "total_logical_blocks", "dtype": "UINT64", "value": 1},
                {"kind": "scalar", "index": 16, "name": "q_tile", "dtype": "UINT64", "value": 1}
              ]
            }
            """
        )
    )

    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()

    pto_isa_root = os.environ.get("PTO_ISA_ROOT")
    cann_home = os.environ.get("CANN_HOME")

    # Step 1: generate workspace
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    gen_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "simpler_setup.tools.insight_trace",
            str(module_file),
            "--case",
            "Tiny",
            "--kernel",
            "PA_AIC",
            "--output-dir",
            str(ws_dir),
            "--arg-spec",
            str(arg_spec),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert gen_result.returncode == 0, f"workspace generation failed: {gen_result.stderr}"

    assert (ws_dir / "replay_kernel.cpp").is_file()
    assert (ws_dir / "replay_launch.cpp").is_file()
    assert (ws_dir / "replay_host.cpp").is_file()
    assert (ws_dir / "run_collect.sh").is_file()

    # Step 2: run_collect.sh (build + collect + export)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    if pto_isa_root:
        env["PTO_ISA_ROOT"] = pto_isa_root
    if cann_home:
        env["CANN_HOME"] = cann_home
    env["REPO_ROOT"] = str(repo_root)

    collect_result = subprocess.run(
        ["bash", str(ws_dir / "run_collect.sh")],
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )
    assert collect_result.returncode == 0, f"collect failed: {collect_result.stderr}"

    # Step 3: validate exported artifacts
    export_root = ws_dir / "insight_export"
    opp_dirs = sorted((export_root).glob("OPPROF_*/simulator"))
    assert opp_dirs, f"No OPPROF simulator dir under {export_root}"
    sim_dir = opp_dirs[-1]

    assert (sim_dir / "trace.json").is_file(), "missing simulator/trace.json"
    assert (sim_dir / "visualize_data.bin").is_file(), "missing simulator/visualize_data.bin"
    csv_files = list(sim_dir.glob("core*/*instr_exe*.csv"))
    assert csv_files, "no instr_exe CSV files"
    assert len(csv_files) >= 3, f"Expected at least 3 CSV files (AIC + 2 AIV lanes), got {len(csv_files)}"

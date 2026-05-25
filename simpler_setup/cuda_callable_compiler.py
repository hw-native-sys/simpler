# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CUDA callable source generation helpers."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from textwrap import indent

from .environment import PROJECT_ROOT

_CUDA_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HOST_SCHEDULE_SOURCE_KIND = "task-body-wrapper"
_PERSISTENT_DAG_ENTRY_NAME = "pto_persistent_dag_f32_executor"
_PERSISTENT_DAG_SOURCE_KIND = "generated-dispatch"
_HOST_SCHEDULE_CACHE_RELATIVE_PATH = Path("build") / "cache" / "cuda" / "onboard" / "host_schedule"
_PERSISTENT_CACHE_RELATIVE_PATH = Path("build") / "cache" / "cuda" / "onboard" / "persistent_device"


@dataclass(frozen=True)
class CudaTaskBody:
    """CUDA PTO task body that can be wrapped for multiple runtimes."""

    name: str
    body: str
    context_type: str = "PtoTaskContext"


@dataclass(frozen=True)
class CudaTaskWrapperSource:
    """Generated CUDA wrappers for one PTO task body."""

    task_name: str
    body_name: str
    host_entry_name: str
    persistent_entry_name: str
    source: str


@dataclass(frozen=True)
class CudaHostScheduleCallableArtifact:
    """Compiled CUDA host-schedule callable artifact."""

    cache_key: str
    cache_hit: bool
    source_path: Path
    ptx_path: Path
    manifest_path: Path
    ptx: bytes
    entry_name: str
    persistent_entry_name: str
    arch: str
    source_kind: str


@dataclass(frozen=True)
class CudaPersistentTaskFunction:
    """Task function lowered into the persistent-device dispatch switch."""

    func_id: int
    name: str
    body: str


@dataclass(frozen=True)
class CudaPersistentCallableArtifact:
    """Compiled CUDA persistent-device callable artifact."""

    cache_key: str
    cache_hit: bool
    source_path: Path
    ptx_path: Path
    manifest_path: Path
    ptx: bytes
    entry_name: str
    arch: str
    source_kind: str


def render_cuda_task_wrappers(task_body: CudaTaskBody) -> CudaTaskWrapperSource:
    """Render host-schedule and persistent-device wrappers for one task body."""

    if not _CUDA_IDENTIFIER_RE.match(task_body.name):
        raise ValueError(f"invalid CUDA task body name: {task_body.name!r}")

    body_name = f"pto_task_body_{task_body.name}"
    host_entry_name = f"pto_kernel_{task_body.name}"
    persistent_entry_name = f"pto_task_{task_body.name}"
    rendered_body = indent(task_body.body.strip() or "(void)ctx;", "    ")
    source = f"""
__device__ void {body_name}({task_body.context_type} *ctx) {{
{rendered_body}
}}

extern "C" __global__ void {host_entry_name}({task_body.context_type} ctx) {{
    {body_name}(&ctx);
}}

__device__ void {persistent_entry_name}({task_body.context_type} *ctx) {{
    {body_name}(ctx);
}}
""".lstrip()
    return CudaTaskWrapperSource(
        task_name=task_body.name,
        body_name=body_name,
        host_entry_name=host_entry_name,
        persistent_entry_name=persistent_entry_name,
        source=source,
    )


def _validate_task_functions(task_functions: Sequence[CudaPersistentTaskFunction]) -> list[CudaPersistentTaskFunction]:
    if not task_functions:
        raise ValueError("at least one CUDA persistent task function is required")

    ordered = sorted(task_functions, key=lambda task: task.func_id)
    seen_func_ids: set[int] = set()
    seen_names: set[str] = set()
    for task in ordered:
        if task.func_id <= 0:
            raise ValueError(f"func_id must be positive: {task.func_id}")
        if task.func_id in seen_func_ids:
            raise ValueError(f"duplicate func_id: {task.func_id}")
        seen_func_ids.add(task.func_id)

        if not _CUDA_IDENTIFIER_RE.match(task.name):
            raise ValueError(f"invalid CUDA task function name: {task.name!r}")
        if task.name in seen_names:
            raise ValueError(f"duplicate CUDA task function name: {task.name}")
        seen_names.add(task.name)
    return ordered


def _render_task_function(task: CudaPersistentTaskFunction) -> str:
    body = indent(task.body.strip() or "(void)task;", "        ")
    return f"""
__device__ void pto_task_{task.name}(const PtoCudaPersistentDagTask *task) {{
    for (unsigned long long i = threadIdx.x; i < task->n; i += blockDim.x) {{
{body}
    }}
}}
""".strip()


def _render_dispatch(task_functions: Sequence[CudaPersistentTaskFunction]) -> str:
    cases = []
    for task in task_functions:
        cases.append(
            f"""
    case {task.func_id}U:
        pto_task_{task.name}(task);
        return;
""".rstrip()
        )
    rendered_cases = "\n".join(cases)
    return f"""
__device__ void pto_dag_dispatch(const PtoCudaPersistentDagTask *task) {{
    switch (task->func_id) {{
{rendered_cases}
    default:
        return;
    }}
}}
""".strip()


def render_persistent_dag_source(task_functions: Sequence[CudaPersistentTaskFunction]) -> str:
    """Render a persistent-device DAG executor with generated task dispatch."""

    ordered = _validate_task_functions(task_functions)
    rendered_tasks = "\n\n".join(_render_task_function(task) for task in ordered)
    rendered_dispatch = _render_dispatch(ordered)

    return f"""
struct PtoCudaPersistentDagTask {{
    unsigned int func_id;
    const float *a;
    const float *b;
    float *out;
    unsigned long long n;
    unsigned int dependent_begin;
    unsigned int dependent_count;
    unsigned int initial_fanin;
    unsigned int rows;
    unsigned int cols;
    unsigned int inner;
    unsigned int lda;
    unsigned int ldb;
    unsigned int ldc;
    unsigned long long a_batch_stride;
    unsigned long long b_batch_stride;
    unsigned long long out_batch_stride;
}};

struct PtoCudaPersistentDagState {{
    const PtoCudaPersistentDagTask *tasks;
    unsigned long long task_count;
    const unsigned int *dependents;
    unsigned int *fanin;
    unsigned int *ready_queue;
    unsigned int *ready_flags;
    unsigned int queue_capacity;
    unsigned int *queue_head;
    unsigned int *queue_tail;
    unsigned int *completed_count;
}};

__device__ void pto_dag_push_ready(const PtoCudaPersistentDagState *state, unsigned int task_id) {{
    unsigned int ticket = atomicAdd(state->queue_tail, 1U);
    unsigned int slot = ticket % state->queue_capacity;
    while (atomicAdd(&state->ready_flags[slot], 0U) != 0U) {{
    }}
    state->ready_queue[slot] = task_id;
    __threadfence();
    atomicExch(&state->ready_flags[slot], ticket + 1U);
}}

__device__ bool pto_dag_pop_ready(const PtoCudaPersistentDagState *state, unsigned int *task_id) {{
    unsigned int ticket = atomicAdd(state->queue_head, 1U);
    if (static_cast<unsigned long long>(ticket) >= state->task_count) {{
        return false;
    }}
    unsigned int slot = ticket % state->queue_capacity;
    unsigned int ready_value = ticket + 1U;
    while (atomicAdd(&state->ready_flags[slot], 0U) != ready_value) {{
    }}
    *task_id = state->ready_queue[slot];
    __threadfence();
    atomicExch(&state->ready_flags[slot], 0U);
    return true;
}}

{rendered_tasks}

{rendered_dispatch}

extern "C" __global__ void pto_persistent_dag_f32_executor(const PtoCudaPersistentDagState *state) {{
    __shared__ unsigned int task_id;
    __shared__ bool has_task;

    if (blockIdx.x == 0) {{
        if (threadIdx.x == 0) {{
            for (unsigned int idx = 0; static_cast<unsigned long long>(idx) < state->task_count; ++idx) {{
                if (state->tasks[idx].initial_fanin == 0U) {{
                    pto_dag_push_ready(state, idx);
                }}
            }}
        }}
        return;
    }}

    while (true) {{
        if (threadIdx.x == 0) {{
            has_task = pto_dag_pop_ready(state, &task_id);
        }}
        __syncthreads();
        if (!has_task) {{
            break;
        }}

        PtoCudaPersistentDagTask task = state->tasks[task_id];
        pto_dag_dispatch(&task);
        __syncthreads();

        if (threadIdx.x == 0) {{
            for (unsigned int idx = 0; idx < task.dependent_count; ++idx) {{
                unsigned int dependent_id = state->dependents[task.dependent_begin + idx];
                unsigned int old = atomicSub(&state->fanin[dependent_id], 1U);
                if (old == 1U) {{
                    pto_dag_push_ready(state, dependent_id);
                }}
            }}
            atomicAdd(state->completed_count, 1U);
        }}
    }}
}}
""".lstrip()


def _task_manifest(task_functions: Sequence[CudaPersistentTaskFunction]) -> list[dict[str, int | str]]:
    return [{"func_id": task.func_id, "name": task.name} for task in task_functions]


def _task_body_manifest(task_body: CudaTaskBody) -> dict[str, str]:
    return {"context_type": task_body.context_type, "name": task_body.name}


def _host_schedule_cache_key(source: str, task_body: CudaTaskBody, arch: str, entry_name: str) -> str:
    payload = {
        "arch": arch,
        "entry_name": entry_name,
        "generator": "cuda-host-schedule-v1",
        "source": source,
        "source_kind": _HOST_SCHEDULE_SOURCE_KIND,
        "task_body": _task_body_manifest(task_body),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _cache_key(source: str, task_functions: Sequence[CudaPersistentTaskFunction], arch: str) -> str:
    payload = {
        "arch": arch,
        "entry_name": _PERSISTENT_DAG_ENTRY_NAME,
        "generator": "cuda-persistent-device-v2",
        "source": source,
        "source_kind": _PERSISTENT_DAG_SOURCE_KIND,
        "task_functions": _task_manifest(task_functions),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _run_nvcc_ptx(source_path: Path, ptx_path: Path, arch: str, nvcc: str) -> None:
    result = subprocess.run(
        [nvcc, "--ptx", "-std=c++17", f"-arch={arch}", str(source_path), "-o", str(ptx_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvcc CUDA PTX compile failed:\n{result.stderr}")


def default_cuda_host_schedule_cache_root() -> Path:
    """Return the repo-local host-schedule callable cache root."""

    return PROJECT_ROOT / _HOST_SCHEDULE_CACHE_RELATIVE_PATH


def default_cuda_persistent_cache_root() -> Path:
    """Return the repo-local persistent-device callable cache root."""

    return PROJECT_ROOT / _PERSISTENT_CACHE_RELATIVE_PATH


def compile_cuda_host_schedule(
    task_body: CudaTaskBody,
    arch: str,
    cache_root: Path | None = None,
    nvcc: str = "nvcc",
) -> CudaHostScheduleCallableArtifact:
    """Compile or reuse a generated host-schedule CUDA callable PTX."""

    wrappers = render_cuda_task_wrappers(task_body)
    cache_key = _host_schedule_cache_key(wrappers.source, task_body, arch, wrappers.host_entry_name)
    resolved_cache_root = default_cuda_host_schedule_cache_root() if cache_root is None else Path(cache_root)
    cache_dir = resolved_cache_root / "callables" / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_path = cache_dir / "generated_host_wrapper.cu"
    ptx_path = cache_dir / "pto_callable.ptx"
    manifest_path = cache_dir / "pto_callable.json"
    source_path.write_text(wrappers.source)

    cache_hit = ptx_path.is_file() and manifest_path.is_file()
    if not cache_hit:
        _run_nvcc_ptx(source_path, ptx_path, arch, nvcc)

    ptx = ptx_path.read_bytes()
    manifest = {
        "arch": arch,
        "cache_key": cache_key,
        "entry_name": wrappers.host_entry_name,
        "persistent_entry_name": wrappers.persistent_entry_name,
        "ptx_path": ptx_path.name,
        "runtime": "host_schedule",
        "source_kind": _HOST_SCHEDULE_SOURCE_KIND,
        "source_path": source_path.name,
        "source_sha256": sha256(wrappers.source.encode("utf-8")).hexdigest(),
        "task_body": _task_body_manifest(task_body),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return CudaHostScheduleCallableArtifact(
        cache_key=cache_key,
        cache_hit=cache_hit,
        source_path=source_path,
        ptx_path=ptx_path,
        manifest_path=manifest_path,
        ptx=ptx,
        entry_name=wrappers.host_entry_name,
        persistent_entry_name=wrappers.persistent_entry_name,
        arch=arch,
        source_kind=_HOST_SCHEDULE_SOURCE_KIND,
    )


def compile_cuda_persistent_device(
    task_functions: Sequence[CudaPersistentTaskFunction],
    arch: str,
    cache_root: Path | None = None,
    nvcc: str = "nvcc",
) -> CudaPersistentCallableArtifact:
    """Compile or reuse a generated persistent-device CUDA callable PTX."""

    ordered = _validate_task_functions(task_functions)
    source = render_persistent_dag_source(ordered)
    cache_key = _cache_key(source, ordered, arch)
    resolved_cache_root = default_cuda_persistent_cache_root() if cache_root is None else Path(cache_root)
    cache_dir = resolved_cache_root / "callables" / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_path = cache_dir / "generated_dispatch.cu"
    ptx_path = cache_dir / "pto_callable.ptx"
    manifest_path = cache_dir / "pto_callable.json"
    source_path.write_text(source)

    cache_hit = ptx_path.is_file() and manifest_path.is_file()
    if not cache_hit:
        _run_nvcc_ptx(source_path, ptx_path, arch, nvcc)

    ptx = ptx_path.read_bytes()
    manifest = {
        "arch": arch,
        "cache_key": cache_key,
        "entry_name": _PERSISTENT_DAG_ENTRY_NAME,
        "ptx_path": ptx_path.name,
        "runtime": "persistent_device",
        "source_kind": _PERSISTENT_DAG_SOURCE_KIND,
        "source_path": source_path.name,
        "source_sha256": sha256(source.encode("utf-8")).hexdigest(),
        "task_functions": _task_manifest(ordered),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return CudaPersistentCallableArtifact(
        cache_key=cache_key,
        cache_hit=cache_hit,
        source_path=source_path,
        ptx_path=ptx_path,
        manifest_path=manifest_path,
        ptx=ptx,
        entry_name=_PERSISTENT_DAG_ENTRY_NAME,
        arch=arch,
        source_kind=_PERSISTENT_DAG_SOURCE_KIND,
    )

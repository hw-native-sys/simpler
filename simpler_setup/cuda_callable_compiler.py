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

import ctypes
import json
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from textwrap import indent
from typing import Any, Union

from .environment import PROJECT_ROOT

_CUDA_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HOST_SCHEDULE_SOURCE_KIND = "task-body-wrapper"
_PERSISTENT_DAG_ENTRY_NAME = "pto_persistent_dag_f32_executor"
_PERSISTENT_DAG_SOURCE_KIND = "generated-dispatch"
_HOST_SCHEDULE_CACHE_RELATIVE_PATH = Path("build") / "cache" / "cuda" / "onboard" / "host_schedule"
_PERSISTENT_CACHE_RELATIVE_PATH = Path("build") / "cache" / "cuda" / "onboard" / "persistent_device"
_CUDA_HOST_OP_VECTOR_ADD_F32 = 1
_CUDA_PERSISTENT_OP_DAG_F32_RING = 1003


@dataclass(frozen=True)
class CudaTaskBody:
    """CUDA PTO task body that can be wrapped for multiple runtimes."""

    name: str
    body: str
    context_type: str = "PtoTaskContext"
    context_definition: str = ""
    host_parameters: tuple[str, ...] = ()
    host_context_initializer: str = ""


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
class CudaPersistentTaskBodyFunction:
    """Shared task body lowered into the persistent-device dispatch switch."""

    func_id: int
    task_body: CudaTaskBody


CudaPersistentFunctionSpec = Union[CudaPersistentTaskFunction, CudaPersistentTaskBodyFunction]  # noqa: UP007


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


class CudaHostScheduleCallable(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("op", ctypes.c_uint32),
        ("image", ctypes.c_void_p),
        ("image_size", ctypes.c_size_t),
        ("entry_name", ctypes.c_char_p),
        ("grid_dim", ctypes.c_uint32),
        ("block_dim", ctypes.c_uint32),
        ("shared_mem_bytes", ctypes.c_size_t),
        ("stream_id", ctypes.c_uint32),
    ]


class CudaVectorAddArgs(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_void_p),
        ("b", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("n", ctypes.c_uint64),
    ]

    def buffer_ptr(self) -> int:
        return ctypes.addressof(self)

    def buffer_size(self) -> int:
        return ctypes.sizeof(self)


class CudaPersistentDeviceCallable(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("op", ctypes.c_uint32),
        ("image", ctypes.c_void_p),
        ("image_size", ctypes.c_size_t),
        ("entry_name", ctypes.c_char_p),
        ("grid_dim", ctypes.c_uint32),
        ("block_dim", ctypes.c_uint32),
        ("shared_mem_bytes", ctypes.c_size_t),
    ]


class CudaPersistentDagTask(ctypes.Structure):
    _fields_ = [
        ("func_id", ctypes.c_uint32),
        ("a", ctypes.c_void_p),
        ("b", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("n", ctypes.c_uint64),
        ("dependent_begin", ctypes.c_uint32),
        ("dependent_count", ctypes.c_uint32),
        ("initial_fanin", ctypes.c_uint32),
        ("rows", ctypes.c_uint32),
        ("cols", ctypes.c_uint32),
        ("inner", ctypes.c_uint32),
        ("lda", ctypes.c_uint32),
        ("ldb", ctypes.c_uint32),
        ("ldc", ctypes.c_uint32),
        ("a_batch_stride", ctypes.c_uint64),
        ("b_batch_stride", ctypes.c_uint64),
        ("out_batch_stride", ctypes.c_uint64),
    ]


class CudaPersistentDagState(ctypes.Structure):
    _fields_ = [
        ("tasks", ctypes.c_void_p),
        ("task_count", ctypes.c_uint64),
        ("dependents", ctypes.c_void_p),
        ("fanin", ctypes.c_void_p),
        ("ready_queue", ctypes.c_void_p),
        ("ready_flags", ctypes.c_void_p),
        ("queue_capacity", ctypes.c_uint32),
        ("queue_head", ctypes.c_void_p),
        ("queue_tail", ctypes.c_void_p),
        ("completed_count", ctypes.c_void_p),
    ]


class CudaPersistentDagArgs(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_void_p),
    ]

    def buffer_ptr(self) -> int:
        return ctypes.addressof(self)

    def buffer_size(self) -> int:
        return ctypes.sizeof(self)


@dataclass(frozen=True)
class PreparedCudaCallable:
    """ctypes callable manifest plus buffers it points into."""

    runtime: str
    artifact: CudaHostScheduleCallableArtifact | CudaPersistentCallableArtifact
    manifest: Any
    image_buffer: Any
    entry_name_buffer: Any

    def byref(self) -> Any:
        return ctypes.byref(self.manifest)

    def buffer_ptr(self) -> int:
        return ctypes.addressof(self.manifest)

    def buffer_size(self) -> int:
        return ctypes.sizeof(self.manifest)

    def to_bytes(self) -> bytes:
        return ctypes.string_at(self.buffer_ptr(), self.buffer_size())


def _create_c_string_buffer(value: bytes) -> Any:
    data = value if value.endswith(b"\0") else value + b"\0"
    return ctypes.create_string_buffer(data, len(data))


def prepare_cuda_host_schedule_callable(
    artifact: CudaHostScheduleCallableArtifact,
    *,
    grid_dim: int,
    block_dim: int,
    shared_mem_bytes: int = 0,
    stream_id: int = 0,
    op: int = _CUDA_HOST_OP_VECTOR_ADD_F32,
) -> PreparedCudaCallable:
    """Build a host-schedule `prepare_callable` manifest from an artifact."""

    image_buffer = _create_c_string_buffer(artifact.ptx)
    entry_name_buffer = _create_c_string_buffer(artifact.entry_name.encode("utf-8"))
    manifest = CudaHostScheduleCallable(
        version=2,
        op=op,
        image=ctypes.cast(image_buffer, ctypes.c_void_p),
        image_size=ctypes.sizeof(image_buffer),
        entry_name=ctypes.cast(entry_name_buffer, ctypes.c_char_p),
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=shared_mem_bytes,
        stream_id=stream_id,
    )
    return PreparedCudaCallable(
        runtime="host_schedule",
        artifact=artifact,
        manifest=manifest,
        image_buffer=image_buffer,
        entry_name_buffer=entry_name_buffer,
    )


def prepare_cuda_persistent_device_callable(
    artifact: CudaPersistentCallableArtifact,
    *,
    grid_dim: int,
    block_dim: int,
    shared_mem_bytes: int = 0,
    op: int = _CUDA_PERSISTENT_OP_DAG_F32_RING,
) -> PreparedCudaCallable:
    """Build a persistent-device `prepare_callable` manifest from an artifact."""

    image_buffer = _create_c_string_buffer(artifact.ptx)
    entry_name_buffer = _create_c_string_buffer(artifact.entry_name.encode("utf-8"))
    manifest = CudaPersistentDeviceCallable(
        version=1,
        op=op,
        image=ctypes.cast(image_buffer, ctypes.c_void_p),
        image_size=ctypes.sizeof(image_buffer),
        entry_name=ctypes.cast(entry_name_buffer, ctypes.c_char_p),
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=shared_mem_bytes,
    )
    return PreparedCudaCallable(
        runtime="persistent_device",
        artifact=artifact,
        manifest=manifest,
        image_buffer=image_buffer,
        entry_name_buffer=entry_name_buffer,
    )


def render_cuda_task_wrappers(
    task_body: CudaTaskBody,
    *,
    include_context_definition: bool = True,
    include_host_wrapper: bool = True,
) -> CudaTaskWrapperSource:
    """Render host-schedule and persistent-device wrappers for one task body."""

    if not _CUDA_IDENTIFIER_RE.match(task_body.name):
        raise ValueError(f"invalid CUDA task body name: {task_body.name!r}")

    body_name = f"pto_task_body_{task_body.name}"
    host_entry_name = f"pto_kernel_{task_body.name}"
    persistent_entry_name = f"pto_task_{task_body.name}"
    rendered_body = indent(task_body.body.strip() or "(void)ctx;", "    ")
    context_definition = task_body.context_definition.strip()
    preamble = f"{context_definition}\n\n" if include_context_definition and context_definition else ""
    if not include_host_wrapper:
        host_wrapper = ""
    elif task_body.host_parameters:
        host_parameters = ", ".join(param.strip() for param in task_body.host_parameters)
        if not all(param.strip() for param in task_body.host_parameters):
            raise ValueError("CUDA host wrapper parameters must be non-empty")
        initializer = task_body.host_context_initializer.strip()
        ctx_initializer = f"{{{initializer}}}" if initializer else "{}"
        host_wrapper = f"""
extern "C" __global__ void {host_entry_name}({host_parameters}) {{
    {task_body.context_type} ctx{ctx_initializer};
    {body_name}(&ctx);
}}
""".strip()
    else:
        host_wrapper = f"""
extern "C" __global__ void {host_entry_name}({task_body.context_type} ctx) {{
    {body_name}(&ctx);
}}
""".strip()
    host_wrapper_block = f"{host_wrapper}\n\n" if host_wrapper else ""
    source = f"""
{preamble}
__device__ void {body_name}({task_body.context_type} *ctx) {{
{rendered_body}
}}

{host_wrapper_block}__device__ void {persistent_entry_name}({task_body.context_type} *ctx) {{
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


def _persistent_func_id(task: CudaPersistentFunctionSpec) -> int:
    return task.func_id


def _persistent_task_name(task: CudaPersistentFunctionSpec) -> str:
    if isinstance(task, CudaPersistentTaskBodyFunction):
        return task.task_body.name
    return task.name


def _validate_task_functions(task_functions: Sequence[CudaPersistentFunctionSpec]) -> list[CudaPersistentFunctionSpec]:
    if not task_functions:
        raise ValueError("at least one CUDA persistent task function is required")

    ordered = sorted(task_functions, key=_persistent_func_id)
    seen_func_ids: set[int] = set()
    seen_names: set[str] = set()
    for task in ordered:
        func_id = _persistent_func_id(task)
        task_name = _persistent_task_name(task)
        if func_id <= 0:
            raise ValueError(f"func_id must be positive: {func_id}")
        if func_id in seen_func_ids:
            raise ValueError(f"duplicate func_id: {func_id}")
        seen_func_ids.add(func_id)

        if not _CUDA_IDENTIFIER_RE.match(task_name):
            raise ValueError(f"invalid CUDA task function name: {task_name!r}")
        if task_name in seen_names:
            raise ValueError(f"duplicate CUDA task function name: {task_name}")
        seen_names.add(task_name)
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


def _render_task_body_function(task: CudaPersistentTaskBodyFunction) -> str:
    wrappers = render_cuda_task_wrappers(
        task.task_body,
        include_context_definition=False,
        include_host_wrapper=False,
    )
    adapter_name = f"pto_dag_task_{task.task_body.name}"
    return f"""
{wrappers.source.strip()}

__device__ void {adapter_name}(const PtoCudaPersistentDagTask *task) {{
    for (unsigned long long i = threadIdx.x; i < task->n; i += blockDim.x) {{
        {task.task_body.context_type} ctx{{task, i}};
        {wrappers.persistent_entry_name}(&ctx);
    }}
}}
""".strip()


def _render_persistent_function(task: CudaPersistentFunctionSpec) -> str:
    if isinstance(task, CudaPersistentTaskBodyFunction):
        return _render_task_body_function(task)
    return _render_task_function(task)


def _persistent_dispatch_entry(task: CudaPersistentFunctionSpec) -> str:
    if isinstance(task, CudaPersistentTaskBodyFunction):
        return f"pto_dag_task_{task.task_body.name}"
    return f"pto_task_{task.name}"


def _render_dispatch(task_functions: Sequence[CudaPersistentFunctionSpec]) -> str:
    cases = []
    for task in task_functions:
        cases.append(
            f"""
    case {_persistent_func_id(task)}U:
        {_persistent_dispatch_entry(task)}(task);
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


def render_persistent_dag_source(task_functions: Sequence[CudaPersistentFunctionSpec]) -> str:
    """Render a persistent-device DAG executor with generated task dispatch."""

    ordered = _validate_task_functions(task_functions)
    context_definitions = []
    seen_context_definitions: set[str] = set()
    for task in ordered:
        if not isinstance(task, CudaPersistentTaskBodyFunction):
            continue
        context_definition = task.task_body.context_definition.strip()
        if context_definition and context_definition not in seen_context_definitions:
            seen_context_definitions.add(context_definition)
            context_definitions.append(context_definition)
    rendered_context_definitions = "\n\n".join(context_definitions)
    rendered_context_block = f"{rendered_context_definitions}\n\n" if rendered_context_definitions else ""
    rendered_tasks = "\n\n".join(_render_persistent_function(task) for task in ordered)
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

    {rendered_context_block}\
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


def _task_manifest(task_functions: Sequence[CudaPersistentFunctionSpec]) -> list[dict[str, int | str]]:
    return [{"func_id": _persistent_func_id(task), "name": _persistent_task_name(task)} for task in task_functions]


def _task_body_function_manifest(task_functions: Sequence[CudaPersistentFunctionSpec]) -> list[dict[str, object]]:
    return [
        {"func_id": task.func_id, "task_body": _task_body_manifest(task.task_body)}
        for task in task_functions
        if isinstance(task, CudaPersistentTaskBodyFunction)
    ]


def _task_body_manifest(task_body: CudaTaskBody) -> dict[str, object]:
    manifest: dict[str, object] = {"context_type": task_body.context_type, "name": task_body.name}
    if task_body.context_definition:
        manifest["context_definition_sha256"] = sha256(task_body.context_definition.encode("utf-8")).hexdigest()
    if task_body.host_parameters:
        manifest["host_parameters"] = list(task_body.host_parameters)
    if task_body.host_context_initializer:
        manifest["host_context_initializer"] = task_body.host_context_initializer
    return manifest


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


def _cache_key(source: str, task_functions: Sequence[CudaPersistentFunctionSpec], arch: str) -> str:
    payload = {
        "arch": arch,
        "entry_name": _PERSISTENT_DAG_ENTRY_NAME,
        "generator": "cuda-persistent-device-v2",
        "source": source,
        "source_kind": _PERSISTENT_DAG_SOURCE_KIND,
        "task_body_functions": _task_body_function_manifest(task_functions),
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
    task_functions: Sequence[CudaPersistentFunctionSpec],
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
        "task_body_functions": _task_body_function_manifest(ordered),
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

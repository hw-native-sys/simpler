# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CUDA integration coverage for the SceneTestCase L2 path."""

from __future__ import annotations

import ctypes
import json
from typing import Any, cast

import pytest
from simpler.task_interface import ArgDirection

from simpler_setup.cuda_callable_compiler import (
    CudaHostScheduleCallableArtifact,
    CudaPersistentCallableArtifact,
    CudaVectorAffineArgs,
    CudaVectorAxpyArgs,
    CudaVectorGenericArgs,
    CudaVectorQuaternaryArgs,
    CudaVectorScaleArgs,
    CudaVectorTernaryArgs,
    CudaVectorUnaryArgs,
    PreparedCudaCallable,
)
from simpler_setup.cuda_preflight import cuda_skip_reason
from simpler_setup.scene_test import (
    Scalar,
    SceneTestCase,
    TaskArgsBuilder,
    Tensor,
    _build_cuda_host_schedule_args,
    _compile_chip_callable_from_spec,
    _CudaPersistentDagSceneBuffers,
    scene_test,
)

_CUDA_SKIP_REASON = cuda_skip_reason(require_nvcc=True)
requires_cuda = pytest.mark.skipif(_CUDA_SKIP_REASON is not None, reason=_CUDA_SKIP_REASON or "")

_VECTOR_ADD_BODY = """
unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < ctx->n) {
    ctx->out[i] = ctx->a[i] + ctx->b[i];
}
""".lstrip()

_VECTOR_MUL_BODY = """
unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < ctx->n) {
    ctx->out[i] = ctx->a[i] * ctx->b[i];
}
""".lstrip()

_VECTOR_SCALE_BODY = """
unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < ctx->n) {
    ctx->out[i] = ctx->a[i] * ctx->alpha;
}
""".lstrip()

_VECTOR_AXPY_BODY = """
unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < ctx->n) {
    ctx->out[i] = ctx->alpha * ctx->a[i] + ctx->b[i];
}
""".lstrip()

_VECTOR_AFFINE_BODY = """
unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < ctx->n) {
    ctx->out[i] = ctx->alpha * ctx->a[i] + ctx->beta * ctx->b[i];
}
""".lstrip()

_VECTOR_TRIAD_BODY = """
unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < ctx->n) {
    ctx->out[i] = ctx->a[i] * ctx->b[i] + ctx->c[i];
}
""".lstrip()

_VECTOR_QUAD_BODY = """
unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < ctx->n) {
    ctx->out[i] = ctx->a[i] * ctx->b[i] + ctx->c[i] * ctx->d[i];
}
""".lstrip()

_VECTOR_GENERIC_ARGS_BODY = """
unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < ctx->n) {
    ctx->out[i] = ctx->scalar0 * ctx->a[i] +
                  ctx->tensor0[i] +
                  ctx->scalar1 * ctx->tensor1[i] +
                  ctx->b[i];
}
""".lstrip()

_VECTOR_GENERIC_ARGS4_BODY = """
unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < ctx->n) {
    ctx->out[i] = ctx->scalar0 * ctx->a[i] +
                  ctx->tensor0[i] +
                  ctx->scalar1 * ctx->tensor1[i] +
                  ctx->scalar2 * ctx->tensor2[i] +
                  ctx->scalar3 * ctx->tensor3[i] +
                  ctx->b[i];
}
""".lstrip()

_VECTOR_SQUARE_BODY = """
unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < ctx->n) {
    ctx->out[i] = ctx->a[i] * ctx->a[i];
}
""".lstrip()

_VECTOR_ADD_CONTEXT = """
struct PtoTaskContext {
    const float *a;
    const float *b;
    float *out;
    unsigned long long n;
};
""".strip()

_VECTOR_SCALE_CONTEXT = """
struct PtoTaskContext {
    const float *a;
    float *out;
    float alpha;
    unsigned long long n;
};
""".strip()

_VECTOR_UNARY_CONTEXT = """
struct PtoTaskContext {
    const float *a;
    float *out;
    unsigned long long n;
};
""".strip()

_VECTOR_AXPY_CONTEXT = """
struct PtoTaskContext {
    const float *a;
    const float *b;
    float *out;
    float alpha;
    unsigned long long n;
};
""".strip()

_VECTOR_AFFINE_CONTEXT = """
struct PtoTaskContext {
    const float *a;
    const float *b;
    float *out;
    float alpha;
    float beta;
    unsigned long long n;
};
""".strip()

_VECTOR_TRIAD_CONTEXT = """
struct PtoTaskContext {
    const float *a;
    const float *b;
    const float *c;
    float *out;
    unsigned long long n;
};
""".strip()

_VECTOR_QUAD_CONTEXT = """
struct PtoTaskContext {
    const float *a;
    const float *b;
    const float *c;
    const float *d;
    float *out;
    unsigned long long n;
};
""".strip()

_VECTOR_GENERIC_ARGS_CONTEXT = """
struct PtoTaskContext {
    const float *a;
    const float *b;
    float *out;
    const float *tensor0;
    const float *tensor1;
    float scalar0;
    float scalar1;
    unsigned long long n;
};
""".strip()

_VECTOR_GENERIC_ARGS4_CONTEXT = """
struct PtoTaskContext {
    const float *a;
    const float *b;
    float *out;
    const float *tensor0;
    const float *tensor1;
    const float *tensor2;
    const float *tensor3;
    float scalar0;
    float scalar1;
    float scalar2;
    float scalar3;
    unsigned long long n;
};
""".strip()

_VECTOR_ADD_HOST_PARAMS = (
    "const float *a",
    "const float *b",
    "float *out",
    "unsigned long long n",
)

_VECTOR_SCALE_HOST_PARAMS = (
    "const float *a",
    "float *out",
    "float alpha",
    "unsigned long long n",
)

_VECTOR_UNARY_HOST_PARAMS = (
    "const float *a",
    "float *out",
    "unsigned long long n",
)

_VECTOR_AXPY_HOST_PARAMS = (
    "const float *a",
    "const float *b",
    "float *out",
    "float alpha",
    "unsigned long long n",
)

_VECTOR_AFFINE_HOST_PARAMS = (
    "const float *a",
    "const float *b",
    "float *out",
    "float alpha",
    "float beta",
    "unsigned long long n",
)

_VECTOR_TRIAD_HOST_PARAMS = (
    "const float *a",
    "const float *b",
    "const float *c",
    "float *out",
    "unsigned long long n",
)

_VECTOR_QUAD_HOST_PARAMS = (
    "const float *a",
    "const float *b",
    "const float *c",
    "const float *d",
    "float *out",
    "unsigned long long n",
)

_VECTOR_GENERIC_ARGS_HOST_PARAMS = (
    "const float *a",
    "const float *b",
    "float *out",
    "const float *tensor0",
    "const float *tensor1",
    "float scalar0",
    "float scalar1",
    "unsigned long long n",
)

_VECTOR_GENERIC_ARGS4_HOST_PARAMS = (
    "const float *a",
    "const float *b",
    "float *out",
    "const float *tensor0",
    "const float *tensor1",
    "const float *tensor2",
    "const float *tensor3",
    "float scalar0",
    "float scalar1",
    "float scalar2",
    "float scalar3",
    "unsigned long long n",
)

_PERSISTENT_CONTEXT = """
struct PtoTaskContext {
    const PtoCudaPersistentDagTask *task;
    unsigned long long i;
};
""".strip()

_PERSISTENT_ADD_BODY = """
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->a[i] + task->b[i];
""".strip()

_PERSISTENT_MUL_BODY = """
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->a[i] * task->b[i];
""".strip()

_PERSISTENT_SCALE_BODY = """
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->scalar0 * task->a[i];
""".strip()

_PERSISTENT_AXPY_BODY = """
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->scalar0 * task->a[i] + task->b[i];
""".strip()

_PERSISTENT_AFFINE_BODY = """
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->scalar0 * task->a[i] + task->scalar1 * task->b[i];
""".strip()

_PERSISTENT_TRIAD_BODY = """
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->a[i] * task->b[i] + task->c[i];
""".strip()

_PERSISTENT_QUAD_BODY = """
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->a[i] * task->b[i] + task->c[i] * task->d[i];
""".strip()

_PERSISTENT_GENERIC_ARGS_BODY = """
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
if (task->tensor_arg_count < 2U || task->scalar_arg_count < 2U) {
  return;
}
task->out[i] = task->scalar_args[0] * task->a[i] +
               task->tensor_args[0][i] +
               task->scalar_args[1] * task->tensor_args[1][i];
if (task->tensor_arg_count >= 4U && task->scalar_arg_count >= 4U) {
  task->out[i] += task->scalar_args[2] * task->tensor_args[2][i] +
                  task->scalar_args[3] * task->tensor_args[3][i];
}
""".strip()

_PERSISTENT_SQUARE_BODY = """
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->a[i] * task->a[i];
""".strip()

_PERSISTENT_MATMUL_TILE_BODY = """
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
unsigned long long rows = static_cast<unsigned long long>(task->rows);
unsigned long long cols = static_cast<unsigned long long>(task->cols);
unsigned long long inner = static_cast<unsigned long long>(task->inner);
if (rows == 0ULL || cols == 0ULL || inner == 0ULL) {
  return;
}
unsigned long long matrix_elems = rows * cols;
unsigned long long tile_id = i / matrix_elems;
unsigned long long elem = i % matrix_elems;
unsigned long long row = elem / cols;
unsigned long long col = elem % cols;
unsigned long long a_base = tile_id * task->a_batch_stride;
unsigned long long b_base = tile_id * task->b_batch_stride;
unsigned long long out_base = tile_id * task->out_batch_stride;
float acc = 0.0f;
for (unsigned long long k = 0; k < inner; ++k) {
  acc += task->a[a_base + row * task->lda + k] * task->b[b_base + k * task->ldb + col];
}
task->out[out_base + row * task->ldc + col] = acc;
""".strip()

_PERSISTENT_WMMA_TILE_BODY = """
if ((task->rows % 16U) != 0U || (task->cols % 16U) != 0U || task->inner == 0U || (task->inner % 8U) != 0U) {
  return;
}
using namespace nvcuda;
unsigned long long tile_count = task->n / task->out_batch_stride;
for (unsigned long long tile_id = 0; tile_id < tile_count; ++tile_id) {
  unsigned long long a_base = tile_id * task->a_batch_stride;
  unsigned long long b_base = tile_id * task->b_batch_stride;
  unsigned long long out_base = tile_id * task->out_batch_stride;
  for (unsigned int row = 0; row < task->rows; row += 16U) {
    for (unsigned int col = 0; col < task->cols; col += 16U) {
      wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
      wmma::fill_fragment(acc_frag, 0.0f);
      for (unsigned int k = 0; k < task->inner; k += 8U) {
        wmma::load_matrix_sync(a_frag, task->a + a_base + row * task->lda + k, task->lda);
        wmma::load_matrix_sync(b_frag, task->b + b_base + k * task->ldb + col, task->ldb);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
      wmma::store_matrix_sync(task->out + out_base + row * task->ldc + col, acc_frag, task->ldc, wmma::mem_row_major);
    }
  }
}
""".strip()


class _FakeTensor:
    def __init__(self, n):
        self._n = n

    def numel(self):
        return self._n

    def element_size(self):
        return 4

    def is_contiguous(self):
        return True


class _CtypesFloatTensor:
    def __init__(self, values):
        self.values = list(values)
        array_t = ctypes.c_float * len(self.values)
        self.array = array_t(*self.values)
        self.device = None

    def numel(self):
        return len(self.values)

    def element_size(self):
        return ctypes.sizeof(ctypes.c_float)

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return ctypes.addressof(self.array)

    def to_list(self):
        return [float(value) for value in self.array]


class _FakeCudaBuffers:
    ptrs = {"a": 101, "b": 202, "c": 404, "d": 505, "e": 606, "f": 707, "out": 303}


class _FakeWorker:
    def __init__(self):
        self.next_ptr = 1000
        self.copy_to_calls = []
        self.freed = []

    def malloc(self, nbytes):
        ptr = self.next_ptr
        self.next_ptr += max(int(nbytes), 1) + 64
        return ptr

    def copy_to(self, dst, src, nbytes):
        self.copy_to_calls.append((dst, src, nbytes))

    def free(self, ptr):
        self.freed.append(ptr)


def _cuda_vector_add_spec(source, *, arch="compute_80", grid_dim=4, block_dim=256):
    return {
        "cuda": {
            "source": str(source),
            "task_name": "vector_add",
            "arch": arch,
            "context_definition": _VECTOR_ADD_CONTEXT,
            "host_parameters": _VECTOR_ADD_HOST_PARAMS,
            "host_context_initializer": "a, b, out, n",
            "grid_dim": grid_dim,
            "block_dim": block_dim,
            "signature": [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "vector_add_f32",
            "args": ["a", "b", "out"],
        }
    }


def _cuda_elementwise_binary_spec(source, *, task_name, arch="compute_80", grid_dim=4, block_dim=256):
    return {
        "cuda": {
            "source": str(source),
            "task_name": task_name,
            "arch": arch,
            "context_definition": _VECTOR_ADD_CONTEXT,
            "host_parameters": _VECTOR_ADD_HOST_PARAMS,
            "host_context_initializer": "a, b, out, n",
            "grid_dim": grid_dim,
            "block_dim": block_dim,
            "signature": [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "elementwise_binary_f32",
            "args": ["a", "b", "out"],
        }
    }


def _cuda_elementwise_scale_spec(source, *, task_name, arch="compute_80", grid_dim=4, block_dim=256):
    return {
        "cuda": {
            "source": str(source),
            "task_name": task_name,
            "arch": arch,
            "context_definition": _VECTOR_SCALE_CONTEXT,
            "host_parameters": _VECTOR_SCALE_HOST_PARAMS,
            "host_context_initializer": "a, out, alpha, n",
            "grid_dim": grid_dim,
            "block_dim": block_dim,
            "op": 2,
            "signature": [ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "elementwise_scale_f32",
            "args": ["a", "out", "alpha"],
        }
    }


def _cuda_elementwise_unary_spec(source, *, task_name, arch="compute_80", grid_dim=4, block_dim=256):
    return {
        "cuda": {
            "source": str(source),
            "task_name": task_name,
            "arch": arch,
            "context_definition": _VECTOR_UNARY_CONTEXT,
            "host_parameters": _VECTOR_UNARY_HOST_PARAMS,
            "host_context_initializer": "a, out, n",
            "grid_dim": grid_dim,
            "block_dim": block_dim,
            "op": 4,
            "signature": [ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "elementwise_unary_f32",
            "args": ["a", "out"],
        }
    }


def _cuda_elementwise_axpy_spec(source, *, task_name, arch="compute_80", grid_dim=4, block_dim=256):
    return {
        "cuda": {
            "source": str(source),
            "task_name": task_name,
            "arch": arch,
            "context_definition": _VECTOR_AXPY_CONTEXT,
            "host_parameters": _VECTOR_AXPY_HOST_PARAMS,
            "host_context_initializer": "a, b, out, alpha, n",
            "grid_dim": grid_dim,
            "block_dim": block_dim,
            "op": 3,
            "signature": [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "elementwise_axpy_f32",
            "args": ["a", "b", "out", "alpha"],
        }
    }


def _cuda_elementwise_affine_spec(source, *, task_name, arch="compute_80", grid_dim=4, block_dim=256):
    return {
        "cuda": {
            "source": str(source),
            "task_name": task_name,
            "arch": arch,
            "context_definition": _VECTOR_AFFINE_CONTEXT,
            "host_parameters": _VECTOR_AFFINE_HOST_PARAMS,
            "host_context_initializer": "a, b, out, alpha, beta, n",
            "grid_dim": grid_dim,
            "block_dim": block_dim,
            "op": 5,
            "signature": [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "elementwise_affine_f32",
            "args": ["a", "b", "out", "alpha", "beta"],
        }
    }


def _cuda_elementwise_triad_spec(source, *, task_name, arch="compute_80", grid_dim=4, block_dim=256):
    return {
        "cuda": {
            "source": str(source),
            "task_name": task_name,
            "arch": arch,
            "context_definition": _VECTOR_TRIAD_CONTEXT,
            "host_parameters": _VECTOR_TRIAD_HOST_PARAMS,
            "host_context_initializer": "a, b, c, out, n",
            "grid_dim": grid_dim,
            "block_dim": block_dim,
            "op": 6,
            "signature": [ArgDirection.IN, ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "elementwise_triad_f32",
            "args": ["a", "b", "c", "out"],
        }
    }


def _cuda_elementwise_quad_spec(source, *, task_name, arch="compute_80", grid_dim=4, block_dim=256):
    return {
        "cuda": {
            "source": str(source),
            "task_name": task_name,
            "arch": arch,
            "context_definition": _VECTOR_QUAD_CONTEXT,
            "host_parameters": _VECTOR_QUAD_HOST_PARAMS,
            "host_context_initializer": "a, b, c, d, out, n",
            "grid_dim": grid_dim,
            "block_dim": block_dim,
            "op": 7,
            "signature": [ArgDirection.IN, ArgDirection.IN, ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "elementwise_quad_f32",
            "args": ["a", "b", "c", "d", "out"],
        }
    }


def _cuda_elementwise_generic_args_spec(source, *, task_name, arch="compute_80", grid_dim=4, block_dim=256):
    return {
        "cuda": {
            "source": str(source),
            "task_name": task_name,
            "arch": arch,
            "context_definition": _VECTOR_GENERIC_ARGS_CONTEXT,
            "host_parameters": _VECTOR_GENERIC_ARGS_HOST_PARAMS,
            "host_context_initializer": "a, b, out, tensor0, tensor1, scalar0, scalar1, n",
            "grid_dim": grid_dim,
            "block_dim": block_dim,
            "op": 8,
            "signature": [ArgDirection.IN, ArgDirection.IN, ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "elementwise_generic_args_f32",
            "args": ["a", "b", "out"],
            "tensor_args": ["c", "d"],
            "scalar_args": ["alpha", "beta"],
        }
    }


def _cuda_elementwise_generic_args4_spec(source, *, task_name, arch="compute_80", grid_dim=4, block_dim=256):
    return {
        "cuda": {
            "source": str(source),
            "task_name": task_name,
            "arch": arch,
            "context_definition": _VECTOR_GENERIC_ARGS4_CONTEXT,
            "host_parameters": _VECTOR_GENERIC_ARGS4_HOST_PARAMS,
            "host_context_initializer": (
                "a, b, out, tensor0, tensor1, tensor2, tensor3, scalar0, scalar1, scalar2, scalar3, n"
            ),
            "grid_dim": grid_dim,
            "block_dim": block_dim,
            "op": 9,
            "signature": [
                ArgDirection.IN,
                ArgDirection.IN,
                ArgDirection.IN,
                ArgDirection.IN,
                ArgDirection.IN,
                ArgDirection.IN,
                ArgDirection.OUT,
            ],
            "arg_builder": "elementwise_generic_args_f32",
            "args": ["a", "b", "out"],
            "tensor_args": ["c", "d", "e", "f"],
            "scalar_args": ["alpha", "beta", "gamma", "delta"],
        }
    }


def _cuda_persistent_dag_spec(add_source, mul_source, *, arch="compute_80", block_dim=256, stream_id=0):
    return {
        "cuda": {
            "runtime": "persistent_device",
            "arch": arch,
            "task_sources": [
                {
                    "func_id": 1,
                    "task_name": "add_f32",
                    "source_path": str(add_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
                {
                    "func_id": 2,
                    "task_name": "mul_f32",
                    "source_path": str(mul_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
            ],
            "grid_dim": 4,
            "block_dim": block_dim,
            "stream_id": stream_id,
            "shared_mem_bytes": 0,
            "signature": [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "persistent_dag_fork_join_f32",
            "args": ["a", "b", "out"],
            "queue_capacity": 2,
        }
    }


def _cuda_persistent_tensor_tile_spec(matmul_source, add_source, mul_source, *, arch="compute_80", block_dim=256):
    return {
        "cuda": {
            "runtime": "persistent_device",
            "arch": arch,
            "task_sources": [
                {
                    "func_id": 3,
                    "task_name": "matmul_tile_f32",
                    "source_path": str(matmul_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
                {
                    "func_id": 1,
                    "task_name": "add_f32",
                    "source_path": str(add_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
                {
                    "func_id": 2,
                    "task_name": "mul_f32",
                    "source_path": str(mul_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
            ],
            "grid_dim": 5,
            "block_dim": block_dim,
            "shared_mem_bytes": 0,
            "signature": [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "persistent_dag_tensor_tile_f32",
            "args": ["a", "b", "out"],
            "queue_capacity": 2,
            "tensor_tile": {"rows": 16, "cols": 16, "inner": 16},
        }
    }


def _cuda_persistent_graph_tensor_tile_spec(
    matmul_source,
    add_source,
    mul_source,
    *,
    arch="compute_80",
    block_dim=256,
):
    spec = _cuda_persistent_tensor_tile_spec(matmul_source, add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["arg_builder"] = "persistent_dag_graph_f32"
    spec["cuda"]["temporaries"] = {"tmp0": "out", "tmp1": "out", "tmp2": "out"}
    spec["cuda"]["graph"] = {
        "tasks": [
            {
                "func_id": 3,
                "a": "a",
                "b": "b",
                "out": "tmp0",
                "rows": 16,
                "cols": 16,
                "inner": 16,
                "lda": 16,
                "ldb": 16,
                "ldc": 16,
                "a_batch_stride": 256,
                "b_batch_stride": 256,
                "out_batch_stride": 256,
            },
            {"func_id": 1, "a": "tmp0", "b": "a", "out": "tmp1"},
            {"func_id": 2, "a": "tmp0", "b": "b", "out": "tmp2"},
            {"func_id": 1, "a": "tmp1", "b": "tmp2", "out": "out"},
        ]
    }
    return spec


def _cuda_persistent_graph_tensor_core_tile_spec(
    wmma_source,
    add_source,
    mul_source,
    *,
    arch="compute_80",
    block_dim=256,
    stream_id=0,
):
    spec = _cuda_persistent_tensor_core_tile_spec(
        wmma_source,
        add_source,
        mul_source,
        arch=arch,
        block_dim=block_dim,
        stream_id=stream_id,
    )
    spec["cuda"]["arg_builder"] = "persistent_dag_graph_f32"
    spec["cuda"]["temporaries"] = {"tmp0": "out", "tmp1": "out", "tmp2": "out"}
    spec["cuda"]["graph"] = {
        "tasks": [
            {
                "func_id": 10,
                "a": "a",
                "b": "b",
                "out": "tmp0",
                "rows": 16,
                "cols": 16,
                "inner": 16,
                "lda": 16,
                "ldb": 16,
                "ldc": 16,
                "a_batch_stride": 256,
                "b_batch_stride": 256,
                "out_batch_stride": 256,
            },
            {"func_id": 1, "a": "tmp0", "b": "a", "out": "tmp1"},
            {"func_id": 2, "a": "tmp0", "b": "b", "out": "tmp2"},
            {"func_id": 1, "a": "tmp1", "b": "tmp2", "out": "out"},
        ]
    }
    return spec


def _cuda_persistent_tensor_core_tile_spec(
    wmma_source,
    add_source,
    mul_source,
    *,
    arch="compute_80",
    block_dim=256,
    stream_id=0,
    tensor_tile=None,
):
    tensor_tile = tensor_tile or {"rows": 16, "cols": 16, "inner": 16}
    return {
        "cuda": {
            "runtime": "persistent_device",
            "arch": arch,
            "task_sources": [
                {
                    "func_id": 10,
                    "task_name": "wmma_m16n16k8_f32",
                    "source_path": str(wmma_source),
                    "threading": "block",
                },
                {
                    "func_id": 1,
                    "task_name": "add_f32",
                    "source_path": str(add_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
                {
                    "func_id": 2,
                    "task_name": "mul_f32",
                    "source_path": str(mul_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
            ],
            "grid_dim": 5,
            "block_dim": block_dim,
            "stream_id": stream_id,
            "shared_mem_bytes": 0,
            "signature": [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "persistent_dag_tensor_core_tile_f32",
            "args": ["a", "b", "out"],
            "queue_capacity": 2,
            "tensor_tile": tensor_tile,
        }
    }


def _cuda_persistent_scalar_axpy_spec(axpy_source, add_source, mul_source, *, arch="compute_80", block_dim=256):
    return {
        "cuda": {
            "runtime": "persistent_device",
            "arch": arch,
            "task_sources": [
                {
                    "func_id": 4,
                    "task_name": "axpy_f32",
                    "source_path": str(axpy_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
                {
                    "func_id": 2,
                    "task_name": "mul_f32",
                    "source_path": str(mul_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
                {
                    "func_id": 1,
                    "task_name": "add_f32",
                    "source_path": str(add_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
            ],
            "grid_dim": 4,
            "block_dim": block_dim,
            "shared_mem_bytes": 0,
            "signature": [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "persistent_dag_scalar_axpy_f32",
            "args": ["a", "b", "out"],
            "queue_capacity": 2,
            "scalar0": 1.5,
        }
    }


def _cuda_persistent_scalar_scale_spec(scale_source, add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_scalar_axpy_spec(scale_source, add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["task_sources"][0]["func_id"] = 11
    spec["cuda"]["task_sources"][0]["task_name"] = "scale_f32"
    spec["cuda"]["arg_builder"] = "persistent_dag_scalar_scale_f32"
    spec["cuda"]["scalar0"] = 2.0
    return spec


def _cuda_persistent_scalar_affine_spec(affine_source, add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_scalar_axpy_spec(affine_source, add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["task_sources"][0]["func_id"] = 5
    spec["cuda"]["task_sources"][0]["task_name"] = "affine_f32"
    spec["cuda"]["arg_builder"] = "persistent_dag_scalar_affine_f32"
    spec["cuda"]["scalar1"] = 0.5
    return spec


def _cuda_persistent_triad_spec(triad_source, add_source, mul_source, *, arch="compute_80", block_dim=256):
    return {
        "cuda": {
            "runtime": "persistent_device",
            "arch": arch,
            "task_sources": [
                {
                    "func_id": 6,
                    "task_name": "triad_f32",
                    "source_path": str(triad_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
                {
                    "func_id": 2,
                    "task_name": "mul_f32",
                    "source_path": str(mul_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
                {
                    "func_id": 1,
                    "task_name": "add_f32",
                    "source_path": str(add_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
            ],
            "grid_dim": 4,
            "block_dim": block_dim,
            "shared_mem_bytes": 0,
            "signature": [ArgDirection.IN, ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "persistent_dag_triad_f32",
            "args": ["a", "b", "c", "out"],
            "queue_capacity": 2,
        }
    }


def _cuda_persistent_quad_spec(quad_source, add_source, mul_source, *, arch="compute_80", block_dim=256):
    return {
        "cuda": {
            "runtime": "persistent_device",
            "arch": arch,
            "task_sources": [
                {
                    "func_id": 8,
                    "task_name": "quad_f32",
                    "source_path": str(quad_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
                {
                    "func_id": 2,
                    "task_name": "mul_f32",
                    "source_path": str(mul_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
                {
                    "func_id": 1,
                    "task_name": "add_f32",
                    "source_path": str(add_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
            ],
            "grid_dim": 4,
            "block_dim": block_dim,
            "shared_mem_bytes": 0,
            "signature": [ArgDirection.IN, ArgDirection.IN, ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "persistent_dag_quad_f32",
            "args": ["a", "b", "c", "d", "out"],
            "queue_capacity": 2,
        }
    }


def _cuda_persistent_generic_args_spec(generic_source, add_source, mul_source, *, arch="compute_80", block_dim=256):
    return {
        "cuda": {
            "runtime": "persistent_device",
            "arch": arch,
            "task_sources": [
                {
                    "func_id": 9,
                    "task_name": "generic_args_f32",
                    "source_path": str(generic_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
                {
                    "func_id": 2,
                    "task_name": "mul_f32",
                    "source_path": str(mul_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
                {
                    "func_id": 1,
                    "task_name": "add_f32",
                    "source_path": str(add_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
            ],
            "grid_dim": 4,
            "block_dim": block_dim,
            "shared_mem_bytes": 0,
            "signature": [ArgDirection.IN, ArgDirection.IN, ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "persistent_dag_generic_args_f32",
            "args": ["a", "b", "c", "d", "out"],
            "queue_capacity": 2,
            "tensor_args": ["c", "d"],
            "scalar_args": [1.5, 0.25],
        }
    }


def _cuda_persistent_generic_args4_spec(generic_source, add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_generic_args_spec(generic_source, add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["signature"] = [
        ArgDirection.IN,
        ArgDirection.IN,
        ArgDirection.IN,
        ArgDirection.IN,
        ArgDirection.IN,
        ArgDirection.IN,
        ArgDirection.OUT,
    ]
    spec["cuda"]["tensor_args"] = ["c", "d", "e", "f"]
    spec["cuda"]["scalar_args"] = [1.5, 0.25, 0.125, 0.0625]
    return spec


def _cuda_persistent_graph_generic_args_spec(
    generic_source,
    add_source,
    mul_source,
    *,
    arch="compute_80",
    block_dim=256,
):
    spec = _cuda_persistent_generic_args_spec(generic_source, add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["arg_builder"] = "persistent_dag_graph_f32"
    spec["cuda"]["temporaries"] = {"tmp0": "out", "tmp1": "out"}
    spec["cuda"]["graph"] = {
        "tasks": [
            {
                "name": "generic",
                "func_id": 9,
                "a": "a",
                "b": "b",
                "out": "tmp0",
                "dependents": "join",
                "tensor_args": ["c", "d"],
                "scalar_args": [1.5, 0.25],
            },
            {
                "name": "mul",
                "func_id": 2,
                "a": "a",
                "b": "b",
                "out": "tmp1",
                "dependents": "join",
            },
            {
                "name": "join",
                "func_id": 1,
                "a": "tmp0",
                "b": "tmp1",
                "out": "out",
                "initial_fanin": 2,
            },
        ]
    }
    return spec


def _cuda_persistent_tagged_graph_generic_args_spec(
    generic_source,
    add_source,
    mul_source,
    *,
    arch="compute_80",
    block_dim=256,
):
    spec = _cuda_persistent_graph_generic_args_spec(
        generic_source,
        add_source,
        mul_source,
        arch=arch,
        block_dim=block_dim,
    )
    spec["cuda"].pop("temporaries", None)
    spec["cuda"]["graph"] = {
        "tasks": [
            {
                "func_id": 9,
                "task_args": [
                    {"tensor": "a", "tag": "input"},
                    {"tensor": "b", "tag": "input"},
                    {"tensor": "tmp0", "tag": "output"},
                    {"scalar": "alpha"},
                    {"scalar": "beta"},
                ],
                "tensor_args": ["c", "d"],
            },
            {
                "func_id": 2,
                "task_args": [
                    {"tensor": "a", "tag": "input"},
                    {"tensor": "b", "tag": "input"},
                    {"tensor": "tmp1", "tag": "output"},
                ],
            },
            {
                "func_id": 1,
                "task_args": [
                    {"tensor": "tmp0", "tag": "input"},
                    {"tensor": "tmp1", "tag": "input"},
                    {"tensor": "out", "tag": "output_existing"},
                ],
            },
        ]
    }
    return spec


def _cuda_persistent_named_callable_graph_spec(
    generic_source,
    add_source,
    mul_source,
    *,
    arch="compute_80",
    block_dim=256,
    callables_as_list=False,
    callable_refs="name",
    callable_list_names=True,
    callable_list_compact=False,
    callable_dict_compact=False,
):
    spec = _cuda_persistent_graph_generic_args_spec(
        generic_source,
        add_source,
        mul_source,
        arch=arch,
        block_dim=block_dim,
    )
    graph_callables = {
        "generic": {"func_id": 9},
        "mul": {"func_id": 2},
        "add": {"func_id": 1},
    }
    if callables_as_list:
        if callable_list_compact:
            graph_callables = [9, 2, 1]
        elif callable_list_names:
            graph_callables = [
                {"name": "generic", "func_id": 9},
                {"name": "mul", "func_id": 2},
                {"name": "add", "func_id": 1},
            ]
        else:
            graph_callables = [{"func_id": 9}, {"func_id": 2}, {"func_id": 1}]
    elif callable_dict_compact:
        graph_callables = {"generic": 9, "mul": 2, "add": 1}
    callable_ids = {"generic": "generic", "mul": "mul", "add": "add"}
    if callable_refs == "index":
        callable_ids = {"generic": 0, "mul": 1, "add": 2}
    spec["cuda"]["graph"] = {
        "callables": graph_callables,
        "tasks": [
            {
                "callable": callable_ids["generic"],
                "task_args": [
                    {"tensor": "a", "tag": "input"},
                    {"tensor": "b", "tag": "input"},
                    {"tensor": "tmp0", "tag": "output"},
                    {"scalar": "alpha"},
                    {"scalar": "beta"},
                ],
                "tensor_args": ["c", "d"],
            },
            {
                "callable": callable_ids["mul"],
                "task_args": [
                    {"tensor": "a", "tag": "input"},
                    {"tensor": "b", "tag": "input"},
                    {"tensor": "tmp1", "tag": "output"},
                ],
            },
            {
                "callable": callable_ids["add"],
                "task_args": [
                    {"tensor": "tmp0", "tag": "input"},
                    {"tensor": "tmp1", "tag": "input"},
                    {"tensor": "out", "tag": "output_existing"},
                ],
            },
        ],
    }
    return spec


def _cuda_persistent_tagged_inout_graph_spec(
    add_source,
    *,
    arch="compute_80",
    block_dim=256,
    task_arg_role_key="tag",
    compact_task_args=False,
    map_task_args=False,
    pair_task_args=False,
):
    def task_arg(name, role):
        if pair_task_args:
            return (role, name)
        if compact_task_args:
            return {role: name}
        return {"tensor": name, task_arg_role_key: role}

    if map_task_args:
        tasks = [
            {"func_id": 1, "task_args": {"inputs": ["a", "b"], "output": "tmp0"}},
            {"func_id": 1, "task_args": {"inout": "tmp0", "input": "b"}},
            {"func_id": 1, "task_args": {"inputs": ["tmp0", "a"], "output_existing": "out"}},
        ]
    else:
        tasks = [
            {
                "func_id": 1,
                "task_args": [
                    task_arg("a", "input"),
                    task_arg("b", "input"),
                    task_arg("tmp0", "output"),
                ],
            },
            {
                "func_id": 1,
                "task_args": [
                    task_arg("tmp0", "inout"),
                    task_arg("b", "input"),
                ],
            },
            {
                "func_id": 1,
                "task_args": [
                    task_arg("tmp0", "input"),
                    task_arg("a", "input"),
                    task_arg("out", "output_existing"),
                ],
            },
        ]

    return {
        "cuda": {
            "runtime": "persistent_device",
            "arch": arch,
            "task_sources": [
                {
                    "func_id": 1,
                    "task_name": "add_f32",
                    "source_path": str(add_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
            ],
            "grid_dim": 4,
            "block_dim": block_dim,
            "shared_mem_bytes": 0,
            "signature": [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "persistent_dag_graph_f32",
            "args": ["a", "b", "out"],
            "queue_capacity": 2,
            "graph": {"tasks": tasks},
        }
    }


def _use_cuda_graph_args_alias(spec):
    for task in spec["cuda"]["graph"]["tasks"]:
        task["args"] = task.pop("task_args")
    return spec


def _use_cuda_graph_submits_alias(spec):
    graph = spec["cuda"]["graph"]
    graph["submits"] = graph.pop("tasks")
    for submit in graph["submits"]:
        submit["args"] = submit.pop("task_args")
    return spec


def _use_cuda_graph_submit_groups(spec):
    graph = spec["cuda"]["graph"]
    tasks = graph.pop("tasks")
    graph["submit_groups"] = [
        {
            "callable": "add",
            "args_list": [tasks[0].pop("task_args"), tasks[1].pop("task_args")],
        },
        {
            "callable": "add",
            "args_list": [tasks[2].pop("task_args")],
        },
    ]
    graph["callables"] = {"add": 1}
    return spec


def _cuda_persistent_graph_generic_args4_spec(
    generic_source,
    add_source,
    mul_source,
    *,
    arch="compute_80",
    block_dim=256,
):
    spec = _cuda_persistent_graph_generic_args_spec(
        generic_source,
        add_source,
        mul_source,
        arch=arch,
        block_dim=block_dim,
    )
    spec["cuda"]["signature"] = [
        ArgDirection.IN,
        ArgDirection.IN,
        ArgDirection.IN,
        ArgDirection.IN,
        ArgDirection.IN,
        ArgDirection.IN,
        ArgDirection.OUT,
    ]
    spec["cuda"]["args"] = ["a", "b", "c", "d", "e", "f", "out"]
    spec["cuda"]["graph"]["tasks"][0]["tensor_args"] = ["c", "d", "e", "f"]
    spec["cuda"]["graph"]["tasks"][0]["scalar_args"] = [1.5, 0.25, 0.125, 0.0625]
    return spec


def _cuda_persistent_inferred_graph_generic_args_spec(
    generic_source,
    add_source,
    mul_source,
    *,
    arch="compute_80",
    block_dim=256,
):
    spec = _cuda_persistent_graph_generic_args_spec(
        generic_source,
        add_source,
        mul_source,
        arch=arch,
        block_dim=block_dim,
    )
    for task in spec["cuda"]["graph"]["tasks"]:
        task.pop("dependents", None)
        task.pop("initial_fanin", None)
    return spec


def _cuda_persistent_mixed_graph_generic_args_spec(
    generic_source,
    add_source,
    mul_source,
    *,
    arch="compute_80",
    block_dim=256,
):
    spec = _cuda_persistent_graph_generic_args_spec(
        generic_source,
        add_source,
        mul_source,
        arch=arch,
        block_dim=block_dim,
    )
    spec["cuda"]["graph"]["tasks"][1].pop("dependents", None)
    spec["cuda"]["graph"]["tasks"][2].pop("initial_fanin", None)
    return spec


def _cuda_persistent_depends_on_graph_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_dag_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["arg_builder"] = "persistent_dag_graph_f32"
    spec["cuda"]["graph"] = {
        "tasks": [
            {"name": "left", "func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
            {"name": "right", "func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
            {"name": "join", "func_id": 1, "a": "a", "b": "b", "out": "out", "depends_on": ["left", "right"]},
        ]
    }
    return spec


def _cuda_persistent_edge_list_graph_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_dag_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["arg_builder"] = "persistent_dag_graph_f32"
    spec["cuda"]["graph"] = {
        "tasks": [
            {"name": "left", "func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
            {"name": "right", "func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
            {"name": "join", "func_id": 1, "a": "a", "b": "b", "out": "out"},
        ],
        "edges": [
            {"from": "left", "to": "join"},
            ["right", "join"],
        ],
    }
    return spec


def _cuda_persistent_string_edge_graph_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_edge_list_graph_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["graph"]["edges"] = [
        "left -> join",
        "right -> join",
    ]
    return spec


def _cuda_persistent_dep_gen_edge_graph_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_edge_list_graph_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["graph"]["edges"] = [
        {"pred": "left", "succ": "join"},
        {"pred": "right", "succ": "join"},
    ]
    return spec


def _cuda_persistent_dep_gen_task_id_graph_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_dep_gen_edge_graph_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["graph"]["tasks"] = [
        {"task_id": "left", "func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
        {"task_id": "right", "func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
        {"task_id": "join", "func_id": 1, "a": "a", "b": "b", "out": "out"},
    ]
    return spec


def _cuda_persistent_dep_gen_annotated_edge_graph_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_dep_gen_task_id_graph_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["graph"]["edges"] = [
        {"pred": "left", "succ": "join", "source": "creator", "arg": 0},
        {"pred": "right", "succ": "join", "source": "tensormap", "arg": 1},
    ]
    return spec


def _cuda_persistent_dep_gen_task_defaults_graph_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_dep_gen_annotated_edge_graph_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["graph"]["task_defaults"] = {"func_id": 1, "a": "a", "b": "b", "out": "out"}
    spec["cuda"]["graph"]["tasks"] = [
        {"task_id": "left", "scope": "auto"},
        {"task_id": "right", "scope": "auto"},
        {"task_id": "join", "scope": "auto"},
    ]
    return spec


def _write_dep_gen_json_file(path):
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "tasks": [
                    {"task_id": "0", "scope": "auto"},
                    {"task_id": "4294967296", "scope": "auto"},
                    {"task_id": "4294967298", "scope": "auto"},
                ],
                "tensors": [
                    {
                        "tensor_id": "13451765318376212391",
                        "buffer_addr": "29204938752",
                        "version": 0,
                        "dtype": "FLOAT32",
                        "ndims": 1,
                        "raw_shapes": [1024],
                    }
                ],
                "edges": [
                    {"pred": "0", "succ": "4294967298", "arg": 0, "source": "creator"},
                    {"pred": "4294967296", "succ": "4294967298", "arg": 1, "source": "tensormap"},
                ],
            }
        )
    )


def _write_dep_gen_task_metadata_file(path):
    path.write_text(
        json.dumps(
            [
                {"task_id": "0", "func_id": 1, "out": "tmp0"},
                {"task_id": "4294967296", "func_id": 2, "out": "tmp1"},
                {"task_id": "4294967298", "func_id": 1, "a": "tmp0", "b": "tmp1", "out": "out"},
            ]
        )
    )


def _write_dep_gen_task_overrides_file(path):
    path.write_text(
        json.dumps(
            {
                "0": {"func_id": 1, "out": "tmp0"},
                "4294967296": {"func_id": 2, "out": "tmp1"},
                "4294967298": {"func_id": 1, "a": "tmp0", "b": "tmp1", "out": "out"},
            }
        )
    )


def _cuda_persistent_dep_gen_json_file_graph_spec(
    add_source,
    mul_source,
    dep_graph_path,
    *,
    arch="compute_80",
    block_dim=256,
):
    spec = _cuda_persistent_dag_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["arg_builder"] = "persistent_dag_graph_f32"
    spec["cuda"]["graph_path"] = str(dep_graph_path)
    spec["cuda"]["graph"] = {"task_defaults": {"func_id": 1, "a": "a", "b": "b", "out": "out"}}
    return spec


def _cuda_persistent_dep_gen_task_overrides_graph_spec(
    add_source,
    mul_source,
    dep_graph_path,
    *,
    arch="compute_80",
    block_dim=256,
):
    spec = _cuda_persistent_dep_gen_json_file_graph_spec(
        add_source,
        mul_source,
        dep_graph_path,
        arch=arch,
        block_dim=block_dim,
    )
    spec["cuda"]["graph"]["task_overrides"] = {
        "0": {"func_id": 1, "out": "tmp0"},
        "4294967296": {"func_id": 2, "out": "tmp1"},
        "4294967298": {"func_id": 1, "a": "tmp0", "b": "tmp1", "out": "out"},
    }
    return spec


def _cuda_persistent_dep_gen_task_overrides_file_graph_spec(
    add_source,
    mul_source,
    dep_graph_path,
    overrides_path,
    *,
    arch="compute_80",
    block_dim=256,
):
    spec = _cuda_persistent_dep_gen_json_file_graph_spec(
        add_source,
        mul_source,
        dep_graph_path,
        arch=arch,
        block_dim=block_dim,
    )
    spec["cuda"]["graph"]["task_overrides_path"] = str(overrides_path)
    return spec


def _cuda_persistent_dep_gen_task_metadata_graph_spec(
    add_source,
    mul_source,
    dep_graph_path,
    *,
    arch="compute_80",
    block_dim=256,
):
    spec = _cuda_persistent_dep_gen_json_file_graph_spec(
        add_source,
        mul_source,
        dep_graph_path,
        arch=arch,
        block_dim=block_dim,
    )
    spec["cuda"]["graph"]["task_metadata"] = [
        {"task_id": "0", "func_id": 1, "out": "tmp0"},
        {"task_id": "4294967296", "func_id": 2, "out": "tmp1"},
        {"task_id": "4294967298", "func_id": 1, "a": "tmp0", "b": "tmp1", "out": "out"},
    ]
    return spec


def _cuda_persistent_dep_gen_task_metadata_file_graph_spec(
    add_source,
    mul_source,
    dep_graph_path,
    metadata_path,
    *,
    arch="compute_80",
    block_dim=256,
):
    spec = _cuda_persistent_dep_gen_json_file_graph_spec(
        add_source,
        mul_source,
        dep_graph_path,
        arch=arch,
        block_dim=block_dim,
    )
    spec["cuda"]["graph"]["task_metadata_path"] = str(metadata_path)
    return spec


def _cuda_persistent_inline_task_metadata_file_graph_spec(
    add_source,
    mul_source,
    metadata_path,
    *,
    arch="compute_80",
    block_dim=256,
):
    spec = _cuda_persistent_dag_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["arg_builder"] = "persistent_dag_graph_f32"
    spec["cuda"]["graph"] = {
        "task_defaults": {"func_id": 1, "a": "a", "b": "b", "out": "out"},
        "task_metadata_path": str(metadata_path),
        "tasks": [
            {"task_id": "0", "scope": "auto"},
            {"task_id": "4294967296", "scope": "auto"},
            {"task_id": "4294967298", "scope": "auto"},
        ],
        "edges": [
            {"pred": "0", "succ": "4294967298", "arg": 0, "source": "creator"},
            {"pred": "4294967296", "succ": "4294967298", "arg": 1, "source": "tensormap"},
        ],
    }
    return spec


def _cuda_persistent_task_dict_edge_graph_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_edge_list_graph_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["graph"]["tasks"] = {
        "left": {"func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
        "right": {"func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
        "join": {"func_id": 1, "a": "a", "b": "b", "out": "out"},
    }
    return spec


def _cuda_persistent_node_graph_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_adjacency_graph_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["graph"]["nodes"] = spec["cuda"]["graph"].pop("tasks")
    return spec


def _cuda_persistent_node_io_graph_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_node_graph_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["graph"]["nodes"] = {
        "left": {"func_id": 1, "inputs": ["a", "b"], "outputs": ["tmp0"]},
        "right": {"func_id": 2, "inputs": ["a", "b"], "outputs": ["tmp1"]},
        "join": {"func_id": 1, "inputs": ["a", "b"], "outputs": ["out"]},
    }
    return spec


def _cuda_persistent_node_id_graph_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_node_io_graph_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["graph"]["nodes"] = [
        {"id": "left", "func_id": 1, "inputs": ["a", "b"], "outputs": ["tmp0"]},
        {"id": "right", "func_id": 2, "inputs": ["a", "b"], "outputs": ["tmp1"]},
        {"id": "join", "func_id": 1, "inputs": ["a", "b"], "outputs": ["out"]},
    ]
    spec["cuda"]["graph"]["edges"] = [
        "left -> join",
        "right -> join",
    ]
    return spec


def _cuda_persistent_node_op_graph_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_node_io_graph_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["graph"]["callables"] = {"add": 1, "mul": 2}
    spec["cuda"]["graph"]["nodes"] = {
        "left": {"op": "add", "inputs": ["a", "b"], "outputs": ["tmp0"]},
        "right": {"op": "mul", "inputs": ["a", "b"], "outputs": ["tmp1"]},
        "join": {"op": "add", "inputs": ["a", "b"], "outputs": ["out"]},
    }
    return spec


def _cuda_persistent_node_data_graph_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_node_io_graph_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["graph"]["callables"] = {"add": 1, "mul": 2}
    spec["cuda"]["graph"]["nodes"] = [
        {"id": "left", "data": {"op": "add", "inputs": ["a", "b"], "outputs": ["tmp0"]}},
        {"id": "right", "data": {"op": "mul", "inputs": ["a", "b"], "outputs": ["tmp1"]}},
        {"id": "join", "data": {"op": "add", "inputs": ["a", "b"], "outputs": ["out"]}},
    ]
    spec["cuda"]["graph"]["edges"] = [
        {"source": "left", "target": "join"},
        {"source": "right", "target": "join"},
    ]
    return spec


def _cuda_persistent_node_link_graph_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_node_data_graph_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["graph"]["links"] = spec["cuda"]["graph"].pop("edges")
    return spec


def _cuda_persistent_node_port_dict_graph_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_node_data_graph_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["graph"]["nodes"] = [
        {
            "id": "left",
            "data": {"op": "add", "inputs": {"lhs": "a", "rhs": "b"}, "outputs": {"value": "tmp0"}},
        },
        {
            "id": "right",
            "data": {"op": "mul", "inputs": {"lhs": "a", "rhs": "b"}, "outputs": {"value": "tmp1"}},
        },
        {
            "id": "join",
            "data": {"op": "add", "inputs": {"lhs": "tmp0", "rhs": "tmp1"}, "outputs": {"value": "out"}},
        },
    ]
    spec["cuda"]["graph"].pop("edges", None)
    return spec


def _cuda_persistent_callable_id_graph_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_node_io_graph_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["graph"]["callables"] = {
        "add": {"callable_id": 1},
        "mul": {"cid": 2},
    }
    spec["cuda"]["graph"]["nodes"] = {
        "left": {"op": "add", "inputs": ["a", "b"], "outputs": ["tmp0"]},
        "right": {"op": "mul", "inputs": ["a", "b"], "outputs": ["tmp1"]},
        "join": {"op": "add", "inputs": ["a", "b"], "outputs": ["out"]},
    }
    return spec


def _cuda_persistent_node_attrs_graph_spec(
    generic_source,
    add_source,
    mul_source,
    *,
    arch="compute_80",
    block_dim=256,
):
    spec = _cuda_persistent_graph_generic_args_spec(
        generic_source,
        add_source,
        mul_source,
        arch=arch,
        block_dim=block_dim,
    )
    spec["cuda"]["graph"] = {
        "callables": {"generic": 9, "mul": 2, "add": 1},
        "nodes": {
            "generic_node": {
                "op": "generic",
                "inputs": ["a", "b"],
                "outputs": ["tmp0"],
                "attrs": {
                    "tensor_args": ["c", "d"],
                    "scalar_args": ["alpha", "beta"],
                },
            },
            "mul_node": {"op": "mul", "inputs": ["a", "b"], "outputs": ["tmp1"]},
            "add_node": {"op": "add", "inputs": ["tmp0", "tmp1"], "outputs": ["out"]},
        },
        "edges": {
            "generic_node": "add_node",
            "mul_node": "add_node",
        },
    }
    return spec


def _cuda_persistent_adjacency_graph_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_task_dict_edge_graph_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["graph"]["edges"] = {
        "left": "join",
        "right": ["join"],
    }
    return spec


def _cuda_persistent_auto_temp_graph_generic_args_spec(
    generic_source,
    add_source,
    mul_source,
    *,
    arch="compute_80",
    block_dim=256,
):
    spec = _cuda_persistent_graph_generic_args_spec(
        generic_source,
        add_source,
        mul_source,
        arch=arch,
        block_dim=block_dim,
    )
    spec["cuda"].pop("temporaries", None)
    return spec


def _cuda_persistent_graph_scratch_reuse_spec(
    add_source,
    mul_source,
    *,
    arch="compute_80",
    block_dim=256,
):
    spec = _cuda_persistent_reuse_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["arg_builder"] = "persistent_dag_graph_f32"
    spec["cuda"]["temporaries"] = {"tmp0": "out", "tmp1": "out", "tmp2": "out", "tmp3": "out"}
    spec["cuda"]["graph"] = {
        "tasks": [
            {"func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
            {"func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
            {"func_id": 1, "a": "tmp0", "b": "tmp1", "out": "tmp2"},
            {"func_id": 2, "a": "tmp2", "b": "b", "out": "tmp3"},
            {
                "func_id": 1,
                "a": "tmp2",
                "b": "a",
                "out": "tmp4",
                "out_storage": "tmp0",
            },
            {"func_id": 1, "a": "tmp4", "b": "tmp3", "out": "out"},
        ]
    }
    return spec


def _cuda_persistent_graph_scalar_scale_spec(
    scale_source,
    add_source,
    mul_source,
    *,
    arch="compute_80",
    block_dim=256,
):
    spec = _cuda_persistent_scalar_scale_spec(scale_source, add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["arg_builder"] = "persistent_dag_graph_f32"
    spec["cuda"]["temporaries"] = {"tmp0": "out", "tmp1": "out"}
    spec["cuda"]["graph"] = {
        "tasks": [
            {"func_id": 11, "a": "a", "out": "tmp0", "scalar0": "alpha"},
            {"func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
            {"func_id": 1, "a": "tmp0", "b": "tmp1", "out": "out"},
        ]
    }
    return spec


def _cuda_persistent_reordered_graph_generic_args_spec(
    generic_source,
    add_source,
    mul_source,
    *,
    arch="compute_80",
    block_dim=256,
):
    spec = _cuda_persistent_inferred_graph_generic_args_spec(
        generic_source,
        add_source,
        mul_source,
        arch=arch,
        block_dim=block_dim,
    )
    tasks = spec["cuda"]["graph"]["tasks"]
    spec["cuda"]["graph"]["tasks"] = [tasks[2], tasks[0], tasks[1]]
    return spec


def _cuda_persistent_unary_square_spec(square_source, add_source, *, arch="compute_80", block_dim=256):
    return {
        "cuda": {
            "runtime": "persistent_device",
            "arch": arch,
            "task_sources": [
                {
                    "func_id": 7,
                    "task_name": "square_f32",
                    "source_path": str(square_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
                {
                    "func_id": 1,
                    "task_name": "add_f32",
                    "source_path": str(add_source),
                    "body_style": "task_body",
                    "context_definition": _PERSISTENT_CONTEXT,
                },
            ],
            "grid_dim": 4,
            "block_dim": block_dim,
            "shared_mem_bytes": 0,
            "signature": [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
            "arg_builder": "persistent_dag_unary_square_f32",
            "args": ["a", "b", "out"],
            "queue_capacity": 2,
        }
    }


def _cuda_persistent_graph_unary_square_spec(square_source, add_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_unary_square_spec(square_source, add_source, arch=arch, block_dim=block_dim)
    spec["cuda"].update(
        {
            "arg_builder": "persistent_dag_graph_f32",
            "temporaries": {"tmp0": "out", "tmp1": "out"},
            "graph": {
                "tasks": [
                    {"func_id": 7, "a": "a", "out": "tmp0"},
                    {"func_id": 1, "a": "tmp0", "b": "b", "out": "tmp1"},
                    {"func_id": 1, "a": "tmp1", "b": "a", "out": "out"},
                ]
            },
        }
    )
    return spec


def _cuda_persistent_chain_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_dag_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["arg_builder"] = "persistent_dag_chain_f32"
    spec["cuda"]["queue_capacity"] = 3
    return spec


def _cuda_persistent_reuse_spec(add_source, mul_source, *, arch="compute_80", block_dim=256):
    spec = _cuda_persistent_dag_spec(add_source, mul_source, arch=arch, block_dim=block_dim)
    spec["cuda"]["arg_builder"] = "persistent_dag_reuse_f32"
    spec["cuda"]["queue_capacity"] = 3
    return spec


def test_scene_test_compiles_cuda_host_schedule_callable(tmp_path, monkeypatch):
    source = tmp_path / "vector_add.pto.cu"
    source.write_text(_VECTOR_ADD_BODY)
    seen = {}

    def fake_compile_cuda_host_schedule(self, source_path, **kwargs):
        seen["platform"] = self.platform
        seen["source_path"] = source_path
        seen["kwargs"] = kwargs
        return CudaHostScheduleCallableArtifact(
            cache_key="scene-test-key",
            cache_hit=False,
            source_path=tmp_path / "generated_host_wrapper.cu",
            ptx_path=tmp_path / "pto_callable.ptx",
            manifest_path=tmp_path / "pto_callable.json",
            ptx=b"fake-scene-ptx",
            entry_name="pto_kernel_vector_add",
            persistent_entry_name="pto_task_vector_add",
            arch=kwargs["arch"],
            source_kind="task-body-wrapper",
        )

    monkeypatch.setattr(
        "simpler_setup.kernel_compiler.KernelCompiler.compile_cuda_host_schedule",
        fake_compile_cuda_host_schedule,
    )

    prepared = _compile_chip_callable_from_spec(
        _cuda_vector_add_spec(source),
        "cuda",
        "host_schedule",
        ("cuda-scene-compile", "cuda", "host_schedule"),
    )

    assert isinstance(prepared, PreparedCudaCallable)
    assert prepared.runtime == "host_schedule"
    assert prepared.manifest.grid_dim == 4
    assert prepared.manifest.block_dim == 256
    assert seen["platform"] == "cuda"
    assert seen["source_path"] == str(source)
    assert seen["kwargs"]["task_name"] == "vector_add"
    assert seen["kwargs"]["arch"] == "compute_80"
    assert seen["kwargs"]["context_definition"] == _VECTOR_ADD_CONTEXT
    assert seen["kwargs"]["host_parameters"] == _VECTOR_ADD_HOST_PARAMS
    assert seen["kwargs"]["host_context_initializer"] == "a, b, out, n"


def test_scene_test_compiles_cuda_persistent_device_callable(tmp_path, monkeypatch):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)
    seen = {}

    def fake_compile_cuda_persistent_device(self, task_sources, **kwargs):
        seen["platform"] = self.platform
        seen["task_sources"] = task_sources
        seen["kwargs"] = kwargs
        return CudaPersistentCallableArtifact(
            cache_key="persistent-scene-test-key",
            cache_hit=False,
            source_path=tmp_path / "generated_dispatch.cu",
            ptx_path=tmp_path / "pto_callable.ptx",
            manifest_path=tmp_path / "pto_callable.json",
            ptx=b"fake-persistent-scene-ptx",
            entry_name="pto_persistent_dag_f32_executor",
            arch=kwargs["arch"],
            source_kind="generated-dispatch",
        )

    monkeypatch.setattr(
        "simpler_setup.kernel_compiler.KernelCompiler.compile_cuda_persistent_device",
        fake_compile_cuda_persistent_device,
    )

    prepared = _compile_chip_callable_from_spec(
        _cuda_persistent_dag_spec(add_source, mul_source, stream_id=1),
        "cuda",
        "persistent_device",
        ("cuda-persistent-scene-compile", "cuda", "persistent_device"),
    )

    assert isinstance(prepared, PreparedCudaCallable)
    assert prepared.runtime == "persistent_device"
    assert prepared.manifest.op == 1003
    assert prepared.manifest.grid_dim == 4
    assert prepared.manifest.block_dim == 256
    assert prepared.manifest.stream_id == 1
    assert seen["platform"] == "cuda"
    assert [item["func_id"] for item in seen["task_sources"]] == [1, 2]
    assert [item["task_name"] for item in seen["task_sources"]] == ["add_f32", "mul_f32"]
    assert seen["kwargs"]["arch"] == "compute_80"


def test_scene_test_builds_cuda_elementwise_binary_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "elementwise_binary_f32",
        "args": ["a", "b", "out"],
    }

    args = _build_cuda_host_schedule_args(test_args, cast(Any, _FakeCudaBuffers()), cuda_spec)

    assert args.a == 101
    assert args.b == 202
    assert args.out == 303
    assert args.n == 17


def test_scene_test_builds_cuda_elementwise_scale_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
        Scalar("alpha", ctypes.c_float(1.5)),
    )
    cuda_spec = {
        "arg_builder": "elementwise_scale_f32",
        "args": ["a", "out", "alpha"],
    }

    args = _build_cuda_host_schedule_args(test_args, cast(Any, _FakeCudaBuffers()), cuda_spec)

    assert isinstance(args, CudaVectorScaleArgs)
    assert args.a == 101
    assert args.out == 303
    assert args.alpha == pytest.approx(1.5)
    assert args.n == 17


def test_scene_test_builds_cuda_elementwise_unary_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "elementwise_unary_f32",
        "args": ["a", "out"],
    }

    args = _build_cuda_host_schedule_args(test_args, cast(Any, _FakeCudaBuffers()), cuda_spec)

    assert isinstance(args, CudaVectorUnaryArgs)
    assert args.a == 101
    assert args.out == 303
    assert args.n == 17


def test_scene_test_builds_cuda_elementwise_axpy_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
        Scalar("alpha", ctypes.c_float(1.5)),
    )
    cuda_spec = {
        "arg_builder": "elementwise_axpy_f32",
        "args": ["a", "b", "out", "alpha"],
    }

    args = _build_cuda_host_schedule_args(test_args, cast(Any, _FakeCudaBuffers()), cuda_spec)

    assert isinstance(args, CudaVectorAxpyArgs)
    assert args.a == 101
    assert args.b == 202
    assert args.out == 303
    assert args.alpha == pytest.approx(1.5)
    assert args.n == 17


def test_scene_test_builds_cuda_elementwise_affine_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
        Scalar("alpha", ctypes.c_float(1.5)),
        Scalar("beta", ctypes.c_float(0.5)),
    )
    cuda_spec = {
        "arg_builder": "elementwise_affine_f32",
        "args": ["a", "b", "out", "alpha", "beta"],
    }

    args = _build_cuda_host_schedule_args(test_args, cast(Any, _FakeCudaBuffers()), cuda_spec)

    assert isinstance(args, CudaVectorAffineArgs)
    assert args.a == 101
    assert args.b == 202
    assert args.out == 303
    assert args.alpha == pytest.approx(1.5)
    assert args.beta == pytest.approx(0.5)
    assert args.n == 17


def test_scene_test_builds_cuda_elementwise_triad_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "elementwise_triad_f32",
        "args": ["a", "b", "c", "out"],
    }

    args = _build_cuda_host_schedule_args(test_args, cast(Any, _FakeCudaBuffers()), cuda_spec)

    assert isinstance(args, CudaVectorTernaryArgs)
    assert args.a == 101
    assert args.b == 202
    assert args.c == 404
    assert args.out == 303
    assert args.n == 17


def test_scene_test_builds_cuda_elementwise_quad_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "elementwise_quad_f32",
        "args": ["a", "b", "c", "d", "out"],
    }

    args = _build_cuda_host_schedule_args(test_args, cast(Any, _FakeCudaBuffers()), cuda_spec)

    assert isinstance(args, CudaVectorQuaternaryArgs)
    assert args.a == 101
    assert args.b == 202
    assert args.c == 404
    assert args.d == 505
    assert args.out == 303
    assert args.n == 17


def test_scene_test_builds_cuda_elementwise_generic_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
        Scalar("alpha", ctypes.c_float(1.5)),
        Scalar("beta", ctypes.c_float(0.25)),
    )
    cuda_spec = {
        "arg_builder": "elementwise_generic_args_f32",
        "args": ["a", "b", "out"],
        "tensor_args": ["c", "d"],
        "scalar_args": ["alpha", "beta"],
    }

    args = _build_cuda_host_schedule_args(test_args, cast(Any, _FakeCudaBuffers()), cuda_spec)

    assert isinstance(args, CudaVectorGenericArgs)
    assert args.a == 101
    assert args.b == 202
    assert args.out == 303
    assert list(args.tensor_args)[:2] == [404, 505]
    assert list(args.scalar_args)[:2] == pytest.approx([1.5, 0.25])
    assert args.tensor_arg_count == 2
    assert args.scalar_arg_count == 2
    assert args.n == 17


def test_scene_test_builds_cuda_elementwise_generic_args_four_slots():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("e", _FakeTensor(17)),
        Tensor("f", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
        Scalar("alpha", ctypes.c_float(1.5)),
        Scalar("beta", ctypes.c_float(0.25)),
        Scalar("gamma", ctypes.c_float(0.125)),
        Scalar("delta", ctypes.c_float(0.0625)),
    )
    cuda_spec = {
        "arg_builder": "elementwise_generic_args_f32",
        "args": ["a", "b", "out"],
        "tensor_args": ["c", "d", "e", "f"],
        "scalar_args": ["alpha", "beta", "gamma", "delta"],
    }

    args = _build_cuda_host_schedule_args(test_args, cast(Any, _FakeCudaBuffers()), cuda_spec)

    assert isinstance(args, CudaVectorGenericArgs)
    assert list(args.tensor_args) == [404, 505, 606, 707]
    assert list(args.scalar_args) == pytest.approx([1.5, 0.25, 0.125, 0.0625])
    assert args.tensor_arg_count == 4
    assert args.scalar_arg_count == 4


def test_scene_test_builds_cuda_persistent_chain_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_chain_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 3,
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 5
    assert list(buffers.host_fanin) == [0, 0, 2, 1, 1]
    assert list(buffers.host_dependents) == [2, 2, 3, 4]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 1),
        (2, 3, 1),
        (1, 4, 0),
    ]
    assert buffers.host_tasks[4].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_reuse_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_reuse_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 3,
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 6
    assert list(buffers.host_fanin) == [0, 0, 2, 1, 1, 2]
    assert list(buffers.host_dependents) == [2, 2, 3, 4, 5, 5]
    assert buffers.host_tasks[0].out == buffers.host_tasks[4].out
    assert buffers.host_tasks[2].dependent_count == 2
    assert buffers.host_tasks[5].a == buffers.host_tasks[4].out
    assert buffers.host_tasks[5].b == buffers.host_tasks[3].out
    assert buffers.host_tasks[5].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_scalar_axpy_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_scalar_axpy_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "scalar0": 1.5,
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 3
    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (4, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[0].scalar0 == pytest.approx(1.5)
    assert buffers.host_tasks[2].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].b == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_scalar_scale_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_scalar_scale_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "scalar0": 2.0,
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 3
    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (11, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[0].a == buffers.tensor_buffers.ptrs["a"]
    assert buffers.host_tasks[0].b is None
    assert buffers.host_tasks[0].scalar0 == pytest.approx(2.0)
    assert buffers.host_tasks[0].out == buffers.dev_tmp0
    assert buffers.host_tasks[1].out == buffers.dev_tmp1
    assert buffers.host_tasks[2].a == buffers.dev_tmp0
    assert buffers.host_tasks[2].b == buffers.dev_tmp1
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_scalar_affine_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_scalar_affine_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "scalar0": 1.5,
        "scalar1": 0.5,
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 3
    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (5, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[0].scalar0 == pytest.approx(1.5)
    assert buffers.host_tasks[0].scalar1 == pytest.approx(0.5)
    assert buffers.host_tasks[2].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].b == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_triad_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_triad_f32",
        "args": ["a", "b", "c", "out"],
        "queue_capacity": 2,
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 3
    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (6, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[0].a == buffers.tensor_buffers.ptrs["a"]
    assert buffers.host_tasks[0].b == buffers.tensor_buffers.ptrs["b"]
    assert buffers.host_tasks[0].c == buffers.tensor_buffers.ptrs["c"]
    assert buffers.host_tasks[2].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].b == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_quad_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_quad_f32",
        "args": ["a", "b", "c", "d", "out"],
        "queue_capacity": 2,
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 3
    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (8, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[0].a == buffers.tensor_buffers.ptrs["a"]
    assert buffers.host_tasks[0].b == buffers.tensor_buffers.ptrs["b"]
    assert buffers.host_tasks[0].c == buffers.tensor_buffers.ptrs["c"]
    assert buffers.host_tasks[0].d == buffers.tensor_buffers.ptrs["d"]
    assert buffers.host_tasks[2].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].b == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_generic_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_generic_args_f32",
        "args": ["a", "b", "c", "d", "out"],
        "queue_capacity": 2,
        "tensor_args": ["c", "d"],
        "scalar_args": [1.5, 0.25],
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 3
    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (9, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[0].a == buffers.tensor_buffers.ptrs["a"]
    assert buffers.host_tasks[0].b == buffers.tensor_buffers.ptrs["b"]
    assert list(buffers.host_tasks[0].tensor_args)[:2] == [
        buffers.tensor_buffers.ptrs["c"],
        buffers.tensor_buffers.ptrs["d"],
    ]
    assert list(buffers.host_tasks[0].scalar_args)[:2] == pytest.approx([1.5, 0.25])
    assert buffers.host_tasks[0].tensor_arg_count == 2
    assert buffers.host_tasks[0].scalar_arg_count == 2
    assert buffers.host_tasks[2].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].b == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_generic_args_four_slots():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("e", _FakeTensor(17)),
        Tensor("f", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_generic_args_f32",
        "args": ["a", "b", "c", "d", "out"],
        "queue_capacity": 2,
        "tensor_args": ["c", "d", "e", "f"],
        "scalar_args": [1.5, 0.25, 0.125, 0.0625],
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 3
    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (9, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert list(buffers.host_tasks[0].tensor_args) == [
        buffers.tensor_buffers.ptrs["c"],
        buffers.tensor_buffers.ptrs["d"],
        buffers.tensor_buffers.ptrs["e"],
        buffers.tensor_buffers.ptrs["f"],
    ]
    assert list(buffers.host_tasks[0].scalar_args) == pytest.approx([1.5, 0.25, 0.125, 0.0625])
    assert buffers.host_tasks[0].tensor_arg_count == 4
    assert buffers.host_tasks[0].scalar_arg_count == 4


def test_scene_test_builds_cuda_persistent_graph_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "c", "d", "out"],
        "queue_capacity": 2,
        "temporaries": {"tmp0": "out", "tmp1": "out"},
        "graph": {
            "tasks": [
                {
                    "func_id": 9,
                    "a": "a",
                    "b": "b",
                    "out": "tmp0",
                    "dependents": [2],
                    "tensor_args": ["c", "d"],
                    "scalar_args": [1.5, 0.25],
                },
                {"func_id": 2, "a": "a", "b": "b", "out": "tmp1", "dependents": [2]},
                {"func_id": 1, "a": "tmp0", "b": "tmp1", "out": "out", "initial_fanin": 2},
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 3
    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (9, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[0].a == buffers.tensor_buffers.ptrs["a"]
    assert buffers.host_tasks[0].b == buffers.tensor_buffers.ptrs["b"]
    assert list(buffers.host_tasks[0].tensor_args)[:2] == [
        buffers.tensor_buffers.ptrs["c"],
        buffers.tensor_buffers.ptrs["d"],
    ]
    assert list(buffers.host_tasks[0].scalar_args)[:2] == pytest.approx([1.5, 0.25])
    assert buffers.host_tasks[0].tensor_arg_count == 2
    assert buffers.host_tasks[0].scalar_arg_count == 2
    assert buffers.host_tasks[2].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].b == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_from_tagged_task_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "c", "d", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {
                    "func_id": 9,
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "tmp0", "tag": "output"},
                    ],
                    "tensor_args": ["c", "d"],
                    "scalar_args": [1.5, 0.25],
                },
                {
                    "func_id": 2,
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "tmp1", "tag": "output"},
                    ],
                },
                {
                    "func_id": 1,
                    "task_args": [
                        {"tensor": "tmp0", "tag": "input"},
                        {"tensor": "tmp1", "tag": "input"},
                        {"tensor": "out", "tag": "output_existing"},
                    ],
                },
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 3
    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (9, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[0].a == buffers.tensor_buffers.ptrs["a"]
    assert buffers.host_tasks[0].b == buffers.tensor_buffers.ptrs["b"]
    assert buffers.host_tasks[0].out == buffers.dev_tmp0
    assert buffers.host_tasks[1].out == buffers.dev_tmp1
    assert list(buffers.host_tasks[0].tensor_args)[:2] == [
        buffers.tensor_buffers.ptrs["c"],
        buffers.tensor_buffers.ptrs["d"],
    ]
    assert list(buffers.host_tasks[0].scalar_args)[:2] == pytest.approx([1.5, 0.25])
    assert buffers.host_tasks[0].tensor_arg_count == 2
    assert buffers.host_tasks[0].scalar_arg_count == 2
    assert buffers.host_tasks[2].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].b == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_from_depends_on_edges():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {"func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
                {"func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
                {"func_id": 1, "a": "a", "b": "b", "out": "out", "depends_on": [0, 1]},
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[2].a == buffers.tensor_buffers.ptrs["a"]
    assert buffers.host_tasks[2].b == buffers.tensor_buffers.ptrs["b"]
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_from_dependencies_alias():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {"func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
                {"func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
                {"func_id": 1, "a": "a", "b": "b", "out": "out", "dependencies": [0, 1]},
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]


def test_scene_test_builds_cuda_persistent_graph_from_named_dependencies():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {"name": "left", "func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
                {"name": "right", "func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
                {"name": "join", "func_id": 1, "a": "a", "b": "b", "out": "out", "depends_on": ["left", "right"]},
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_graph_edges():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {"name": "left", "func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
                {"name": "right", "func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
                {"name": "join", "func_id": 1, "a": "a", "b": "b", "out": "out"},
            ],
            "edges": [
                {"from": "left", "to": "join"},
                ["right", "join"],
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[2].a == buffers.tensor_buffers.ptrs["a"]
    assert buffers.host_tasks[2].b == buffers.tensor_buffers.ptrs["b"]


def test_scene_test_builds_cuda_persistent_graph_from_string_edges():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {"name": "left", "func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
                {"name": "right", "func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
                {"name": "join", "func_id": 1, "a": "a", "b": "b", "out": "out"},
            ],
            "edges": [
                "left -> join",
                "right -> join",
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_dep_gen_edges():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {"name": "left", "func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
                {"name": "right", "func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
                {"name": "join", "func_id": 1, "a": "a", "b": "b", "out": "out"},
            ],
            "edges": [
                {"pred": "left", "succ": "join"},
                {"pred": "right", "succ": "join"},
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_dep_gen_task_id_edges():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {"task_id": "left", "func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
                {"task_id": "right", "func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
                {"task_id": "join", "func_id": 1, "a": "a", "b": "b", "out": "out"},
            ],
            "edges": [
                {"pred": "left", "succ": "join"},
                {"pred": "right", "succ": "join"},
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_annotated_dep_gen_edges():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {"task_id": "left", "func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
                {"task_id": "right", "func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
                {"task_id": "join", "func_id": 1, "a": "a", "b": "b", "out": "out"},
            ],
            "edges": [
                {"pred": "left", "succ": "join", "source": "creator", "arg": 0},
                {"pred": "right", "succ": "join", "source": "tensormap", "arg": 1},
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_dep_gen_task_defaults():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "task_defaults": {"func_id": 1, "a": "a", "b": "b", "out": "out"},
            "tasks": [
                {"task_id": "left", "scope": "auto"},
                {"task_id": "right", "scope": "auto"},
                {"task_id": "join", "scope": "auto"},
            ],
            "edges": [
                {"pred": "left", "succ": "join", "source": "creator", "arg": 0},
                {"pred": "right", "succ": "join", "source": "tensormap", "arg": 1},
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (1, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_dep_gen_json_file(tmp_path):
    dep_graph_path = tmp_path / "deps.json"
    _write_dep_gen_json_file(dep_graph_path)
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph_path": str(dep_graph_path),
        "graph": {"task_defaults": {"func_id": 1, "a": "a", "b": "b", "out": "out"}},
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (1, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_dep_gen_task_overrides(tmp_path):
    dep_graph_path = tmp_path / "deps.json"
    _write_dep_gen_json_file(dep_graph_path)
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = _cuda_persistent_dep_gen_task_overrides_graph_spec(
        "add.pto.cu",
        "mul.pto.cu",
        dep_graph_path,
    )["cuda"]

    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_dep_gen_task_overrides_file(tmp_path):
    dep_graph_path = tmp_path / "deps.json"
    overrides_path = tmp_path / "task_overrides.json"
    _write_dep_gen_json_file(dep_graph_path)
    _write_dep_gen_task_overrides_file(overrides_path)
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = _cuda_persistent_dep_gen_task_overrides_file_graph_spec(
        "add.pto.cu",
        "mul.pto.cu",
        dep_graph_path,
        overrides_path,
    )["cuda"]

    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_dep_gen_task_metadata_list(tmp_path):
    dep_graph_path = tmp_path / "deps.json"
    _write_dep_gen_json_file(dep_graph_path)
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = _cuda_persistent_dep_gen_task_metadata_graph_spec(
        "add.pto.cu",
        "mul.pto.cu",
        dep_graph_path,
    )["cuda"]

    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_dep_gen_task_metadata_file(tmp_path):
    dep_graph_path = tmp_path / "deps.json"
    metadata_path = tmp_path / "task_metadata.json"
    _write_dep_gen_json_file(dep_graph_path)
    _write_dep_gen_task_metadata_file(metadata_path)
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = _cuda_persistent_dep_gen_task_metadata_file_graph_spec(
        "add.pto.cu",
        "mul.pto.cu",
        dep_graph_path,
        metadata_path,
    )["cuda"]

    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_inline_task_metadata_file(tmp_path):
    metadata_path = tmp_path / "task_metadata.json"
    _write_dep_gen_task_metadata_file(metadata_path)
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = _cuda_persistent_inline_task_metadata_file_graph_spec(
        "add.pto.cu",
        "mul.pto.cu",
        metadata_path,
    )["cuda"]

    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_task_dict_and_graph_edges():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": {
                "left": {"func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
                "right": {"func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
                "join": {"func_id": 1, "a": "a", "b": "b", "out": "out"},
            },
            "edges": [
                {"from": "left", "to": "join"},
                ["right", "join"],
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_node_dict_and_adjacency_edges():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "nodes": {
                "left": {"func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
                "right": {"func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
                "join": {"func_id": 1, "a": "a", "b": "b", "out": "out"},
            },
            "edges": {
                "left": "join",
                "right": ["join"],
            },
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_node_input_output_fields():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "nodes": {
                "left": {"func_id": 1, "inputs": ["a", "b"], "outputs": ["tmp0"]},
                "right": {"func_id": 2, "inputs": ["a", "b"], "outputs": ["tmp1"]},
                "join": {"func_id": 1, "inputs": ["a", "b"], "outputs": ["out"]},
            },
            "edges": {
                "left": "join",
                "right": ["join"],
            },
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.a, task.b, task.out) for task in buffers.host_tasks] == [
        (1, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.dev_tmp0),
        (2, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.dev_tmp1),
        (1, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.tensor_buffers.ptrs["out"]),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_node_id_fields():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "nodes": [
                {"id": "left", "func_id": 1, "inputs": ["a", "b"], "outputs": ["tmp0"]},
                {"id": "right", "func_id": 2, "inputs": ["a", "b"], "outputs": ["tmp1"]},
                {"id": "join", "func_id": 1, "inputs": ["a", "b"], "outputs": ["out"]},
            ],
            "edges": [
                "left -> join",
                "right -> join",
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.a, task.b, task.out) for task in buffers.host_tasks] == [
        (1, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.dev_tmp0),
        (2, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.dev_tmp1),
        (1, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.tensor_buffers.ptrs["out"]),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_node_ops():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "callables": {"add": 1, "mul": 2},
            "nodes": {
                "left": {"op": "add", "inputs": ["a", "b"], "outputs": ["tmp0"]},
                "right": {"op": "mul", "inputs": ["a", "b"], "outputs": ["tmp1"]},
                "join": {"op": "add", "inputs": ["a", "b"], "outputs": ["out"]},
            },
            "edges": {
                "left": "join",
                "right": ["join"],
            },
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.a, task.b, task.out) for task in buffers.host_tasks] == [
        (1, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.dev_tmp0),
        (2, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.dev_tmp1),
        (1, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.tensor_buffers.ptrs["out"]),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_node_data_payloads():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "callables": {"add": 1, "mul": 2},
            "nodes": [
                {"id": "left", "data": {"op": "add", "inputs": ["a", "b"], "outputs": ["tmp0"]}},
                {"id": "right", "data": {"op": "mul", "inputs": ["a", "b"], "outputs": ["tmp1"]}},
                {"id": "join", "data": {"op": "add", "inputs": ["a", "b"], "outputs": ["out"]}},
            ],
            "edges": [
                {"source": "left", "target": "join"},
                {"source": "right", "target": "join"},
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.a, task.b, task.out) for task in buffers.host_tasks] == [
        (1, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.dev_tmp0),
        (2, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.dev_tmp1),
        (1, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.tensor_buffers.ptrs["out"]),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_node_links():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "callables": {"add": 1, "mul": 2},
            "nodes": [
                {"id": "left", "data": {"op": "add", "inputs": ["a", "b"], "outputs": ["tmp0"]}},
                {"id": "right", "data": {"op": "mul", "inputs": ["a", "b"], "outputs": ["tmp1"]}},
                {"id": "join", "data": {"op": "add", "inputs": ["a", "b"], "outputs": ["out"]}},
            ],
            "links": [
                {"source": "left", "target": "join"},
                {"source": "right", "target": "join"},
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.a, task.b, task.out) for task in buffers.host_tasks] == [
        (1, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.dev_tmp0),
        (2, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.dev_tmp1),
        (1, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.tensor_buffers.ptrs["out"]),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_node_port_dicts():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "callables": {"add": 1, "mul": 2},
            "nodes": [
                {
                    "id": "left",
                    "data": {
                        "op": "add",
                        "inputs": {"lhs": "a", "rhs": "b"},
                        "outputs": {"value": "tmp0"},
                    },
                },
                {
                    "id": "right",
                    "data": {
                        "op": "mul",
                        "inputs": {"lhs": "a", "rhs": "b"},
                        "outputs": {"value": "tmp1"},
                    },
                },
                {
                    "id": "join",
                    "data": {
                        "op": "add",
                        "inputs": {"lhs": "tmp0", "rhs": "tmp1"},
                        "outputs": {"value": "out"},
                    },
                },
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.a, task.b, task.out) for task in buffers.host_tasks] == [
        (1, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.dev_tmp0),
        (2, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.dev_tmp1),
        (1, buffers.dev_tmp0, buffers.dev_tmp1, buffers.tensor_buffers.ptrs["out"]),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_callable_id_aliases():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "callables": {
                "add": {"callable_id": 1},
                "mul": {"cid": 2},
            },
            "nodes": {
                "left": {"op": "add", "inputs": ["a", "b"], "outputs": ["tmp0"]},
                "right": {"op": "mul", "inputs": ["a", "b"], "outputs": ["tmp1"]},
                "join": {"op": "add", "inputs": ["a", "b"], "outputs": ["out"]},
            },
            "edges": {
                "left": "join",
                "right": ["join"],
            },
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.a, task.b, task.out) for task in buffers.host_tasks] == [
        (1, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.dev_tmp0),
        (2, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.dev_tmp1),
        (1, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.tensor_buffers.ptrs["out"]),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_node_attrs():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
        Scalar("alpha", ctypes.c_float(1.5)),
        Scalar("beta", ctypes.c_float(0.25)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "c", "d", "out"],
        "queue_capacity": 2,
        "graph": {
            "callables": {"generic": 9, "mul": 2, "add": 1},
            "nodes": {
                "generic_node": {
                    "op": "generic",
                    "inputs": ["a", "b"],
                    "outputs": ["tmp0"],
                    "attrs": {
                        "tensor_args": ["c", "d"],
                        "scalar_args": ["alpha", "beta"],
                    },
                },
                "mul_node": {"op": "mul", "inputs": ["a", "b"], "outputs": ["tmp1"]},
                "add_node": {"op": "add", "inputs": ["tmp0", "tmp1"], "outputs": ["out"]},
            },
            "edges": {
                "generic_node": "add_node",
                "mul_node": ["add_node"],
            },
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.a, task.b, task.out) for task in buffers.host_tasks] == [
        (9, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.dev_tmp0),
        (2, buffers.tensor_buffers.ptrs["a"], buffers.tensor_buffers.ptrs["b"], buffers.dev_tmp1),
        (1, buffers.dev_tmp0, buffers.dev_tmp1, buffers.tensor_buffers.ptrs["out"]),
    ]
    assert list(buffers.host_tasks[0].tensor_args)[:2] == [
        buffers.tensor_buffers.ptrs["c"],
        buffers.tensor_buffers.ptrs["d"],
    ]
    assert list(buffers.host_tasks[0].scalar_args)[:2] == pytest.approx([1.5, 0.25])


def test_scene_test_builds_cuda_persistent_graph_from_adjacency_edges():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": {
                "left": {"func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
                "right": {"func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
                "join": {"func_id": 1, "a": "a", "b": "b", "out": "out"},
            },
            "edges": {
                "left": "join",
                "right": ["join"],
            },
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_named_dependents():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {"name": "left", "func_id": 1, "a": "a", "b": "b", "out": "tmp0", "dependents": ["join"]},
                {"name": "right", "func_id": 2, "a": "a", "b": "b", "out": "tmp1", "dependents": ["join"]},
                {"name": "join", "func_id": 1, "a": "tmp0", "b": "tmp1", "out": "out", "initial_fanin": 2},
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_builds_cuda_persistent_graph_from_scalar_dependents():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {"name": "left", "func_id": 1, "a": "a", "b": "b", "out": "tmp0", "dependents": "join"},
                {"name": "right", "func_id": 2, "a": "a", "b": "b", "out": "tmp1", "dependents": 2},
                {"name": "join", "func_id": 1, "a": "tmp0", "b": "tmp1", "out": "out", "initial_fanin": 2},
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_rejects_cuda_persistent_graph_unknown_named_dependent():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {"name": "left", "func_id": 1, "a": "a", "b": "b", "out": "tmp0", "dependents": ["missing"]},
                {"name": "join", "func_id": 1, "a": "tmp0", "b": "b", "out": "out", "initial_fanin": 1},
            ]
        },
    }

    with pytest.raises(ValueError, match="unknown dependent task name.*missing"):
        _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)


def test_scene_test_rejects_cuda_persistent_graph_unknown_named_dependency():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {"name": "left", "func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
                {"func_id": 1, "a": "tmp0", "b": "b", "out": "out", "depends_on": "missing"},
            ]
        },
    }

    with pytest.raises(ValueError, match="unknown dependency task name.*missing"):
        _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)


def test_scene_test_rejects_cuda_persistent_graph_depends_on_out_of_range():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {"func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
                {"func_id": 1, "a": "tmp0", "b": "b", "out": "out", "depends_on": [7]},
            ]
        },
    }

    with pytest.raises(ValueError, match="dependency task id 7 for task 1 is outside the graph"):
        _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)


def test_scene_test_builds_cuda_persistent_graph_scalar_task_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
        Scalar("alpha", ctypes.c_float(1.5)),
        Scalar("beta", ctypes.c_float(0.25)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "c", "d", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {
                    "func_id": 9,
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "tmp0", "tag": "output"},
                        {"scalar": "alpha"},
                        {"scalar": "beta"},
                    ],
                    "tensor_args": ["c", "d"],
                },
                {
                    "func_id": 2,
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "tmp1", "tag": "output"},
                    ],
                },
                {
                    "func_id": 1,
                    "task_args": [
                        {"tensor": "tmp0", "tag": "input"},
                        {"tensor": "tmp1", "tag": "input"},
                        {"tensor": "out", "tag": "output_existing"},
                    ],
                },
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert list(buffers.host_tasks[0].scalar_args)[:2] == pytest.approx([1.5, 0.25])
    assert buffers.host_tasks[0].scalar_arg_count == 2


def test_scene_test_builds_cuda_persistent_graph_from_named_callables_and_task_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
        Scalar("alpha", ctypes.c_float(1.5)),
        Scalar("beta", ctypes.c_float(0.25)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "c", "d", "out"],
        "queue_capacity": 2,
        "graph": {
            "callables": {
                "generic": {"func_id": 9},
                "mul": {"func_id": 2},
                "add": {"func_id": 1},
            },
            "tasks": [
                {
                    "callable": "generic",
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "tmp0", "tag": "output"},
                        {"scalar": "alpha"},
                        {"scalar": "beta"},
                    ],
                    "tensor_args": ["c", "d"],
                },
                {
                    "callable": "mul",
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "tmp1", "tag": "output"},
                    ],
                },
                {
                    "callable": "add",
                    "task_args": [
                        {"tensor": "tmp0", "tag": "input"},
                        {"tensor": "tmp1", "tag": "input"},
                        {"tensor": "out", "tag": "output_existing"},
                    ],
                },
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 3
    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (9, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert list(buffers.host_tasks[0].scalar_args)[:2] == pytest.approx([1.5, 0.25])
    assert buffers.host_tasks[0].scalar_arg_count == 2
    assert buffers.host_tasks[2].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].b == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_from_compact_callable_dict():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
        Scalar("alpha", ctypes.c_float(1.5)),
        Scalar("beta", ctypes.c_float(0.25)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "c", "d", "out"],
        "queue_capacity": 2,
        "graph": {
            "callables": {"generic": 9, "mul": 2, "add": 1},
            "tasks": [
                {
                    "callable": "generic",
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "tmp0", "tag": "output"},
                        {"scalar": "alpha"},
                        {"scalar": "beta"},
                    ],
                    "tensor_args": ["c", "d"],
                },
                {
                    "callable": "mul",
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "tmp1", "tag": "output"},
                    ],
                },
                {
                    "callable": "add",
                    "task_args": [
                        {"tensor": "tmp0", "tag": "input"},
                        {"tensor": "tmp1", "tag": "input"},
                        {"tensor": "out", "tag": "output_existing"},
                    ],
                },
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (9, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert list(buffers.host_tasks[0].scalar_args)[:2] == pytest.approx([1.5, 0.25])


def test_scene_test_builds_cuda_persistent_graph_from_named_callable_list():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
        Scalar("alpha", ctypes.c_float(1.5)),
        Scalar("beta", ctypes.c_float(0.25)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "c", "d", "out"],
        "queue_capacity": 2,
        "graph": {
            "callables": [
                {"name": "generic", "func_id": 9},
                {"name": "mul", "func_id": 2},
                {"name": "add", "func_id": 1},
            ],
            "tasks": [
                {
                    "callable": "generic",
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "tmp0", "tag": "output"},
                        {"scalar": "alpha"},
                        {"scalar": "beta"},
                    ],
                    "tensor_args": ["c", "d"],
                },
                {
                    "callable": "mul",
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "tmp1", "tag": "output"},
                    ],
                },
                {
                    "callable": "add",
                    "task_args": [
                        {"tensor": "tmp0", "tag": "input"},
                        {"tensor": "tmp1", "tag": "input"},
                        {"tensor": "out", "tag": "output_existing"},
                    ],
                },
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (9, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert list(buffers.host_tasks[0].scalar_args)[:2] == pytest.approx([1.5, 0.25])
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_from_callable_list_indexes():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
        Scalar("alpha", ctypes.c_float(1.5)),
        Scalar("beta", ctypes.c_float(0.25)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "c", "d", "out"],
        "queue_capacity": 2,
        "graph": {
            "callables": [
                {"name": "generic", "func_id": 9},
                {"name": "mul", "func_id": 2},
                {"name": "add", "func_id": 1},
            ],
            "tasks": [
                {
                    "callable": 0,
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "tmp0", "tag": "output"},
                        {"scalar": "alpha"},
                        {"scalar": "beta"},
                    ],
                    "tensor_args": ["c", "d"],
                },
                {
                    "callable": 1,
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "tmp1", "tag": "output"},
                    ],
                },
                {
                    "callable": 2,
                    "task_args": [
                        {"tensor": "tmp0", "tag": "input"},
                        {"tensor": "tmp1", "tag": "input"},
                        {"tensor": "out", "tag": "output_existing"},
                    ],
                },
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (9, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert list(buffers.host_tasks[0].scalar_args)[:2] == pytest.approx([1.5, 0.25])
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_from_unnamed_callable_list_indexes():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
        Scalar("alpha", ctypes.c_float(1.5)),
        Scalar("beta", ctypes.c_float(0.25)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "c", "d", "out"],
        "queue_capacity": 2,
        "graph": {
            "callables": [{"func_id": 9}, {"func_id": 2}, {"func_id": 1}],
            "tasks": [
                {
                    "callable": 0,
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "tmp0", "tag": "output"},
                        {"scalar": "alpha"},
                        {"scalar": "beta"},
                    ],
                    "tensor_args": ["c", "d"],
                },
                {
                    "callable": 1,
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "tmp1", "tag": "output"},
                    ],
                },
                {
                    "callable": 2,
                    "task_args": [
                        {"tensor": "tmp0", "tag": "input"},
                        {"tensor": "tmp1", "tag": "input"},
                        {"tensor": "out", "tag": "output_existing"},
                    ],
                },
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (9, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_from_compact_callable_list_indexes():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
        Scalar("alpha", ctypes.c_float(1.5)),
        Scalar("beta", ctypes.c_float(0.25)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "c", "d", "out"],
        "queue_capacity": 2,
        "graph": {
            "callables": [9, 2, 1],
            "tasks": [
                {
                    "callable": 0,
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "tmp0", "tag": "output"},
                        {"scalar": "alpha"},
                        {"scalar": "beta"},
                    ],
                    "tensor_args": ["c", "d"],
                },
                {
                    "callable": 1,
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "tmp1", "tag": "output"},
                    ],
                },
                {
                    "callable": 2,
                    "task_args": [
                        {"tensor": "tmp0", "tag": "input"},
                        {"tensor": "tmp1", "tag": "input"},
                        {"tensor": "out", "tag": "output_existing"},
                    ],
                },
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (9, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert list(buffers.host_tasks[0].scalar_args)[:2] == pytest.approx([1.5, 0.25])
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_rejects_malformed_cuda_persistent_graph_callable_list_entry():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "callables": ["add"],
            "tasks": [
                {
                    "callable": "add",
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "out", "tag": "output_existing"},
                    ],
                }
            ],
        },
    }

    with pytest.raises(ValueError, match="graph callable list entries must be dictionaries"):
        _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)


def test_scene_test_rejects_unknown_cuda_persistent_graph_callable_name():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "callables": {"add": {"func_id": 1}},
            "tasks": [
                {
                    "callable": "missing",
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "out", "tag": "output_existing"},
                    ],
                }
            ],
        },
    }

    with pytest.raises(ValueError, match="unknown graph callable.*missing"):
        _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)


def test_scene_test_builds_cuda_persistent_graph_from_tagged_inout_task_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {
                    "func_id": 1,
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "tmp0", "tag": "output"},
                    ],
                },
                {
                    "func_id": 1,
                    "task_args": [
                        {"tensor": "tmp0", "tag": "inout"},
                        {"tensor": "b", "tag": "input"},
                    ],
                },
                {
                    "func_id": 1,
                    "task_args": [
                        {"tensor": "tmp0", "tag": "input"},
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "out", "tag": "output_existing"},
                    ],
                },
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 1, 1]
    assert list(buffers.host_dependents) == [1, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (1, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[1].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[1].out == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].a == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_from_role_keyed_task_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {
                    "func_id": 1,
                    "task_args": [
                        {"tensor": "a", "role": "input"},
                        {"tensor": "b", "role": "input"},
                        {"tensor": "tmp0", "role": "output"},
                    ],
                },
                {
                    "func_id": 1,
                    "task_args": [
                        {"tensor": "tmp0", "role": "inout"},
                        {"tensor": "b", "role": "input"},
                    ],
                },
                {
                    "func_id": 1,
                    "task_args": [
                        {"tensor": "tmp0", "role": "input"},
                        {"tensor": "a", "role": "input"},
                        {"tensor": "out", "role": "output_existing"},
                    ],
                },
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 1, 1]
    assert list(buffers.host_dependents) == [1, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (1, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[1].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[1].out == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].a == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_from_args_alias():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {
                    "func_id": 1,
                    "args": [
                        {"tensor": "a", "role": "input"},
                        {"tensor": "b", "role": "input"},
                        {"tensor": "tmp0", "role": "output"},
                    ],
                },
                {
                    "func_id": 1,
                    "args": [
                        {"tensor": "tmp0", "role": "inout"},
                        {"tensor": "b", "role": "input"},
                    ],
                },
                {
                    "func_id": 1,
                    "args": [
                        {"tensor": "tmp0", "role": "input"},
                        {"tensor": "a", "role": "input"},
                        {"tensor": "out", "role": "output_existing"},
                    ],
                },
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 1, 1]
    assert list(buffers.host_dependents) == [1, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (1, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[1].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[1].out == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].a == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_from_task_arg_role_maps():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {
                    "func_id": 1,
                    "task_args": {"inputs": ["a", "b"], "output": "tmp0"},
                },
                {
                    "func_id": 1,
                    "task_args": {"inout": "tmp0", "input": "b"},
                },
                {
                    "func_id": 1,
                    "task_args": {
                        "inputs": {"left": "tmp0", "right": "a"},
                        "output_existing": "out",
                    },
                },
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 1, 1]
    assert list(buffers.host_dependents) == [1, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (1, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[1].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[1].out == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].a == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_from_submits_alias():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "callables": {"add": 1},
            "submits": [
                {
                    "callable": "add",
                    "args": [
                        {"input": "a"},
                        {"input": "b"},
                        {"output": "tmp0"},
                    ],
                },
                {
                    "callable": "add",
                    "args": [
                        {"inout": "tmp0"},
                        {"input": "b"},
                    ],
                },
                {
                    "callable": "add",
                    "args": [
                        {"input": "tmp0"},
                        {"input": "a"},
                        {"output_existing": "out"},
                    ],
                },
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 1, 1]
    assert list(buffers.host_dependents) == [1, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (1, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[1].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[1].out == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].a == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_from_submit_groups():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "callables": {"add": 1},
            "submit_groups": [
                {
                    "name": "roots",
                    "callable": "add",
                    "args_list": [
                        [
                            {"input": "a"},
                            {"input": "b"},
                            {"output": "tmp0"},
                        ],
                        [
                            {"input": "a"},
                            {"input": "b"},
                            {"output": "tmp1"},
                        ],
                    ],
                },
                {
                    "name": "join",
                    "callable": "add",
                    "args_list": [
                        [
                            {"input": "tmp0"},
                            {"input": "tmp1"},
                            {"output_existing": "out"},
                        ],
                    ],
                },
            ],
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (1, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[2].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].b == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_from_compact_role_task_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {
                    "func_id": 1,
                    "task_args": [
                        {"input": "a"},
                        {"input": "b"},
                        {"output": "tmp0"},
                    ],
                },
                {
                    "func_id": 1,
                    "task_args": [
                        {"inout": "tmp0"},
                        {"input": "b"},
                    ],
                },
                {
                    "func_id": 1,
                    "task_args": [
                        {"input": "tmp0"},
                        {"input": "a"},
                        {"output_existing": "out"},
                    ],
                },
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 1, 1]
    assert list(buffers.host_dependents) == [1, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (1, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[1].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[1].out == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].a == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_from_pair_task_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {
                    "func_id": 1,
                    "task_args": [
                        ["input", "a"],
                        ("input", "b"),
                        ("output", "tmp0"),
                    ],
                },
                {
                    "func_id": 1,
                    "task_args": [
                        ("inout", "tmp0"),
                        ("input", "b"),
                    ],
                },
                {
                    "func_id": 1,
                    "task_args": [
                        ("input", "tmp0"),
                        ("input", "a"),
                        ("output_existing", "out"),
                    ],
                },
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 1, 1]
    assert list(buffers.host_dependents) == [1, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (1, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[1].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[1].out == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].a == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_rejects_unknown_tagged_output_existing_task_arg():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {
                    "func_id": 1,
                    "task_args": [
                        {"tensor": "a", "tag": "input"},
                        {"tensor": "b", "tag": "input"},
                        {"tensor": "missing", "tag": "output_existing"},
                    ],
                },
            ]
        },
    }

    with pytest.raises(ValueError, match="output_existing.*missing"):
        _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)


def test_scene_test_rejects_unknown_tagged_inout_task_arg():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {
                    "func_id": 1,
                    "task_args": [
                        {"tensor": "missing", "tag": "inout"},
                        {"tensor": "b", "tag": "input"},
                    ],
                },
            ]
        },
    }

    with pytest.raises(ValueError, match="inout.*missing"):
        _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)


def test_scene_test_rejects_mixed_cuda_persistent_compact_role_task_arg():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {
                    "func_id": 1,
                    "task_args": [
                        {"input": "a", "tensor": "a"},
                        {"input": "b"},
                        {"output_existing": "out"},
                    ],
                }
            ]
        },
    }

    with pytest.raises(ValueError, match="mixes compact and expanded tensor roles"):
        _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)


def test_scene_test_builds_cuda_persistent_graph_generic_args_four_slots():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("e", _FakeTensor(17)),
        Tensor("f", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "c", "d", "e", "f", "out"],
        "queue_capacity": 2,
        "temporaries": {"tmp0": "out", "tmp1": "out"},
        "graph": {
            "tasks": [
                {
                    "func_id": 9,
                    "a": "a",
                    "b": "b",
                    "out": "tmp0",
                    "dependents": [2],
                    "tensor_args": ["c", "d", "e", "f"],
                    "scalar_args": [1.5, 0.25, 0.125, 0.0625],
                },
                {"func_id": 2, "a": "a", "b": "b", "out": "tmp1", "dependents": [2]},
                {"func_id": 1, "a": "tmp0", "b": "tmp1", "out": "out", "initial_fanin": 2},
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 3
    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (9, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]
    assert list(buffers.host_tasks[0].tensor_args) == [
        buffers.tensor_buffers.ptrs["c"],
        buffers.tensor_buffers.ptrs["d"],
        buffers.tensor_buffers.ptrs["e"],
        buffers.tensor_buffers.ptrs["f"],
    ]
    assert list(buffers.host_tasks[0].scalar_args) == pytest.approx([1.5, 0.25, 0.125, 0.0625])
    assert buffers.host_tasks[0].tensor_arg_count == 4
    assert buffers.host_tasks[0].scalar_arg_count == 4
    assert buffers.host_tasks[2].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].b == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_infers_cuda_persistent_graph_edges_from_tensor_flow():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "c", "d", "out"],
        "queue_capacity": 2,
        "temporaries": {"tmp0": "out", "tmp1": "out"},
        "graph": {
            "tasks": [
                {
                    "func_id": 9,
                    "a": "a",
                    "b": "b",
                    "out": "tmp0",
                    "tensor_args": ["c", "d"],
                    "scalar_args": [1.5, 0.25],
                },
                {"func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
                {"func_id": 1, "a": "tmp0", "b": "tmp1", "out": "out"},
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (9, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_infers_missing_cuda_persistent_graph_edges_per_task():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "c", "d", "out"],
        "queue_capacity": 2,
        "temporaries": {"tmp0": "out", "tmp1": "out"},
        "graph": {
            "tasks": [
                {
                    "func_id": 9,
                    "a": "a",
                    "b": "b",
                    "out": "tmp0",
                    "dependents": [2],
                    "tensor_args": ["c", "d"],
                    "scalar_args": [1.5, 0.25],
                },
                {"func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
                {"func_id": 1, "a": "tmp0", "b": "tmp1", "out": "out"},
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (9, 0, 1),
        (2, 1, 1),
        (1, 2, 0),
    ]


def test_scene_test_infers_cuda_persistent_graph_edges_independent_of_task_order():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "c", "d", "out"],
        "queue_capacity": 2,
        "temporaries": {"tmp0": "out", "tmp1": "out"},
        "graph": {
            "tasks": [
                {"func_id": 1, "a": "tmp0", "b": "tmp1", "out": "out"},
                {
                    "func_id": 9,
                    "a": "a",
                    "b": "b",
                    "out": "tmp0",
                    "tensor_args": ["c", "d"],
                    "scalar_args": [1.5, 0.25],
                },
                {"func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert list(buffers.host_fanin) == [2, 0, 0]
    assert list(buffers.host_dependents) == [0, 0]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 0),
        (9, 0, 1),
        (2, 1, 1),
    ]


def test_scene_test_builds_cuda_persistent_graph_tensor_tile_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(256)),
        Tensor("b", _FakeTensor(256)),
        Tensor("out", _FakeTensor(256)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "temporaries": {"tmp0": "out", "tmp1": "out", "tmp2": "out"},
        "graph": {
            "tasks": [
                {
                    "func_id": 3,
                    "a": "a",
                    "b": "b",
                    "out": "tmp0",
                    "rows": 16,
                    "cols": 16,
                    "inner": 16,
                    "lda": 16,
                    "ldb": 16,
                    "ldc": 16,
                    "a_batch_stride": 256,
                    "b_batch_stride": 256,
                    "out_batch_stride": 256,
                },
                {"func_id": 1, "a": "tmp0", "b": "a", "out": "tmp1"},
                {"func_id": 2, "a": "tmp0", "b": "b", "out": "tmp2"},
                {"func_id": 1, "a": "tmp1", "b": "tmp2", "out": "out"},
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 4
    assert list(buffers.host_fanin) == [0, 1, 1, 2]
    assert list(buffers.host_dependents) == [1, 2, 3, 3]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (3, 0, 2),
        (1, 2, 1),
        (2, 3, 1),
        (1, 4, 0),
    ]
    assert buffers.host_tasks[0].rows == 16
    assert buffers.host_tasks[0].cols == 16
    assert buffers.host_tasks[0].inner == 16
    assert buffers.host_tasks[0].lda == 16
    assert buffers.host_tasks[0].ldb == 16
    assert buffers.host_tasks[0].ldc == 16
    assert buffers.host_tasks[0].a_batch_stride == 256
    assert buffers.host_tasks[0].b_batch_stride == 256
    assert buffers.host_tasks[0].out_batch_stride == 256
    assert buffers.host_tasks[3].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_auto_allocates_cuda_persistent_graph_temporaries():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("c", _FakeTensor(17)),
        Tensor("d", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "c", "d", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {
                    "func_id": 9,
                    "a": "a",
                    "b": "b",
                    "out": "tmp0",
                    "dependents": [2],
                    "tensor_args": ["c", "d"],
                    "scalar_args": [1.5, 0.25],
                },
                {"func_id": 2, "a": "a", "b": "b", "out": "tmp1", "dependents": [2]},
                {"func_id": 1, "a": "tmp0", "b": "tmp1", "out": "out", "initial_fanin": 2},
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert buffers.host_tasks[0].out == buffers.dev_tmp0
    assert buffers.host_tasks[1].out == buffers.dev_tmp1
    assert buffers.host_tasks[2].a == buffers.dev_tmp0
    assert buffers.host_tasks[2].b == buffers.dev_tmp1
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_with_reused_output_storage():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 3,
        "temporaries": {"tmp0": "out", "tmp1": "out", "tmp2": "out", "tmp3": "out"},
        "graph": {
            "tasks": [
                {"func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
                {"func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
                {"func_id": 1, "a": "tmp0", "b": "tmp1", "out": "tmp2"},
                {"func_id": 2, "a": "tmp2", "b": "b", "out": "tmp3"},
                {
                    "func_id": 1,
                    "a": "tmp2",
                    "b": "a",
                    "out": "tmp4",
                    "out_storage": "tmp0",
                },
                {"func_id": 1, "a": "tmp4", "b": "tmp3", "out": "out"},
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 6
    assert list(buffers.host_fanin) == [0, 0, 2, 1, 1, 2]
    assert list(buffers.host_dependents) == [2, 2, 3, 4, 5, 5]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 2),
        (2, 4, 1),
        (1, 5, 1),
        (1, 6, 0),
    ]
    assert buffers.host_tasks[0].out == buffers.host_tasks[4].out
    assert buffers.host_tasks[5].a == buffers.host_tasks[4].out
    assert buffers.host_tasks[5].b == buffers.host_tasks[3].out
    assert buffers.host_tasks[5].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_rejects_unknown_cuda_persistent_graph_output_storage():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "graph": {
            "tasks": [
                {"func_id": 1, "a": "a", "b": "b", "out": "tmp0"},
                {
                    "func_id": 1,
                    "a": "tmp0",
                    "b": "a",
                    "out": "tmp1",
                    "out_storage": "typo_storage",
                },
                {"func_id": 1, "a": "tmp1", "b": "b", "out": "out"},
            ]
        },
    }

    with pytest.raises(ValueError, match="out_storage.*typo_storage"):
        _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)


def test_scene_test_resolves_cuda_persistent_graph_scalar_field_names():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
        Scalar("alpha", ctypes.c_float(2.0)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "temporaries": {"tmp0": "out", "tmp1": "out"},
        "graph": {
            "tasks": [
                {"func_id": 11, "a": "a", "out": "tmp0", "scalar0": "alpha"},
                {"func_id": 2, "a": "a", "b": "b", "out": "tmp1"},
                {"func_id": 1, "a": "tmp0", "b": "tmp1", "out": "out"},
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 3
    assert list(buffers.host_fanin) == [0, 0, 2]
    assert list(buffers.host_dependents) == [2, 2]
    assert buffers.host_tasks[0].func_id == 11
    assert buffers.host_tasks[0].scalar0 == pytest.approx(2.0)
    assert buffers.host_tasks[2].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[2].b == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_unary_square_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_unary_square_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 3
    assert list(buffers.host_fanin) == [0, 1, 1]
    assert list(buffers.host_dependents) == [1, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (7, 0, 1),
        (1, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[0].a == buffers.tensor_buffers.ptrs["a"]
    assert buffers.host_tasks[0].b is None
    assert buffers.host_tasks[1].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[1].b == buffers.tensor_buffers.ptrs["b"]
    assert buffers.host_tasks[2].a == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].b == buffers.tensor_buffers.ptrs["a"]
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_unary_square_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(17)),
        Tensor("b", _FakeTensor(17)),
        Tensor("out", _FakeTensor(17)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "temporaries": {"tmp0": "out", "tmp1": "out"},
        "graph": {
            "tasks": [
                {"func_id": 7, "a": "a", "out": "tmp0"},
                {"func_id": 1, "a": "tmp0", "b": "b", "out": "tmp1"},
                {"func_id": 1, "a": "tmp1", "b": "a", "out": "out"},
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 3
    assert list(buffers.host_fanin) == [0, 1, 1]
    assert list(buffers.host_dependents) == [1, 2]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (7, 0, 1),
        (1, 1, 1),
        (1, 2, 0),
    ]
    assert buffers.host_tasks[0].a == buffers.tensor_buffers.ptrs["a"]
    assert buffers.host_tasks[0].b is None
    assert buffers.host_tasks[1].a == buffers.host_tasks[0].out
    assert buffers.host_tasks[1].b == buffers.tensor_buffers.ptrs["b"]
    assert buffers.host_tasks[2].a == buffers.host_tasks[1].out
    assert buffers.host_tasks[2].b == buffers.tensor_buffers.ptrs["a"]
    assert buffers.host_tasks[2].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_tensor_core_tile_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(256)),
        Tensor("b", _FakeTensor(256)),
        Tensor("out", _FakeTensor(256)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_tensor_core_tile_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "tensor_tile": {"rows": 16, "cols": 16, "inner": 16},
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 4
    assert list(buffers.host_fanin) == [0, 1, 1, 2]
    assert list(buffers.host_dependents) == [1, 2, 3, 3]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (10, 0, 2),
        (1, 2, 1),
        (2, 3, 1),
        (1, 4, 0),
    ]
    assert buffers.host_tasks[0].rows == 16
    assert buffers.host_tasks[0].cols == 16
    assert buffers.host_tasks[0].inner == 16
    assert buffers.host_tasks[3].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_multi_fragment_tensor_core_tile_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(512)),
        Tensor("b", _FakeTensor(512)),
        Tensor("out", _FakeTensor(512)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_tensor_core_tile_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "tensor_tile": {"rows": 32, "cols": 16, "inner": 16},
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 4
    assert buffers.host_tasks[0].func_id == 10
    assert buffers.host_tasks[0].rows == 32
    assert buffers.host_tasks[0].cols == 16
    assert buffers.host_tasks[0].inner == 16
    assert buffers.host_tasks[0].a_batch_stride == 512
    assert buffers.host_tasks[0].b_batch_stride == 256
    assert buffers.host_tasks[0].out_batch_stride == 512
    assert buffers.host_tasks[3].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_tensor_core_tile_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(256)),
        Tensor("b", _FakeTensor(256)),
        Tensor("out", _FakeTensor(256)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "temporaries": {"tmp0": "out", "tmp1": "out", "tmp2": "out"},
        "graph": {
            "tasks": [
                {
                    "func_id": 10,
                    "a": "a",
                    "b": "b",
                    "out": "tmp0",
                    "rows": 16,
                    "cols": 16,
                    "inner": 16,
                    "lda": 16,
                    "ldb": 16,
                    "ldc": 16,
                    "a_batch_stride": 256,
                    "b_batch_stride": 256,
                    "out_batch_stride": 256,
                },
                {"func_id": 1, "a": "tmp0", "b": "a", "out": "tmp1"},
                {"func_id": 2, "a": "tmp0", "b": "b", "out": "tmp2"},
                {"func_id": 1, "a": "tmp1", "b": "tmp2", "out": "out"},
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 4
    assert list(buffers.host_fanin) == [0, 1, 1, 2]
    assert list(buffers.host_dependents) == [1, 2, 3, 3]
    assert [(task.func_id, task.dependent_begin, task.dependent_count) for task in buffers.host_tasks] == [
        (10, 0, 2),
        (1, 2, 1),
        (2, 3, 1),
        (1, 4, 0),
    ]
    assert buffers.host_tasks[0].rows == 16
    assert buffers.host_tasks[0].cols == 16
    assert buffers.host_tasks[0].inner == 16
    assert buffers.host_tasks[3].out == buffers.tensor_buffers.ptrs["out"]


def test_scene_test_builds_cuda_persistent_graph_multi_fragment_tensor_core_tile_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(512)),
        Tensor("b", _FakeTensor(512)),
        Tensor("out", _FakeTensor(512)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "temporaries": {"tmp0": "out"},
        "graph": {
            "tasks": [
                {
                    "func_id": 10,
                    "a": "a",
                    "b": "b",
                    "out": "tmp0",
                    "rows": 32,
                    "cols": 16,
                    "inner": 16,
                    "lda": 16,
                    "ldb": 16,
                    "ldc": 16,
                    "a_batch_stride": 512,
                    "b_batch_stride": 256,
                    "out_batch_stride": 512,
                }
            ]
        },
    }
    buffers = _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)

    assert len(buffers.host_tasks) == 1
    assert buffers.host_tasks[0].func_id == 10
    assert buffers.host_tasks[0].rows == 32
    assert buffers.host_tasks[0].cols == 16
    assert buffers.host_tasks[0].inner == 16
    assert buffers.host_tasks[0].out == buffers.dev_tmp0


def test_scene_test_rejects_incompatible_cuda_persistent_graph_tensor_core_tile_args():
    test_args = TaskArgsBuilder(
        Tensor("a", _FakeTensor(128)),
        Tensor("b", _FakeTensor(256)),
        Tensor("out", _FakeTensor(128)),
    )
    cuda_spec = {
        "arg_builder": "persistent_dag_graph_f32",
        "args": ["a", "b", "out"],
        "queue_capacity": 2,
        "temporaries": {"tmp0": "out"},
        "graph": {
            "tasks": [
                {
                    "func_id": 10,
                    "a": "a",
                    "b": "b",
                    "out": "tmp0",
                    "rows": 8,
                    "cols": 16,
                    "inner": 16,
                    "lda": 16,
                    "ldb": 16,
                    "ldc": 16,
                    "a_batch_stride": 128,
                    "b_batch_stride": 256,
                    "out_batch_stride": 128,
                }
            ]
        },
    }

    with pytest.raises(ValueError, match="graph.*tensor.*core.*rows/cols multiples of 16"):
        _CudaPersistentDagSceneBuffers(_FakeWorker(), test_args, cuda_spec)


@requires_cuda
def test_scene_test_runs_cuda_host_schedule_vector_add_with_real_data(tmp_path):
    torch = pytest.importorskip("torch")

    source = tmp_path / "vector_add.pto.cu"
    source.write_text(_VECTOR_ADD_BODY)

    @scene_test(level=2, runtime="host_schedule")
    class CudaVectorAddScene(SceneTestCase):
        CALLABLE = _cuda_vector_add_spec(source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            return TaskArgsBuilder(
                Tensor("a", torch.arange(n, dtype=torch.float32)),
                Tensor("b", torch.arange(n, dtype=torch.float32) * 2.0),
                Tensor("out", torch.zeros(n, dtype=torch.float32)),
            )

        def compute_golden(self, args, params):
            args.out[:] = args.a + args.b

    scene = CudaVectorAddScene()
    worker = CudaVectorAddScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(worker, callable_obj, CudaVectorAddScene.CASES[0])
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_host_schedule_elementwise_binary_with_real_data(tmp_path):
    torch = pytest.importorskip("torch")

    source = tmp_path / "vector_mul.pto.cu"
    source.write_text(_VECTOR_MUL_BODY)

    @scene_test(level=2, runtime="host_schedule")
    class CudaVectorMulScene(SceneTestCase):
        CALLABLE = _cuda_elementwise_binary_spec(source, task_name="vector_mul")
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            return TaskArgsBuilder(
                Tensor("a", torch.arange(n, dtype=torch.float32) + 1.0),
                Tensor("b", torch.arange(n, dtype=torch.float32) * 0.5),
                Tensor("out", torch.zeros(n, dtype=torch.float32)),
            )

        def compute_golden(self, args, params):
            args.out[:] = args.a * args.b

    scene = CudaVectorMulScene()
    worker = CudaVectorMulScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(worker, callable_obj, CudaVectorMulScene.CASES[0])
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_host_schedule_elementwise_unary_with_real_data(tmp_path):
    torch = pytest.importorskip("torch")

    source = tmp_path / "vector_square.pto.cu"
    source.write_text(_VECTOR_SQUARE_BODY)

    @scene_test(level=2, runtime="host_schedule")
    class CudaVectorSquareScene(SceneTestCase):
        CALLABLE = _cuda_elementwise_unary_spec(source, task_name="vector_square")
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            return TaskArgsBuilder(
                Tensor("a", torch.arange(n, dtype=torch.float32) + 1.0),
                Tensor("out", torch.zeros(n, dtype=torch.float32)),
            )

        def compute_golden(self, args, params):
            args.out[:] = args.a * args.a

    scene = CudaVectorSquareScene()
    worker = CudaVectorSquareScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(worker, callable_obj, CudaVectorSquareScene.CASES[0])
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_host_schedule_elementwise_scale_with_real_data(tmp_path):
    torch = pytest.importorskip("torch")

    source = tmp_path / "vector_scale.pto.cu"
    source.write_text(_VECTOR_SCALE_BODY)

    @scene_test(level=2, runtime="host_schedule")
    class CudaVectorScaleScene(SceneTestCase):
        CALLABLE = _cuda_elementwise_scale_spec(source, task_name="vector_scale")
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024, "alpha": 1.5},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            return TaskArgsBuilder(
                Tensor("a", torch.arange(n, dtype=torch.float32) + 1.0),
                Tensor("out", torch.zeros(n, dtype=torch.float32)),
                Scalar("alpha", ctypes.c_float(params["alpha"])),
            )

        def compute_golden(self, args, params):
            args.out[:] = args.a * params["alpha"]

    scene = CudaVectorScaleScene()
    worker = CudaVectorScaleScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(worker, callable_obj, CudaVectorScaleScene.CASES[0])
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_host_schedule_elementwise_axpy_with_real_data(tmp_path):
    torch = pytest.importorskip("torch")

    source = tmp_path / "vector_axpy.pto.cu"
    source.write_text(_VECTOR_AXPY_BODY)

    @scene_test(level=2, runtime="host_schedule")
    class CudaVectorAxpyScene(SceneTestCase):
        CALLABLE = _cuda_elementwise_axpy_spec(source, task_name="vector_axpy")
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024, "alpha": 1.5},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            return TaskArgsBuilder(
                Tensor("a", torch.arange(n, dtype=torch.float32) + 1.0),
                Tensor("b", torch.arange(n, dtype=torch.float32) * 0.5),
                Tensor("out", torch.zeros(n, dtype=torch.float32)),
                Scalar("alpha", ctypes.c_float(params["alpha"])),
            )

        def compute_golden(self, args, params):
            args.out[:] = params["alpha"] * args.a + args.b

    scene = CudaVectorAxpyScene()
    worker = CudaVectorAxpyScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(worker, callable_obj, CudaVectorAxpyScene.CASES[0])
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_host_schedule_elementwise_affine_with_real_data(tmp_path):
    torch = pytest.importorskip("torch")

    source = tmp_path / "vector_affine.pto.cu"
    source.write_text(_VECTOR_AFFINE_BODY)

    @scene_test(level=2, runtime="host_schedule")
    class CudaVectorAffineScene(SceneTestCase):
        CALLABLE = _cuda_elementwise_affine_spec(source, task_name="vector_affine")
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024, "alpha": 1.5, "beta": 0.5},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            return TaskArgsBuilder(
                Tensor("a", torch.arange(n, dtype=torch.float32) + 1.0),
                Tensor("b", torch.arange(n, dtype=torch.float32) * 0.5),
                Tensor("out", torch.zeros(n, dtype=torch.float32)),
                Scalar("alpha", ctypes.c_float(params["alpha"])),
                Scalar("beta", ctypes.c_float(params["beta"])),
            )

        def compute_golden(self, args, params):
            args.out[:] = params["alpha"] * args.a + params["beta"] * args.b

    scene = CudaVectorAffineScene()
    worker = CudaVectorAffineScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(worker, callable_obj, CudaVectorAffineScene.CASES[0])
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_host_schedule_elementwise_triad_with_real_data(tmp_path):
    torch = pytest.importorskip("torch")

    source = tmp_path / "vector_triad.pto.cu"
    source.write_text(_VECTOR_TRIAD_BODY)

    @scene_test(level=2, runtime="host_schedule")
    class CudaVectorTriadScene(SceneTestCase):
        CALLABLE = _cuda_elementwise_triad_spec(source, task_name="vector_triad")
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            return TaskArgsBuilder(
                Tensor("a", torch.arange(n, dtype=torch.float32) + 1.0),
                Tensor("b", torch.arange(n, dtype=torch.float32) * 0.5),
                Tensor("c", torch.arange(n, dtype=torch.float32) * 0.25),
                Tensor("out", torch.zeros(n, dtype=torch.float32)),
            )

        def compute_golden(self, args, params):
            args.out[:] = args.a * args.b + args.c

    scene = CudaVectorTriadScene()
    worker = CudaVectorTriadScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(worker, callable_obj, CudaVectorTriadScene.CASES[0])
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_host_schedule_elementwise_quad_with_real_data(tmp_path):
    torch = pytest.importorskip("torch")

    source = tmp_path / "vector_quad.pto.cu"
    source.write_text(_VECTOR_QUAD_BODY)

    @scene_test(level=2, runtime="host_schedule")
    class CudaVectorQuadScene(SceneTestCase):
        CALLABLE = _cuda_elementwise_quad_spec(source, task_name="vector_quad")
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            return TaskArgsBuilder(
                Tensor("a", torch.arange(n, dtype=torch.float32) + 1.0),
                Tensor("b", torch.arange(n, dtype=torch.float32) * 0.5),
                Tensor("c", torch.arange(n, dtype=torch.float32) * 0.25),
                Tensor("d", torch.arange(n, dtype=torch.float32) * 0.125),
                Tensor("out", torch.zeros(n, dtype=torch.float32)),
            )

        def compute_golden(self, args, params):
            args.out[:] = args.a * args.b + args.c * args.d

    scene = CudaVectorQuadScene()
    worker = CudaVectorQuadScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(worker, callable_obj, CudaVectorQuadScene.CASES[0])
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_host_schedule_elementwise_generic_args_with_ctypes_data(tmp_path):
    source = tmp_path / "vector_generic_args.pto.cu"
    source.write_text(_VECTOR_GENERIC_ARGS_BODY)

    @scene_test(level=2, runtime="host_schedule")
    class CudaVectorGenericArgsScene(SceneTestCase):
        CALLABLE = _cuda_elementwise_generic_args_spec(source, task_name="vector_generic_args")
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024, "alpha": 1.5, "beta": 0.25},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
                Scalar("alpha", ctypes.c_float(params["alpha"])),
                Scalar("beta", ctypes.c_float(params["beta"])),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaVectorGenericArgsScene()
    worker = CudaVectorGenericArgsScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaVectorGenericArgsScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx] + c_values[idx] + 0.25 * d_values[idx] + b_values[idx] for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_host_schedule_elementwise_generic_args_four_slots_with_ctypes_data(tmp_path):
    source = tmp_path / "vector_generic_args4.pto.cu"
    source.write_text(_VECTOR_GENERIC_ARGS4_BODY)

    @scene_test(level=2, runtime="host_schedule")
    class CudaVectorGenericArgs4Scene(SceneTestCase):
        CALLABLE = _cuda_elementwise_generic_args4_spec(source, task_name="vector_generic_args4")
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024, "alpha": 1.5, "beta": 0.25, "gamma": 0.125, "delta": 0.0625},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("e", _CtypesFloatTensor(float(i) * 0.0625 for i in range(n))),
                Tensor("f", _CtypesFloatTensor(float(i) * 0.03125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
                Scalar("alpha", ctypes.c_float(params["alpha"])),
                Scalar("beta", ctypes.c_float(params["beta"])),
                Scalar("gamma", ctypes.c_float(params["gamma"])),
                Scalar("delta", ctypes.c_float(params["delta"])),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaVectorGenericArgs4Scene()
    worker = CudaVectorGenericArgs4Scene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaVectorGenericArgs4Scene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        e_values = args.e.to_list()
        f_values = args.f.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx]
            + c_values[idx]
            + 0.25 * d_values[idx]
            + 0.125 * e_values[idx]
            + 0.0625 * f_values[idx]
            + b_values[idx]
            for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_dag_with_real_data(tmp_path):
    torch = pytest.importorskip("torch")

    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentDagScene(SceneTestCase):
        CALLABLE = _cuda_persistent_dag_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            return TaskArgsBuilder(
                Tensor("a", torch.arange(n, dtype=torch.float32)),
                Tensor("b", torch.arange(n, dtype=torch.float32) * 2.0),
                Tensor("out", torch.zeros(n, dtype=torch.float32)),
            )

        def compute_golden(self, args, params):
            args.out[:] = args.a + args.b + args.a * args.b

    scene = CudaPersistentDagScene()
    worker = CudaPersistentDagScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(worker, callable_obj, CudaPersistentDagScene.CASES[0])
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_chain_with_real_data(tmp_path):
    torch = pytest.importorskip("torch")

    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentChainScene(SceneTestCase):
        CALLABLE = _cuda_persistent_chain_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            return TaskArgsBuilder(
                Tensor("a", torch.arange(n, dtype=torch.float32) + 1.0),
                Tensor("b", torch.arange(n, dtype=torch.float32) * 0.25),
                Tensor("out", torch.zeros(n, dtype=torch.float32)),
            )

        def compute_golden(self, args, params):
            tmp0 = args.a + args.b
            tmp1 = args.a * args.b
            tmp2 = tmp0 + tmp1
            tmp3 = tmp2 * args.b
            args.out[:] = tmp2 + tmp3

    scene = CudaPersistentChainScene()
    worker = CudaPersistentChainScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(worker, callable_obj, CudaPersistentChainScene.CASES[0])
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_reuse_with_real_data(tmp_path):
    torch = pytest.importorskip("torch")

    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentReuseScene(SceneTestCase):
        CALLABLE = _cuda_persistent_reuse_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            return TaskArgsBuilder(
                Tensor("a", torch.arange(n, dtype=torch.float32) + 1.0),
                Tensor("b", torch.arange(n, dtype=torch.float32) * 0.25),
                Tensor("out", torch.zeros(n, dtype=torch.float32)),
            )

        def compute_golden(self, args, params):
            tmp0_initial = args.a + args.b
            tmp1 = args.a * args.b
            tmp2 = tmp0_initial + tmp1
            tmp3 = tmp2 * args.b
            tmp0_reused = tmp2 + args.a
            args.out[:] = tmp0_reused + tmp3

    scene = CudaPersistentReuseScene()
    worker = CudaPersistentReuseScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(worker, callable_obj, CudaPersistentReuseScene.CASES[0])
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_scalar_axpy_with_real_data(tmp_path):
    torch = pytest.importorskip("torch")

    axpy_source = tmp_path / "axpy.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    axpy_source.write_text(_PERSISTENT_AXPY_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentScalarAxpyScene(SceneTestCase):
        CALLABLE = _cuda_persistent_scalar_axpy_spec(axpy_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024, "scalar0": 1.5},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            return TaskArgsBuilder(
                Tensor("a", torch.arange(n, dtype=torch.float32) + 1.0),
                Tensor("b", torch.arange(n, dtype=torch.float32) * 0.25),
                Tensor("out", torch.zeros(n, dtype=torch.float32)),
            )

        def compute_golden(self, args, params):
            tmp0 = params["scalar0"] * args.a + args.b
            tmp1 = args.a * args.b
            args.out[:] = tmp0 + tmp1

    scene = CudaPersistentScalarAxpyScene()
    worker = CudaPersistentScalarAxpyScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(worker, callable_obj, CudaPersistentScalarAxpyScene.CASES[0])
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_scalar_scale_with_real_data(tmp_path):
    scale_source = tmp_path / "scale.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    scale_source.write_text(_PERSISTENT_SCALE_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentScalarScaleScene(SceneTestCase):
        CALLABLE = _cuda_persistent_scalar_scale_spec(scale_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024, "scalar0": 2.0},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentScalarScaleScene()
    worker = CudaPersistentScalarScaleScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentScalarScaleScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [2.0 * a_values[idx] + a_values[idx] * b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_scalar_affine_with_real_data(tmp_path):
    torch = pytest.importorskip("torch")

    affine_source = tmp_path / "affine.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    affine_source.write_text(_PERSISTENT_AFFINE_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentScalarAffineScene(SceneTestCase):
        CALLABLE = _cuda_persistent_scalar_affine_spec(affine_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024, "scalar0": 1.5, "scalar1": 0.5},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            return TaskArgsBuilder(
                Tensor("a", torch.arange(n, dtype=torch.float32) + 1.0),
                Tensor("b", torch.arange(n, dtype=torch.float32) * 0.25),
                Tensor("out", torch.zeros(n, dtype=torch.float32)),
            )

        def compute_golden(self, args, params):
            tmp0 = params["scalar0"] * args.a + params["scalar1"] * args.b
            tmp1 = args.a * args.b
            args.out[:] = tmp0 + tmp1

    scene = CudaPersistentScalarAffineScene()
    worker = CudaPersistentScalarAffineScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(worker, callable_obj, CudaPersistentScalarAffineScene.CASES[0])
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_triad_with_real_data(tmp_path):
    torch = pytest.importorskip("torch")

    triad_source = tmp_path / "triad.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    triad_source.write_text(_PERSISTENT_TRIAD_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentTriadScene(SceneTestCase):
        CALLABLE = _cuda_persistent_triad_spec(triad_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            return TaskArgsBuilder(
                Tensor("a", torch.arange(n, dtype=torch.float32) + 1.0),
                Tensor("b", torch.arange(n, dtype=torch.float32) * 0.5),
                Tensor("c", torch.arange(n, dtype=torch.float32) * 0.25),
                Tensor("out", torch.zeros(n, dtype=torch.float32)),
            )

        def compute_golden(self, args, params):
            tmp0 = args.a * args.b + args.c
            tmp1 = args.a * args.b
            args.out[:] = tmp0 + tmp1

    scene = CudaPersistentTriadScene()
    worker = CudaPersistentTriadScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(worker, callable_obj, CudaPersistentTriadScene.CASES[0])
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_triad_with_ctypes_data(tmp_path):
    triad_source = tmp_path / "triad.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    triad_source.write_text(_PERSISTENT_TRIAD_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentTriadCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_triad_spec(triad_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentTriadCtypesScene()
    worker = CudaPersistentTriadCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentTriadCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        actual = args.out.to_list()
        expected = [
            2.0 * args.a.to_list()[idx] * args.b.to_list()[idx] + args.c.to_list()[idx] for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_quad_with_ctypes_data(tmp_path):
    quad_source = tmp_path / "quad.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    quad_source.write_text(_PERSISTENT_QUAD_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentQuadCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_quad_spec(quad_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentQuadCtypesScene()
    worker = CudaPersistentQuadCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentQuadCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        actual = args.out.to_list()
        expected = [2.0 * a_values[idx] * b_values[idx] + c_values[idx] * d_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_generic_args_with_ctypes_data(tmp_path):
    generic_source = tmp_path / "generic_args.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    generic_source.write_text(_PERSISTENT_GENERIC_ARGS_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentGenericArgsCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_generic_args_spec(generic_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentGenericArgsCtypesScene()
    worker = CudaPersistentGenericArgsCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentGenericArgsCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx] + c_values[idx] + 0.25 * d_values[idx] + a_values[idx] * b_values[idx]
            for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_generic_args_four_slots_with_ctypes_data(tmp_path):
    generic_source = tmp_path / "generic_args4.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    generic_source.write_text(_PERSISTENT_GENERIC_ARGS_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentGenericArgs4CtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_generic_args4_spec(generic_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("e", _CtypesFloatTensor(float(i) * 0.0625 for i in range(n))),
                Tensor("f", _CtypesFloatTensor(float(i) * 0.03125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentGenericArgs4CtypesScene()
    worker = CudaPersistentGenericArgs4CtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentGenericArgs4CtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        e_values = args.e.to_list()
        f_values = args.f.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx]
            + c_values[idx]
            + 0.25 * d_values[idx]
            + 0.125 * e_values[idx]
            + 0.0625 * f_values[idx]
            + a_values[idx] * b_values[idx]
            for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_graph_with_ctypes_data(tmp_path):
    generic_source = tmp_path / "generic_args.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    generic_source.write_text(_PERSISTENT_GENERIC_ARGS_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_graph_generic_args_spec(generic_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentGraphCtypesScene()
    worker = CudaPersistentGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx] + c_values[idx] + 0.25 * d_values[idx] + a_values[idx] * b_values[idx]
            for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_tagged_graph_with_ctypes_data(tmp_path):
    generic_source = tmp_path / "generic_args.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    generic_source.write_text(_PERSISTENT_GENERIC_ARGS_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentTaggedGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_tagged_graph_generic_args_spec(generic_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
                Scalar("alpha", ctypes.c_float(1.5)),
                Scalar("beta", ctypes.c_float(0.25)),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentTaggedGraphCtypesScene()
    worker = CudaPersistentTaggedGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentTaggedGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx] + c_values[idx] + 0.25 * d_values[idx] + a_values[idx] * b_values[idx]
            for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_named_callable_graph_with_ctypes_data(tmp_path):
    generic_source = tmp_path / "generic_args.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    generic_source.write_text(_PERSISTENT_GENERIC_ARGS_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentNamedCallableGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_named_callable_graph_spec(generic_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
                Scalar("alpha", ctypes.c_float(1.5)),
                Scalar("beta", ctypes.c_float(0.25)),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentNamedCallableGraphCtypesScene()
    worker = CudaPersistentNamedCallableGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentNamedCallableGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx] + c_values[idx] + 0.25 * d_values[idx] + a_values[idx] * b_values[idx]
            for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_compact_callable_dict_graph_with_ctypes_data(tmp_path):
    generic_source = tmp_path / "generic_args.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    generic_source.write_text(_PERSISTENT_GENERIC_ARGS_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentCompactCallableDictGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_named_callable_graph_spec(
            generic_source,
            add_source,
            mul_source,
            callable_dict_compact=True,
        )
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
                Scalar("alpha", ctypes.c_float(1.5)),
                Scalar("beta", ctypes.c_float(0.25)),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentCompactCallableDictGraphCtypesScene()
    worker = CudaPersistentCompactCallableDictGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentCompactCallableDictGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx] + c_values[idx] + 0.25 * d_values[idx] + a_values[idx] * b_values[idx]
            for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_named_callable_list_graph_with_ctypes_data(tmp_path):
    generic_source = tmp_path / "generic_args.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    generic_source.write_text(_PERSISTENT_GENERIC_ARGS_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentNamedCallableListGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_named_callable_graph_spec(
            generic_source,
            add_source,
            mul_source,
            callables_as_list=True,
        )
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
                Scalar("alpha", ctypes.c_float(1.5)),
                Scalar("beta", ctypes.c_float(0.25)),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentNamedCallableListGraphCtypesScene()
    worker = CudaPersistentNamedCallableListGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentNamedCallableListGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx] + c_values[idx] + 0.25 * d_values[idx] + a_values[idx] * b_values[idx]
            for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_callable_index_graph_with_ctypes_data(tmp_path):
    generic_source = tmp_path / "generic_args.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    generic_source.write_text(_PERSISTENT_GENERIC_ARGS_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentCallableIndexGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_named_callable_graph_spec(
            generic_source,
            add_source,
            mul_source,
            callables_as_list=True,
            callable_refs="index",
        )
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
                Scalar("alpha", ctypes.c_float(1.5)),
                Scalar("beta", ctypes.c_float(0.25)),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentCallableIndexGraphCtypesScene()
    worker = CudaPersistentCallableIndexGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentCallableIndexGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx] + c_values[idx] + 0.25 * d_values[idx] + a_values[idx] * b_values[idx]
            for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_unnamed_callable_index_graph_with_ctypes_data(tmp_path):
    generic_source = tmp_path / "generic_args.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    generic_source.write_text(_PERSISTENT_GENERIC_ARGS_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentUnnamedCallableIndexGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_named_callable_graph_spec(
            generic_source,
            add_source,
            mul_source,
            callables_as_list=True,
            callable_refs="index",
            callable_list_names=False,
        )
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
                Scalar("alpha", ctypes.c_float(1.5)),
                Scalar("beta", ctypes.c_float(0.25)),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentUnnamedCallableIndexGraphCtypesScene()
    worker = CudaPersistentUnnamedCallableIndexGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentUnnamedCallableIndexGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx] + c_values[idx] + 0.25 * d_values[idx] + a_values[idx] * b_values[idx]
            for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_compact_callable_index_graph_with_ctypes_data(tmp_path):
    generic_source = tmp_path / "generic_args.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    generic_source.write_text(_PERSISTENT_GENERIC_ARGS_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentCompactCallableIndexGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_named_callable_graph_spec(
            generic_source,
            add_source,
            mul_source,
            callables_as_list=True,
            callable_refs="index",
            callable_list_compact=True,
        )
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
                Scalar("alpha", ctypes.c_float(1.5)),
                Scalar("beta", ctypes.c_float(0.25)),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentCompactCallableIndexGraphCtypesScene()
    worker = CudaPersistentCompactCallableIndexGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentCompactCallableIndexGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx] + c_values[idx] + 0.25 * d_values[idx] + a_values[idx] * b_values[idx]
            for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_tagged_inout_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentTaggedInoutGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_tagged_inout_graph_spec(add_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentTaggedInoutGraphCtypesScene()
    worker = CudaPersistentTaggedInoutGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentTaggedInoutGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [2.0 * a_values[idx] + 2.0 * b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_role_keyed_inout_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentRoleKeyedInoutGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_tagged_inout_graph_spec(add_source, task_arg_role_key="role")
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentRoleKeyedInoutGraphCtypesScene()
    worker = CudaPersistentRoleKeyedInoutGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentRoleKeyedInoutGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [2.0 * a_values[idx] + 2.0 * b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_args_alias_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentArgsAliasGraphCtypesScene(SceneTestCase):
        CALLABLE = _use_cuda_graph_args_alias(
            _cuda_persistent_tagged_inout_graph_spec(add_source, task_arg_role_key="role")
        )
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentArgsAliasGraphCtypesScene()
    worker = CudaPersistentArgsAliasGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentArgsAliasGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [2.0 * a_values[idx] + 2.0 * b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_submits_alias_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentSubmitsAliasGraphCtypesScene(SceneTestCase):
        CALLABLE = _use_cuda_graph_submits_alias(
            _cuda_persistent_tagged_inout_graph_spec(add_source, task_arg_role_key="role")
        )
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentSubmitsAliasGraphCtypesScene()
    worker = CudaPersistentSubmitsAliasGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentSubmitsAliasGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [2.0 * a_values[idx] + 2.0 * b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_submit_groups_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentSubmitGroupsGraphCtypesScene(SceneTestCase):
        CALLABLE = _use_cuda_graph_submit_groups(
            _cuda_persistent_tagged_inout_graph_spec(add_source, task_arg_role_key="role")
        )
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentSubmitGroupsGraphCtypesScene()
    worker = CudaPersistentSubmitGroupsGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentSubmitGroupsGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [2.0 * a_values[idx] + 2.0 * b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_compact_role_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentCompactRoleGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_tagged_inout_graph_spec(add_source, compact_task_args=True)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentCompactRoleGraphCtypesScene()
    worker = CudaPersistentCompactRoleGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentCompactRoleGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [2.0 * a_values[idx] + 2.0 * b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_pair_task_args_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentPairTaskArgsGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_tagged_inout_graph_spec(add_source, pair_task_args=True)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentPairTaskArgsGraphCtypesScene()
    worker = CudaPersistentPairTaskArgsGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentPairTaskArgsGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [2.0 * a_values[idx] + 2.0 * b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_role_map_task_args_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentRoleMapTaskArgsGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_tagged_inout_graph_spec(add_source, map_task_args=True)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentRoleMapTaskArgsGraphCtypesScene()
    worker = CudaPersistentRoleMapTaskArgsGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentRoleMapTaskArgsGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [2.0 * a_values[idx] + 2.0 * b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_graph_generic_args_four_slots_with_ctypes_data(tmp_path):
    generic_source = tmp_path / "generic_args.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    generic_source.write_text(_PERSISTENT_GENERIC_ARGS_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentGraphGenericArgs4CtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_graph_generic_args4_spec(generic_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("e", _CtypesFloatTensor(float(i) * 0.0625 for i in range(n))),
                Tensor("f", _CtypesFloatTensor(float(i) * 0.03125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentGraphGenericArgs4CtypesScene()
    worker = CudaPersistentGraphGenericArgs4CtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentGraphGenericArgs4CtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        e_values = args.e.to_list()
        f_values = args.f.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx]
            + c_values[idx]
            + 0.25 * d_values[idx]
            + 0.125 * e_values[idx]
            + 0.0625 * f_values[idx]
            + a_values[idx] * b_values[idx]
            for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_inferred_graph_with_ctypes_data(tmp_path):
    generic_source = tmp_path / "generic_args.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    generic_source.write_text(_PERSISTENT_GENERIC_ARGS_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentInferredGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_inferred_graph_generic_args_spec(generic_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentInferredGraphCtypesScene()
    worker = CudaPersistentInferredGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentInferredGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx] + c_values[idx] + 0.25 * d_values[idx] + a_values[idx] * b_values[idx]
            for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_reordered_graph_with_ctypes_data(tmp_path):
    generic_source = tmp_path / "generic_args.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    generic_source.write_text(_PERSISTENT_GENERIC_ARGS_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentReorderedGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_reordered_graph_generic_args_spec(generic_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentReorderedGraphCtypesScene()
    worker = CudaPersistentReorderedGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentReorderedGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx] + c_values[idx] + 0.25 * d_values[idx] + a_values[idx] * b_values[idx]
            for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_mixed_graph_with_ctypes_data(tmp_path):
    generic_source = tmp_path / "generic_args.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    generic_source.write_text(_PERSISTENT_GENERIC_ARGS_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentMixedGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_mixed_graph_generic_args_spec(generic_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentMixedGraphCtypesScene()
    worker = CudaPersistentMixedGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentMixedGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx] + c_values[idx] + 0.25 * d_values[idx] + a_values[idx] * b_values[idx]
            for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_depends_on_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentDependsOnGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_depends_on_graph_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentDependsOnGraphCtypesScene()
    worker = CudaPersistentDependsOnGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentDependsOnGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_edge_list_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentEdgeListGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_edge_list_graph_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentEdgeListGraphCtypesScene()
    worker = CudaPersistentEdgeListGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentEdgeListGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_string_edge_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentStringEdgeGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_string_edge_graph_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentStringEdgeGraphCtypesScene()
    worker = CudaPersistentStringEdgeGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentStringEdgeGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_task_dict_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentTaskDictGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_task_dict_edge_graph_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentTaskDictGraphCtypesScene()
    worker = CudaPersistentTaskDictGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentTaskDictGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_node_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentNodeGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_node_graph_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentNodeGraphCtypesScene()
    worker = CudaPersistentNodeGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentNodeGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_node_io_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentNodeIoGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_node_io_graph_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentNodeIoGraphCtypesScene()
    worker = CudaPersistentNodeIoGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentNodeIoGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_node_id_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentNodeIdGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_node_id_graph_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentNodeIdGraphCtypesScene()
    worker = CudaPersistentNodeIdGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentNodeIdGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_node_op_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentNodeOpGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_node_op_graph_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentNodeOpGraphCtypesScene()
    worker = CudaPersistentNodeOpGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentNodeOpGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_node_data_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentNodeDataGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_node_data_graph_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentNodeDataGraphCtypesScene()
    worker = CudaPersistentNodeDataGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentNodeDataGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_node_link_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentNodeLinkGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_node_link_graph_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentNodeLinkGraphCtypesScene()
    worker = CudaPersistentNodeLinkGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentNodeLinkGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_node_port_dict_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentNodePortDictGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_node_port_dict_graph_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentNodePortDictGraphCtypesScene()
    worker = CudaPersistentNodePortDictGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentNodePortDictGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] + a_values[idx] * b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_callable_id_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentCallableIdGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_callable_id_graph_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentCallableIdGraphCtypesScene()
    worker = CudaPersistentCallableIdGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentCallableIdGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_node_attrs_graph_with_ctypes_data(tmp_path):
    generic_source = tmp_path / "generic_args.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    generic_source.write_text(_PERSISTENT_GENERIC_ARGS_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentNodeAttrsGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_node_attrs_graph_spec(generic_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
                Scalar("alpha", ctypes.c_float(1.5)),
                Scalar("beta", ctypes.c_float(0.25)),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentNodeAttrsGraphCtypesScene()
    worker = CudaPersistentNodeAttrsGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentNodeAttrsGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx] + c_values[idx] + 0.25 * d_values[idx] + a_values[idx] * b_values[idx]
            for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_adjacency_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentAdjacencyGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_adjacency_graph_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentAdjacencyGraphCtypesScene()
    worker = CudaPersistentAdjacencyGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentAdjacencyGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_dep_gen_edge_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentDepGenEdgeGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_dep_gen_edge_graph_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentDepGenEdgeGraphCtypesScene()
    worker = CudaPersistentDepGenEdgeGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentDepGenEdgeGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_dep_gen_task_id_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentDepGenTaskIdGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_dep_gen_task_id_graph_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentDepGenTaskIdGraphCtypesScene()
    worker = CudaPersistentDepGenTaskIdGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentDepGenTaskIdGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_annotated_dep_gen_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentAnnotatedDepGenGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_dep_gen_annotated_edge_graph_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentAnnotatedDepGenGraphCtypesScene()
    worker = CudaPersistentAnnotatedDepGenGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentAnnotatedDepGenGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_dep_gen_task_defaults_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentDepGenDefaultsGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_dep_gen_task_defaults_graph_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentDepGenDefaultsGraphCtypesScene()
    worker = CudaPersistentDepGenDefaultsGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentDepGenDefaultsGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_dep_gen_json_file_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    dep_graph_path = tmp_path / "deps.json"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)
    _write_dep_gen_json_file(dep_graph_path)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentDepGenJsonFileGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_dep_gen_json_file_graph_spec(add_source, mul_source, dep_graph_path)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentDepGenJsonFileGraphCtypesScene()
    worker = CudaPersistentDepGenJsonFileGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentDepGenJsonFileGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_dep_gen_task_overrides_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    dep_graph_path = tmp_path / "deps.json"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)
    _write_dep_gen_json_file(dep_graph_path)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentDepGenTaskOverridesGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_dep_gen_task_overrides_graph_spec(add_source, mul_source, dep_graph_path)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentDepGenTaskOverridesGraphCtypesScene()
    worker = CudaPersistentDepGenTaskOverridesGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentDepGenTaskOverridesGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] + a_values[idx] * b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_dep_gen_task_overrides_file_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    dep_graph_path = tmp_path / "deps.json"
    overrides_path = tmp_path / "task_overrides.json"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)
    _write_dep_gen_json_file(dep_graph_path)
    _write_dep_gen_task_overrides_file(overrides_path)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentDepGenTaskOverridesFileGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_dep_gen_task_overrides_file_graph_spec(
            add_source,
            mul_source,
            dep_graph_path,
            overrides_path,
        )
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentDepGenTaskOverridesFileGraphCtypesScene()
    worker = CudaPersistentDepGenTaskOverridesFileGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentDepGenTaskOverridesFileGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] + a_values[idx] * b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_dep_gen_task_metadata_list_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    dep_graph_path = tmp_path / "deps.json"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)
    _write_dep_gen_json_file(dep_graph_path)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentDepGenTaskMetadataListGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_dep_gen_task_metadata_graph_spec(add_source, mul_source, dep_graph_path)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentDepGenTaskMetadataListGraphCtypesScene()
    worker = CudaPersistentDepGenTaskMetadataListGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentDepGenTaskMetadataListGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] + a_values[idx] * b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_dep_gen_task_metadata_file_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    dep_graph_path = tmp_path / "deps.json"
    metadata_path = tmp_path / "task_metadata.json"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)
    _write_dep_gen_json_file(dep_graph_path)
    _write_dep_gen_task_metadata_file(metadata_path)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentDepGenTaskMetadataFileGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_dep_gen_task_metadata_file_graph_spec(
            add_source,
            mul_source,
            dep_graph_path,
            metadata_path,
        )
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentDepGenTaskMetadataFileGraphCtypesScene()
    worker = CudaPersistentDepGenTaskMetadataFileGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentDepGenTaskMetadataFileGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] + a_values[idx] * b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_inline_task_metadata_file_graph_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    metadata_path = tmp_path / "task_metadata.json"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)
    _write_dep_gen_task_metadata_file(metadata_path)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentInlineTaskMetadataFileGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_inline_task_metadata_file_graph_spec(add_source, mul_source, metadata_path)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentInlineTaskMetadataFileGraphCtypesScene()
    worker = CudaPersistentInlineTaskMetadataFileGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentInlineTaskMetadataFileGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] + b_values[idx] + a_values[idx] * b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_auto_temp_graph_with_ctypes_data(tmp_path):
    generic_source = tmp_path / "generic_args.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    generic_source.write_text(_PERSISTENT_GENERIC_ARGS_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentAutoTempGraphCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_auto_temp_graph_generic_args_spec(generic_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("c", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("d", _CtypesFloatTensor(float(i) * 0.125 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentAutoTempGraphCtypesScene()
    worker = CudaPersistentAutoTempGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentAutoTempGraphCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        c_values = args.c.to_list()
        d_values = args.d.to_list()
        actual = args.out.to_list()
        expected = [
            1.5 * a_values[idx] + c_values[idx] + 0.25 * d_values[idx] + a_values[idx] * b_values[idx]
            for idx in range(len(actual))
        ]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_graph_scratch_reuse_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentGraphScratchReuseCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_graph_scratch_reuse_spec(add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentGraphScratchReuseCtypesScene()
    worker = CudaPersistentGraphScratchReuseCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentGraphScratchReuseCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = []
        for idx in range(len(actual)):
            tmp2 = (a_values[idx] + b_values[idx]) + (a_values[idx] * b_values[idx])
            tmp3 = tmp2 * b_values[idx]
            tmp4 = tmp2 + a_values[idx]
            expected.append(tmp4 + tmp3)
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_graph_scalar_scale_with_ctypes_data(tmp_path):
    scale_source = tmp_path / "scale.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    scale_source.write_text(_PERSISTENT_SCALE_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentGraphScalarScaleCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_graph_scalar_scale_spec(scale_source, add_source, mul_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024, "scalar0": 2.0},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
                Scalar("alpha", ctypes.c_float(params["scalar0"])),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentGraphScalarScaleCtypesScene()
    worker = CudaPersistentGraphScalarScaleCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentGraphScalarScaleCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [2.0 * a_values[idx] + a_values[idx] * b_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_reports_cuda_persistent_scheduler_errors_with_ctypes_data(tmp_path):
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)
    callable_spec = _cuda_persistent_dag_spec(add_source, mul_source)
    callable_spec["cuda"]["arg_builder"] = "persistent_dag_graph_f32"
    callable_spec["cuda"]["temporaries"] = {"tmp0": "out"}
    callable_spec["cuda"]["graph"] = {
        "tasks": [
            {"func_id": 99, "a": "a", "b": "b", "out": "tmp0", "dependents": [1]},
            {"func_id": 1, "a": "tmp0", "b": "b", "out": "out", "initial_fanin": 1},
        ]
    }

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentBadGraphCtypesScene(SceneTestCase):
        CALLABLE = callable_spec
        CASES = [
            {
                "name": "n64",
                "platforms": ["cuda"],
                "params": {"n": 64},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            return TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.5 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit scheduler diagnostics")

    scene = CudaPersistentBadGraphCtypesScene()
    worker = CudaPersistentBadGraphCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        with pytest.raises(RuntimeError, match="CUDA persistent DAG scheduler error code=1 task_id=0 count=1"):
            scene._run_and_validate_l2(
                worker,
                callable_obj,
                CudaPersistentBadGraphCtypesScene.CASES[0],
                skip_golden=True,
            )
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_unary_square_with_ctypes_data(tmp_path):
    square_source = tmp_path / "square.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    square_source.write_text(_PERSISTENT_SQUARE_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentUnarySquareCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_unary_square_spec(square_source, add_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentUnarySquareCtypesScene()
    worker = CudaPersistentUnarySquareCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentUnarySquareCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] * a_values[idx] + b_values[idx] + a_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_graph_unary_square_with_ctypes_data(tmp_path):
    square_source = tmp_path / "square.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    square_source.write_text(_PERSISTENT_SQUARE_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentGraphUnarySquareCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_graph_unary_square_spec(square_source, add_source)
        CASES = [
            {
                "name": "n1024",
                "platforms": ["cuda"],
                "params": {"n": 1024},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            n = params["n"]
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float(i + 1) for i in range(n))),
                Tensor("b", _CtypesFloatTensor(float(i) * 0.25 for i in range(n))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(n))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentGraphUnarySquareCtypesScene()
    worker = CudaPersistentGraphUnarySquareCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentGraphUnarySquareCtypesScene.CASES[0],
            skip_golden=True,
        )
        args = scene.last_args
        a_values = args.a.to_list()
        b_values = args.b.to_list()
        actual = args.out.to_list()
        expected = [a_values[idx] * a_values[idx] + b_values[idx] + a_values[idx] for idx in range(len(actual))]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_tensor_tile_with_real_data(tmp_path):
    torch = pytest.importorskip("torch")

    matmul_source = tmp_path / "matmul.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    matmul_source.write_text(_PERSISTENT_MATMUL_TILE_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentTensorTileScene(SceneTestCase):
        CALLABLE = _cuda_persistent_tensor_tile_spec(matmul_source, add_source, mul_source)
        CASES = [
            {
                "name": "tile16",
                "platforms": ["cuda"],
                "params": {"rows": 16, "cols": 16, "inner": 16},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            rows = params["rows"]
            cols = params["cols"]
            inner = params["inner"]
            return TaskArgsBuilder(
                Tensor("a", (torch.arange(rows * inner, dtype=torch.float32) % 5.0) + 1.0),
                Tensor("b", (torch.arange(inner * cols, dtype=torch.float32) % 3.0) + 1.0),
                Tensor("out", torch.zeros(rows * cols, dtype=torch.float32)),
            )

        def compute_golden(self, args, params):
            rows = params["rows"]
            cols = params["cols"]
            inner = params["inner"]
            matmul = args.a.reshape(rows, inner) @ args.b.reshape(inner, cols)
            flat = matmul.reshape(rows * cols)
            args.out[:] = flat + args.a[: rows * cols] + flat * args.b[: rows * cols]

    scene = CudaPersistentTensorTileScene()
    worker = CudaPersistentTensorTileScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(worker, callable_obj, CudaPersistentTensorTileScene.CASES[0])
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_graph_tensor_tile_with_ctypes_data(tmp_path):
    matmul_source = tmp_path / "matmul.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    matmul_source.write_text(_PERSISTENT_MATMUL_TILE_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentGraphTensorTileCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_graph_tensor_tile_spec(matmul_source, add_source, mul_source)
        CASES = [
            {
                "name": "tile16",
                "platforms": ["cuda"],
                "params": {"rows": 16, "cols": 16, "inner": 16},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            rows = params["rows"]
            cols = params["cols"]
            inner = params["inner"]
            b_elems = max(rows * cols, inner * cols)
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float((i % 5) + 1) for i in range(rows * inner))),
                Tensor("b", _CtypesFloatTensor(float((i % 3) + 1) for i in range(b_elems))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(rows * cols))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentGraphTensorTileCtypesScene()
    worker = CudaPersistentGraphTensorTileCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentGraphTensorTileCtypesScene.CASES[0],
            skip_golden=True,
        )
        params = CudaPersistentGraphTensorTileCtypesScene.CASES[0]["params"]
        rows = params["rows"]
        cols = params["cols"]
        inner = params["inner"]
        a_values = scene.last_args.a.to_list()
        b_values = scene.last_args.b.to_list()
        actual = scene.last_args.out.to_list()
        matmul = []
        for row in range(rows):
            for col in range(cols):
                acc = 0.0
                for k in range(inner):
                    acc += a_values[row * inner + k] * b_values[k * cols + col]
                matmul.append(acc)
        expected = [matmul[i] + a_values[i] + matmul[i] * b_values[i] for i in range(rows * cols)]
        assert actual == pytest.approx(expected)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_graph_tensor_core_tile_with_ctypes_data(tmp_path):
    wmma_source = tmp_path / "wmma.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    wmma_source.write_text(_PERSISTENT_WMMA_TILE_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentGraphTensorCoreTileCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_graph_tensor_core_tile_spec(
            wmma_source,
            add_source,
            mul_source,
            stream_id=1,
        )
        CASES = [
            {
                "name": "tile16",
                "platforms": ["cuda"],
                "params": {"rows": 16, "cols": 16, "inner": 16},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            rows = params["rows"]
            cols = params["cols"]
            inner = params["inner"]
            b_elems = max(rows * cols, inner * cols)
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float((i % 5) + 1) for i in range(rows * inner))),
                Tensor("b", _CtypesFloatTensor(float((i % 3) + 1) for i in range(b_elems))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(rows * cols))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentGraphTensorCoreTileCtypesScene()
    worker = CudaPersistentGraphTensorCoreTileCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentGraphTensorCoreTileCtypesScene.CASES[0],
            skip_golden=True,
        )
        params = CudaPersistentGraphTensorCoreTileCtypesScene.CASES[0]["params"]
        rows = params["rows"]
        cols = params["cols"]
        inner = params["inner"]
        a_values = scene.last_args.a.to_list()
        b_values = scene.last_args.b.to_list()
        actual = scene.last_args.out.to_list()
        matmul = []
        for row in range(rows):
            for col in range(cols):
                acc = 0.0
                for k in range(inner):
                    acc += a_values[row * inner + k] * b_values[k * cols + col]
                matmul.append(acc)
        expected = [matmul[i] + a_values[i] + matmul[i] * b_values[i] for i in range(rows * cols)]
        assert actual == pytest.approx(expected, rel=1e-4, abs=1e-4)
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_tensor_core_tile_with_real_data(tmp_path):
    torch = pytest.importorskip("torch")

    wmma_source = tmp_path / "wmma.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    wmma_source.write_text(_PERSISTENT_WMMA_TILE_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentTensorCoreTileScene(SceneTestCase):
        RTOL = 1e-4
        ATOL = 1e-4
        CALLABLE = _cuda_persistent_tensor_core_tile_spec(wmma_source, add_source, mul_source)
        CASES = [
            {
                "name": "tile16",
                "platforms": ["cuda"],
                "params": {"rows": 16, "cols": 16, "inner": 16},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            rows = params["rows"]
            cols = params["cols"]
            inner = params["inner"]
            return TaskArgsBuilder(
                Tensor("a", (torch.arange(rows * inner, dtype=torch.float32) % 5.0) + 1.0),
                Tensor("b", (torch.arange(inner * cols, dtype=torch.float32) % 3.0) + 1.0),
                Tensor("out", torch.zeros(rows * cols, dtype=torch.float32)),
            )

        def compute_golden(self, args, params):
            rows = params["rows"]
            cols = params["cols"]
            inner = params["inner"]
            matmul = args.a.reshape(rows, inner) @ args.b.reshape(inner, cols)
            flat = matmul.reshape(rows * cols)
            args.out[:] = flat + args.a[: rows * cols] + flat * args.b[: rows * cols]

    scene = CudaPersistentTensorCoreTileScene()
    worker = CudaPersistentTensorCoreTileScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(worker, callable_obj, CudaPersistentTensorCoreTileScene.CASES[0])
    finally:
        worker.close()


@requires_cuda
def test_scene_test_runs_cuda_persistent_device_tensor_core_tile_with_ctypes_data(tmp_path):
    wmma_source = tmp_path / "wmma.pto.cu"
    add_source = tmp_path / "add.pto.cu"
    mul_source = tmp_path / "mul.pto.cu"
    wmma_source.write_text(_PERSISTENT_WMMA_TILE_BODY)
    add_source.write_text(_PERSISTENT_ADD_BODY)
    mul_source.write_text(_PERSISTENT_MUL_BODY)

    @scene_test(level=2, runtime="persistent_device")
    class CudaPersistentTensorCoreTileCtypesScene(SceneTestCase):
        CALLABLE = _cuda_persistent_tensor_core_tile_spec(
            wmma_source,
            add_source,
            mul_source,
            stream_id=1,
            tensor_tile={"rows": 32, "cols": 16, "inner": 16},
        )
        CASES = [
            {
                "name": "tile32x16",
                "platforms": ["cuda"],
                "params": {"rows": 32, "cols": 16, "inner": 16},
                "config": {"block_dim": 256},
            }
        ]

        def generate_args(self, params):
            rows = params["rows"]
            cols = params["cols"]
            inner = params["inner"]
            b_elems = max(rows * cols, inner * cols)
            args = TaskArgsBuilder(
                Tensor("a", _CtypesFloatTensor(float((i % 5) + 1) for i in range(rows * inner))),
                Tensor("b", _CtypesFloatTensor(float((i % 3) + 1) for i in range(b_elems))),
                Tensor("out", _CtypesFloatTensor(0.0 for _ in range(rows * cols))),
            )
            self.last_args = args
            return args

        def compute_golden(self, args, params):
            raise AssertionError("ctypes scene uses explicit post-run validation")

    scene = CudaPersistentTensorCoreTileCtypesScene()
    worker = CudaPersistentTensorCoreTileCtypesScene._create_worker("cuda", device_id=0, build=False)
    try:
        callable_obj = scene.build_callable("cuda")
        scene._run_and_validate_l2(
            worker,
            callable_obj,
            CudaPersistentTensorCoreTileCtypesScene.CASES[0],
            skip_golden=True,
        )
        params = CudaPersistentTensorCoreTileCtypesScene.CASES[0]["params"]
        rows = params["rows"]
        cols = params["cols"]
        inner = params["inner"]
        a_values = scene.last_args.a.to_list()
        b_values = scene.last_args.b.to_list()
        actual = scene.last_args.out.to_list()
        matmul = []
        for row in range(rows):
            for col in range(cols):
                acc = 0.0
                for k in range(inner):
                    acc += a_values[row * inner + k] * b_values[k * cols + col]
                matmul.append(acc)
        expected = [matmul[i] + a_values[i] + matmul[i] * b_values[i] for i in range(rows * cols)]
        assert actual == pytest.approx(expected, rel=1e-4, abs=1e-4)
    finally:
        worker.close()

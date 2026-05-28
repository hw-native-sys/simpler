/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef SRC_CUDA_PLATFORM_INCLUDE_HOST_PTO_CUDA_HOST_SCHEDULE_ABI_H_
#define SRC_CUDA_PLATFORM_INCLUDE_HOST_PTO_CUDA_HOST_SCHEDULE_ABI_H_

#include <stddef.h>
#include <stdint.h>

enum PtoCudaHostScheduleOp : uint32_t {
    PTO_CUDA_HOST_OP_VECTOR_ADD_F32 = 1,
    PTO_CUDA_HOST_OP_VECTOR_SCALE_F32 = 2,
    PTO_CUDA_HOST_OP_VECTOR_AXPY_F32 = 3,
    PTO_CUDA_HOST_OP_VECTOR_UNARY_F32 = 4,
    PTO_CUDA_HOST_OP_VECTOR_AFFINE_F32 = 5,
    PTO_CUDA_HOST_OP_VECTOR_TRIAD_F32 = 6,
    PTO_CUDA_HOST_OP_VECTOR_QUAD_F32 = 7,
    PTO_CUDA_HOST_OP_VECTOR_GENERIC_ARGS_F32 = 8,
};

struct PtoCudaHostCallable {
    uint32_t version;
    uint32_t op;
    const void *image;
    size_t image_size;
    const char *entry_name;
    uint32_t grid_dim;
    uint32_t block_dim;
    size_t shared_mem_bytes;
    uint32_t stream_id;
};

struct PtoCudaVectorAddArgs {
    const float *a;
    const float *b;
    float *out;
    uint64_t n;
};

struct PtoCudaVectorScaleArgs {
    const float *a;
    float *out;
    float alpha;
    uint64_t n;
};

struct PtoCudaVectorUnaryArgs {
    const float *a;
    float *out;
    uint64_t n;
};

struct PtoCudaVectorAxpyArgs {
    const float *a;
    const float *b;
    float *out;
    float alpha;
    uint64_t n;
};

struct PtoCudaVectorAffineArgs {
    const float *a;
    const float *b;
    float *out;
    float alpha;
    float beta;
    uint64_t n;
};

struct PtoCudaVectorTernaryArgs {
    const float *a;
    const float *b;
    const float *c;
    float *out;
    uint64_t n;
};

struct PtoCudaVectorQuaternaryArgs {
    const float *a;
    const float *b;
    const float *c;
    const float *d;
    float *out;
    uint64_t n;
};

struct PtoCudaVectorGenericArgs {
    const float *a;
    const float *b;
    float *out;
    const float *tensor_args[4];
    float scalar_args[4];
    uint32_t tensor_arg_count;
    uint32_t scalar_arg_count;
    uint64_t n;
};

#endif  // SRC_CUDA_PLATFORM_INCLUDE_HOST_PTO_CUDA_HOST_SCHEDULE_ABI_H_

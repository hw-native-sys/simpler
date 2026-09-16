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
#include "worker/runtime_c_api.h"
#include "platform_comm/comm.h"
#include "common/host_log_state.h"

// Sequential loader fixture: only the declaration and factory counter are live.
// Every loaded symbol uses its production signature; unexpected execution fails.
namespace {
PipelineContract contract{};
int create_count = 0;
}  // namespace
extern "C" {
void test_set_pipeline_contract(const PipelineContract *value) {
    contract = *value;
    create_count = 0;
}
int test_context_create_count() { return create_count; }
int simpler_host_log_bind_state(SimplerHostLogState *) { return 0; }

DeviceContextHandle create_device_context(void) {
    ++create_count;
    return nullptr;
}

void destroy_device_context(DeviceContextHandle ctx) {}

void *device_malloc_ctx(DeviceContextHandle ctx, size_t size) { return nullptr; }

void device_free_ctx(DeviceContextHandle ctx, void *dev_ptr) {}

size_t committed_device_memory_ctx(DeviceContextHandle ctx) { return 0; }

int device_memory_info_ctx(DeviceContextHandle ctx, DeviceMemoryInfo *info) { return PTO_RUNTIME_ERR_INTERNAL; }

int copy_to_device_ctx(DeviceContextHandle ctx, void *dev_ptr, const void *host_ptr, size_t size) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

int copy_from_device_ctx(DeviceContextHandle ctx, void *host_ptr, const void *dev_ptr, size_t size) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

size_t get_runtime_size(void) { return 0; }

size_t get_runtime_alignment(void) { return 0; }

int simpler_init(
    DeviceContextHandle ctx, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size,
    const uint8_t *aicore_binary, size_t aicore_size, const uint8_t *dispatcher_binary, size_t dispatcher_size,
    const CallConfig *prewarm_config, int enable_sdma, const void *sdma_warmup_binary, uint64_t sdma_warmup_size
) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

int simpler_register_callable(DeviceContextHandle ctx, int32_t callable_id, const void *callable) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

int simpler_run(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, const CallConfig *config,
    const NativeRunDescriptor *descriptor
) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

int simpler_prepare_run(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, const CallConfig *config,
    const NativeRunDescriptor *descriptor
) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

int simpler_launch_run(DeviceContextHandle ctx, RuntimeHandle runtime) { return PTO_RUNTIME_ERR_INTERNAL; }

int simpler_poll_run(DeviceContextHandle ctx, RuntimeHandle runtime) { return PTO_RUNTIME_ERR_INTERNAL; }

int simpler_wait_run(DeviceContextHandle ctx, RuntimeHandle runtime) { return PTO_RUNTIME_ERR_INTERNAL; }

int simpler_finalize_run(DeviceContextHandle ctx, RuntimeHandle runtime) { return PTO_RUNTIME_ERR_INTERNAL; }

int supports_concurrent_native_prepare_ctx(DeviceContextHandle ctx) { return PTO_RUNTIME_ERR_INTERNAL; }

uint64_t get_arena_bank_gm_heap_base_ctx(DeviceContextHandle ctx, uint32_t bank_id) { return 0; }

uint64_t get_retained_temp_addr_ctx(DeviceContextHandle ctx, uint32_t slot_id) { return 0; }

const PipelineContract *get_pipeline_contract(void) { return &contract; }

int simpler_unregister_callable(DeviceContextHandle ctx, int32_t callable_id) { return PTO_RUNTIME_ERR_INTERNAL; }

size_t get_aicpu_dlopen_count(DeviceContextHandle ctx) { return 0; }

size_t get_host_dlopen_count(DeviceContextHandle ctx) { return 0; }

size_t get_run_stream_set_create_count(DeviceContextHandle ctx) { return 0; }

int finalize_device(DeviceContextHandle ctx) { return PTO_RUNTIME_ERR_INTERNAL; }

int ensure_acl_ready_ctx(void *, int) { return PTO_RUNTIME_ERR_INTERNAL; }

void *create_comm_stream_ctx(void *) { return nullptr; }

int destroy_comm_stream_ctx(void *, void *) { return PTO_RUNTIME_ERR_INTERNAL; }

CommHandle comm_init(int rank, int nranks, void *stream, const char *rootinfo_path) { return nullptr; }

int comm_alloc_windows(CommHandle h, size_t win_size, uint64_t *device_ctx_out) { return PTO_RUNTIME_ERR_INTERNAL; }

int comm_get_local_window_base(CommHandle h, uint64_t *base_out) { return PTO_RUNTIME_ERR_INTERNAL; }

int comm_get_window_size(CommHandle h, size_t *size_out) { return PTO_RUNTIME_ERR_INTERNAL; }

int comm_derive_context(
    CommHandle h, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank, size_t window_offset,
    size_t window_size, uint64_t *device_ctx_out
) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

int comm_alloc_domain_windows(
    CommHandle h, uint64_t allocation_id, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank,
    size_t window_size, uint64_t *device_ctx_out, uint64_t *local_window_base_out
) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

int comm_release_domain_windows(CommHandle h, uint64_t allocation_id, size_t rank_count, uint32_t domain_rank) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

int comm_global_domain_prepare(
    uint64_t domain_id, uint32_t domain_rank, uint32_t rank_count, size_t window_size, uint32_t profile,
    CommGlobalDomainDescriptor *descriptor_out, uint64_t *local_window_base_out
) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

int comm_global_domain_import(
    uint64_t domain_id, const CommGlobalDomainDescriptor *descriptors, size_t descriptor_count, uint64_t *device_ctx_out
) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

int comm_global_domain_release(uint64_t domain_id) { return PTO_RUNTIME_ERR_INTERNAL; }

int comm_barrier(CommHandle h) { return PTO_RUNTIME_ERR_INTERNAL; }

int comm_destroy(CommHandle h) { return PTO_RUNTIME_ERR_INTERNAL; }

int simpler_kernel_mode_supported(DeviceContextHandle ctx) { return PTO_RUNTIME_ERR_INTERNAL; }

int simpler_kernel_mode_init(
    DeviceContextHandle ctx, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size,
    const uint8_t *aicore_binary, size_t aicore_size, const uint8_t *dispatcher_binary, size_t dispatcher_size,
    const CallConfig *config, uint64_t context_generation
) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

int simpler_kernel_mode_prepare_callable(
    DeviceContextHandle ctx, int32_t callable_id, const void *callable, size_t callable_size, void *caller_stream
) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

int simpler_kernel_mode_launch(DeviceContextHandle ctx, int32_t callable_id, const void *args, void *caller_stream) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

}  // extern "C"

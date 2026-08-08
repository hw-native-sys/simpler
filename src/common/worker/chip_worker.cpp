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

#include "chip_worker.h"

#include "chip_run_lane.h"
#include "pipeline_contract.h"

#include <dlfcn.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

template <typename T>
T load_symbol(void *handle, const char *name) {
    dlerror();  // clear any existing error
    void *sym = dlsym(handle, name);
    const char *err = dlerror();
    if (err) {
        std::string msg = "dlsym failed for '";
        msg += name;
        msg += "': ";
        msg += err;
        msg += "; every host runtime built from this source tree exports it, so rebuild the runtime";
        throw std::runtime_error(msg);
    }
    return reinterpret_cast<T>(sym);
}

uint64_t next_native_run_epoch() {
    static std::atomic<uint64_t> epoch{0};
    uint64_t current = epoch.load(std::memory_order_relaxed);
    while (current != std::numeric_limits<uint64_t>::max()) {
        if (epoch.compare_exchange_weak(current, current + 1, std::memory_order_relaxed)) {
            return current + 1;
        }
    }
    throw std::overflow_error("native-run epoch space is exhausted");
}

std::string format_native_run_identity(const ChipWorkerNativeRun &run) {
    return "(run_id=" + std::to_string(run.run_id) + " slot=" + std::to_string(run.slot_id) +
           " generation=" + std::to_string(run.generation) + " dispatch_id=" + std::to_string(run.dispatch_id) +
           " run_epoch=" + std::to_string(run.run_epoch) + ")";
}

std::vector<uint8_t> read_binary_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("Failed to open binary file: " + path);
    }
    auto size = f.tellg();
    if (size < 0) {
        throw std::runtime_error("Failed to determine size of binary file: " + path);
    }
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    f.seekg(0);
    if (size > 0 && !f.read(reinterpret_cast<char *>(buf.data()), size)) {
        throw std::runtime_error("Failed to read binary file: " + path);
    }
    return buf;
}

}  // namespace

ChipWorker::RuntimeStorage::RuntimeStorage(size_t size, size_t alignment) {
    void *storage = nullptr;
    if (posix_memalign(&storage, alignment, size) != 0) {
        throw std::bad_alloc();
    }
    std::memset(storage, 0, size);
    data_ = storage;
}

ChipWorker::RuntimeStorage::~RuntimeStorage() { std::free(data_); }

ChipWorker::RuntimeStorage::RuntimeStorage(RuntimeStorage &&other) noexcept :
    data_(other.data_) {
    other.data_ = nullptr;
}

ChipWorker::RuntimeStorage &ChipWorker::RuntimeStorage::operator=(RuntimeStorage &&other) noexcept {
    if (this == &other) return *this;
    std::free(data_);
    data_ = other.data_;
    other.data_ = nullptr;
    return *this;
}

ChipWorker::~ChipWorker() { finalize(); }

void ChipWorker::init(
    const std::string &host_lib_path, const std::string &aicpu_path, const std::string &aicore_path,
    const std::string &dispatcher_path, int device_id, const CallConfig *prewarm_config, uint32_t dma_workspace_mask
) {
    if (finalized_) {
        throw std::runtime_error("ChipWorker already finalized; cannot reinitialize");
    }
    if (initialized_) {
        throw std::runtime_error("ChipWorker already initialized; runtime cannot be changed");
    }
    if (device_id < 0) {
        throw std::runtime_error("ChipWorker::init requires a non-negative device_id");
    }
    pipeline_contract_ = {PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, 1, {}};

    // libsimpler_log.so (RTLD_GLOBAL, with HostLogger already seeded via
    // simpler_log_init) and — on sim — libcpu_sim_context.so (RTLD_GLOBAL) must
    // already be loaded by the caller; host_runtime.so resolves its undefined
    // HostLogger / unified_log_* (and, on sim, sim_context_*) symbols against
    // those globals. The Python `ChipWorker` wrapper does this preload.
    //
    // Host runtime SO is loaded with RTLD_LOCAL so that different runtimes'
    // identically-named symbols (simpler_init, simpler_register_callable,
    // simpler_run, etc.) do not collide when switching runtimes within the
    // same process.
    // Cross-runtime isolation relies on -fno-gnu-unique (#453) allowing
    // dlclose to actually unload the previous runtime's SO before loading
    // the next one.
    dlerror();
    void *handle = dlopen(host_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::string err = "dlopen failed: ";
        const char *msg = dlerror();
        err += msg ? msg : "unknown error";
        throw std::runtime_error(err);
    }

    GetPipelineContractFn get_pipeline_contract_fn = nullptr;
    try {
        create_device_context_fn_ = load_symbol<CreateDeviceContextFn>(handle, "create_device_context");
        destroy_device_context_fn_ = load_symbol<DestroyDeviceContextFn>(handle, "destroy_device_context");
        device_malloc_ctx_fn_ = load_symbol<DeviceMallocCtxFn>(handle, "device_malloc_ctx");
        device_free_ctx_fn_ = load_symbol<DeviceFreeCtxFn>(handle, "device_free_ctx");
        device_committed_memory_fn_ = load_symbol<GetCommittedDeviceMemoryFn>(handle, "committed_device_memory_ctx");
        copy_to_device_ctx_fn_ = load_symbol<CopyToDeviceCtxFn>(handle, "copy_to_device_ctx");
        copy_from_device_ctx_fn_ = load_symbol<CopyFromDeviceCtxFn>(handle, "copy_from_device_ctx");
        get_runtime_size_fn_ = load_symbol<GetRuntimeSizeFn>(handle, "get_runtime_size");
        get_runtime_alignment_fn_ = load_symbol<GetRuntimeAlignmentFn>(handle, "get_runtime_alignment");
        simpler_init_fn_ = load_symbol<SimplerInitFn>(handle, "simpler_init");
        register_callable_fn_ = load_symbol<SimplerRegisterCallableFn>(handle, "simpler_register_callable");
        run_fn_ = load_symbol<SimplerRunFn>(handle, "simpler_run");
        prepare_run_fn_ = load_symbol<SimplerPrepareRunFn>(handle, "simpler_prepare_run");
        launch_run_fn_ = load_symbol<SimplerNativeRunFn>(handle, "simpler_launch_run");
        poll_run_fn_ = load_symbol<SimplerNativeRunFn>(handle, "simpler_poll_run");
        wait_run_fn_ = load_symbol<SimplerNativeRunFn>(handle, "simpler_wait_run");
        finalize_run_fn_ = load_symbol<SimplerNativeRunFn>(handle, "simpler_finalize_run");
        supports_concurrent_native_prepare_fn_ =
            load_symbol<SupportsConcurrentNativePrepareFn>(handle, "supports_concurrent_native_prepare_ctx");
        get_arena_bank_gm_heap_base_fn_ =
            load_symbol<GetArenaBankGmHeapBaseFn>(handle, "get_arena_bank_gm_heap_base_ctx");
        get_retained_temp_addr_fn_ = load_symbol<GetRetainedTempAddrFn>(handle, "get_retained_temp_addr_ctx");
        get_pipeline_contract_fn = load_symbol<GetPipelineContractFn>(handle, "get_pipeline_contract");
        unregister_callable_fn_ = load_symbol<SimplerUnregisterCallableFn>(handle, "simpler_unregister_callable");
        get_aicpu_dlopen_count_fn_ = load_symbol<GetAicpuDlopenCountFn>(handle, "get_aicpu_dlopen_count");
        get_host_dlopen_count_fn_ = load_symbol<GetAicpuDlopenCountFn>(handle, "get_host_dlopen_count");
        get_run_stream_set_create_count_fn_ =
            load_symbol<GetAicpuDlopenCountFn>(handle, "get_run_stream_set_create_count");
        simpler_provision_dma_workspace_fn_ =
            load_symbol<SimplerProvisionDmaWorkspaceFn>(handle, "simpler_provision_dma_workspace");
        finalize_device_fn_ = load_symbol<FinalizeDeviceFn>(handle, "finalize_device");
        // ACL lifecycle + comm_* are part of the uniform host_runtime.so ABI.
        // Every platform runtime exports all of them — runtimes that do not
        // have a real backend (today: a5) ship not-supported stubs rather
        // than omitting the symbols.  This keeps ChipWorker.init platform-
        // agnostic: no per-symbol probing, no half-loaded extension groups.
        ensure_acl_ready_fn_ = load_symbol<EnsureAclReadyFn>(handle, "ensure_acl_ready_ctx");
        create_comm_stream_fn_ = load_symbol<CreateCommStreamFn>(handle, "create_comm_stream_ctx");
        destroy_comm_stream_fn_ = load_symbol<DestroyCommStreamFn>(handle, "destroy_comm_stream_ctx");
        comm_init_fn_ = load_symbol<CommInitFn>(handle, "comm_init");
        comm_alloc_windows_fn_ = load_symbol<CommAllocWindowsFn>(handle, "comm_alloc_windows");
        comm_get_local_window_base_fn_ = load_symbol<CommGetLocalWindowBaseFn>(handle, "comm_get_local_window_base");
        comm_get_window_size_fn_ = load_symbol<CommGetWindowSizeFn>(handle, "comm_get_window_size");
        comm_derive_context_fn_ = load_symbol<CommDeriveContextFn>(handle, "comm_derive_context");
        comm_alloc_domain_windows_fn_ = load_symbol<CommAllocDomainWindowsFn>(handle, "comm_alloc_domain_windows");
        comm_release_domain_windows_fn_ =
            load_symbol<CommReleaseDomainWindowsFn>(handle, "comm_release_domain_windows");
        comm_global_domain_prepare_fn_ = load_symbol<CommGlobalDomainPrepareFn>(handle, "comm_global_domain_prepare");
        comm_global_domain_import_fn_ = load_symbol<CommGlobalDomainImportFn>(handle, "comm_global_domain_import");
        comm_global_domain_release_fn_ = load_symbol<CommGlobalDomainReleaseFn>(handle, "comm_global_domain_release");
        comm_barrier_fn_ = load_symbol<CommBarrierFn>(handle, "comm_barrier");
        comm_destroy_fn_ = load_symbol<CommDestroyFn>(handle, "comm_destroy");
    } catch (...) {
        dlclose(handle);
        throw;
    }

    const PipelineContract *contract = get_pipeline_contract_fn();
    if (!is_valid_pipeline_contract(contract) || !has_serviceable_arena_topology(*contract)) {
        dlclose(handle);
        throw std::runtime_error("host runtime returned a PipelineContract this build cannot accept");
    }
    const PipelineContract resolved_contract = *contract;

    lib_handle_ = handle;

    device_ctx_ = create_device_context_fn_();
    if (device_ctx_ == nullptr) {
        dlclose(handle);
        lib_handle_ = nullptr;
        throw std::runtime_error("create_device_context returned null");
    }

    try {
        // One opaque native-run storage buffer per slot, always. The host
        // runtime constructs its per-run Runtime + phase state behind this
        // ABI boundary. This storage is not the RUNTIME_IMAGE resource: the
        // contract classifies the device-resident image, while this owns the
        // host-side tensor leases, launch arguments, and validation/finalize
        // state. It is per-run even when the device image is DEVICE_SCRATCH.
        const size_t runtime_size = get_runtime_size_fn_();
        const size_t runtime_alignment = get_runtime_alignment_fn_();
        if (runtime_size < sizeof(uint64_t) || runtime_alignment < sizeof(void *) ||
            (runtime_alignment & (runtime_alignment - 1)) != 0) {
            throw std::runtime_error("host runtime returned unsupported native-run storage size/alignment");
        }
        std::vector<RuntimeStorage> runtime_bufs;
        runtime_bufs.reserve(resolved_contract.pipeline_depth);
        for (uint32_t slot = 0; slot < resolved_contract.pipeline_depth; ++slot) {
            runtime_bufs.emplace_back(runtime_size, runtime_alignment);
        }
        runtime_bufs_.swap(runtime_bufs);
    } catch (...) {
        destroy_device_context_fn_(device_ctx_);
        device_ctx_ = nullptr;
        dlclose(handle);
        lib_handle_ = nullptr;
        throw;
    }

    // One-shot platform-side init: attach the calling thread to `device_id`
    // (rtSetDevice on onboard, sim bind+acquire on sim), transfer ownership
    // of the executor binaries to the DeviceRunner, and (onboard) sync CANN
    // dlog from HostLogger. Subsequent device-ops re-attach their caller
    // threads idempotently against the recorded device id; subsequent
    // register_callable / run invocations reuse the cached binaries.
    //
    // read_binary_file may throw — defer the dlsym/dlclose rollback to the
    // catch block so the buffers and any partially-resolved handle are torn
    // down symmetrically.
    int init_rc = 0;
    try {
        std::vector<uint8_t> aicpu_bytes = read_binary_file(aicpu_path);
        std::vector<uint8_t> aicore_bytes = read_binary_file(aicore_path);
        // dispatcher_path is empty on sim (no dispatcher) and on tests that
        // exercise _ChipWorker.init directly without a RuntimeBinaries.
        // simpler_init treats a null/empty buffer as "no dispatcher" — onboard
        // ensure_binaries_loaded raises with a clear message if the bootstrap
        // is actually attempted, sim ignores it entirely.
        std::vector<uint8_t> dispatcher_bytes;
        if (!dispatcher_path.empty()) {
            dispatcher_bytes = read_binary_file(dispatcher_path);
        }
        const uint8_t *dispatcher_ptr = dispatcher_bytes.empty() ? nullptr : dispatcher_bytes.data();
        // `prewarm_config` (fork-constant, COW-delivered) rides simpler_init: the
        // platform builds + caches the prebuilt runtime-arena for its ring sizing
        // right after the device comes up. Null => no prewarm.
        init_rc = simpler_init_fn_(
            device_ctx_, device_id, aicpu_bytes.data(), aicpu_bytes.size(), aicore_bytes.data(), aicore_bytes.size(),
            dispatcher_ptr, dispatcher_bytes.size(), prewarm_config
        );
    } catch (...) {
        destroy_device_context_fn_(device_ctx_);
        device_ctx_ = nullptr;
        dlclose(handle);
        lib_handle_ = nullptr;
        create_device_context_fn_ = nullptr;
        destroy_device_context_fn_ = nullptr;
        device_malloc_ctx_fn_ = nullptr;
        device_free_ctx_fn_ = nullptr;
        device_committed_memory_fn_ = nullptr;
        copy_to_device_ctx_fn_ = nullptr;
        copy_from_device_ctx_fn_ = nullptr;
        get_runtime_size_fn_ = nullptr;
        get_runtime_alignment_fn_ = nullptr;
        simpler_init_fn_ = nullptr;
        register_callable_fn_ = nullptr;
        run_fn_ = nullptr;
        prepare_run_fn_ = nullptr;
        launch_run_fn_ = nullptr;
        poll_run_fn_ = nullptr;
        wait_run_fn_ = nullptr;
        finalize_run_fn_ = nullptr;
        supports_concurrent_native_prepare_fn_ = nullptr;
        get_arena_bank_gm_heap_base_fn_ = nullptr;
        get_retained_temp_addr_fn_ = nullptr;
        unregister_callable_fn_ = nullptr;
        get_aicpu_dlopen_count_fn_ = nullptr;
        get_host_dlopen_count_fn_ = nullptr;
        get_run_stream_set_create_count_fn_ = nullptr;
        simpler_provision_dma_workspace_fn_ = nullptr;
        finalize_device_fn_ = nullptr;
        ensure_acl_ready_fn_ = nullptr;
        create_comm_stream_fn_ = nullptr;
        destroy_comm_stream_fn_ = nullptr;
        comm_init_fn_ = nullptr;
        comm_alloc_windows_fn_ = nullptr;
        comm_get_local_window_base_fn_ = nullptr;
        comm_get_window_size_fn_ = nullptr;
        comm_alloc_domain_windows_fn_ = nullptr;
        comm_release_domain_windows_fn_ = nullptr;
        comm_global_domain_prepare_fn_ = nullptr;
        comm_global_domain_import_fn_ = nullptr;
        comm_global_domain_release_fn_ = nullptr;
        comm_barrier_fn_ = nullptr;
        comm_destroy_fn_ = nullptr;
        runtime_bufs_.clear();
        throw;
    }
    if (init_rc != 0) {
        // Symmetric teardown: drop the device context, clear all dlsym'd
        // function pointers, dlclose, and discard cached binaries so the
        // ChipWorker is back to its zero-initialized state. Mirror finalize()
        // exactly minus finalize_device_fn_ (we never reached the
        // initialized_=true point, so device-side teardown is unnecessary).
        destroy_device_context_fn_(device_ctx_);
        device_ctx_ = nullptr;
        dlclose(handle);
        lib_handle_ = nullptr;
        create_device_context_fn_ = nullptr;
        destroy_device_context_fn_ = nullptr;
        device_malloc_ctx_fn_ = nullptr;
        device_free_ctx_fn_ = nullptr;
        device_committed_memory_fn_ = nullptr;
        copy_to_device_ctx_fn_ = nullptr;
        copy_from_device_ctx_fn_ = nullptr;
        get_runtime_size_fn_ = nullptr;
        get_runtime_alignment_fn_ = nullptr;
        simpler_init_fn_ = nullptr;
        register_callable_fn_ = nullptr;
        run_fn_ = nullptr;
        prepare_run_fn_ = nullptr;
        launch_run_fn_ = nullptr;
        poll_run_fn_ = nullptr;
        wait_run_fn_ = nullptr;
        finalize_run_fn_ = nullptr;
        supports_concurrent_native_prepare_fn_ = nullptr;
        get_arena_bank_gm_heap_base_fn_ = nullptr;
        get_retained_temp_addr_fn_ = nullptr;
        unregister_callable_fn_ = nullptr;
        get_aicpu_dlopen_count_fn_ = nullptr;
        get_host_dlopen_count_fn_ = nullptr;
        get_run_stream_set_create_count_fn_ = nullptr;
        simpler_provision_dma_workspace_fn_ = nullptr;
        finalize_device_fn_ = nullptr;
        ensure_acl_ready_fn_ = nullptr;
        create_comm_stream_fn_ = nullptr;
        destroy_comm_stream_fn_ = nullptr;
        comm_init_fn_ = nullptr;
        comm_alloc_windows_fn_ = nullptr;
        comm_get_local_window_base_fn_ = nullptr;
        comm_get_window_size_fn_ = nullptr;
        comm_derive_context_fn_ = nullptr;
        comm_alloc_domain_windows_fn_ = nullptr;
        comm_release_domain_windows_fn_ = nullptr;
        comm_global_domain_prepare_fn_ = nullptr;
        comm_global_domain_import_fn_ = nullptr;
        comm_global_domain_release_fn_ = nullptr;
        comm_barrier_fn_ = nullptr;
        comm_destroy_fn_ = nullptr;
        runtime_bufs_.clear();
        throw std::runtime_error("simpler_init failed with code " + std::to_string(init_rc));
    }

    device_id_ = device_id;
    // Published only once the runtime is up: the rollback paths above leave the
    // default K=1 contract in place, so a failed init never reports the counts
    // of a runtime this worker is not bound to.
    pipeline_contract_ = resolved_contract;
    initialized_ = true;

    // Provision async-DMA workspaces (SDMA) once, now that the device is up. The
    // addresses are latched into the resident KernelArgs so every run carries
    // them. On failure, roll the whole Worker back through finalize() so no
    // half-provisioned state leaks, then surface the error.
    if (dma_workspace_mask != 0) {
        int prov_rc = simpler_provision_dma_workspace_fn_(device_ctx_, dma_workspace_mask);
        if (prov_rc != 0) {
            finalize();
            throw std::runtime_error(
                "async-DMA workspace provisioning (mask=" + std::to_string(dma_workspace_mask) + ") failed with code " +
                std::to_string(prov_rc)
            );
        }
    }
    run_lane_ = std::make_unique<ChipRunLane>(*this);
}

void ChipWorker::finalize() {
    if (run_lane_ != nullptr) {
        try {
            run_lane_->close();
        } catch (...) {
            // Final device teardown below is the last-resort reclamation path;
            // destructors must not terminate while reporting a prior run error.
        }
        run_lane_.reset();
    }
    cleanup_native_runs_noexcept();
    // Global domains are independent of the legacy communicator sessions.
    // Release them while the host runtime and device context are still alive.
    if (comm_global_domain_release_fn_ != nullptr) {
        for (uint64_t domain_id : global_domain_ids_) {
            comm_global_domain_release_fn_(domain_id);
        }
    }
    global_domain_ids_.clear();

    // Defensive: if the user never called comm_destroy, reclaim all owned
    // communicator handles and streams before tearing down the device context.
    clear_comm_sessions();

    if (device_ctx_ != nullptr && finalize_device_fn_ != nullptr && initialized_) {
        finalize_device_fn_(device_ctx_);
    }
    if (device_ctx_ != nullptr && destroy_device_context_fn_ != nullptr) {
        destroy_device_context_fn_(device_ctx_);
        device_ctx_ = nullptr;
    }
    if (lib_handle_) {
        dlclose(lib_handle_);
    }
    lib_handle_ = nullptr;
    create_device_context_fn_ = nullptr;
    destroy_device_context_fn_ = nullptr;
    device_malloc_ctx_fn_ = nullptr;
    device_free_ctx_fn_ = nullptr;
    device_committed_memory_fn_ = nullptr;
    copy_to_device_ctx_fn_ = nullptr;
    copy_from_device_ctx_fn_ = nullptr;
    get_runtime_size_fn_ = nullptr;
    get_runtime_alignment_fn_ = nullptr;
    register_callable_fn_ = nullptr;
    run_fn_ = nullptr;
    prepare_run_fn_ = nullptr;
    launch_run_fn_ = nullptr;
    poll_run_fn_ = nullptr;
    wait_run_fn_ = nullptr;
    finalize_run_fn_ = nullptr;
    supports_concurrent_native_prepare_fn_ = nullptr;
    get_arena_bank_gm_heap_base_fn_ = nullptr;
    get_retained_temp_addr_fn_ = nullptr;
    unregister_callable_fn_ = nullptr;
    get_aicpu_dlopen_count_fn_ = nullptr;
    get_host_dlopen_count_fn_ = nullptr;
    get_run_stream_set_create_count_fn_ = nullptr;
    simpler_provision_dma_workspace_fn_ = nullptr;
    finalize_device_fn_ = nullptr;
    ensure_acl_ready_fn_ = nullptr;
    create_comm_stream_fn_ = nullptr;
    destroy_comm_stream_fn_ = nullptr;
    comm_init_fn_ = nullptr;
    comm_alloc_windows_fn_ = nullptr;
    comm_get_local_window_base_fn_ = nullptr;
    comm_get_window_size_fn_ = nullptr;
    comm_derive_context_fn_ = nullptr;
    comm_alloc_domain_windows_fn_ = nullptr;
    comm_release_domain_windows_fn_ = nullptr;
    comm_global_domain_prepare_fn_ = nullptr;
    comm_global_domain_import_fn_ = nullptr;
    comm_global_domain_release_fn_ = nullptr;
    comm_barrier_fn_ = nullptr;
    comm_destroy_fn_ = nullptr;
    runtime_bufs_.clear();
    pipeline_generations_.reset();
    pipeline_contract_ = {PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, 1, {}};
    initialized_ = false;
    device_id_ = -1;
    finalized_ = true;
}

void ChipWorker::register_callable(int32_t callable_id, const void *callable) {
    if (!initialized_) {
        throw std::runtime_error("ChipWorker not initialized; call init() first");
    }
    if (callable == nullptr) {
        throw std::runtime_error("register_callable: callable must not be null");
    }
    int rc = register_callable_fn_(device_ctx_, callable_id, callable);
    if (rc != 0) {
        throw std::runtime_error("register_callable failed with code " + std::to_string(rc));
    }
}

void ChipWorker::run(int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config) {
    run(callable_id, args, config, nullptr, 0);
}

void ChipWorker::run(
    int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config, volatile int32_t *accepted_state,
    int32_t accepted_value
) {
    if (args == nullptr) throw std::runtime_error("run requires task args");
    ChipRun run = submit_chip_run(callable_id, *args, config, accepted_state, accepted_value);
    (void)run.wait_until(ChipRun::Deadline::max());
}

uint32_t ChipWorker::arena_bank_for_slot(uint32_t slot_id) const {
    // init() rejected any contract whose three arena kinds disagree, so
    // whichever of them the runtime declares yields the same bank.
    const PipelineSlotLease selector{slot_id, 0, 0};
    uint32_t arena_bank = 0;
    for (uint32_t kind : {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_GM_SM, PTO_PIPELINE_RUNTIME_IMAGE}) {
        const PipelineResource *resource = find_pipeline_resource(pipeline_contract_, kind);
        if (resource == nullptr) continue;
        arena_bank = pipeline_resource_slot(pipeline_contract_, *resource, selector);
        break;
    }
    return arena_bank;
}

std::vector<uint64_t> ChipWorker::runtime_buffer_addrs() const {
    std::vector<uint64_t> addrs;
    addrs.reserve(runtime_bufs_.size());
    for (const auto &buf : runtime_bufs_) {
        addrs.push_back(reinterpret_cast<uint64_t>(buf.data()));
    }
    return addrs;
}

uint64_t ChipWorker::arena_bank_gm_heap_base(uint32_t bank_id) const {
    if (!initialized_) return 0;
    return get_arena_bank_gm_heap_base_fn_(device_ctx_, bank_id);
}

uint64_t ChipWorker::retained_temp_addr(uint32_t slot_id) const {
    if (!initialized_) return 0;
    return get_retained_temp_addr_fn_(device_ctx_, slot_id);
}

void ChipWorker::run_with_lease(
    int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config, const PipelineSlotLease &lease,
    volatile int32_t *accepted_state, int32_t accepted_value
) {
    if (lease.reserved != 0 || lease.generation == 0 || lease.slot_id >= pipeline_contract_.pipeline_depth) {
        throw std::runtime_error("run pipeline lease is outside the runtime PipelineContract");
    }
    if (args == nullptr) throw std::runtime_error("run_with_lease requires task args");
    if (run_lane_ == nullptr) throw std::runtime_error("ChipWorker run lane is not initialized");
    ChipRun run = run_lane_->submit(
        callable_id, *args, config, lease, /*run_id=*/0, /*dispatch_id=*/0, accepted_state, accepted_value, true
    );
    (void)run.wait_until(ChipRun::Deadline::max());
}

ChipWorkerNativeRun ChipWorker::prepare_native_run(
    int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config, const PipelineSlotLease &lease,
    uint64_t run_id, uint64_t dispatch_id, volatile int32_t *accepted_state, int32_t accepted_value
) {
    if (lease.reserved != 0 || lease.generation == 0 || lease.slot_id >= pipeline_contract_.pipeline_depth) {
        throw std::runtime_error("native-run pipeline lease is outside the runtime PipelineContract");
    }
    return prepare_native_run_on_slot(
        callable_id, args, config, lease.slot_id, lease.generation, run_id, dispatch_id, accepted_state, accepted_value,
        true
    );
}

ChipWorkerNativeRun ChipWorker::prepare_native_run_for_lane(
    int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config, const PipelineSlotLease &lease,
    uint64_t run_id, uint64_t dispatch_id, volatile int32_t *accepted_state, int32_t accepted_value,
    bool pipeline_leased
) {
    return prepare_native_run_on_slot(
        callable_id, args, config, lease.slot_id, lease.generation, run_id, dispatch_id, accepted_state, accepted_value,
        pipeline_leased
    );
}

bool ChipWorker::supports_concurrent_native_prepare() const {
    return initialized_ && pipeline_contract_.pipeline_depth > 1 &&
           supports_concurrent_native_prepare_fn_(device_ctx_) > 0;
}

ChipWorkerNativeRun ChipWorker::prepare_native_run_on_slot(
    int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config, uint32_t slot_id,
    uint64_t generation, uint64_t run_id, uint64_t dispatch_id, volatile int32_t *accepted_state,
    int32_t accepted_value, bool admit_pipeline_generation
) {
    config.validate();
    if (!initialized_) {
        throw std::runtime_error("ChipWorker not initialized; call init() first");
    }
    if (args == nullptr) {
        throw std::runtime_error("prepare_native_run requires task args");
    }
    if (slot_id >= runtime_bufs_.size()) {
        throw std::runtime_error("prepare_native_run slot is outside the runtime PipelineContract");
    }
    const uint64_t run_epoch = next_native_run_epoch();
    const ChipWorkerNativeRun run_identity{slot_id, generation, run_epoch, run_id, dispatch_id};
    const bool allow_prepared_successor = supports_concurrent_native_prepare() && !config.diagnostics_any();
    {
        std::lock_guard<std::mutex> lk(native_run_mu_);
        NativeRunSlotState &state = native_run_states_[slot_id];
        if (state.phase != NativeRunPhase::EMPTY) {
            throw std::runtime_error(
                "prepare_native_run slot already owns an unfinished native run " +
                format_native_run_identity(run_identity)
            );
        }
        size_t occupied = 0;
        for (const NativeRunSlotState &candidate : native_run_states_) {
            if (candidate.phase == NativeRunPhase::EMPTY) continue;
            ++occupied;
            if (!allow_prepared_successor || !candidate.permits_prepared_successor ||
                (candidate.phase != NativeRunPhase::LAUNCHED && candidate.phase != NativeRunPhase::REAPED)) {
                throw std::runtime_error(
                    "prepare_native_run requires an active predecessor before staging a successor " +
                    format_native_run_identity(run_identity)
                );
            }
        }
        if (occupied > 1) {
            throw std::runtime_error(
                "prepare_native_run already owns a prepared successor " + format_native_run_identity(run_identity)
            );
        }
        if (admit_pipeline_generation &&
            !pipeline_generations_.is_admissible(PipelineSlotLease{slot_id, 0, generation})) {
            throw std::runtime_error(
                "native-run pipeline lease generation is stale " + format_native_run_identity(run_identity)
            );
        }
        state.lease_generation = generation;
        state.run_epoch = run_epoch;
        state.phase = NativeRunPhase::PREPARING;
        state.wait_rc = 0;
        state.permits_prepared_successor = allow_prepared_successor;
    }

    int rc = -1;
    try {
        const NativeRunDescriptor descriptor{slot_id,        arena_bank_for_slot(slot_id),
                                             run_id,         generation,
                                             dispatch_id,    run_epoch,
                                             accepted_state, accepted_value};
        rc = prepare_run_fn_(device_ctx_, runtime_bufs_[slot_id].data(), callable_id, args, &config, &descriptor);
    } catch (...) {
        std::lock_guard<std::mutex> lk(native_run_mu_);
        NativeRunSlotState &state = native_run_states_[slot_id];
        if (state.run_epoch == run_epoch && state.phase == NativeRunPhase::PREPARING) {
            state = NativeRunSlotState{};
        }
        throw;
    }
    if (rc != 0) {
        std::lock_guard<std::mutex> lk(native_run_mu_);
        NativeRunSlotState &state = native_run_states_[slot_id];
        if (state.run_epoch == run_epoch && state.phase == NativeRunPhase::PREPARING) {
            state = NativeRunSlotState{};
        }
        if (rc == PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE) {
            throw PreparedRunIncompatible(
                "native prepare requires depth-one fallback " + format_native_run_identity(run_identity)
            );
        }
        throw std::runtime_error(
            "prepare_native_run failed with code " + std::to_string(rc) + " " + format_native_run_identity(run_identity)
        );
    }

    {
        std::lock_guard<std::mutex> lk(native_run_mu_);
        NativeRunSlotState &state = native_run_states_[slot_id];
        if (state.run_epoch != run_epoch || state.phase != NativeRunPhase::PREPARING) {
            (void)finalize_run_fn_(device_ctx_, runtime_bufs_[slot_id].data());
            state = NativeRunSlotState{};
            throw std::runtime_error("native-run identity changed while prepare was in progress");
        }
        if (admit_pipeline_generation && !pipeline_generations_.admit(PipelineSlotLease{slot_id, 0, generation})) {
            (void)finalize_run_fn_(device_ctx_, runtime_bufs_[slot_id].data());
            state = NativeRunSlotState{};
            throw std::runtime_error(
                "native-run pipeline lease generation became stale while prepare was in progress " +
                format_native_run_identity(run_identity)
            );
        }
        state.phase = NativeRunPhase::PREPARED;
    }
    return run_identity;
}

ChipRun ChipWorker::submit_chip_run(
    int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config, const PipelineSlotLease &lease,
    uint64_t run_id, uint64_t dispatch_id, volatile int32_t *accepted_state, int32_t accepted_value, bool activated
) {
    if (run_lane_ == nullptr) throw std::runtime_error("ChipWorker run lane is not initialized");
    return run_lane_->submit(
        callable_id, args, config, lease, run_id, dispatch_id, accepted_state, accepted_value, activated
    );
}

ChipRun ChipWorker::submit_chip_run(
    int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config, volatile int32_t *accepted_state,
    int32_t accepted_value
) {
    if (run_lane_ == nullptr) throw std::runtime_error("ChipWorker run lane is not initialized");
    return run_lane_->submit(callable_id, args, config, accepted_state, accepted_value);
}

void ChipWorker::close_chip_run_lane() {
    if (run_lane_ != nullptr) run_lane_->close();
}

void ChipWorker::launch_native_run(const ChipWorkerNativeRun &run) {
    {
        std::lock_guard<std::mutex> lk(native_run_mu_);
        if (run.slot_id >= runtime_bufs_.size()) {
            throw std::runtime_error("native-run token slot is outside the runtime PipelineContract");
        }
        NativeRunSlotState &state = native_run_states_[run.slot_id];
        if (state.run_epoch != run.run_epoch || state.phase != NativeRunPhase::PREPARED) {
            throw std::runtime_error("native-run token is stale or used in the wrong phase");
        }
    }

    int rc = launch_run_fn_(device_ctx_, runtime_bufs_[run.slot_id].data());
    if (rc != 0) {
        int poll_rc = poll_run_fn_(device_ctx_, runtime_bufs_[run.slot_id].data());
        std::lock_guard<std::mutex> lk(native_run_mu_);
        NativeRunSlotState &state = native_run_states_[run.slot_id];
        state.phase = poll_rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE ? NativeRunPhase::REAPED : NativeRunPhase::PREPARED;
        state.wait_rc = poll_rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE ? rc : 0;
        throw std::runtime_error(
            "launch_native_run failed with code " + std::to_string(rc) + " " + format_native_run_identity(run)
        );
    }
    std::lock_guard<std::mutex> lk(native_run_mu_);
    NativeRunSlotState &state = native_run_states_[run.slot_id];
    state.phase = NativeRunPhase::LAUNCHED;
}

bool ChipWorker::poll_native_run(const ChipWorkerNativeRun &run) {
    {
        std::lock_guard<std::mutex> lk(native_run_mu_);
        if (run.slot_id >= runtime_bufs_.size()) {
            throw std::runtime_error("native-run token slot is outside the runtime PipelineContract");
        }
        const NativeRunSlotState &state = native_run_states_[run.slot_id];
        if (state.run_epoch != run.run_epoch ||
            (state.phase != NativeRunPhase::LAUNCHED && state.phase != NativeRunPhase::REAPED)) {
            throw std::runtime_error("native-run token is stale or used in the wrong phase");
        }
        if (state.phase == NativeRunPhase::REAPED) return true;
    }
    int rc = poll_run_fn_(device_ctx_, runtime_bufs_[run.slot_id].data());
    if (rc == SIMPLER_NATIVE_RUN_POLL_NOT_READY) {
        return false;
    }
    if (rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE) {
        std::lock_guard<std::mutex> lk(native_run_mu_);
        NativeRunSlotState &state = native_run_states_[run.slot_id];
        state.phase = NativeRunPhase::REAPED;
        return true;
    }
    throw std::runtime_error(
        "poll_native_run failed with code " + std::to_string(rc) + " " + format_native_run_identity(run)
    );
}

void ChipWorker::wait_native_run(const ChipWorkerNativeRun &run) {
    {
        std::lock_guard<std::mutex> lk(native_run_mu_);
        if (run.slot_id >= runtime_bufs_.size()) {
            throw std::runtime_error("native-run token slot is outside the runtime PipelineContract");
        }
        const NativeRunSlotState &state = native_run_states_[run.slot_id];
        if (state.run_epoch != run.run_epoch ||
            (state.phase != NativeRunPhase::LAUNCHED && state.phase != NativeRunPhase::REAPED)) {
            throw std::runtime_error("native-run token is stale or used in the wrong phase");
        }
        if (state.phase == NativeRunPhase::REAPED) return;
    }
    int wait_rc = wait_run_fn_(device_ctx_, runtime_bufs_[run.slot_id].data());
    std::lock_guard<std::mutex> lk(native_run_mu_);
    NativeRunSlotState &state = native_run_states_[run.slot_id];
    state.wait_rc = wait_rc;
    state.phase = NativeRunPhase::REAPED;
}

void ChipWorker::finalize_native_run(const ChipWorkerNativeRun &run) {
    NativeRunPhase phase;
    int wait_rc;
    {
        std::lock_guard<std::mutex> lk(native_run_mu_);
        if (run.slot_id >= runtime_bufs_.size()) {
            throw std::runtime_error("native-run token slot is outside the runtime PipelineContract");
        }
        NativeRunSlotState &state = native_run_states_[run.slot_id];
        if (state.run_epoch != run.run_epoch ||
            (state.phase != NativeRunPhase::PREPARED && state.phase != NativeRunPhase::LAUNCHED &&
             state.phase != NativeRunPhase::REAPED)) {
            throw std::runtime_error("native-run token is stale or already finalized");
        }
        phase = state.phase;
        wait_rc = state.wait_rc;
        state.phase = NativeRunPhase::FINALIZING;
    }

    try {
        if (phase == NativeRunPhase::LAUNCHED) {
            wait_rc = wait_run_fn_(device_ctx_, runtime_bufs_[run.slot_id].data());
        }
    } catch (...) {
        std::lock_guard<std::mutex> lk(native_run_mu_);
        native_run_states_[run.slot_id].phase = phase;
        throw;
    }
    int finalize_rc = finalize_run_fn_(device_ctx_, runtime_bufs_[run.slot_id].data());
    {
        std::lock_guard<std::mutex> lk(native_run_mu_);
        native_run_states_[run.slot_id] = NativeRunSlotState{};
    }
    int rc = finalize_rc != 0 ? finalize_rc : wait_rc;
    if (rc != 0) {
        throw std::runtime_error(
            "finalize_native_run failed with code " + std::to_string(rc) + " " + format_native_run_identity(run)
        );
    }
}

void ChipWorker::cleanup_native_runs_noexcept() noexcept {
    if (device_ctx_ == nullptr || finalize_run_fn_ == nullptr) {
        return;
    }
    for (size_t slot_id = 0; slot_id < runtime_bufs_.size(); ++slot_id) {
        NativeRunPhase phase;
        {
            std::lock_guard<std::mutex> lk(native_run_mu_);
            NativeRunSlotState &state = native_run_states_[slot_id];
            phase = state.phase;
            if (phase == NativeRunPhase::EMPTY) continue;
            if (phase == NativeRunPhase::PREPARING || phase == NativeRunPhase::FINALIZING) {
                continue;
            }
            state.phase = NativeRunPhase::FINALIZING;
        }
        if (phase == NativeRunPhase::LAUNCHED && wait_run_fn_ != nullptr) {
            (void)wait_run_fn_(device_ctx_, runtime_bufs_[slot_id].data());
        }
        (void)finalize_run_fn_(device_ctx_, runtime_bufs_[slot_id].data());
        std::lock_guard<std::mutex> lk(native_run_mu_);
        native_run_states_[slot_id] = NativeRunSlotState{};
    }
}

void ChipWorker::unregister_callable(int32_t callable_id) {
    if (!initialized_) {
        throw std::runtime_error("ChipWorker not initialized; call init() first");
    }
    int rc = unregister_callable_fn_(device_ctx_, callable_id);
    if (rc != 0) {
        throw std::runtime_error("unregister_callable failed with code " + std::to_string(rc));
    }
}

size_t ChipWorker::aicpu_dlopen_count() const {
    if (!initialized_) {
        return 0;
    }
    return get_aicpu_dlopen_count_fn_(device_ctx_);
}

size_t ChipWorker::committed_device_memory() const {
    if (!initialized_) {
        return 0;
    }
    return device_committed_memory_fn_(device_ctx_);
}

size_t ChipWorker::host_dlopen_count() const {
    if (!initialized_) {
        return 0;
    }
    return get_host_dlopen_count_fn_(device_ctx_);
}

size_t ChipWorker::run_stream_set_create_count() const {
    if (!initialized_) {
        return 0;
    }
    return get_run_stream_set_create_count_fn_(device_ctx_);
}

void *ChipWorker::create_comm_stream_checked(const char *op_name) {
    int rc = ensure_acl_ready_fn_(device_ctx_, device_id_);
    if (rc != 0) {
        std::string msg = op_name;
        msg += ": ensure_acl_ready failed with code ";
        msg += std::to_string(rc);
        throw std::runtime_error(msg);
    }
    return create_comm_stream_fn_(device_ctx_);
}

void ChipWorker::destroy_comm_stream_best_effort(void *stream, int *rc) {
    if (stream == nullptr || device_ctx_ == nullptr || destroy_comm_stream_fn_ == nullptr) {
        return;
    }
    int srv = destroy_comm_stream_fn_(device_ctx_, stream);
    if (srv != 0 && rc != nullptr && *rc == 0) {
        *rc = srv;
    }
}

ChipWorker::CommSession *ChipWorker::find_comm_session(uint64_t comm_handle) {
    auto it = comm_session_index_.find(comm_handle);
    if (it == comm_session_index_.end() || it->second >= comm_sessions_.size()) {
        return nullptr;
    }
    CommSession &session = comm_sessions_[it->second];
    if (reinterpret_cast<uint64_t>(session.handle) != comm_handle) {
        return nullptr;
    }
    return &session;
}

ChipWorker::CommSession *ChipWorker::create_comm_session(void *handle, void *stream, bool is_base) {
    if (handle == nullptr) {
        return nullptr;
    }
    uint64_t key = reinterpret_cast<uint64_t>(handle);
    if (comm_session_index_.find(key) != comm_session_index_.end()) {
        return nullptr;
    }
    CommSession session{};
    session.handle = handle;
    session.stream = stream;
    session.is_base = is_base;
    comm_sessions_.push_back(session);
    size_t index = comm_sessions_.size() - 1;
    comm_session_index_[key] = index;
    return &comm_sessions_[index];
}

int ChipWorker::destroy_comm_session(CommSession &session) {
    int rc = 0;
    if (session.handle != nullptr && comm_destroy_fn_ != nullptr) {
        rc = comm_destroy_fn_(session.handle);
    }
    destroy_comm_stream_best_effort(session.stream, &rc);
    if (reinterpret_cast<uint64_t>(session.handle) == base_comm_handle_) {
        base_comm_handle_ = 0;
    }
    comm_session_index_.erase(reinterpret_cast<uint64_t>(session.handle));
    session.handle = nullptr;
    session.stream = nullptr;
    session.device_ctx = 0;
    session.local_window_base = 0;
    session.window_size = 0;
    return rc;
}

uint64_t ChipWorker::create_base_comm(int rank, int nranks, const std::string &rootinfo_path) {
    void *stream = create_comm_stream_checked("comm_init");
    void *handle = comm_init_fn_(rank, nranks, stream, rootinfo_path.c_str());
    if (handle == nullptr) {
        int rc = 0;
        destroy_comm_stream_best_effort(stream, &rc);
        throw std::runtime_error("comm_init failed");
    }
    CommSession *session = create_comm_session(handle, stream, true);
    if (session == nullptr) {
        int rc = comm_destroy_fn_(handle);
        destroy_comm_stream_best_effort(stream, &rc);
        throw std::runtime_error("comm_init: duplicate comm handle");
    }
    base_comm_handle_ = reinterpret_cast<uint64_t>(handle);
    return base_comm_handle_;
}

void ChipWorker::clear_comm_sessions() {
    for (auto it = comm_sessions_.rbegin(); it != comm_sessions_.rend(); ++it) {
        if (it->handle == nullptr && it->stream == nullptr) {
            continue;
        }
        destroy_comm_session(*it);
    }
    comm_sessions_.clear();
    comm_session_index_.clear();
    base_comm_handle_ = 0;
}

uint64_t ChipWorker::malloc(size_t size) {
    if (!initialized_) {
        throw std::runtime_error("ChipWorker not initialized; call init() first");
    }
    void *ptr = device_malloc_ctx_fn_(device_ctx_, size);
    if (ptr == nullptr) {
        throw std::runtime_error("malloc failed");
    }
    return reinterpret_cast<uint64_t>(ptr);
}

void ChipWorker::free(uint64_t ptr) {
    if (!initialized_) {
        throw std::runtime_error("ChipWorker not initialized; call init() first");
    }
    device_free_ctx_fn_(device_ctx_, reinterpret_cast<void *>(ptr));
}

void ChipWorker::copy_to(uint64_t dst, uint64_t src, size_t size) {
    if (!initialized_) {
        throw std::runtime_error("ChipWorker not initialized; call init() first");
    }
    int rc =
        copy_to_device_ctx_fn_(device_ctx_, reinterpret_cast<void *>(dst), reinterpret_cast<const void *>(src), size);
    if (rc != 0) {
        throw std::runtime_error("copy_to failed with code " + std::to_string(rc));
    }
}

void ChipWorker::copy_from(uint64_t dst, uint64_t src, size_t size) {
    if (!initialized_) {
        throw std::runtime_error("ChipWorker not initialized; call init() first");
    }
    int rc =
        copy_from_device_ctx_fn_(device_ctx_, reinterpret_cast<void *>(dst), reinterpret_cast<const void *>(src), size);
    if (rc != 0) {
        throw std::runtime_error("copy_from failed with code " + std::to_string(rc));
    }
}

uint64_t ChipWorker::comm_init(int rank, int nranks, const std::string &rootinfo_path) {
    if (!initialized_) {
        throw std::runtime_error("ChipWorker not initialized; call init() first");
    }
    if (base_comm_handle_ != 0) {
        return base_comm_handle_;
    }

    return create_base_comm(rank, nranks, rootinfo_path);
}

uint64_t ChipWorker::comm_alloc_windows(uint64_t comm_handle, size_t win_size) {
    uint64_t device_ctx = 0;
    int rc = comm_alloc_windows_fn_(reinterpret_cast<void *>(comm_handle), win_size, &device_ctx);
    if (rc != 0) {
        throw std::runtime_error("comm_alloc_windows failed with code " + std::to_string(rc));
    }
    CommSession *session = find_comm_session(comm_handle);
    if (session != nullptr) {
        session->device_ctx = device_ctx;
    }
    return device_ctx;
}

uint64_t ChipWorker::comm_get_local_window_base(uint64_t comm_handle) {
    uint64_t base = 0;
    int rc = comm_get_local_window_base_fn_(reinterpret_cast<void *>(comm_handle), &base);
    if (rc != 0) {
        throw std::runtime_error("comm_get_local_window_base failed with code " + std::to_string(rc));
    }
    CommSession *session = find_comm_session(comm_handle);
    if (session != nullptr) {
        session->local_window_base = base;
    }
    return base;
}

size_t ChipWorker::comm_get_window_size(uint64_t comm_handle) {
    size_t win_size = 0;
    int rc = comm_get_window_size_fn_(reinterpret_cast<void *>(comm_handle), &win_size);
    if (rc != 0) {
        throw std::runtime_error("comm_get_window_size failed with code " + std::to_string(rc));
    }
    CommSession *session = find_comm_session(comm_handle);
    if (session != nullptr) {
        session->window_size = win_size;
    }
    return win_size;
}

uint64_t ChipWorker::comm_derive_context(
    uint64_t comm_handle, const std::vector<uint32_t> &rank_ids, uint32_t domain_rank, size_t window_offset,
    size_t window_size
) {
    if (comm_derive_context_fn_ == nullptr) {
        throw std::runtime_error("comm_derive_context is not supported by this runtime");
    }
    if (rank_ids.empty()) {
        throw std::runtime_error("comm_derive_context: rank_ids must not be empty");
    }
    uint64_t device_ctx = 0;
    int rc = comm_derive_context_fn_(
        reinterpret_cast<void *>(comm_handle), rank_ids.data(), rank_ids.size(), domain_rank, window_offset,
        window_size, &device_ctx
    );
    if (rc != 0) {
        throw std::runtime_error("comm_derive_context failed with code " + std::to_string(rc));
    }
    if (device_ctx == 0) {
        throw std::runtime_error("comm_derive_context returned null device_ctx");
    }
    return device_ctx;
}

std::pair<uint64_t, uint64_t> ChipWorker::comm_alloc_domain_windows(
    uint64_t comm_handle, uint64_t allocation_id, const std::vector<uint32_t> &rank_ids, uint32_t domain_rank,
    size_t window_size
) {
    if (comm_alloc_domain_windows_fn_ == nullptr) {
        throw std::runtime_error("comm_alloc_domain_windows is not supported by this runtime");
    }
    if (rank_ids.empty()) {
        throw std::runtime_error("comm_alloc_domain_windows: rank_ids must not be empty");
    }
    if (domain_rank >= rank_ids.size()) {
        throw std::runtime_error("comm_alloc_domain_windows: domain_rank out of range");
    }
    if (window_size == 0) {
        throw std::runtime_error("comm_alloc_domain_windows: window_size must be positive");
    }
    uint64_t device_ctx = 0;
    uint64_t local_window_base = 0;
    int rc = comm_alloc_domain_windows_fn_(
        reinterpret_cast<void *>(comm_handle), allocation_id, rank_ids.data(), rank_ids.size(), domain_rank,
        window_size, &device_ctx, &local_window_base
    );
    if (rc != 0) {
        throw std::runtime_error("comm_alloc_domain_windows failed with code " + std::to_string(rc));
    }
    if (device_ctx == 0 || local_window_base == 0) {
        throw std::runtime_error("comm_alloc_domain_windows returned null device_ctx / local_window_base");
    }
    return {device_ctx, local_window_base};
}

void ChipWorker::comm_release_domain_windows(
    uint64_t comm_handle, uint64_t allocation_id, size_t rank_count, uint32_t domain_rank
) {
    if (comm_release_domain_windows_fn_ == nullptr) {
        throw std::runtime_error("comm_release_domain_windows is not supported by this runtime");
    }
    int rc =
        comm_release_domain_windows_fn_(reinterpret_cast<void *>(comm_handle), allocation_id, rank_count, domain_rank);
    if (rc != 0) {
        throw std::runtime_error("comm_release_domain_windows failed with code " + std::to_string(rc));
    }
}

std::tuple<std::vector<uint8_t>, uint64_t, size_t> ChipWorker::comm_global_domain_prepare(
    uint64_t domain_id, uint32_t domain_rank, uint32_t rank_count, size_t window_size, uint32_t profile
) {
    if (comm_global_domain_prepare_fn_ == nullptr) {
        throw std::runtime_error("comm_global_domain_prepare is not supported by this runtime");
    }
    auto [tracked, inserted] = global_domain_ids_.insert(domain_id);
    if (!inserted) {
        throw std::runtime_error("comm_global_domain_prepare received a duplicate domain_id");
    }
    CommGlobalDomainDescriptor descriptor{};
    uint64_t local_window_base = 0;
    int rc = comm_global_domain_prepare_fn_(
        domain_id, domain_rank, rank_count, window_size, profile, &descriptor, &local_window_base
    );
    if (rc != 0) {
        global_domain_ids_.erase(tracked);
        throw std::runtime_error("comm_global_domain_prepare failed with code " + std::to_string(rc));
    }
    if (local_window_base == 0 || descriptor.mapping_size == 0) {
        comm_global_domain_release_fn_(domain_id);
        global_domain_ids_.erase(domain_id);
        throw std::runtime_error("comm_global_domain_prepare returned an invalid window");
    }
    const auto *begin = reinterpret_cast<const uint8_t *>(&descriptor);
    std::vector<uint8_t> descriptor_bytes(begin, begin + sizeof(descriptor));
    return {std::move(descriptor_bytes), local_window_base, static_cast<size_t>(descriptor.mapping_size)};
}

uint64_t ChipWorker::comm_global_domain_import(uint64_t domain_id, const std::vector<uint8_t> &descriptors) {
    if (comm_global_domain_import_fn_ == nullptr) {
        throw std::runtime_error("comm_global_domain_import is not supported by this runtime");
    }
    if (descriptors.empty() || descriptors.size() % sizeof(CommGlobalDomainDescriptor) != 0) {
        throw std::runtime_error("comm_global_domain_import descriptor table size is invalid");
    }
    uint64_t device_ctx = 0;
    int rc = comm_global_domain_import_fn_(
        domain_id, reinterpret_cast<const CommGlobalDomainDescriptor *>(descriptors.data()),
        descriptors.size() / sizeof(CommGlobalDomainDescriptor), &device_ctx
    );
    if (rc != 0) {
        throw std::runtime_error("comm_global_domain_import failed with code " + std::to_string(rc));
    }
    if (device_ctx == 0) {
        throw std::runtime_error("comm_global_domain_import returned a null device context");
    }
    return device_ctx;
}

void ChipWorker::comm_global_domain_release(uint64_t domain_id) {
    if (comm_global_domain_release_fn_ == nullptr) {
        throw std::runtime_error("comm_global_domain_release is not supported by this runtime");
    }
    int rc = comm_global_domain_release_fn_(domain_id);
    if (rc != 0) {
        throw std::runtime_error("comm_global_domain_release failed with code " + std::to_string(rc));
    }
    global_domain_ids_.erase(domain_id);
}

void ChipWorker::comm_barrier(uint64_t comm_handle) {
    int rc = comm_barrier_fn_(reinterpret_cast<void *>(comm_handle));
    if (rc != 0) {
        throw std::runtime_error("comm_barrier failed with code " + std::to_string(rc));
    }
}

void ChipWorker::comm_destroy(uint64_t comm_handle) {
    CommSession *session = find_comm_session(comm_handle);
    if (session == nullptr) {
        int rc = comm_destroy_fn_(reinterpret_cast<void *>(comm_handle));
        if (rc != 0) {
            throw std::runtime_error("comm_destroy failed with code " + std::to_string(rc));
        }
        return;
    }
    if (session->is_base) {
        comm_destroy_all();
        return;
    }

    int rc = destroy_comm_session(*session);
    while (!comm_sessions_.empty() && comm_sessions_.back().handle == nullptr &&
           comm_sessions_.back().stream == nullptr) {
        comm_sessions_.pop_back();
    }

    if (rc != 0) {
        throw std::runtime_error("comm_destroy failed with code " + std::to_string(rc));
    }
}

void ChipWorker::comm_destroy_all() {
    int first_rc = 0;
    for (auto it = comm_sessions_.rbegin(); it != comm_sessions_.rend(); ++it) {
        if (it->handle == nullptr && it->stream == nullptr) {
            continue;
        }
        int rc = destroy_comm_session(*it);
        if (rc != 0 && first_rc == 0) {
            first_rc = rc;
        }
    }
    comm_sessions_.clear();
    comm_session_index_.clear();
    base_comm_handle_ = 0;
    if (first_rc != 0) {
        throw std::runtime_error("comm_destroy_all failed with code " + std::to_string(first_rc));
    }
}

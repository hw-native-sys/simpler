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

#include <dlfcn.h>

#include <cstring>
#include <fstream>
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
        throw std::runtime_error(msg);
    }
    return reinterpret_cast<T>(sym);
}

template <typename T>
T load_optional_symbol(void *handle, const char *name) {
    dlerror();
    void *sym = dlsym(handle, name);
    if (dlerror() != nullptr) return nullptr;
    return reinterpret_cast<T>(sym);
}

bool is_valid_k1_pipeline_contract(const PipelineContract *contract) {
    if (contract == nullptr || contract->abi_version != PTO_PIPELINE_CONTRACT_ABI_VERSION ||
        contract->resource_count > PTO_PIPELINE_MAX_RESOURCES || contract->pipeline_slots != 1 ||
        contract->arena_banks != 1) {
        return false;
    }
    for (uint32_t i = 0; i < contract->resource_count; ++i) {
        const PipelineResource &resource = contract->resources[i];
        if (resource.kind > PTO_PIPELINE_AICORE_STREAM || resource.resource_class > PTO_PIPELINE_EXEC_HANDLE) {
            return false;
        }
    }
    return true;
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
    pipeline_contract_ = {PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, 1, 1, {}};

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
        copy_to_device_ctx_fn_ = load_symbol<CopyToDeviceCtxFn>(handle, "copy_to_device_ctx");
        copy_from_device_ctx_fn_ = load_symbol<CopyFromDeviceCtxFn>(handle, "copy_from_device_ctx");
        get_runtime_size_fn_ = load_symbol<GetRuntimeSizeFn>(handle, "get_runtime_size");
        simpler_init_fn_ = load_symbol<SimplerInitFn>(handle, "simpler_init");
        register_callable_fn_ = load_symbol<SimplerRegisterCallableFn>(handle, "simpler_register_callable");
        run_fn_ = load_symbol<SimplerRunFn>(handle, "simpler_run");
        get_pipeline_contract_fn = load_optional_symbol<GetPipelineContractFn>(handle, "get_pipeline_contract");
        unregister_callable_fn_ = load_symbol<SimplerUnregisterCallableFn>(handle, "simpler_unregister_callable");
        get_aicpu_dlopen_count_fn_ = load_symbol<GetAicpuDlopenCountFn>(handle, "get_aicpu_dlopen_count");
        get_host_dlopen_count_fn_ = load_symbol<GetAicpuDlopenCountFn>(handle, "get_host_dlopen_count");
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
        comm_barrier_fn_ = load_symbol<CommBarrierFn>(handle, "comm_barrier");
        comm_destroy_fn_ = load_symbol<CommDestroyFn>(handle, "comm_destroy");
    } catch (...) {
        dlclose(handle);
        throw;
    }

    PipelineContract resolved_contract{PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, 1, 1, {}};
    if (get_pipeline_contract_fn != nullptr) {
        const PipelineContract *contract = get_pipeline_contract_fn();
        if (!is_valid_k1_pipeline_contract(contract)) {
            dlclose(handle);
            throw std::runtime_error("host runtime returned an invalid K=1 PipelineContract");
        }
        resolved_contract = *contract;
    }

    lib_handle_ = handle;
    pipeline_contract_ = resolved_contract;

    device_ctx_ = create_device_context_fn_();
    if (device_ctx_ == nullptr) {
        dlclose(handle);
        lib_handle_ = nullptr;
        throw std::runtime_error("create_device_context returned null");
    }

    runtime_buf_.resize(get_runtime_size_fn_());

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
        copy_to_device_ctx_fn_ = nullptr;
        copy_from_device_ctx_fn_ = nullptr;
        get_runtime_size_fn_ = nullptr;
        simpler_init_fn_ = nullptr;
        register_callable_fn_ = nullptr;
        run_fn_ = nullptr;
        unregister_callable_fn_ = nullptr;
        get_aicpu_dlopen_count_fn_ = nullptr;
        get_host_dlopen_count_fn_ = nullptr;
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
        comm_barrier_fn_ = nullptr;
        comm_destroy_fn_ = nullptr;
        runtime_buf_.clear();
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
        copy_to_device_ctx_fn_ = nullptr;
        copy_from_device_ctx_fn_ = nullptr;
        get_runtime_size_fn_ = nullptr;
        simpler_init_fn_ = nullptr;
        register_callable_fn_ = nullptr;
        run_fn_ = nullptr;
        unregister_callable_fn_ = nullptr;
        get_aicpu_dlopen_count_fn_ = nullptr;
        get_host_dlopen_count_fn_ = nullptr;
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
        comm_barrier_fn_ = nullptr;
        comm_destroy_fn_ = nullptr;
        runtime_buf_.clear();
        throw std::runtime_error("simpler_init failed with code " + std::to_string(init_rc));
    }

    device_id_ = device_id;
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
}

void ChipWorker::finalize() {
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
    copy_to_device_ctx_fn_ = nullptr;
    copy_from_device_ctx_fn_ = nullptr;
    get_runtime_size_fn_ = nullptr;
    register_callable_fn_ = nullptr;
    run_fn_ = nullptr;
    unregister_callable_fn_ = nullptr;
    get_aicpu_dlopen_count_fn_ = nullptr;
    get_host_dlopen_count_fn_ = nullptr;
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
    comm_barrier_fn_ = nullptr;
    comm_destroy_fn_ = nullptr;
    runtime_buf_.clear();
    pipeline_contract_ = {PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, 1, 1, {}};
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

void ChipWorker::run(int32_t callable_id, TaskArgsView args, const CallConfig &config) {
    ChipStorageTaskArgs chip_storage = view_to_chip_storage(args);
    run(callable_id, &chip_storage, config);
}

void ChipWorker::run(int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config) {
    config.validate();
    if (!initialized_) {
        throw std::runtime_error("ChipWorker not initialized; call init() first");
    }

    void *rt = runtime_buf_.data();
    // Per-stage timing is emitted by the platform as `[STRACE]` log markers, not
    // returned (see chip_worker.h::run). CallConfig is threaded through to the C
    // ABI as a single pointer rather than unpacked into per-field args.
    int rc = run_fn_(device_ctx_, rt, callable_id, args, &config);
    if (rc != 0) {
        throw std::runtime_error("run failed with code " + std::to_string(rc));
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

size_t ChipWorker::host_dlopen_count() const {
    if (!initialized_) {
        return 0;
    }
    return get_host_dlopen_count_fn_(device_ctx_);
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

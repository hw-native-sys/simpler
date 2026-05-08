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

#include <fstream>
#include <mutex>
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

// Process-wide singleton: libcpu_sim_context.so is loaded once with
// RTLD_GLOBAL so that host_runtime.so can resolve sim_context_set_* and
// pto_sim_get_* symbols at runtime.  Never dlclosed.
std::once_flag g_sim_context_once;
void *g_sim_context_handle = nullptr;

void ensure_sim_context_loaded(const std::string &path) {
    std::call_once(g_sim_context_once, [&]() {
        dlerror();
        g_sim_context_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!g_sim_context_handle) {
            std::string err = "dlopen sim_context failed: ";
            const char *msg = dlerror();
            err += msg ? msg : "unknown error";
            throw std::runtime_error(err);
        }
    });
}

// Process-wide singleton: libsimpler_log.so is loaded once with RTLD_GLOBAL
// so every consumer .so (host_runtime, cpu_sim_context, the binding, sim
// AICore) shares one HostLogger across the process.  Must be loaded BEFORE
// any consumer that has unresolved HostLogger / unified_log_* symbols.
// Never dlclosed.
std::once_flag g_simpler_log_once;
void *g_simpler_log_handle = nullptr;

void ensure_simpler_log_loaded(const std::string &path) {
    std::call_once(g_simpler_log_once, [&]() {
        dlerror();
        g_simpler_log_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!g_simpler_log_handle) {
            std::string err = "dlopen simpler_log failed: ";
            const char *msg = dlerror();
            err += msg ? msg : "unknown error";
            throw std::runtime_error(err);
        }
    });
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
    const std::string &simpler_log_lib_path, int device_id, const std::string &sim_context_lib_path, int log_level,
    int log_info_v
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

    // Load libsimpler_log FIRST with RTLD_GLOBAL so that subsequent host_runtime
    // / cpu_sim_context / sim aicore .so loads can resolve their HostLogger and
    // unified_log_* references against the single process-wide instance.
    if (simpler_log_lib_path.empty()) {
        throw std::runtime_error("ChipWorker::init requires simpler_log_lib_path");
    }
    ensure_simpler_log_loaded(simpler_log_lib_path);

    // Load the sim context SO with RTLD_GLOBAL (once per process) so that
    // PTO ISA TPUSH/TPOP can resolve pto_sim_get_subblock_id and
    // pto_sim_get_pipe_shared_state via dlsym(RTLD_DEFAULT).
    if (!sim_context_lib_path.empty()) {
        ensure_sim_context_loaded(sim_context_lib_path);
    }

    // Host runtime SO is loaded with RTLD_LOCAL so that different runtimes'
    // identically-named symbols (init_runtime_impl, run_runtime, etc.) do
    // not collide when switching runtimes within the same process.
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

    try {
        create_device_context_fn_ = load_symbol<CreateDeviceContextFn>(handle, "create_device_context");
        destroy_device_context_fn_ = load_symbol<DestroyDeviceContextFn>(handle, "destroy_device_context");
        device_malloc_ctx_fn_ = load_symbol<DeviceMallocCtxFn>(handle, "device_malloc_ctx");
        device_free_ctx_fn_ = load_symbol<DeviceFreeCtxFn>(handle, "device_free_ctx");
        copy_to_device_ctx_fn_ = load_symbol<CopyToDeviceCtxFn>(handle, "copy_to_device_ctx");
        copy_from_device_ctx_fn_ = load_symbol<CopyFromDeviceCtxFn>(handle, "copy_from_device_ctx");
        get_runtime_size_fn_ = load_symbol<GetRuntimeSizeFn>(handle, "get_runtime_size");
        run_runtime_fn_ = load_symbol<RunRuntimeFn>(handle, "run_runtime");
        simpler_init_fn_ = load_symbol<SimplerInitFn>(handle, "simpler_init");
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
        comm_barrier_fn_ = load_symbol<CommBarrierFn>(handle, "comm_barrier");
        comm_destroy_fn_ = load_symbol<CommDestroyFn>(handle, "comm_destroy");
    } catch (...) {
        dlclose(handle);
        throw;
    }

    lib_handle_ = handle;

    device_ctx_ = create_device_context_fn_();
    if (device_ctx_ == nullptr) {
        dlclose(handle);
        lib_handle_ = nullptr;
        throw std::runtime_error("create_device_context returned null");
    }

    // Read platform binaries from files
    aicpu_binary_ = read_binary_file(aicpu_path);
    aicore_binary_ = read_binary_file(aicore_path);

    runtime_buf_.resize(get_runtime_size_fn_());

    // One-shot platform-side init: attach the calling thread to `device_id`
    // (rtSetDevice on onboard, sim bind+acquire on sim) and push the user's
    // simpler-logger choice into HostLogger + runner state (and CANN dlog
    // onboard). Subsequent device-ops re-attach their caller threads
    // idempotently against the recorded device id.
    int init_rc = simpler_init_fn_(device_ctx_, device_id, log_level, log_info_v);
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
        run_runtime_fn_ = nullptr;
        simpler_init_fn_ = nullptr;
        finalize_device_fn_ = nullptr;
        ensure_acl_ready_fn_ = nullptr;
        create_comm_stream_fn_ = nullptr;
        destroy_comm_stream_fn_ = nullptr;
        comm_init_fn_ = nullptr;
        comm_alloc_windows_fn_ = nullptr;
        comm_get_local_window_base_fn_ = nullptr;
        comm_get_window_size_fn_ = nullptr;
        comm_barrier_fn_ = nullptr;
        comm_destroy_fn_ = nullptr;
        runtime_buf_.clear();
        aicpu_binary_.clear();
        aicore_binary_.clear();
        throw std::runtime_error("simpler_init failed with code " + std::to_string(init_rc));
    }

    device_id_ = device_id;
    initialized_ = true;
}

void ChipWorker::finalize() {
    // Defensive: if the user never called comm_destroy, reclaim the stream
    // before we tear down the device context (otherwise the stream-backing
    // ACL state outlives its owning context).
    if (comm_stream_ != nullptr && device_ctx_ != nullptr && destroy_comm_stream_fn_ != nullptr) {
        destroy_comm_stream_fn_(device_ctx_, comm_stream_);
    }
    comm_stream_ = nullptr;

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
    run_runtime_fn_ = nullptr;
    finalize_device_fn_ = nullptr;
    ensure_acl_ready_fn_ = nullptr;
    create_comm_stream_fn_ = nullptr;
    destroy_comm_stream_fn_ = nullptr;
    comm_init_fn_ = nullptr;
    comm_alloc_windows_fn_ = nullptr;
    comm_get_local_window_base_fn_ = nullptr;
    comm_get_window_size_fn_ = nullptr;
    comm_barrier_fn_ = nullptr;
    comm_destroy_fn_ = nullptr;
    runtime_buf_.clear();
    aicpu_binary_.clear();
    aicore_binary_.clear();
    initialized_ = false;
    device_id_ = -1;
    finalized_ = true;
}

void ChipWorker::run(uint64_t callable, TaskArgsView args, const CallConfig &config) {
    // L2 ABI edge: assemble the fixed-size ChipStorageTaskArgs POD from the
    // view and hand it to the runtime. This conversion used to happen at
    // submit time (stored on the slot); it now runs lazily in the worker so
    // the slot can carry a single TaskArgs irrespective of the destination.
    ChipStorageTaskArgs chip_storage = view_to_chip_storage(args);
    run(reinterpret_cast<const void *>(callable), &chip_storage, config);
}

void ChipWorker::run(const void *callable, const void *args, const CallConfig &config) {
    config.validate();
    if (!initialized_) {
        throw std::runtime_error("ChipWorker not initialized; call init() first");
    }

    void *rt = runtime_buf_.data();

    int rc = run_runtime_fn_(
        device_ctx_, rt, callable, args, config.block_dim, config.aicpu_thread_num, device_id_, aicpu_binary_.data(),
        aicpu_binary_.size(), aicore_binary_.data(), aicore_binary_.size(), config.enable_l2_swimlane,
        config.enable_dump_tensor, config.enable_pmu, config.output_prefix
    );
    if (rc != 0) {
        throw std::runtime_error("run_runtime failed with code " + std::to_string(rc));
    }
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
    if (comm_stream_ != nullptr) {
        throw std::runtime_error("comm_init: a comm session is already active on this ChipWorker");
    }

    // Bring ACL up on the calling thread before stream creation.  Onboard
    // runs aclInit (idempotent) + per-thread aclrtSetDevice; sim's stub is
    // a no-op; platforms with no distributed backend (a5 today) also no-op.
    int rc = ensure_acl_ready_fn_(device_ctx_, device_id_);
    if (rc != 0) {
        throw std::runtime_error("ensure_acl_ready failed with code " + std::to_string(rc));
    }

    // Create an aclrtStream owned by this ChipWorker.  Sim / a5 stubs
    // return NULL; their raw comm_init ignores the stream arg.  A NULL
    // from a runtime that has a real comm backend is a genuine failure —
    // we can't distinguish "stub returned NULL on purpose" from "onboard
    // create failed" here, so we defer the check to comm_init's own
    // return value below (a stub returns NULL from comm_init anyway).
    void *stream = create_comm_stream_fn_(device_ctx_);

    void *handle = comm_init_fn_(rank, nranks, stream, rootinfo_path.c_str());
    if (handle == nullptr) {
        // Roll back the stream we just created — otherwise the ChipWorker
        // leaks it and the next comm_init attempt trips the
        // "session already active" guard above.
        if (stream != nullptr) {
            destroy_comm_stream_fn_(device_ctx_, stream);
        }
        throw std::runtime_error("comm_init failed");
    }

    comm_stream_ = stream;
    return reinterpret_cast<uint64_t>(handle);
}

uint64_t ChipWorker::comm_alloc_windows(uint64_t comm_handle, size_t win_size) {
    uint64_t device_ctx = 0;
    int rc = comm_alloc_windows_fn_(reinterpret_cast<void *>(comm_handle), win_size, &device_ctx);
    if (rc != 0) {
        throw std::runtime_error("comm_alloc_windows failed with code " + std::to_string(rc));
    }
    return device_ctx;
}

uint64_t ChipWorker::comm_get_local_window_base(uint64_t comm_handle) {
    uint64_t base = 0;
    int rc = comm_get_local_window_base_fn_(reinterpret_cast<void *>(comm_handle), &base);
    if (rc != 0) {
        throw std::runtime_error("comm_get_local_window_base failed with code " + std::to_string(rc));
    }
    return base;
}

size_t ChipWorker::comm_get_window_size(uint64_t comm_handle) {
    size_t win_size = 0;
    int rc = comm_get_window_size_fn_(reinterpret_cast<void *>(comm_handle), &win_size);
    if (rc != 0) {
        throw std::runtime_error("comm_get_window_size failed with code " + std::to_string(rc));
    }
    return win_size;
}

void ChipWorker::comm_barrier(uint64_t comm_handle) {
    int rc = comm_barrier_fn_(reinterpret_cast<void *>(comm_handle));
    if (rc != 0) {
        throw std::runtime_error("comm_barrier failed with code " + std::to_string(rc));
    }
}

void ChipWorker::comm_destroy(uint64_t comm_handle) {
    int rc = comm_destroy_fn_(reinterpret_cast<void *>(comm_handle));

    // Destroy our comm-owned stream regardless of the handle-destroy result —
    // leaking the stream is the worse outcome (it keeps the device attached
    // and blocks the next session).  We still surface the underlying rc
    // below so callers see the original failure.
    if (comm_stream_ != nullptr) {
        int srv = destroy_comm_stream_fn_(device_ctx_, comm_stream_);
        if (srv != 0 && rc == 0) rc = srv;
    }
    comm_stream_ = nullptr;

    if (rc != 0) {
        throw std::runtime_error("comm_destroy failed with code " + std::to_string(rc));
    }
}

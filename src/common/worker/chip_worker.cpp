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
#include "common/host_log_binding.h"
#include "device_fault_monitor_host.h"
#include "host_log.h"
#include "pipeline_contract.h"

#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

class DlHandleGuard {
public:
    explicit DlHandleGuard(void *handle = nullptr) :
        handle_(handle) {}
    ~DlHandleGuard() {
        if (handle_ != nullptr) dlclose(handle_);
    }
    DlHandleGuard(const DlHandleGuard &) = delete;
    DlHandleGuard &operator=(const DlHandleGuard &) = delete;
    void *release() {
        void *handle = handle_;
        handle_ = nullptr;
        return handle;
    }

private:
    void *handle_;
};

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
        msg += "; every compatible module built from this source tree exports it, so rebuild the module";
        throw std::runtime_error(msg);
    }
    return reinterpret_cast<T>(sym);
}

void bind_host_log_state(void *handle, const char *module_name) {
    const char *error = nullptr;
    if (simpler::log::bind_loaded_host_log_state(handle, HostLogger::get_instance().state(), &error) != 0) {
        throw std::runtime_error(
            std::string(module_name) + " failed to bind host-log state: " + (error != nullptr ? error : "unknown error")
        );
    }
}

/**
 * Hand the loaded module the process's device-fault monitor.
 *
 * The monitor and the trampoline the driver retains live in this module, which
 * the interpreter never unloads; a host runtime is opened `RTLD_LOCAL` and
 * `dlclose`d, so it can only hold a pointer. Binding is mandatory rather than
 * best-effort for the same reason the host-log binding is: every module built
 * from this source tree exports the setter, so a missing one means a stale
 * build, and a silently unbound runtime would report no device fault at all.
 */
void bind_device_fault_monitor(void *handle, const char *module_name) {
    const char *error = nullptr;
    if (bind_loaded_device_fault_monitor(handle, &error) != 0) {
        throw std::runtime_error(
            std::string(module_name) +
            " failed to bind the device-fault monitor: " + (error != nullptr ? error : "unknown error")
        );
    }
}

void load_sim_context(const std::string &path) {
    static std::mutex registry_mutex;
    // Raw handles keep simulator globals and pthread keys alive until process
    // exit; the registry container itself owns no dlclose responsibility.
    static std::unordered_map<std::string, void *> registry;

    std::scoped_lock lock(registry_mutex);
    if (registry.find(path) != registry.end()) return;

    dlerror();
    void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char *error = dlerror();
        throw std::runtime_error(
            std::string("dlopen sim context failed: ") + (error != nullptr ? error : "unknown error")
        );
    }
    DlHandleGuard guard(handle);
    bind_host_log_state(handle, "sim context");
    registry.emplace(path, handle);
    (void)guard.release();
}

uint64_t next_environment_incarnation() {
    static std::atomic<uint64_t> incarnation{0};
    uint64_t current = incarnation.load(std::memory_order_relaxed);
    while (current != std::numeric_limits<uint64_t>::max()) {
        if (incarnation.compare_exchange_weak(current, current + 1, std::memory_order_relaxed)) {
            return current + 1;
        }
    }
    throw std::overflow_error("execution environment identity space is exhausted");
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

std::string format_device_address(uint64_t addr) {
    std::ostringstream os;
    os << "0x" << std::hex << addr;
    return os.str();
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
    size_ = size;
}

ChipWorker::RuntimeStorage::~RuntimeStorage() { std::free(data_); }

ChipWorker::RuntimeStorage::RuntimeStorage(RuntimeStorage &&other) noexcept :
    data_(other.data_),
    size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

ChipWorker::RuntimeStorage &ChipWorker::RuntimeStorage::operator=(RuntimeStorage &&other) noexcept {
    if (this == &other) return *this;
    std::free(data_);
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
}

ChipWorker::~ChipWorker() {
    try {
        finalize();
    } catch (const std::exception &error) {
        std::fprintf(stderr, "ChipWorker::~ChipWorker: teardown failed: %s\n", error.what());
        abandon_after_failed_teardown(std::current_exception());
    } catch (...) {
        std::fprintf(stderr, "ChipWorker::~ChipWorker: teardown failed with an unknown error\n");
        abandon_after_failed_teardown(std::current_exception());
    }
}

void ChipWorker::init(
    const std::string &host_lib_path, const std::string &aicpu_path, const std::string &aicore_path,
    const std::string &dispatcher_path, int device_id, const CallConfig *prewarm_config, bool enable_sdma,
    const std::string &sim_context_path, const std::string &sdma_warmup_path, bool collect_across_runs,
    uint64_t workspace_budget_bytes, bool manage_workspace, uint32_t requested_pipeline_depth,
    std::optional<uint64_t> borrowed_context_generation
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
    if (device_ctx_ != nullptr) {
        throw std::runtime_error("ChipWorker initialization requires finalize after an incomplete rollback");
    }
    if (borrowed_context_generation.has_value()) {
        if (*borrowed_context_generation == 0)
            throw std::invalid_argument("borrowed context generation must be nonzero");
        if (enable_sdma || !sim_context_path.empty() || !sdma_warmup_path.empty()) {
            throw std::invalid_argument("borrowed initialization does not support SDMA or simulator options");
        }
    }
    pipeline_contract_ = {PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, 1, {}};

    if (!sim_context_path.empty()) {
        load_sim_context(sim_context_path);
    }

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
    DlHandleGuard host_guard(handle);
    bind_host_log_state(handle, "host runtime");
    bind_device_fault_monitor(handle, "host runtime");

    GetPipelineContractFn get_pipeline_contract_fn = nullptr;
    KernelSupportedFn kernel_supported_fn = nullptr;
    KernelInitFn kernel_init_fn = nullptr;
    KernelPrepareCallableFn kernel_prepare_callable_fn = nullptr;
    KernelLaunchFn kernel_launch_fn = nullptr;
    try {
        create_device_context_fn_ = load_symbol<CreateDeviceContextFn>(handle, "create_device_context");
        destroy_device_context_fn_ = load_symbol<DestroyDeviceContextFn>(handle, "destroy_device_context");
        device_malloc_ctx_fn_ = load_symbol<DeviceMallocCtxFn>(handle, "device_malloc_ctx");
        device_free_ctx_fn_ = load_symbol<DeviceFreeCtxFn>(handle, "device_free_ctx");
        device_free_caller_buffer_ctx_fn_ =
            load_symbol<DeviceFreeCallerBufferCtxFn>(handle, "device_free_caller_buffer_ctx");
        device_borrow_caller_buffers_ctx_fn_ =
            load_symbol<DeviceBorrowCallerBuffersCtxFn>(handle, "device_borrow_caller_buffers_ctx");
        device_release_caller_buffers_ctx_fn_ =
            load_symbol<DeviceReleaseCallerBuffersCtxFn>(handle, "device_release_caller_buffers_ctx");
        device_committed_memory_fn_ = load_symbol<GetCommittedDeviceMemoryFn>(handle, "committed_device_memory_ctx");
        device_memory_info_fn_ = load_symbol<GetDeviceMemoryInfoFn>(handle, "device_memory_info_ctx");
        copy_to_device_ctx_fn_ = load_symbol<CopyToDeviceCtxFn>(handle, "copy_to_device_ctx");
        copy_from_device_ctx_fn_ = load_symbol<CopyFromDeviceCtxFn>(handle, "copy_from_device_ctx");
        get_runtime_size_fn_ = load_symbol<GetRuntimeSizeFn>(handle, "get_runtime_size");
        get_runtime_alignment_fn_ = load_symbol<GetRuntimeAlignmentFn>(handle, "get_runtime_alignment");
        simpler_init_fn_ = load_symbol<SimplerInitFn>(handle, "simpler_init");
        register_callable_fn_ = load_symbol<SimplerRegisterCallableFn>(handle, "simpler_register_callable");
        run_fn_ = load_symbol<SimplerRunFn>(handle, "simpler_run");
        prepare_run_fn_ = load_symbol<SimplerPrepareRunFn>(handle, "simpler_prepare_run");
        launch_run_fn_ = load_symbol<SimplerNativeRunFn>(handle, "simpler_launch_run");
        validate_execution_context_fn_ =
            reinterpret_cast<ValidateExecutionContextFn>(dlsym(handle, "simpler_validate_execution_context"));
        // Older runtime modules have neither entry; ordinary submissions need neither.
        validate_caller_stream_fn_ =
            reinterpret_cast<ValidateCallerStreamFn>(dlsym(handle, "simpler_validate_caller_stream"));
        launch_run_on_stream_fn_ =
            reinterpret_cast<CallerStreamLaunchFn>(dlsym(handle, "simpler_launch_run_on_stream"));
        launch_run_joined_fn_ = load_symbol<SimplerJoinedLaunchFn>(handle, "simpler_launch_run_joined");
        poll_run_fn_ = load_symbol<SimplerNativeRunFn>(handle, "simpler_poll_run");
        wait_run_fn_ = load_symbol<SimplerNativeRunFn>(handle, "simpler_wait_run");
        finalize_run_fn_ = load_symbol<SimplerNativeRunFn>(handle, "simpler_finalize_run");
        // Absent on every simulated and stand-in module by construction, so a
        // null here is a fact about the backend rather than a stale build. The
        // caller reports it.
        probe_run_retention_fn_ =
            reinterpret_cast<SimplerProbeRunRetentionFn>(dlsym(handle, "simpler_probe_run_retention"));
        // Optional for the same reason: a module that records no teardown does
        // not export this, and a null is that fact rather than a stale build.
        get_teardown_report_fn_ = reinterpret_cast<GetTeardownReportFn>(dlsym(handle, "get_teardown_report"));
        // Cross-run collection entry points. Resolved by name rather than added
        // to `simpler_init`'s signature, which every runtime module must match
        // exactly; a module built without them simply cannot retain runs.
        //
        // Two names, because this one is the only symbol here that was renamed
        // after it shipped: the canonical spelling first, then the one it
        // shipped under, so a module from either build is usable.
        set_retain_runs_fn_ = reinterpret_cast<SimplerSetRetainRunsFn>(dlsym(handle, "simpler_set_retain_runs_ctx"));
        if (set_retain_runs_fn_ == nullptr) {
            set_retain_runs_fn_ =
                reinterpret_cast<SimplerSetRetainRunsFn>(dlsym(handle, "simpler_set_dfx_session_ctx"));
        }
        set_workspace_budget_fn_ =
            reinterpret_cast<SimplerSetWorkspaceBudgetFn>(dlsym(handle, "simpler_set_workspace_budget_ctx"));
        // Automatic admission, resolved by name for the same reason as the
        // entries above: a module built without them cannot size itself, and a
        // caller that never asks for it never looks for them.
        enable_automatic_admission_fn_ = reinterpret_cast<SimplerEnableAutomaticAdmissionFn>(
            dlsym(handle, "simpler_enable_automatic_admission_ctx")
        );
        admissible_run_resource_sets_fn_ = reinterpret_cast<SimplerAdmissibleRunResourceSetsFn>(
            dlsym(handle, "simpler_admissible_run_resource_sets_ctx")
        );
        abort_run_plan_fn_ = reinterpret_cast<SimplerAbortRunPlanFn>(dlsym(handle, "simpler_abort_run_plan_ctx"));
        enable_workspace_management_fn_ = reinterpret_cast<SimplerEnableWorkspaceManagementFn>(
            dlsym(handle, "simpler_enable_workspace_management_ctx")
        );
        get_workspace_report_fn_ =
            reinterpret_cast<SimplerGetWorkspaceReportFn>(dlsym(handle, "simpler_get_workspace_report_ctx"));
        flush_diagnostics_fn_ =
            reinterpret_cast<SimplerFlushDiagnosticsFn>(dlsym(handle, "simpler_flush_diagnostics_ctx"));
        supports_concurrent_native_prepare_fn_ =
            load_symbol<SupportsConcurrentNativePrepareFn>(handle, "supports_concurrent_native_prepare_ctx");
        supports_joined_native_launch_fn_ =
            load_symbol<SupportsConcurrentNativePrepareFn>(handle, "supports_joined_native_launch_ctx");
        // Optional: a runtime built before the diagnostic join exports no
        // such symbol, and a null pointer is the refusal.
        supports_joined_diagnostic_launch_fn_ =
            reinterpret_cast<SupportsJoinedDiagnosticLaunchFn>(dlsym(handle, "supports_joined_diagnostic_launch_ctx"));
        // Optional for the same reason: a runtime built before the retained
        // capacity grant exports nothing, and a null pointer is the refusal.
        set_retained_epoch_limit_fn_ =
            reinterpret_cast<SimplerSetRetainedEpochLimitFn>(dlsym(handle, "simpler_set_retained_epoch_limit_ctx"));
        // Optional on the same terms: a module built before per-run resources
        // were sized by the grant exports nothing and keeps its own sizing.
        set_run_resource_grant_fn_ =
            reinterpret_cast<SimplerSetRunResourceGrantFn>(dlsym(handle, "simpler_set_run_resource_grant_ctx"));
        get_arena_bank_gm_heap_base_fn_ =
            load_symbol<GetArenaBankGmHeapBaseFn>(handle, "get_arena_bank_gm_heap_base_ctx");
        get_retained_temp_addr_fn_ = load_symbol<GetRetainedTempAddrFn>(handle, "get_retained_temp_addr_ctx");
        get_pipeline_contract_fn = load_symbol<GetPipelineContractFn>(handle, "get_pipeline_contract");
        unregister_callable_fn_ = load_symbol<SimplerUnregisterCallableFn>(handle, "simpler_unregister_callable");
        get_aicpu_dlopen_count_fn_ = load_symbol<GetAicpuDlopenCountFn>(handle, "get_aicpu_dlopen_count");
        get_host_dlopen_count_fn_ = load_symbol<GetAicpuDlopenCountFn>(handle, "get_host_dlopen_count");
        get_run_stream_set_create_count_fn_ =
            load_symbol<GetAicpuDlopenCountFn>(handle, "get_run_stream_set_create_count");
        finalize_device_fn_ = load_symbol<FinalizeDeviceFn>(handle, "finalize_device");
        // ACL lifecycle + comm_* + kernel mode are part of the uniform
        // host_runtime.so ABI. Every platform runtime exports all of them —
        // a runtime without a real backend (today: a5 for comm) ships
        // not-supported stubs rather than omitting the symbols, so these groups
        // resolve unconditionally and never half-load.
        //
        // simpler_kernel_mode_supported does not answer which runtime has a
        // kernel backend: it answers whether kernel-mode launches are
        // available, and stays zero while simpler_kernel_mode_launch is a
        // rejecting stub — so it is zero on a runtime that can already
        // establish a kernel context. The runtime-capability question is
        // answered by simpler_kernel_mode_init, which refuses with
        // PTO_RUNTIME_ERR_UNSUPPORTED and establishes nothing when its runtime
        // cannot size kernel resources. Neither gates symbol resolution.
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
        kernel_supported_fn = load_symbol<KernelSupportedFn>(handle, "simpler_kernel_mode_supported");
        kernel_init_fn = load_symbol<KernelInitFn>(handle, "simpler_kernel_mode_init");
        kernel_prepare_callable_fn =
            load_symbol<KernelPrepareCallableFn>(handle, "simpler_kernel_mode_prepare_callable");
        kernel_launch_fn = load_symbol<KernelLaunchFn>(handle, "simpler_kernel_mode_launch");
    } catch (...) {
        throw;
    }

    if (borrowed_context_generation.has_value() &&
        (validate_execution_context_fn_ == nullptr || validate_caller_stream_fn_ == nullptr ||
         launch_run_on_stream_fn_ == nullptr)) {
        throw std::runtime_error("borrowed initialization requires native caller ordering and context validation");
    }

    const PipelineContract *contract = get_pipeline_contract_fn();
    if (!is_valid_pipeline_contract(contract) || !has_serviceable_arena_topology(*contract) ||
        !has_serviceable_stream_topology(*contract)) {
        throw std::runtime_error("host runtime returned a PipelineContract this build cannot accept");
    }
    PipelineContract resolved_contract = *contract;
    // The runtime publishes what it *supports*; this decides how much of it this
    // context uses. A caller that asked for nothing gets the standing default
    // rather than the maximum, so raising a runtime's published support moves no
    // existing route's capacity or footprint. Granted here, before any per-slot
    // storage is built or prewarmed, because nothing downstream re-derives it and
    // no caller can resize these pools once they exist.
    const uint32_t wanted = requested_pipeline_depth != 0 ? requested_pipeline_depth : kDefaultRunResourceSets;
    if (wanted < resolved_contract.pipeline_depth) {
        resolved_contract.pipeline_depth = wanted;
    }
    // The generation filter is sized by the grant for the same reason the run
    // storage below is: a set exists because a run can hold it.
    pipeline_generations_.resize(resolved_contract.pipeline_depth);

    const uint64_t incarnation = next_environment_incarnation();
    device_ctx_ = create_device_context_fn_();
    if (device_ctx_ == nullptr) {
        throw std::runtime_error("create_device_context returned null");
    }

    execution_environment_ = ExecutionEnvironment{
        static_cast<uint64_t>(getpid()), incarnation, borrowed_context_generation.value_or(incarnation), device_id,
        borrowed_context_generation.has_value()
    };

    try {
        // The grant reaches the runtime before anything can name a set, so every
        // per-run resource behind the ABI is built to the count this context
        // actually uses. A module that exports no such entry keeps its own
        // sizing and is held to it by the slot bound every prepare checks.
        if (set_run_resource_grant_fn_ != nullptr) {
            const int grant_rc = set_run_resource_grant_fn_(device_ctx_, resolved_contract.pipeline_depth);
            if (grant_rc != 0) {
                throw std::runtime_error(
                    "host runtime refused a grant of " + std::to_string(resolved_contract.pipeline_depth) +
                    " run-resource set(s) with code " + std::to_string(grant_rc)
                );
            }
        }
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
        execution_environment_.reset();
        throw;
    }

    // Cold-path init attaches an owned device or adopts the caller's current
    // RTS context. Both paths retain executor binaries on the runner; later
    // operations attach or validate their calling thread against that binding.
    // Only owned init provisions the optional SDMA workspaces and log level.
    //
    // read_binary_file may throw — defer the dlsym/dlclose rollback to the
    // catch block so the buffers and any partially-resolved handle are torn
    // down symmetrically.
    bool borrowed_init_attempted = false;
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
        // The warmup ELF rides simpler_init because it warms the workspace that
        // init provisions. Absent on arches that build no such ELF and on tests
        // driving _ChipWorker.init directly; the platform then skips the warmup
        // and the first TPREFETCH_ASYNC pays the cold control path instead. Read
        // only when the Worker opted in, so a stale path on a non-SDMA Worker is
        // never touched.
        std::vector<uint8_t> warmup_bytes;
        if (enable_sdma && !sdma_warmup_path.empty()) {
            warmup_bytes = read_binary_file(sdma_warmup_path);
        }
        const uint8_t *warmup_ptr = warmup_bytes.empty() ? nullptr : warmup_bytes.data();
        // `prewarm_config` (fork-constant, COW-delivered) rides simpler_init: the
        // platform builds + caches the prebuilt runtime-arena for its ring sizing
        // right after the device comes up. Null => no prewarm.
        // Latched before the first run so the collector's lazy initialize()
        // sees it; a module without the symbol reports the refusal rather than
        // silently running without retaining anything.
        if (collect_across_runs) {
            if (set_retain_runs_fn_ == nullptr) {
                throw std::runtime_error("ChipWorker::init: this runtime module cannot retain runs across boundaries");
            }
            if (set_retain_runs_fn_(device_ctx_, 1) != 0) {
                throw std::runtime_error("ChipWorker::init: retaining runs across boundaries could not be enabled");
            }
            // How many runs this Worker could ever have open at once: its own
            // launch budget and the resource sets this context was actually
            // granted, never the request and never a host-build constant --
            // the runtime is loaded, so only it knows its device bound.
            //
            // A depth below two asks for nothing: that is L2's shape, whose
            // launch budget is one, and it keeps the shipped capacity without
            // being asked about separately. Only a success return records the
            // grant; absence or refusal leaves this Worker exactly as a module
            // without the entry leaves it.
            const uint32_t depth =
                launch_depth_ < resolved_contract.pipeline_depth ? launch_depth_ : resolved_contract.pipeline_depth;
            if (depth > simpler::dfx::runs::kMaxOpenEpochs && set_retained_epoch_limit_fn_ != nullptr &&
                set_retained_epoch_limit_fn_(device_ctx_, static_cast<int32_t>(depth)) == 0) {
                retained_epoch_limit_granted_ = depth;
            }
        }
        // Onboard init installs workspace ownership for both owned and borrowed
        // contexts. A finite limit is staged before either init entry runs.
        // Simulator backends manage no device workspace. `manage_workspace` is
        // a compatibility argument and does not select ownership policy.
        const bool simulated = !sim_context_path.empty();
        (void)manage_workspace;
        if (!simulated) {
            if (workspace_budget_bytes != 0) {
                if (set_workspace_budget_fn_ == nullptr) {
                    throw std::runtime_error("ChipWorker::init: this runtime module has no workspace budget support");
                }
                if (set_workspace_budget_fn_(device_ctx_, workspace_budget_bytes) != 0) {
                    throw std::runtime_error("ChipWorker::init: workspace budget could not be enabled");
                }
                workspace_limit_latched_ = true;
            }
            // A module that manages the four regions is one that can report on
            // them, and that accessor is what `workspace_report` needs to tell
            // "not managed" from "managed but unreadable". Its absence is the
            // only way a non-sim module can be unmanaged here.
            workspace_managed_ = get_workspace_report_fn_ != nullptr;
        } else if (workspace_budget_bytes != 0) {
            throw std::runtime_error("ChipWorker::init: a workspace budget is not supported on a simulated backend");
        }
        if (borrowed_context_generation.has_value()) {
            const CallConfig default_config{};
            borrowed_init_attempted = true;
            init_rc = kernel_init_fn(
                device_ctx_, device_id, aicpu_bytes.data(), aicpu_bytes.size(), aicore_bytes.data(),
                aicore_bytes.size(), dispatcher_ptr, dispatcher_bytes.size(),
                prewarm_config != nullptr ? prewarm_config : &default_config, *borrowed_context_generation
            );
            // UNSUPPORTED establishes no resource owner and owes no close.
            if (init_rc == PTO_RUNTIME_ERR_UNSUPPORTED) borrowed_init_attempted = false;
        } else {
            init_rc = simpler_init_fn_(
                device_ctx_, device_id, aicpu_bytes.data(), aicpu_bytes.size(), aicore_bytes.data(),
                aicore_bytes.size(), dispatcher_ptr, dispatcher_bytes.size(), prewarm_config, enable_sdma ? 1 : 0,
                warmup_ptr, warmup_bytes.size()
            );
        }
        if (init_rc != 0) {
            throw std::runtime_error(
                std::string(borrowed_context_generation.has_value() ? "borrowed init" : "simpler_init") +
                " failed with code " + std::to_string(init_rc)
            );
        }
    } catch (...) {
        // A borrowed runner's destroy entry refuses live owners. Only a
        // successful explicit close makes destroying and unloading it safe.
        if (borrowed_init_attempted) {
            const int close_rc = finalize_device_fn_(device_ctx_);
            if (close_rc != 0) {
                lib_handle_ = host_guard.release();
                device_id_ = device_id;
                throw std::runtime_error(
                    "ChipWorker borrowed init failed with code " + std::to_string(init_rc) + "; rollback failed (" +
                    std::to_string(close_rc) + "); keeping the context and host runtime loaded; finalize is required"
                );
            }
        }
        // Program rollback runs the runner destructor without an explicit
        // finalize: a failed sim attach must not release a sibling's context.
        destroy_device_context_fn_(device_ctx_);
        device_ctx_ = nullptr;
        clear_runtime_binding();
        throw;
    }

    lib_handle_ = host_guard.release();
    kernel_supported_fn_ = kernel_supported_fn;
    kernel_init_fn_ = kernel_init_fn;
    kernel_prepare_callable_fn_ = kernel_prepare_callable_fn;
    kernel_launch_fn_ = kernel_launch_fn;
    device_id_ = device_id;
    // Published only once the runtime is up: the rollback paths above leave the
    // default K=1 contract in place, so a failed init never reports the counts
    // of a runtime this worker is not bound to.
    pipeline_contract_ = resolved_contract;
    initialized_ = true;

    run_lane_ = std::make_unique<ChipRunLane>(*this);
}

void ChipWorker::capture_teardown_report_noexcept() noexcept {
    if (teardown_report_captured_ || get_teardown_report_fn_ == nullptr || device_ctx_ == nullptr) return;
    SimplerTeardownReport report{};
    int rc = PTO_RUNTIME_ERR_INTERNAL;
    try {
        rc = get_teardown_report_fn_(device_ctx_, &report, sizeof(report));
    } catch (...) {
        // Observation must never displace the teardown outcome the caller is
        // about to be told about, so a capture failure stays silent and leaves
        // the record absent.
        return;
    }
    if (rc != 0 || report.schema != TEARDOWN_REPORT_SCHEMA) return;
    teardown_report_ = report;
    teardown_report_captured_ = true;
}

ChipWorker::WorkspaceReportStatus ChipWorker::workspace_report(SimplerWorkspaceReport *out) const noexcept {
    if (!workspace_managed_) return WorkspaceReportStatus::Disabled;
    if (out == nullptr || device_ctx_ == nullptr || get_workspace_report_fn_ == nullptr) {
        return WorkspaceReportStatus::Unavailable;
    }
    SimplerWorkspaceReport report{};
    int rc = PTO_RUNTIME_ERR_INTERNAL;
    try {
        rc = get_workspace_report_fn_(device_ctx_, &report, sizeof(report));
    } catch (...) {
        return WorkspaceReportStatus::Unavailable;
    }
    // An unrecognised schema is the whole record absent, not a partial read.
    if (rc != 0 || report.schema != WORKSPACE_REPORT_SCHEMA) return WorkspaceReportStatus::Unavailable;
    *out = report;
    return WorkspaceReportStatus::Available;
}

bool ChipWorker::teardown_report(SimplerTeardownReport *out) const {
    if (out == nullptr || !teardown_report_captured_) return false;
    *out = teardown_report_;
    return true;
}

void ChipWorker::finalize() {
    validate_execution_context();
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
    // A run whose destruction could not be proved still owns whatever its carrier holds, and the
    // carrier itself still lives in this worker's runtime storage. Everything below this point —
    // the device teardown, the context destruction, the `dlclose` and the release of that storage
    // — would either free those resources a second time or unload the code that owns their release
    // routines. So the close stops here and keeps all of it, which is the same contract the device
    // teardown's own failure already states.
    const size_t unproven = unproven_native_run_count();
    if (unproven != 0) {
        // The storage leaves this object's ownership before anything else
        // happens. Everything below would free those resources a second time or
        // unload the code that owns their release routines, and this object's
        // own destructor would free the carrier's bytes whatever the run table
        // says — so the one durable retention is to detach the block, which no
        // later destructor then returns to the allocator.
        const size_t abandoned = abandon_unproven_native_run_storage();
        throw std::runtime_error(
            "ChipWorker::finalize: " + std::to_string(unproven) +
            " native run(s) could not be proved destroyed; abandoned " + std::to_string(abandoned) +
            " run storage block(s) and kept the context and the host runtime loaded"
        );
    }
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

    int device_finalize_rc = 0;
    if (device_ctx_ != nullptr && finalize_device_fn_ != nullptr && lib_handle_ != nullptr) {
        device_finalize_rc = finalize_device_fn_(device_ctx_);
        // Immediately after the teardown call and before both the throw below
        // and the context destruction / dlclose a success runs into: that is
        // the only window in which the recording owner is still reachable.
        // `finalize_device` catches everything and returns an rc, so a
        // sequential call here covers the failing teardown too.
        capture_teardown_report_noexcept();
    }
    // A context whose teardown did not complete still owns device resources,
    // and unloading the library that owns their release routines — or
    // destroying the context that holds them — is unrecoverable. Both the
    // context and the handle stay, so an explicit retry can finish the job.
    if (device_finalize_rc != 0) {
        throw std::runtime_error(
            "ChipWorker::finalize: device teardown failed (" + std::to_string(device_finalize_rc) +
            "); keeping the context and host runtime loaded"
        );
    }
    if (device_ctx_ != nullptr && destroy_device_context_fn_ != nullptr) {
        destroy_device_context_fn_(device_ctx_);
        device_ctx_ = nullptr;
    }
    if (lib_handle_) {
        // The destroyed context owns the retained orchestration scratch, whose
        // shared_ptr deleter and control-block code live in this host runtime.
        // Native runs and their borrowed references are drained before teardown.
        assert(device_ctx_ == nullptr);
        dlclose(lib_handle_);
    }
    clear_runtime_binding();
    finalized_ = true;
}

void ChipWorker::clear_runtime_binding() {
    execution_environment_.reset();
    lib_handle_ = nullptr;
    create_device_context_fn_ = nullptr;
    simpler_init_fn_ = nullptr;
    destroy_device_context_fn_ = nullptr;
    device_malloc_ctx_fn_ = nullptr;
    device_free_ctx_fn_ = nullptr;
    device_free_caller_buffer_ctx_fn_ = nullptr;
    device_borrow_caller_buffers_ctx_fn_ = nullptr;
    device_release_caller_buffers_ctx_fn_ = nullptr;
    device_committed_memory_fn_ = nullptr;
    device_memory_info_fn_ = nullptr;
    copy_to_device_ctx_fn_ = nullptr;
    copy_from_device_ctx_fn_ = nullptr;
    get_runtime_size_fn_ = nullptr;
    get_runtime_alignment_fn_ = nullptr;
    register_callable_fn_ = nullptr;
    run_fn_ = nullptr;
    prepare_run_fn_ = nullptr;
    launch_run_fn_ = nullptr;
    validate_execution_context_fn_ = nullptr;
    validate_caller_stream_fn_ = nullptr;
    launch_run_on_stream_fn_ = nullptr;
    launch_run_joined_fn_ = nullptr;
    poll_run_fn_ = nullptr;
    wait_run_fn_ = nullptr;
    finalize_run_fn_ = nullptr;
    probe_run_retention_fn_ = nullptr;
    get_teardown_report_fn_ = nullptr;
    // Reached only past the throw a failed teardown takes, which keeps the
    // latch — and with it a readable report — for the retry. Past it, every
    // block this generation owned has been released or quarantined, so the
    // latch is resolved and must not outlive the generation that set it.
    workspace_managed_ = false;
    workspace_limit_latched_ = false;
    automatic_admission_ = false;
    set_workspace_budget_fn_ = nullptr;
    enable_workspace_management_fn_ = nullptr;
    get_workspace_report_fn_ = nullptr;
    supports_concurrent_native_prepare_fn_ = nullptr;
    supports_joined_native_launch_fn_ = nullptr;
    supports_joined_diagnostic_launch_fn_ = nullptr;
    set_run_resource_grant_fn_ = nullptr;
    get_arena_bank_gm_heap_base_fn_ = nullptr;
    get_retained_temp_addr_fn_ = nullptr;
    unregister_callable_fn_ = nullptr;
    get_aicpu_dlopen_count_fn_ = nullptr;
    get_host_dlopen_count_fn_ = nullptr;
    get_run_stream_set_create_count_fn_ = nullptr;
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
    set_retain_runs_fn_ = nullptr;
    set_retained_epoch_limit_fn_ = nullptr;
    retained_epoch_limit_granted_ = simpler::dfx::runs::kMaxOpenEpochs;
    flush_diagnostics_fn_ = nullptr;
    kernel_supported_fn_ = nullptr;
    kernel_init_fn_ = nullptr;
    kernel_prepare_callable_fn_ = nullptr;
    kernel_launch_fn_ = nullptr;
    runtime_bufs_.clear();
    pipeline_generations_.reset();
    pipeline_contract_ = {PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, 1, {}};
    initialized_ = false;
    device_id_ = -1;
}

void ChipWorker::register_callable(int32_t callable_id, const void *callable) {
    if (!initialized_) {
        throw std::runtime_error("ChipWorker not initialized; call init() first");
    }
    if (callable == nullptr) {
        throw std::runtime_error("register_callable: callable must not be null");
    }
    validate_execution_context();
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
    volatile int32_t *accepted_state, int32_t accepted_value, uint64_t run_id, uint64_t dispatch_id,
    NativeRunProvenance provenance
) {
    if (lease.reserved != 0 || lease.generation == 0 || lease.slot_id >= pipeline_contract_.pipeline_depth) {
        throw std::runtime_error("run pipeline lease is outside the runtime PipelineContract");
    }
    if (args == nullptr) throw std::runtime_error("run_with_lease requires task args");
    if (run_lane_ == nullptr) throw std::runtime_error("ChipWorker run lane is not initialized");
    ChipRun run = run_lane_->submit(
        callable_id, *args, config, lease, run_id, dispatch_id, accepted_state, accepted_value, true, provenance
    );
    (void)run.wait_until(ChipRun::Deadline::max());
}

ChipWorkerNativeRun ChipWorker::prepare_native_run(
    int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config, const PipelineSlotLease &lease,
    uint64_t run_id, uint64_t dispatch_id, volatile int32_t *accepted_state, int32_t accepted_value,
    NativeRunProvenance provenance
) {
    if (lease.reserved != 0 || lease.generation == 0 || lease.slot_id >= pipeline_contract_.pipeline_depth) {
        throw std::runtime_error("native-run pipeline lease is outside the runtime PipelineContract");
    }
    return prepare_native_run_on_slot(
        callable_id, args, config, lease.slot_id, lease.generation, run_id, dispatch_id, accepted_state, accepted_value,
        true, provenance
    );
}

ChipWorkerNativeRun ChipWorker::prepare_native_run_for_lane(
    int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config, const PipelineSlotLease &lease,
    uint64_t run_id, uint64_t dispatch_id, volatile int32_t *accepted_state, int32_t accepted_value,
    bool pipeline_leased, NativeRunProvenance provenance
) {
    return prepare_native_run_on_slot(
        callable_id, args, config, lease.slot_id, lease.generation, run_id, dispatch_id, accepted_state, accepted_value,
        pipeline_leased, provenance
    );
}

bool ChipWorker::enable_automatic_admission() {
    if (!initialized_ || device_ctx_ == nullptr) return false;
    if (automatic_admission_) return true;
    if (enable_automatic_admission_fn_ == nullptr) return false;
    automatic_admission_ = enable_automatic_admission_fn_(device_ctx_) == 0;
    return automatic_admission_;
}

uint32_t ChipWorker::admissible_run_resource_sets() const {
    if (!automatic_admission_ || admissible_run_resource_sets_fn_ == nullptr || device_ctx_ == nullptr) {
        return pipeline_contract_.pipeline_depth;
    }
    const uint32_t reported = admissible_run_resource_sets_fn_(device_ctx_);
    // A context always holds the run it is executing, and it can never hold
    // more sets than it was granted: a report outside that is not a capacity
    // this worker can act on.
    if (reported == 0) return 1;
    return reported > pipeline_contract_.pipeline_depth ? pipeline_contract_.pipeline_depth : reported;
}

int ChipWorker::abort_run_plan(uint32_t pipeline_slot, uint64_t run_epoch) {
    if (!automatic_admission_ || abort_run_plan_fn_ == nullptr || device_ctx_ == nullptr) return 0;
    return abort_run_plan_fn_(device_ctx_, pipeline_slot, run_epoch);
}

bool ChipWorker::supports_concurrent_native_prepare() const {
    return initialized_ && pipeline_contract_.pipeline_depth > 1 &&
           supports_concurrent_native_prepare_fn_(device_ctx_) > 0;
}

bool ChipWorker::supports_joined_native_launch() const {
    return initialized_ && launch_depth_ > 1 && pipeline_contract_.pipeline_depth > 1 &&
           supports_joined_native_launch_fn_(device_ctx_) > 0;
}

bool ChipWorker::supports_joined_diagnostic_launch(
    const ChipWorkerNativeRun &run, const ChipWorkerNativeRun &predecessor
) {
    if (!initialized_) return false;
    // As deep as this collector was actually granted, and no deeper. The
    // backend answers how many runs it can hold open at once; ordering a
    // further one behind them would leave the last with no bucket to be
    // admitted into, and admission has no refusal for that — it waits. A
    // module that exports no capacity entry, or refused the request, leaves
    // this at the shipped two. Ordinary non-diagnostic joins keep the
    // backend's own depth, which this does not touch.
    if (launch_depth_ < 2 || launch_depth_ > retained_epoch_limit_granted_) return false;
    if (pipeline_contract_.pipeline_depth <= 1) return false;
    // An older `libhost_runtime.so` exports no such symbol. Absent means the
    // refusal this call is asking to lift, so the caller keeps today's
    // behaviour.
    if (supports_joined_diagnostic_launch_fn_ == nullptr) return false;
    if (run.slot_id >= runtime_bufs_.size() || predecessor.slot_id >= runtime_bufs_.size()) return false;
    if (run.slot_id == predecessor.slot_id) return false;
    // Both carriers are dereferenced by the backend call, so both are leased — the same protection
    // the joined launch itself takes. A phase check under the bookkeeping mutex is not enough: a
    // finalizer reserves a run and then releases that mutex for the whole runtime destruction,
    // leaving the phase it reserved intact, so a query admitted on the phase alone can enter while
    // a carrier is being destroyed.
    //
    // Immediate, because this answers one scheduling round. A run another operation is inside is a
    // refusal for this round and the question is asked again next round; nothing is remembered and
    // no thread waits here.
    void *storage = nullptr;
    void *predecessor_storage = nullptr;
    NativeRunOpLease held;
    NativeRunOpLease predecessor_held;
    const NativeRunLease lease = acquire_native_run_pair(
        run, predecessor, NativeRunOp::ProbeSuccessor, &storage, &predecessor_storage, &held, &predecessor_held,
        LeaseWait::Immediate
    );
    if (lease != NativeRunLease::Acquired) return false;
    return supports_joined_diagnostic_launch_fn_(device_ctx_, storage, predecessor_storage) > 0;
}

bool ChipWorker::holds_live_comm_resources() const {
    // Unsynchronized: comm sessions, global domains and the exported-region
    // declaration are all written from the same thread that drives the run lane
    // — a child's control commands and its task frames arrive on one loop.
    return !comm_sessions_.empty() || !global_domain_ids_.empty() || exported_device_regions_live_;
}

void ChipWorker::configure_launch_depth(unsigned depth) {
    if (initialized_) {
        throw std::runtime_error("ChipWorker::configure_launch_depth after init");
    }
    if (depth == 0) {
        throw std::runtime_error("ChipWorker::configure_launch_depth requires a depth of at least one");
    }
    // Not clamped here: the contract's pipeline depth is unknown until init
    // binds the runtime, and `supports_joined_native_launch` is where the two
    // meet. A launched run holds its pipeline slot for its whole lifetime, so
    // the slot count is the real ceiling.
    launch_depth_ = depth;
}

ChipWorkerNativeRun ChipWorker::prepare_native_run_on_slot(
    int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config, uint32_t slot_id,
    uint64_t generation, uint64_t run_id, uint64_t dispatch_id, volatile int32_t *accepted_state,
    int32_t accepted_value, bool admit_pipeline_generation, NativeRunProvenance provenance
) {
    validate_execution_context();
    try {
        config.validate();
    } catch (const std::invalid_argument &e) {
        throw RunPreparationRefused(e.what());
    }
    if (!initialized_) {
        throw RunPreparationRefused("ChipWorker not initialized; call init() first");
    }
    if (args == nullptr) {
        throw RunPreparationRefused("prepare_native_run requires task args");
    }
    if (slot_id >= runtime_bufs_.size()) {
        throw RunPreparationRefused("prepare_native_run slot is outside the runtime PipelineContract");
    }
    const uint64_t run_epoch = next_native_run_epoch();
    const ChipWorkerNativeRun run_identity{slot_id, generation, run_epoch, run_id, dispatch_id};
    const bool allow_prepared_successor = supports_concurrent_native_prepare();
    {
        std::lock_guard<std::mutex> lk(native_run_mu_);
        size_t occupied = 0;
        for (const auto &candidate : native_runs_) {
            const NativeRunEntry &entry = *candidate.second;
            if (entry.phase == NativeRunPhase::EMPTY) continue;
            if (entry.slot_id == slot_id) {
                throw RunPreparationRefused(
                    "prepare_native_run slot already owns an unfinished native run " +
                    format_native_run_identity(run_identity)
                );
            }
            ++occupied;
            if (!allow_prepared_successor || !entry.permits_prepared_successor ||
                (entry.phase != NativeRunPhase::LAUNCHED && entry.phase != NativeRunPhase::REAPED)) {
                throw RunPreparationRefused(
                    "prepare_native_run requires an active predecessor before staging a successor " +
                    format_native_run_identity(run_identity)
                );
            }
        }
        // Every run already here holds a live claim, checked above, so the only remaining
        // question is whether this context granted a resource set for one more. The granted
        // depth is that number — not a constant — and the run past it is refused rather than
        // handed a set another run owns.
        if (occupied + 1 > pipeline_contract_.pipeline_depth) {
            throw RunPreparationRefused(
                "prepare_native_run has no free run resource set at its granted depth " +
                format_native_run_identity(run_identity)
            );
        }
        if (admit_pipeline_generation &&
            !pipeline_generations_.is_admissible(PipelineSlotLease{slot_id, 0, generation})) {
            throw RunPreparationRefused(
                "native-run pipeline lease generation is stale " + format_native_run_identity(run_identity)
            );
        }
        auto entry = std::make_unique<NativeRunEntry>();
        entry->slot_id = slot_id;
        entry->lease_generation = generation;
        entry->run_epoch = run_epoch;
        entry->phase = NativeRunPhase::PREPARING;
        entry->permits_prepared_successor = allow_prepared_successor;
        native_runs_.emplace(run_epoch, std::move(entry));
    }

    int rc = -1;
    try {
        const NativeRunDescriptor descriptor{
            slot_id,        arena_bank_for_slot(slot_id), run_id,    generation, dispatch_id, run_epoch, accepted_state,
            accepted_value, launch_depth_ > 1 ? 1u : 0u,  provenance
        };
        rc = prepare_run_fn_(device_ctx_, runtime_bufs_[slot_id].data(), callable_id, args, &config, &descriptor);
    } catch (...) {
        drop_preparing_native_run(run_epoch);
        throw;
    }
    if (rc != 0) {
        const bool released =
            consume_released_preparation(runtime_bufs_[slot_id].data(), runtime_bufs_[slot_id].size(), run_epoch);
        drop_preparing_native_run(run_epoch);
        if (rc == PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE) {
            throw PreparedRunIncompatible(
                "native prepare requires depth-one fallback " + format_native_run_identity(run_identity)
            );
        }
        if (rc == SIMPLER_RUNTIME_ERR_RESOURCE_WAIT) {
            throw PreparedRunResourceWait(
                "native prepare is waiting for run resources a live consumer still holds " +
                format_native_run_identity(run_identity)
            );
        }
        const std::string message = "prepare_native_run failed with code " + std::to_string(rc) + " " +
                                    format_native_run_identity(run_identity);
        if (released) throw RunPreparationRefused(message);
        throw std::runtime_error(message);
    }

    {
        std::unique_lock<std::mutex> lk(native_run_mu_);
        auto it = native_runs_.find(run_epoch);
        if (it == native_runs_.end() || it->second->phase != NativeRunPhase::PREPARING) {
            lk.unlock();
            (void)finalize_run_fn_(device_ctx_, runtime_bufs_[slot_id].data());
            drop_preparing_native_run(run_epoch);
            throw std::runtime_error("native-run identity changed while prepare was in progress");
        }
        if (admit_pipeline_generation && !pipeline_generations_.admit(PipelineSlotLease{slot_id, 0, generation})) {
            lk.unlock();
            (void)finalize_run_fn_(device_ctx_, runtime_bufs_[slot_id].data());
            drop_preparing_native_run(run_epoch);
            throw std::runtime_error(
                "native-run pipeline lease generation became stale while prepare was in progress " +
                format_native_run_identity(run_identity)
            );
        }
        it->second->phase = NativeRunPhase::PREPARED;
    }
    return run_identity;
}

ChipRun ChipWorker::submit_chip_run(
    int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config, const PipelineSlotLease &lease,
    uint64_t run_id, uint64_t dispatch_id, volatile int32_t *accepted_state, int32_t accepted_value, bool activated,
    NativeRunProvenance provenance
) {
    if (run_lane_ == nullptr) throw std::runtime_error("ChipWorker run lane is not initialized");
    return run_lane_->submit(
        callable_id, args, config, lease, run_id, dispatch_id, accepted_state, accepted_value, activated, provenance
    );
}

ChipRun ChipWorker::submit_chip_run(
    int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config, volatile int32_t *accepted_state,
    int32_t accepted_value, void *caller_stream
) {
    if (run_lane_ == nullptr) throw std::runtime_error("ChipWorker run lane is not initialized");
    return run_lane_->submit(callable_id, args, config, accepted_state, accepted_value, caller_stream);
}

void ChipWorker::stop_chip_run_lane_admission() noexcept {
    if (run_lane_ != nullptr) run_lane_->stop_admission();
}

void ChipWorker::close_chip_run_lane() {
    if (run_lane_ != nullptr) run_lane_->close();
}

void ChipWorker::validate_execution_context() const {
    // Foreign-process bindings are rejected before entry to the runtime or its mutexes.
    if (execution_environment_ && execution_environment_->process_id != static_cast<uint64_t>(getpid())) {
        throw ExecutionContextRefused("execution environment belongs to another process");
    }
    if (device_ctx_ == nullptr || validate_execution_context_fn_ == nullptr) return;
    const int rc = validate_execution_context_fn_(device_ctx_);
    if (rc != 0) throw ExecutionContextRefused("execution context validation failed with code " + std::to_string(rc));
}

void ChipWorker::abandon_after_failed_teardown(std::exception_ptr error) noexcept {
    if (run_lane_ != nullptr) {
        run_lane_->detach_after_failed_teardown(error);
        // The retained lane keeps any run-side host snapshots alive as well.
        (void)run_lane_.release();
    }
    for (auto &storage : runtime_bufs_)
        storage.abandon();
    // device_ctx_ and lib_handle_ remain live for the process lifetime.
}

void ChipWorker::validate_caller_stream(void *caller_stream) {
    if (!initialized_ || validate_caller_stream_fn_ == nullptr || launch_run_on_stream_fn_ == nullptr) {
        throw std::runtime_error("caller-stream submission requires an initialized runtime with caller ordering");
    }
    const int rc = validate_caller_stream_fn_(device_ctx_, caller_stream);
    if (rc != 0) throw std::runtime_error("caller-stream validation failed with code " + std::to_string(rc));
}

void ChipWorker::drop_preparing_native_run(uint64_t run_epoch) noexcept {
    std::lock_guard<std::mutex> lk(native_run_mu_);
    const auto it = native_runs_.find(run_epoch);
    if (it != native_runs_.end() && it->second->phase == NativeRunPhase::PREPARING) {
        native_runs_.erase(it);
    }
    native_run_ops_done_.notify_all();
}

ChipWorker::RunStorageTag ChipWorker::read_run_storage_tag(const void *storage) const noexcept {
    // A byte read of storage this worker owns, never a use of the object the runtime placed in
    // it: after `destroy_native_run_context` that object's lifetime has ended, and only the raw
    // bytes remain. Zero is the tag that call leaves behind; the canonical value means it never
    // ran; anything else is a partially torn-down or corrupted tag and is never treated as either.
    if (storage == nullptr) return RunStorageTag::Noncanonical;
    uint64_t tag = 0;
    std::memcpy(&tag, storage, sizeof(tag));
    if (tag == 0) return RunStorageTag::Destroyed;
    if (tag == kNativeRunContextMagic) return RunStorageTag::Canonical;
    return RunStorageTag::Noncanonical;
}

bool ChipWorker::native_run_op_admits_phase(NativeRunOp op, NativeRunPhase phase) {
    switch (op) {
    case NativeRunOp::Launch:
        return phase == NativeRunPhase::PREPARED;
    case NativeRunOp::Poll:
    case NativeRunOp::Wait:
        return phase == NativeRunPhase::LAUNCHED || phase == NativeRunPhase::REAPED;
    case NativeRunOp::Finalize:
        return phase == NativeRunPhase::PREPARED || phase == NativeRunPhase::LAUNCHED ||
               phase == NativeRunPhase::REAPED;
    case NativeRunOp::Probe:
        return phase == NativeRunPhase::LAUNCHED;
    case NativeRunOp::Predecessor:
        // Only a run still executing is something to be ordered behind. A reaped one is already
        // over, so the ordinary launch is both correct and cheaper.
        return phase == NativeRunPhase::LAUNCHED;
    case NativeRunOp::ProbeSuccessor:
        // The retention probe launches and drains this one itself, so it takes it unlaunched.
        return phase == NativeRunPhase::PREPARED;
    }
    return false;
}

ChipWorker::NativeRunLease
ChipWorker::inspect_native_run_locked(const ChipWorkerNativeRun &run, NativeRunOp op, NativeRunEntry **entry_out) {
    const auto it = native_runs_.find(run.run_epoch);
    if (it == native_runs_.end()) return NativeRunLease::Absent;
    NativeRunEntry &entry = *it->second;
    if (entry.destruction_unknown) return NativeRunLease::Stale;
    if (entry.lease_generation != run.generation || entry.slot_id != run.slot_id) return NativeRunLease::Stale;
    // The already-finished answer, taken before a lease and before anything can block: a completed
    // run's poll and wait are satisfied by the recorded fact.
    if ((op == NativeRunOp::Poll || op == NativeRunOp::Wait) && entry.phase == NativeRunPhase::REAPED) {
        return NativeRunLease::AlreadyComplete;
    }
    if (!native_run_op_admits_phase(op, entry.phase)) return NativeRunLease::Stale;
    if (entry.finalize_reserved) return NativeRunLease::Reserved;
    if (entry.ops_in_flight != 0) return NativeRunLease::Busy;
    if (entry_out != nullptr) *entry_out = &entry;
    return NativeRunLease::Acquired;
}

ChipWorker::NativeRunLease ChipWorker::acquire_native_run_op(
    const ChipWorkerNativeRun &run, NativeRunOp op, void **storage_out, NativeRunPhase *phase_out,
    NativeRunOpLease *lease_out
) {
    std::unique_lock<std::mutex> lk(native_run_mu_);
    for (;;) {
        // Revalidated from scratch here and after every wake: the wait releases this mutex, so a
        // finalizer may reserve the run and the operation ahead may remove it. Nothing but the
        // epoch is carried across the release.
        NativeRunEntry *entry = nullptr;
        const NativeRunLease result = inspect_native_run_locked(run, op, &entry);
        if (result == NativeRunLease::Busy && op != NativeRunOp::Poll) {
            native_run_ops_done_.wait(lk);
            continue;
        }
        if (result != NativeRunLease::Acquired) return result;
        entry->ops_in_flight = 1;
        if (phase_out != nullptr) *phase_out = entry->phase;
        if (storage_out != nullptr) *storage_out = runtime_bufs_[entry->slot_id].data();
        if (lease_out != nullptr) *lease_out = NativeRunOpLease(this, run.run_epoch);
        return NativeRunLease::Acquired;
    }
}

ChipWorker::NativeRunLease ChipWorker::acquire_native_run_pair(
    const ChipWorkerNativeRun &run, const ChipWorkerNativeRun &predecessor, NativeRunOp op, void **storage_out,
    void **predecessor_storage_out, NativeRunOpLease *lease_out, NativeRunOpLease *predecessor_lease_out, LeaseWait wait
) {
    if (run.run_epoch == predecessor.run_epoch) return NativeRunLease::Stale;
    std::unique_lock<std::mutex> lk(native_run_mu_);
    for (;;) {
        NativeRunEntry *entry = nullptr;
        NativeRunEntry *ahead = nullptr;
        // Both are decided under one hold of the mutex, so neither can change between them, and
        // both leases are taken together or not at all.
        const NativeRunLease own = inspect_native_run_locked(run, op, &entry);
        const NativeRunLease other = inspect_native_run_locked(
            predecessor, op == NativeRunOp::ProbeSuccessor ? NativeRunOp::Probe : NativeRunOp::Predecessor, &ahead
        );
        if (own == NativeRunLease::Busy || other == NativeRunLease::Busy) {
            if (wait == LeaseWait::Immediate) return NativeRunLease::Busy;
            native_run_ops_done_.wait(lk);
            continue;
        }
        if (own != NativeRunLease::Acquired) return own;
        if (other != NativeRunLease::Acquired) return other;
        entry->ops_in_flight = 1;
        ahead->ops_in_flight = 1;
        *lease_out = NativeRunOpLease(this, run.run_epoch);
        *predecessor_lease_out = NativeRunOpLease(this, predecessor.run_epoch);
        if (storage_out != nullptr) *storage_out = runtime_bufs_[entry->slot_id].data();
        if (predecessor_storage_out != nullptr) *predecessor_storage_out = runtime_bufs_[ahead->slot_id].data();
        return NativeRunLease::Acquired;
    }
}

void ChipWorker::release_native_run_op(uint64_t run_epoch) noexcept {
    std::lock_guard<std::mutex> lk(native_run_mu_);
    const auto it = native_runs_.find(run_epoch);
    if (it != native_runs_.end()) it->second->ops_in_flight = 0;
    native_run_ops_done_.notify_all();
}

[[noreturn]] void
ChipWorker::throw_native_run_lease_error(const ChipWorkerNativeRun &run, NativeRunLease lease, const char *operation) {
    if (lease == NativeRunLease::Reserved) {
        throw std::runtime_error(
            std::string(operation) + ": another finalization is in progress " + format_native_run_identity(run)
        );
    }
    throw std::runtime_error(
        std::string(operation) + ": native-run token is stale or used in the wrong phase " +
        format_native_run_identity(run)
    );
}

void ChipWorker::note_native_run_launched(uint64_t run_epoch) noexcept {
    std::lock_guard<std::mutex> lk(native_run_mu_);
    const auto it = native_runs_.find(run_epoch);
    if (it != native_runs_.end()) it->second->phase = NativeRunPhase::LAUNCHED;
}

void ChipWorker::note_native_run_completion(uint64_t run_epoch, int rc) noexcept {
    // The execution fact, written whether or not a finalizer has reserved this run: completion
    // and the error it carries are what the run established, not a phase a reservation may hold.
    // A reservation that froze them would make a later finalize repeat a wait that already ran.
    std::lock_guard<std::mutex> lk(native_run_mu_);
    const auto it = native_runs_.find(run_epoch);
    if (it == native_runs_.end()) return;
    NativeRunEntry &entry = *it->second;
    entry.phase = NativeRunPhase::REAPED;
    if (rc != 0 && !entry.pending_rc_set) {
        entry.pending_rc = rc;
        entry.pending_rc_set = true;
    }
}

void ChipWorker::launch_native_run(const ChipWorkerNativeRun &run, void *caller_stream) {
    validate_execution_context();
    if (run.slot_id >= runtime_bufs_.size()) {
        throw std::runtime_error("native-run token slot is outside the runtime PipelineContract");
    }
    // Validated before the lease: the check is about the caller's stream, not
    // about this run, and a refusal should not have held a run's lease to reach.
    if (caller_stream != nullptr) validate_caller_stream(caller_stream);
    void *storage = nullptr;
    NativeRunOpLease held;
    const NativeRunLease lease = acquire_native_run_op(run, NativeRunOp::Launch, &storage, nullptr, &held);
    if (lease != NativeRunLease::Acquired) throw_native_run_lease_error(run, lease, "launch_native_run");

    int rc = 0;
    std::exception_ptr failure;
    try {
        // Both entries take the storage the lease handed out, which is what keeps
        // the carrier alive for the duration of the call — a caller-stream launch
        // dereferences it exactly as the ordinary one does.
        rc = caller_stream != nullptr ? launch_run_on_stream_fn_(device_ctx_, storage, caller_stream, nullptr) :
                                        launch_run_fn_(device_ctx_, storage);
    } catch (...) {
        failure = std::current_exception();
    }
    if (failure == nullptr && rc != 0) {
        // A launch that failed may still have reached the device. The probe decides which, and
        // the answer is an execution fact: a completed one carries this launch's own error into
        // the record a later finalize reports, and an uncompleted one leaves the run preparable.
        int poll_rc = 0;
        try {
            poll_rc = poll_run_fn_(device_ctx_, storage);
        } catch (...) {
            poll_rc = SIMPLER_NATIVE_RUN_POLL_ERROR;
        }
        if (poll_rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE) {
            note_native_run_completion(run.run_epoch, rc);
        }
        failure = std::make_exception_ptr(
            std::runtime_error(
                "launch_native_run failed with code " + std::to_string(rc) + " " + format_native_run_identity(run)
            )
        );
    } else if (failure == nullptr) {
        note_native_run_launched(run.run_epoch);
    }
    if (failure != nullptr) std::rethrow_exception(failure);
}

bool ChipWorker::launch_native_run_joined(
    const ChipWorkerNativeRun &run, const ChipWorkerNativeRun &predecessor, void *caller_stream
) {
    validate_execution_context();
    if (!supports_joined_native_launch()) return false;
    if (run.slot_id >= runtime_bufs_.size() || predecessor.slot_id >= runtime_bufs_.size()) {
        throw std::runtime_error("native-run token slot is outside the runtime PipelineContract");
    }
    if (run.slot_id == predecessor.slot_id) {
        throw std::runtime_error("a native run cannot be ordered behind its own pipeline slot");
    }
    // Both carriers are dereferenced by the call below, so both are leased. Holding only the
    // successor's would leave the predecessor free to be finalized and its object destroyed while
    // the joined entry is still reading it.
    void *storage = nullptr;
    void *predecessor_storage = nullptr;
    NativeRunOpLease held;
    NativeRunOpLease predecessor_held;
    const NativeRunLease lease = acquire_native_run_pair(
        run, predecessor, NativeRunOp::Launch, &storage, &predecessor_storage, &held, &predecessor_held,
        LeaseWait::Block
    );
    // A predecessor that is no longer there, or no longer executing, is not an error: this run
    // launches ordinarily once it reaches the front, which is what the backend's own refusal does.
    if (lease == NativeRunLease::Absent || lease == NativeRunLease::Stale) return false;
    if (lease != NativeRunLease::Acquired) throw_native_run_lease_error(run, lease, "launch_native_run_joined");

    int rc = 0;
    bool declined = false;
    std::exception_ptr failure;
    try {
        // Both carriers come from the pair lease, so a concurrent finalization of
        // either cannot destroy one under this call — which the caller-stream
        // entry needs as much as the joined one, since it reads both.
        rc = caller_stream != nullptr ?
                 launch_run_on_stream_fn_(device_ctx_, storage, caller_stream, predecessor_storage) :
                 launch_run_joined_fn_(device_ctx_, storage, predecessor_storage);
    } catch (...) {
        failure = std::current_exception();
    }
    // The backend refuses this way when it cannot order the two right now, and it changes nothing
    // when it does: the run is still prepared, and the caller launches it ordinarily once it
    // reaches the front.
    if (failure == nullptr && rc == PTO_RUNTIME_ERR_UNSUPPORTED) {
        declined = true;
    } else if (failure == nullptr && rc != 0) {
        int poll_rc = 0;
        try {
            poll_rc = poll_run_fn_(device_ctx_, storage);
        } catch (...) {
            poll_rc = SIMPLER_NATIVE_RUN_POLL_ERROR;
        }
        if (poll_rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE) {
            note_native_run_completion(run.run_epoch, rc);
        }
        failure = std::make_exception_ptr(
            std::runtime_error(
                "launch_native_run_joined failed with code " + std::to_string(rc) + " " +
                format_native_run_identity(run)
            )
        );
    } else if (failure == nullptr) {
        note_native_run_launched(run.run_epoch);
    }
    if (failure != nullptr) std::rethrow_exception(failure);
    return !declined;
}

bool ChipWorker::poll_native_run(const ChipWorkerNativeRun &run) {
    validate_execution_context();
    if (run.slot_id >= runtime_bufs_.size()) {
        throw std::runtime_error("native-run token slot is outside the runtime PipelineContract");
    }
    void *storage = nullptr;
    NativeRunOpLease held;
    const NativeRunLease lease = acquire_native_run_op(run, NativeRunOp::Poll, &storage, nullptr, &held);
    // Completion already established answers the question without a device call. A run another
    // operation is inside reads as "not yet": a poll reports progress and never holds a thread.
    if (lease == NativeRunLease::AlreadyComplete) return true;
    if (lease == NativeRunLease::Busy) return false;
    if (lease != NativeRunLease::Acquired) throw_native_run_lease_error(run, lease, "poll_native_run");

    int rc = SIMPLER_NATIVE_RUN_POLL_ERROR;
    std::exception_ptr failure;
    try {
        rc = poll_run_fn_(device_ctx_, storage);
    } catch (...) {
        failure = std::current_exception();
    }
    // Zero is "not ready" and one is "complete"; a complete poll carries no error, so nothing of
    // its value reaches the record a finalize reports. Every other value is this poll's failure.
    if (failure == nullptr && rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE) {
        note_native_run_completion(run.run_epoch, 0);
    } else if (failure == nullptr && rc != SIMPLER_NATIVE_RUN_POLL_NOT_READY) {
        failure = std::make_exception_ptr(
            std::runtime_error(
                "poll_native_run failed with code " + std::to_string(rc) + " " + format_native_run_identity(run)
            )
        );
    }
    if (failure != nullptr) std::rethrow_exception(failure);
    return rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE;
}

void ChipWorker::wait_native_run(const ChipWorkerNativeRun &run) {
    validate_execution_context();
    if (run.slot_id >= runtime_bufs_.size()) {
        throw std::runtime_error("native-run token slot is outside the runtime PipelineContract");
    }
    void *storage = nullptr;
    NativeRunOpLease held;
    const NativeRunLease lease = acquire_native_run_op(run, NativeRunOp::Wait, &storage, nullptr, &held);
    if (lease == NativeRunLease::AlreadyComplete) return;
    if (lease != NativeRunLease::Acquired) throw_native_run_lease_error(run, lease, "wait_native_run");

    int wait_rc = 0;
    std::exception_ptr failure;
    try {
        wait_rc = wait_run_fn_(device_ctx_, storage);
    } catch (...) {
        failure = std::current_exception();
    }
    // A wait that returned establishes completion whatever its code says, and the code is the
    // error a later finalize reports. A wait that threw established nothing, so the run keeps the
    // phase it had and this call's own exception is what the caller sees.
    if (failure == nullptr) note_native_run_completion(run.run_epoch, wait_rc);
    if (failure != nullptr) std::rethrow_exception(failure);
}

void ChipWorker::finalize_native_run(const ChipWorkerNativeRun &run) { finalize_native_run_impl(run, nullptr); }

void ChipWorker::finalize_native_run_impl(const ChipWorkerNativeRun &run, bool *no_submission_released) {
    if (no_submission_released != nullptr) *no_submission_released = false;
    validate_execution_context();
    if (run.slot_id >= runtime_bufs_.size()) {
        throw std::runtime_error("native-run token slot is outside the runtime PipelineContract");
    }
    const NativeRunFinalizeOutcome outcome = finalize_native_run_epoch(run.run_epoch, &run, FinalizeMode::Caller);
    if (no_submission_released != nullptr && outcome.captured == nullptr &&
        (outcome.fault == FinalizeFault::None || outcome.fault == FinalizeFault::Failed)) {
        *no_submission_released = outcome.no_submission_released;
    }
    // The original exception outranks any reason this call could name for it: it is what actually
    // happened, and replacing it would discard the runtime's own account of the failure.
    if (outcome.captured != nullptr) std::rethrow_exception(outcome.captured);
    switch (outcome.fault) {
    case FinalizeFault::None:
        return;
    case FinalizeFault::Stale:
        throw std::runtime_error(
            "finalize_native_run: native-run token is stale or already finalized " + format_native_run_identity(run)
        );
    case FinalizeFault::Reserved:
        throw std::runtime_error(
            "finalize_native_run: another finalization is in progress " + format_native_run_identity(run)
        );
    case FinalizeFault::NonDestroying:
        throw std::runtime_error(
            "finalize_native_run returned success without destroying the run context " + format_native_run_identity(run)
        );
    case FinalizeFault::UnknownDestruction:
        throw std::runtime_error(
            "finalize_native_run left the run context in an unknown state " + format_native_run_identity(run)
        );
    case FinalizeFault::Failed:
        throw std::runtime_error(
            "finalize_native_run failed with code " + std::to_string(outcome.rc) + " " + format_native_run_identity(run)
        );
    }
}

void ChipWorker::clear_finalize_reservation_noexcept(uint64_t run_epoch) noexcept {
    try {
        std::lock_guard<std::mutex> lk(native_run_mu_);
        const auto it = native_runs_.find(run_epoch);
        if (it != native_runs_.end()) it->second->finalize_reserved = false;
        native_run_ops_done_.notify_all();
    } catch (...) {
        // The reservation stands and the run stays unreachable to every other operation, which is
        // the isolated state this would otherwise have to reach by a path that can also fail.
    }
}

/**
 * The one finalization protocol, shared by the caller's finalize and by teardown.
 *
 * Nothing in here allocates: every reason it can report is an enumerator and a code, every
 * exception it can meet is either the runtime's own — captured, which `std::current_exception` does
 * without allocating — or the bookkeeping mutex's, which is caught and leaves the reservation
 * disposed of either way. Teardown differs in exactly two ways, both stated below: it identifies
 * the run by epoch alone, and it never leaves a run eligible for another attempt.
 */
ChipWorker::NativeRunFinalizeOutcome ChipWorker::finalize_native_run_epoch(
    uint64_t run_epoch, const ChipWorkerNativeRun *token, FinalizeMode mode
) noexcept {
    const bool teardown = mode == FinalizeMode::Teardown;
    NativeRunFinalizeOutcome outcome;
    void *storage = nullptr;
    size_t storage_bytes = 0;
    bool reserved = false;
    try {
        std::unique_lock<std::mutex> lk(native_run_mu_);
        const auto found = native_runs_.find(run_epoch);
        if (found == native_runs_.end() || found->second->destruction_unknown ||
            (token != nullptr &&
             (found->second->lease_generation != token->generation || found->second->slot_id != token->slot_id)) ||
            !native_run_op_admits_phase(NativeRunOp::Finalize, found->second->phase)) {
            if (!teardown) outcome.fault = FinalizeFault::Stale;
            return outcome;
        }
        // Exactly one finalizer, decided here and independently of the run's execution phase. The
        // phase itself is left alone: an operation finishing during the drain below still has to
        // be able to advance it, and overwriting it is what would make this call repeat a wait
        // that already ran and discard its error.
        if (found->second->finalize_reserved) {
            if (!teardown) outcome.fault = FinalizeFault::Reserved;
            return outcome;
        }
        found->second->finalize_reserved = true;
        reserved = true;

        // Drain whatever operation is already inside the runtime for this run. The wait releases
        // this mutex so that operation can reach its own tail, and the reservation above stops a
        // new one starting. Re-looked-up after every wake, never carried.
        for (;;) {
            const auto current = native_runs_.find(run_epoch);
            if (current == native_runs_.end()) return outcome;
            if (current->second->ops_in_flight == 0) {
                storage = runtime_bufs_[current->second->slot_id].data();
                storage_bytes = runtime_bufs_[current->second->slot_id].size();
                break;
            }
            native_run_ops_done_.wait(lk);
        }
    } catch (...) {
        // The bookkeeping mutex or the drain wait failed. The run never reached the runtime, so it
        // is certainly intact; the reservation this call took is given back so another finalization
        // can still close it.
        outcome.captured = std::current_exception();
        if (reserved) clear_finalize_reservation_noexcept(run_epoch);
        return outcome;
    }

    // Read after the drain, not before it: an operation that completed while this call was
    // waiting established completion and may have recorded the error this call has to report.
    NativeRunPhase phase = NativeRunPhase::PREPARED;
    int persisted_rc = 0;
    bool persisted = false;
    try {
        std::lock_guard<std::mutex> lk(native_run_mu_);
        const auto it = native_runs_.find(run_epoch);
        if (it == native_runs_.end()) return outcome;
        phase = it->second->phase;
        persisted_rc = it->second->pending_rc;
        persisted = it->second->pending_rc_set;
    } catch (...) {
        outcome.captured = std::current_exception();
        clear_finalize_reservation_noexcept(run_epoch);
        return outcome;
    }
    // Only a run still launched has a device wait outstanding; one already reaped had its wait,
    // and its error is the persisted one rather than anything this call would produce.
    const bool needs_wait = phase == NativeRunPhase::LAUNCHED && wait_run_fn_ != nullptr;

    bool entered_finalize = false;
    int finalize_rc = 0;

    // This call's own wait records its outcome where every other operation records one, and before
    // the teardown below can run. A later call on a retained handle then reads completion and this
    // error instead of waiting again and replacing it.
    if (needs_wait) {
        int wait_rc = 0;
        try {
            wait_rc = wait_run_fn_(device_ctx_, storage);
        } catch (...) {
            outcome.captured = std::current_exception();
        }
        if (outcome.captured == nullptr) {
            note_native_run_completion(run_epoch, wait_rc);
            if (wait_rc != 0) {
                persisted_rc = wait_rc;
                persisted = true;
            }
        }
    }

    if (outcome.captured == nullptr) {
        entered_finalize = true;
        try {
            finalize_rc = finalize_run_fn_(device_ctx_, storage);
        } catch (...) {
            outcome.captured = std::current_exception();
        }
    }

    // The tag is only meaningful for a call that returned. A throw out of the teardown says
    // nothing about how far it got — it may already have released resources while the tag still
    // reads as a live context — so that case is isolated rather than offered for a retry.
    if (entered_finalize && outcome.captured == nullptr) {
        outcome.no_submission_released = consume_released_preparation(storage, storage_bytes, run_epoch);
    }
    const RunStorageTag tag = entered_finalize ? read_run_storage_tag(storage) : RunStorageTag::Canonical;
    const bool threw = outcome.captured != nullptr;
    try {
        std::lock_guard<std::mutex> lk(native_run_mu_);
        const auto it = native_runs_.find(run_epoch);
        if (it != native_runs_.end()) {
            NativeRunEntry &entry = *it->second;
            if (tag == RunStorageTag::Destroyed && !threw) {
                native_runs_.erase(it);
            } else if (!entered_finalize && !teardown) {
                // The wait threw: the teardown was never reached, so the run is certainly intact
                // and may be finalized again. Its phase is left as it stands rather than restored
                // from the pre-wait capture — anything recorded since is an established fact.
                entry.finalize_reserved = false;
            } else if (tag == RunStorageTag::Canonical && !threw && finalize_rc != 0 && !teardown) {
                // The context refusal: the run is intact and this caller may try again on a valid
                // context. The phase keeps what the wait recorded, so that retry reads completion
                // and the persisted error instead of waiting a second time.
                entry.finalize_reserved = false;
            } else {
                // Everything else leaves teardown progress unknown, and teardown mode treats every
                // non-destroyed outcome this way so a refused run cannot be attempted forever.
                // Nothing further reaches this run and nothing it holds is recycled.
                entry.destruction_unknown = true;
                entry.finalize_reserved = false;
                if (!threw && finalize_rc == 0 && !persisted) {
                    outcome.fault = tag == RunStorageTag::Canonical ? FinalizeFault::NonDestroying :
                                                                      FinalizeFault::UnknownDestruction;
                }
            }
        }
        native_run_ops_done_.notify_all();
    } catch (...) {
        // The outcome above is what the runtime established and is reported unchanged; only this
        // call's own bookkeeping failed, and a reservation left behind isolates the run rather
        // than offering it to a second teardown.
        if (!threw) outcome.captured = std::current_exception();
        return outcome;
    }

    if (threw || outcome.fault != FinalizeFault::None) return outcome;
    const int rc = finalize_rc != 0 ? finalize_rc : (persisted ? persisted_rc : 0);
    if (rc != 0) {
        outcome.fault = FinalizeFault::Failed;
        outcome.rc = rc;
    }
    return outcome;
}

RunRetentionProbeReport ChipWorker::probe_run_retention(
    const ChipWorkerNativeRun &run, const ChipWorkerNativeRun &successor, const RunRetentionProbeConfig &config
) {
    if (probe_run_retention_fn_ == nullptr) {
        throw std::runtime_error(
            "this runtime module exports no run-retention fixture; it measures a stream-level property and only "
            "onboard modules carry it"
        );
    }
    const bool want_successor = config.launch_successor != 0;
    if (run.slot_id >= runtime_bufs_.size()) {
        throw std::runtime_error("native-run token slot is outside the runtime PipelineContract");
    }
    if (want_successor && (successor.slot_id >= runtime_bufs_.size() || successor.slot_id == run.slot_id)) {
        throw std::runtime_error("probe_run_retention needs the successor on a different pipeline slot");
    }

    // Leased like any other operation, and both runs when the fixture drives both: the call
    // dereferences each carrier it is handed, so neither may be finalized underneath it. The lease
    // also checks each token's generation, which this path did not.
    void *storage = nullptr;
    void *successor_storage = nullptr;
    NativeRunOpLease held;
    NativeRunOpLease successor_held;
    const NativeRunLease lease = want_successor ?
                                     acquire_native_run_pair(
                                         successor, run, NativeRunOp::ProbeSuccessor, &successor_storage, &storage,
                                         &successor_held, &held, LeaseWait::Block
                                     ) :
                                     acquire_native_run_op(run, NativeRunOp::Probe, &storage, nullptr, &held);
    if (lease != NativeRunLease::Acquired) {
        throw std::runtime_error(
            "probe_run_retention needs a launched, undrained predecessor and a prepared successor " +
            format_native_run_identity(run)
        );
    }

    RunRetentionProbeReport report{};
    const int rc =
        probe_run_retention_fn_(device_ctx_, storage, want_successor ? successor_storage : nullptr, &config, &report);
    if (rc != 0) {
        throw std::runtime_error(
            "probe_run_retention refused its arguments with code " + std::to_string(rc) + " " +
            format_native_run_identity(run)
        );
    }
    if (want_successor) {
        // The fixture drained the successor, so it is where an ordinary wait
        // would have left it. The predecessor stays LAUNCHED on purpose: its
        // drain, copy-back and DFX teardown are the production path, and
        // finalize reaches all three from that phase.
        note_native_run_completion(
            successor.run_epoch,
            report.successor_launch_rc != 0 ? report.successor_launch_rc : report.successor_drain_rc
        );
    }
    return report;
}

void ChipWorker::cleanup_native_runs_noexcept() noexcept {
    if (device_ctx_ == nullptr || finalize_run_fn_ == nullptr) {
        return;
    }
    // One eligible run at a time, re-found under the lock each round: no container is built, so
    // this path allocates nothing, and the finalization below is the same protocol a caller's
    // finalize runs. Progress is guaranteed because teardown mode leaves every run it touches
    // either gone or isolated, so no run can be selected twice.
    for (;;) {
        uint64_t run_epoch = 0;
        bool found = false;
        {
            std::lock_guard<std::mutex> lk(native_run_mu_);
            for (const auto &candidate : native_runs_) {
                const NativeRunEntry &entry = *candidate.second;
                if (entry.destruction_unknown || entry.finalize_reserved) continue;
                if (!native_run_op_admits_phase(NativeRunOp::Finalize, entry.phase)) continue;
                run_epoch = candidate.first;
                found = true;
                break;
            }
        }
        if (!found) break;
        // The outcome is dropped: there is no caller left to report it to, and it is already
        // recorded against the run it belongs to.
        (void)finalize_native_run_epoch(run_epoch, nullptr, FinalizeMode::Teardown);
    }
}

size_t ChipWorker::abandon_unproven_native_run_storage() noexcept {
    std::lock_guard<std::mutex> lk(native_run_mu_);
    size_t abandoned = 0;
    for (const auto &entry : native_runs_) {
        const uint32_t slot = entry.second->slot_id;
        if (slot >= runtime_bufs_.size()) continue;
        if (runtime_bufs_[slot].data() == nullptr) continue;  // already abandoned by an earlier close
        runtime_bufs_[slot].abandon();
        ++abandoned;
        std::fprintf(
            stderr,
            "ChipWorker: abandoning the run storage of set %u; its carrier could not be proved destroyed, so the "
            "block is not returned to the allocator\n",
            slot
        );
    }
    return abandoned;
}

size_t ChipWorker::unproven_native_run_count() const noexcept {
    // Every run whose destruction was proved is erased from this map by the finalization that
    // proved it, so whatever is left after a teardown sweep is exactly what could not be closed —
    // isolated, reserved by a finalization still running, or never past preparation.
    std::lock_guard<std::mutex> lk(native_run_mu_);
    return native_runs_.size();
}

void ChipWorker::unregister_callable(int32_t callable_id) {
    if (!initialized_) {
        throw std::runtime_error("ChipWorker not initialized; call init() first");
    }
    validate_execution_context();
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

void ChipWorker::flush_diagnostics(int timeout_ms) {
    if (!initialized_) {
        throw std::runtime_error("ChipWorker not initialized; call init() first");
    }
    validate_execution_context();
    if (flush_diagnostics_fn_ == nullptr) return;  // this module retains no run: nothing is deferred
    char error[512] = {};
    const int rc = flush_diagnostics_fn_(device_ctx_, timeout_ms, error, sizeof(error));
    if (rc != 0) {
        throw std::runtime_error(std::string("ChipWorker::flush_diagnostics failed: ") + error);
    }
}

size_t ChipWorker::committed_device_memory() const {
    if (!initialized_) {
        return 0;
    }
    return device_committed_memory_fn_(device_ctx_);
}

DeviceMemoryInfo ChipWorker::device_memory_info() const {
    if (!initialized_) {
        throw std::runtime_error("ChipWorker not initialized; call init() first");
    }
    validate_execution_context();
    DeviceMemoryInfo info{};
    int rc = device_memory_info_fn_(device_ctx_, &info);
    if (rc == PTO_RUNTIME_ERR_UNSUPPORTED) {
        throw UnsupportedRuntimeOperation("device_memory_info is not supported by this runtime");
    }
    if (rc != 0) {
        throw std::runtime_error("device_memory_info failed with code " + std::to_string(rc));
    }
    return info;
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
    validate_execution_context();
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
    validate_execution_context();
    // Through the guarded entry when the runtime has one: a caller allocation a run may still be
    // reading or writing is not releasable yet, and the refusal has to reach the caller rather than
    // be swallowed by the void-returning free. The check and the release are one step inside the
    // device context, so a borrow taken in between cannot be missed.
    if (device_free_caller_buffer_ctx_fn_ != nullptr) {
        int rc = device_free_caller_buffer_ctx_fn_(device_ctx_, reinterpret_cast<void *>(ptr));
        if (rc == PTO_RUNTIME_ERR_INVALID_STATE) {
            throw std::runtime_error(
                "free: device address " + format_device_address(ptr) +
                " is still in use by a submitted run and cannot be released yet"
            );
        }
        if (rc != 0) {
            // Any other code is a failure of the release itself, not a refusal, and must not be
            // reported as one: the address may be gone, or the context may be unusable.
            throw std::runtime_error(
                "free: releasing device address " + format_device_address(ptr) + " failed with code " +
                std::to_string(rc)
            );
        }
        return;
    }
    device_free_ctx_fn_(device_ctx_, reinterpret_cast<void *>(ptr));
}

bool ChipWorker::borrow_caller_device_spans(uint64_t borrow_id, const CallerDeviceSpan *spans, size_t count) {
    if (!initialized_ || device_borrow_caller_buffers_ctx_fn_ == nullptr) return false;
    std::vector<CallerBufferSpan> wire(count);
    for (size_t i = 0; i < count; ++i) {
        wire[i] = CallerBufferSpan{spans[i].addr, spans[i].bytes};
    }
    return device_borrow_caller_buffers_ctx_fn_(
               device_ctx_, wire.data(), static_cast<uint32_t>(wire.size()), borrow_id
           ) == 0;
}

void ChipWorker::release_caller_device_borrow(uint64_t borrow_id, bool keep) noexcept {
    if (device_release_caller_buffers_ctx_fn_ == nullptr) return;
    device_release_caller_buffers_ctx_fn_(device_ctx_, borrow_id, keep ? 1 : 0);
}

void ChipWorker::copy_to(uint64_t dst, uint64_t src, size_t size) {
    if (!initialized_) {
        throw std::runtime_error("ChipWorker not initialized; call init() first");
    }
    validate_execution_context();
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
    validate_execution_context();
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

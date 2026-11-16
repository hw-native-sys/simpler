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
#include "device_runner.h"
#include "common/dma_workspace.h"
#include "cpu_sim_context.h"
#include "host/raii_scope_guard.h"
#include "host/sim_thread_start_gate.h"
#include "host_log.h"
#include <chrono>
#include <cstdio>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <thread>

struct DeviceRunner::ActiveRun {
    SimThreadStartGate startup;
    WseRun *state{nullptr};
    uint64_t epoch{0};
    uint64_t expected{0};
    std::chrono::steady_clock::time_point started;
    std::vector<std::thread> threads;
    void join() noexcept {
        for (auto &thread : threads)
            if (thread.joinable()) thread.join();
    }
};

DeviceRunner::DeviceRunner() = default;
DeviceRunner::~DeviceRunner() { finalize(); }
void *DeviceRunner::allocate_tensor(size_t bytes) { return host_graph_storage_.alloc(bytes); }
void *DeviceRunner::allocate_caller_storage(size_t bytes) { return sram_.claim(bytes); }
void *DeviceRunner::allocate_graph_storage(size_t bytes) { return host_graph_storage_.alloc(bytes); }
void DeviceRunner::free_graph_storage(void *ptr) { host_graph_storage_.free(ptr); }
size_t DeviceRunner::committed_device_memory() const { return sram_.committed_bytes(); }
void *DeviceRunner::acquire_child_memory_host_view(void *ptr, size_t bytes) {
    sram_.check_application(reinterpret_cast<uint64_t>(ptr), bytes);
    return ptr;
}
int DeviceRunner::device_memset(void *ptr, int value, size_t bytes) {
    sram_.check_application(reinterpret_cast<uint64_t>(ptr), bytes);
    return WseHostRunner::device_memset(ptr, value, bytes);
}
void DeviceRunner::free_tensor(void *ptr) {
    if (sram_.owns(ptr)) sram_.release(ptr);
    else host_graph_storage_.free(ptr);
}
int DeviceRunner::copy_to_device(void *ptr, const void *host, size_t bytes) {
    sram_.check_application(reinterpret_cast<uint64_t>(ptr), bytes);
    return WseHostRunner::copy_to_device(ptr, host, bytes);
}
int DeviceRunner::copy_from_device(void *host, const void *ptr, size_t bytes) {
    sram_.check_application(reinterpret_cast<uint64_t>(ptr), bytes);
    return WseHostRunner::copy_from_device(host, ptr, bytes);
}

int DeviceRunner::prepare_launch_shape(Runtime &runtime, const CallConfig &config) {
    const uint32_t cores = sram_.core_count();
    if (cores < 1) {
        LOG_ERROR("WSE requires at least one configured Core (core_count=%u)", cores);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (config.aicpu_thread_num != 0) {
        LOG_ERROR("WSE requires aicpu_thread_num=0 (got %d)", config.aicpu_thread_num);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (config.enable_chip_swimlane || config.enable_dump_args || config.enable_pmu || config.enable_scope_stats) {
        LOG_ERROR("WSE exposes PE events through output_prefix; legacy device diagnostic collectors are unsupported");
        return PTO_RUNTIME_ERR_UNSUPPORTED;
    }
    sram_.freeze();
    block_dim_ = cores;
    worker_count_ = cores * WSE_PE_PER_CORE;
    runtime.set_worker_count(worker_count_);
    runtime.set_aicpu_thread_num(0);
    // The common host callable interface retains a type vector; it does not route WSE execution.
    runtime.set_core_type_rule(worker_count_, worker_count_);
    runtime.dev.chip_swimlane_level = config.enable_chip_swimlane;
    runtime.dev.dump_args_level = config.enable_dump_args;
    return 0;
}

int DeviceRunner::ensure_binaries_loaded() {
    if (aicore_so_handle_) return 0;
    if (!aicpu_so_binary_.empty() || aicore_kernel_binary_.empty()) return PTO_RUNTIME_ERR_INTERNAL;
    if (!simpler::common::sim_host::create_temp_so_file(
            "/tmp/wse_pe_XXXXXX", aicore_kernel_binary_.data(), aicore_kernel_binary_.size(), &aicore_so_path_
        ))
        return PTO_RUNTIME_ERR_INTERNAL;
    aicore_so_handle_ = dlopen(aicore_so_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!aicore_so_handle_) {
        LOG_ERROR("WSE PE runtime load failed: %s", dlerror());
        unload_executor();
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    create_run_ = reinterpret_cast<decltype(create_run_)>(dlsym(aicore_so_handle_, "wse_create_run"));
    destroy_run_ = reinterpret_cast<decltype(destroy_run_)>(dlsym(aicore_so_handle_, "wse_destroy_run"));
    execute_pe_ = reinterpret_cast<decltype(execute_pe_)>(dlsym(aicore_so_handle_, "wse_execute_pe"));
    run_events_ = reinterpret_cast<decltype(run_events_)>(dlsym(aicore_so_handle_, "wse_run_events"));
    completed_tasks_ = reinterpret_cast<decltype(completed_tasks_)>(dlsym(aicore_so_handle_, "wse_completed_tasks"));
    runtime_used_ = reinterpret_cast<decltype(runtime_used_)>(dlsym(aicore_so_handle_, "wse_runtime_used"));
    memory_sections_ = reinterpret_cast<decltype(memory_sections_)>(dlsym(aicore_so_handle_, "wse_memory_sections"));
    control_address_ = reinterpret_cast<decltype(control_address_)>(dlsym(aicore_so_handle_, "wse_control_address"));
    if (!create_run_ || !destroy_run_ || !execute_pe_ || !run_events_ || !completed_tasks_ || !runtime_used_ ||
        !memory_sections_ || !control_address_) {
        LOG_ERROR("WSE PE runtime is missing a required entry point");
        unload_executor();
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    return 0;
}

int DeviceRunner::prepare_execution(
    Runtime &runtime, const CallConfig &config, uint32_t slot, const NativeRunIdentity &identity,
    std::unique_ptr<PreparedExecution> *out
) {
    if (!out || *out || active_run_) return PTO_RUNTIME_ERR_INTERNAL;
    if (int rc = ensure_device_initialized(); rc != 0) return rc;
    if (int rc = ensure_binaries_loaded(); rc != 0) return rc;
    if (int rc = validate_active_callable(runtime); rc != 0) return rc;
    apply_call_config(config);
    auto prepared = std::make_unique<PreparedExecution>(identity, runtime, config, slot);
    prepared->num_aicore = runtime.get_worker_count();
    prepared->launch_aicpu_num = 0;
    auto *descriptor =
        reinterpret_cast<const WseGraphDescriptor *>(runtime.dev.scheduler_bootstrap.worker_context_base);
    auto run = std::make_unique<ActiveRun>();
    run->threads.reserve(prepared->num_aicore);
    const auto regions = sram_.regions();
    const auto allocations = sram_.allocations();
    char error[512]{};
    run->state = create_run_(
        descriptor, prepared->num_aicore / WSE_PE_PER_CORE, identity.run_epoch, regions.data(), regions.size(),
        allocations.data(), allocations.size(), error, sizeof(error)
    );
    if (!run->state) {
        if (config.output_prefix[0]) {
            std::filesystem::create_directories(config.output_prefix);
            std::ofstream(std::filesystem::path(config.output_prefix) / "wse_prepare_error.txt") << error << "\n";
        }
        LOG_ERROR("WSE SRAM run admission rejected: %s", error);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    std::vector<uint64_t> used;
    for (uint32_t core = 0; core < regions.size(); ++core)
        used.push_back(runtime_used_(run->state, core));
    sram_.set_runtime_used(used);
    run->epoch = identity.run_epoch;
    run->expected = descriptor->expected_tasks;
    active_run_ = std::move(run);
    completion_.reset(prepared->num_aicore);
    *out = std::move(prepared);
    return 0;
}

WseLaunchTransactionResult DeviceRunner::launch_execution_impl(PreparedExecution &prepared, WseLaunchPermit permit) {
    auto *run = active_run_.get();
    auto rollback = RAIIScopeGuard([&] {
        run->startup.cancel();
        run->join();
        uint64_t device_events = 0;
        for (uint32_t worker = 0; worker < run->threads.size(); ++worker) {
            uint64_t count = 0;
            (void)run_events_(run->state, worker, &count);
            device_events += count;
        }
        LOG_INFO(
            "WSE startup rollback: joined %zu PE threads, device events=%llu", run->threads.size(),
            static_cast<unsigned long long>(device_events)
        );
        completion_.abandon(PTO_RUNTIME_ERR_INTERNAL);
    });
    auto result = wse_launch_transaction(prepared.identity, std::move(permit), [&](LaunchProgressSink &) -> int {
        try {
            for (int worker = 0; worker < prepared.num_aicore; ++worker) {
                run->threads.push_back(create_thread(
                    [this, run, worker] {
                        if (!run->startup.wait()) return;
                        completion_.task_finished(execute_pe_(run->state, worker));
                    },
                    "wse-" + std::to_string(worker / WSE_PE_PER_CORE) + "-" + std::to_string(worker % WSE_PE_PER_CORE)
                ));
            }
        } catch (...) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        return 0;
    });
    if (result.rc == 0) {
        run->started = std::chrono::steady_clock::now();
        run->startup.release();
        rollback.dismiss();
    }
    return result;
}

int DeviceRunner::poll_execution(const ActiveExecution &) { return completion_.poll(); }

DrainOutcome DeviceRunner::drain_execution(ActiveExecution &active) {
    if (!active_run_ || !active.prepared) return DrainOutcome::device_error(PTO_RUNTIME_ERR_INTERNAL);
    auto cleanup = RAIIScopeGuard([&] {
        cleanup_run();
    });
    active_run_->join();
    device_wall_ns_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - active_run_->started)
            .count();
    if (const int rc = completion_.first_error(); rc != 0) return DrainOutcome::device_error(rc, true);
    if (completed_tasks_(active_run_->state) != active_run_->expected)
        return DrainOutcome::device_error(PTO_RUNTIME_ERR_INTERNAL, true);
    const auto &config = active.prepared->config;
    if (config.output_prefix[0]) {
        std::filesystem::create_directories(config.output_prefix);
        std::ofstream trace(std::filesystem::path(config.output_prefix) / "wse_events.jsonl");
        trace << "{\"epoch\":" << active_run_->epoch << ",\"expected_tasks\":" << active_run_->expected
              << ",\"core_count\":" << active.prepared->num_aicore / WSE_PE_PER_CORE << "}\n";
        for (uint32_t worker = 0; worker < active_run_->threads.size(); ++worker) {
            uint64_t count = 0;
            const auto *events = run_events_(active_run_->state, worker, &count);
            for (uint64_t i = 0; i < count; ++i) {
                const auto &e = events[i];
                trace << "{\"seq\":" << e.sequence << ",\"kind\":" << static_cast<uint32_t>(e.kind)
                      << ",\"core\":" << e.core << ",\"pe\":" << e.pe << ",\"task\":" << e.task
                      << ",\"block\":" << e.block << ",\"value\":" << e.value << ",\"slot\":" << e.slot
                      << ",\"generation\":" << e.generation << "}\n";
            }
        }
        if (!trace) return DrainOutcome::device_complete(PTO_RUNTIME_ERR_INTERNAL);
        std::ofstream memory(std::filesystem::path(config.output_prefix) / "wse_sram.json");
        memory << "{\"configuration\":[";
        const auto effective = sram_.configuration();
        for (size_t i = 0; i < effective.size(); ++i)
            memory << (i ? "," : "") << effective[i];
        memory << "],\"cores\":[";
        for (uint32_t core = 0; core < sram_.core_count(); ++core) {
            const auto info = sram_.info(core);
            memory << (core ? "," : "") << "{\"owner\":" << core << ",\"base\":" << info[0]
                   << ",\"capacity\":" << info[1] << ",\"reserve\":" << info[2] << ",\"application_used\":" << info[3]
                   << ",\"runtime_base\":" << sram_.regions()[core].runtime_base << ",\"runtime_used\":" << info[6]
                   << ",\"control_address\":" << control_address_(active_run_->state, core) << "}";
        }
        memory << "],\"sections\":[";
        uint64_t section_count = 0;
        const auto *sections = memory_sections_(active_run_->state, &section_count);
        for (uint64_t i = 0; i < section_count; ++i) {
            const auto &section = sections[i];
            memory << (i ? "," : "") << "{\"owner\":" << section.owner << ",\"kind\":" << section.kind
                   << ",\"address\":" << section.address << ",\"bytes\":" << section.bytes << "}";
        }
        memory << "]}";
        if (!memory) return DrainOutcome::device_complete(PTO_RUNTIME_ERR_INTERNAL);
    }
    return DrainOutcome::device_complete(0);
}

void DeviceRunner::cleanup_run() noexcept {
    if (!active_run_) return;
    active_run_->join();
    destroy_run_(active_run_->state);
    active_run_.reset();
    sram_.reset_runtime_used();
}
void DeviceRunner::abandon_prepared_execution(PreparedExecution &) noexcept {
    completion_.abandon();
    cleanup_run();
}
void DeviceRunner::unload_executor() {
    if (aicore_so_handle_) dlclose(aicore_so_handle_);
    aicore_so_handle_ = nullptr;
    create_run_ = nullptr;
    destroy_run_ = nullptr;
    execute_pe_ = nullptr;
    run_events_ = nullptr;
    completed_tasks_ = nullptr;
    runtime_used_ = nullptr;
    memory_sections_ = nullptr;
    control_address_ = nullptr;
    if (!aicore_so_path_.empty()) std::remove(aicore_so_path_.c_str());
    aicore_so_path_.clear();
}
int DeviceRunner::finalize() {
    cleanup_run();
    if (device_id_ == -1 && aicore_so_handle_ == nullptr) return 0;
    release_callable_state();

    unload_executor();
    release_graph_definition_staging();
    // Host allocations here, so a failure has no device meaning; this
    // finalize reports none of its releases, as its siblings above do not.
    (void)release_scheduler_state_storage();
    release_sm_mirrors();
    release_run_image_stagings();

    for (auto &bank : arena_banks_) {
        bank->gm_heap.release();
        bank->gm_sm.release();
        bank->runtime_pool.release();
    }
    clear_temporary_buffer();
    for (auto &bank : arena_banks_) {
        bank->cached_gm_heap_size = 0;
        bank->cached_gm_sm_size = 0;
        bank->cached_runtime_arena_size = 0;
    }

    if (dma_workspace_handle_ != nullptr) {
        dma_workspace_release(dma_workspace_handle_);
        dma_workspace_handle_ = nullptr;
    }
    for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind) {
        dma_workspace_addr_[kind] = 0;
    }

    sram_.clear();
    host_graph_storage_.finalize();
    clear_cpu_sim_shared_storage();

    device_id_ = -1;
    worker_count_ = 0;
    return 0;
}

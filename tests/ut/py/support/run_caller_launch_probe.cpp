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

#include <acl/acl.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <map>
#include <tuple>
#include <utility>
#include <vector>
#include <unordered_map>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#include "native_run_context.h"
#include "task_args.h"
#include "common/support/native_run_execution_peer.h"
#include "device_runner.h"
#include "shared_memory.h"
#include "common/host_span.h"
#include "common/device_phase.h"
#include "common/host_log_state.h"
#include "common/log_level.h"
#include "host_log.h"

static int tokens[64], next_stream, next_event;
static int fail_at, submits, syncs, copies;
static int allocation_attempts, fail_allocation;
static std::unordered_map<void *, size_t> live_allocations;
static bool capture;
static int metadata_extensions, sync_error, device_drains;
static int aicore_sync_error;
static int event_query_error, event_queries;
static void *event_query_error_event;
static void *sync_error_stream;
static bool observation_probe;
static uint32_t observation_mode;
static void *observation_record, *observation_output_host, *observation_core_stream;
static int private_copy_error, output_copy_error, private_reads, output_reads;
static std::vector<std::pair<void *, size_t>> observation_outputs;
static std::vector<std::string> observation_steps;
static std::vector<std::string> runtime_diagnostics;
static SimplerHostLogState runtime_log_state{};

static int capture_runtime_record(void *, SimplerHostLogState *state, const char *record, uint32_t size) {
    runtime_diagnostics.emplace_back(record, size);
    if (state->log_directory_bound == 1) {
        std::ofstream file(
            std::filesystem::path(state->log_directory) / ("host." + std::to_string(getpid()) + ".log"), std::ios::app
        );
        file.write(record, size);
        file.close();
        assert(file);
    }
    // The synchronous sink completes the producer claim without queueing work.
    __atomic_fetch_sub(&state->sink_producer_state, UINT64_C(1), __ATOMIC_RELEASE);
    return 1;
}

static void bind_runtime_log(const std::string &directory = {}) {
    pthread_key_t trace_key;
    assert(pthread_key_create(&trace_key, std::free) == 0);
    runtime_log_state.threshold = static_cast<int>(simpler::log::LogLevel::TIMING);
    runtime_log_state.sink_owner_pid = getpid();
    runtime_log_state.sink_process_pid = getpid();
    runtime_log_state.sink_context = &runtime_diagnostics;
    runtime_log_state.sink_enqueue = capture_runtime_record;
    runtime_log_state.trace_key_plus_one = static_cast<uint64_t>(trace_key) + 1;
    if (!directory.empty()) {
        assert(directory.size() < sizeof(runtime_log_state.log_directory));
        std::memcpy(runtime_log_state.log_directory, directory.c_str(), directory.size() + 1);
        runtime_log_state.log_directory_bound = 1;
    }
    assert(HostLogger::get_instance().bind_state(&runtime_log_state) == 0);
    void *runtime_dso = dlopen("libhost_runtime.so", RTLD_NOW | RTLD_NOLOAD);
    assert(runtime_dso != nullptr);
    void *runtime_api = dlsym(runtime_dso, "get_runtime_size");
    void *runtime_bind = dlsym(runtime_dso, "simpler_host_log_bind_state");
    Dl_info api_owner{}, bind_owner{};
    assert(runtime_api && runtime_bind && dladdr(runtime_api, &api_owner) && dladdr(runtime_bind, &bind_owner));
    assert(api_owner.dli_fbase == bind_owner.dli_fbase);
    using BindState = int (*)(SimplerHostLogState *);
    assert(reinterpret_cast<BindState>(runtime_bind)(&runtime_log_state) == 0);
    assert(dlclose(runtime_dso) == 0);
}

static void check_claim_release(uint64_t invocation, uint64_t callable_hash, bool print_record = false) {
    size_t count = 0;
    std::ostringstream expected_hash;
    expected_hash << "hid=" << std::hex << callable_hash << ' ';
    for (const auto &record : runtime_diagnostics) {
        if (record.find("name=chip.run.claim_release ") == std::string::npos) continue;
        ++count;
        assert(record.find("inv=" + std::to_string(invocation) + " ") != std::string::npos);
        assert(record.find(expected_hash.str()) != std::string::npos);
        assert(record.find("depth=1 ") != std::string::npos);
        if (print_record) std::fwrite(record.data(), 1, record.size(), stderr);
    }
    assert(count == 1);
    assert(runtime_log_state.sink_producer_state == 0 && runtime_log_state.pending_record_count == 0);
    assert(runtime_log_state.dropped_record_count == 0);
}

using ProofFields = std::map<std::string, std::string>;

static ProofFields proof_fields(const std::string &line) {
    const auto start = line.find("[STRACE] ");
    if (start == std::string::npos) return {};
    std::istringstream input(line.substr(start + 9));
    ProofFields fields;
    std::string word;
    while (input >> word) {
        const auto equal = word.find('=');
        if (equal != std::string::npos && !fields.emplace(word.substr(0, equal), word.substr(equal + 1)).second)
            return {};
    }
    return fields;
}

static bool complete_proof(
    const std::vector<std::string> &records, const NativeRunIdentity &identity, uint64_t invocation, uint64_t hash,
    void *aicore, void *aicpu
) {
    if (runtime_log_state.dropped_record_count != 0 || runtime_log_state.pending_record_count != 0) return false;
    std::map<std::string, ProofFields> bindings;
    std::map<std::string, ProofFields> raw_by_thread;
    unsigned completed = 0;
    std::ostringstream expected_hash;
    expected_hash << std::hex << hash;
    const auto same_identity = [&](const ProofFields &row) {
        return row.at("ep") == std::to_string(identity.run_epoch) &&
               row.at("gen") == std::to_string(identity.generation) &&
               row.at("disp") == std::to_string(identity.dispatch_id) &&
               row.at("slot") == std::to_string(identity.pipeline_slot);
    };
    try {
        for (const auto &line : records) {
            auto row = proof_fields(line);
            const auto name = row.find("name");
            if (name == row.end() || name->second.find("chip.run.proof.") != 0) continue;
            if (name->second == "chip.run.proof.invalid" || std::stoul(row.at("depth")) == 0) return false;
            const std::string thread = row.at("pid") + ":" + row.at("tid");
            if (name->second == "chip.run.proof.record") {
                std::ostringstream stream;
                stream << (row.at("role") == "0" ? aicore : aicpu);
                if (same_identity(row) && (row.at("inv") != std::to_string(invocation) ||
                                           row.at("hid") != expected_hash.str() || row.at("s") != stream.str()))
                    return false;
                if (same_identity(row) && bindings.count(row.at("e"))) completed &= ~(row.at("role") == "0" ? 1U : 2U);
                bindings[row.at("e")] = row;
                raw_by_thread.erase(thread);
            } else if (name->second == "chip.run.proof.event_query" || name->second == "chip.run.proof.event_wait") {
                if (raw_by_thread.count(thread) && raw_by_thread.at(thread).at("e") == row.at("e")) return false;
                raw_by_thread[thread] = row;
            } else if (name->second == "chip.run.proof.consume") {
                const auto raw = raw_by_thread.find(thread);
                const auto binding = bindings.find(row.at("e"));
                if (raw == raw_by_thread.end() || binding == bindings.end()) return false;
                const auto &sdk = raw->second;
                const auto &bound = binding->second;
                if (sdk.at("e") != row.at("e") || sdk.at("rc") != row.at("rc") || sdk.at("fresh") != "1" ||
                    row.at("ep") != bound.at("ep") || row.at("gen") != bound.at("gen") ||
                    row.at("disp") != bound.at("disp") || row.at("slot") != bound.at("slot") ||
                    row.at("role") != bound.at("role") || row.at("f") != bound.at("f") ||
                    row.at("pid") != bound.at("pid"))
                    return false;
                const bool query = row.at("op") == "query";
                if (!query && row.at("op") != "wait") return false;
                if (sdk.at("name") != (query ? "chip.run.proof.event_query" : "chip.run.proof.event_wait"))
                    return false;
                if (same_identity(row) && row.at("rc") == "0" && row.at("c") == "1" && row.at("owner") == "1") {
                    if (query && (sdk.at("status_valid") != "1" ||
                                  sdk.at("status") != std::to_string(ACL_EVENT_RECORDED_STATUS_COMPLETE)))
                        return false;
                    if (!query && sdk.at("status_valid") != "0") return false;
                    const unsigned bit = row.at("role") == "0" ? 1 : row.at("role") == "1" ? 2 : 0;
                    const unsigned cache = std::stoul(row.at("cache"));
                    if (cache > 3 || (cache & bit) || (cache & ~completed)) return false;
                    if (bit == 0 || (completed & bit)) return false;
                    completed |= bit;
                } else if (row.at("rc") != "0" && sdk.at("status_valid") != "0") return false;
                raw_by_thread.erase(raw);
            }
        }
    } catch (const std::exception &) {
        return false;
    }
    return completed == 3;
}

static void check_completion_proof_records(
    const NativeRunIdentity &identity, uint64_t invocation, uint64_t hash, void *aicore, void *aicpu, bool waited
) {
    const bool complete = complete_proof(runtime_diagnostics, identity, invocation, hash, aicore, aicpu);
    if (!complete) {
        constexpr size_t limit = 64;
        std::fprintf(
            stderr,
            "COMPLETION_PROOF_READER_FAILURE ep=%llu gen=%llu disp=%llu slot=%u inv=%llu hid=%llx "
            "aicore=%p aicpu=%p waited=%d host_strace=%d sdk_complete=%d sdk_pending=%d "
            "threshold=%d producer=%llu pending=%llu dropped=%llu records=%zu truncated=%d\n",
            static_cast<unsigned long long>(identity.run_epoch), static_cast<unsigned long long>(identity.generation),
            static_cast<unsigned long long>(identity.dispatch_id), identity.pipeline_slot,
            static_cast<unsigned long long>(invocation), static_cast<unsigned long long>(hash), aicore, aicpu,
            waited ? 1 : 0, SIMPLER_HOST_STRACE, static_cast<int>(ACL_EVENT_RECORDED_STATUS_COMPLETE),
            static_cast<int>(ACL_EVENT_RECORDED_STATUS_NOT_READY), runtime_log_state.threshold,
            static_cast<unsigned long long>(runtime_log_state.sink_producer_state),
            static_cast<unsigned long long>(runtime_log_state.pending_record_count),
            static_cast<unsigned long long>(runtime_log_state.dropped_record_count), runtime_diagnostics.size(),
            runtime_diagnostics.size() > limit ? 1 : 0
        );
        for (size_t index = 0; index < std::min(limit, runtime_diagnostics.size()); ++index) {
            const auto &record = runtime_diagnostics[index];
            std::fprintf(stderr, "COMPLETION_PROOF_RAW index=%zu bytes=%zu ", index, record.size());
            std::fwrite(record.data(), 1, std::min(record.size(), size_t{1024}), stderr);
            if (record.size() > 1024) std::fputs(" [truncated]", stderr);
            if (record.empty() || record.back() != '\n') std::fputc('\n', stderr);
        }
        std::fflush(stderr);
    }
    assert(complete);
    // These mutations start from actual SDK/owner records, not generated JSON.
    const char *raw_marker = waited ? "chip.run.proof.event_wait" : "chip.run.proof.event_query";
    for (const char *marker : {"chip.run.proof.record", raw_marker, "chip.run.proof.consume"}) {
        auto missing = runtime_diagnostics;
        const auto found = std::find_if(missing.begin(), missing.end(), [&](const auto &line) {
            const auto row = proof_fields(line);
            if (row.find("name") == row.end() || row.at("name") != marker) return false;
            if (row.at("name") == "chip.run.proof.record") return true;
            return std::any_of(runtime_diagnostics.begin(), runtime_diagnostics.end(), [&](const auto &record) {
                const auto owner = proof_fields(record);
                return owner.count("name") && owner.at("name") == "chip.run.proof.consume" &&
                       owner.at("owner") == "1" && owner.at("e") == row.at("e");
            });
        });
        assert(found != missing.end());
        missing.erase(found);
        assert(!complete_proof(missing, identity, invocation, hash, aicore, aicpu));
    }
    for (const char *key : {"ep", "gen", "disp", "slot", "role", "f", "e", "s", "inv", "hid", "pid"}) {
        auto changed = runtime_diagnostics;
        const auto found = std::find_if(changed.begin(), changed.end(), [](const auto &line) {
            return line.find("name=chip.run.proof.record ") != std::string::npos;
        });
        assert(found != changed.end());
        const std::string field = std::string(key) + '=';
        const auto start = found->find(' ' + field);
        assert(start != std::string::npos);
        const auto end = found->find_first_of(" \n", start + 1);
        found->replace(start + 1, end - start - 1, field + "999999");
        assert(!complete_proof(changed, identity, invocation, hash, aicore, aicpu));
    }
    for (const auto &[marker, key, value] : std::vector<std::tuple<const char *, const char *, const char *>>{
             {raw_marker, "status_valid", waited ? "1" : "0"},
             {raw_marker, waited ? "name" : "status", "999999"},
             {raw_marker, "rc", "507018"},
             {raw_marker, "tid", "999999"},
             {"chip.run.proof.consume", "c", "0"},
             {"chip.run.proof.consume", "owner", "0"},
             {"chip.run.proof.consume", "cache", "3"},
         }) {
        auto changed = runtime_diagnostics;
        const auto found = std::find_if(changed.begin(), changed.end(), [&](const auto &line) {
            const auto row = proof_fields(line);
            if (!row.count("name") || row.at("name") != marker) return false;
            return std::any_of(runtime_diagnostics.begin(), runtime_diagnostics.end(), [&](const auto &record) {
                const auto owner = proof_fields(record);
                return owner.count("name") && owner.at("name") == "chip.run.proof.consume" &&
                       owner.at("owner") == "1" && owner.at("e") == row.at("e");
            });
        });
        assert(found != changed.end());
        const std::string field = std::string(key) + '=';
        const auto start = found->find(' ' + field);
        const auto end = found->find_first_of(" \n", start + 1);
        found->replace(start + 1, end - start - 1, field + value);
        assert(!complete_proof(changed, identity, invocation, hash, aicore, aicpu));
    }
    runtime_log_state.dropped_record_count = 1;
    assert(!complete_proof(runtime_diagnostics, identity, invocation, hash, aicore, aicpu));
    runtime_log_state.dropped_record_count = 0;
    runtime_log_state.pending_record_count = 1;
    assert(!complete_proof(runtime_diagnostics, identity, invocation, hash, aicore, aicpu));
    runtime_log_state.pending_record_count = 0;
}

static void check_stream_proof(const NativeRunIdentity &identity, void *aicore, void *aicpu, int rc, bool cached) {
    std::vector<ProofFields> waits;
    ProofFields result;
    for (const auto &record : runtime_diagnostics) {
        auto row = proof_fields(record);
        if (!row.count("name")) continue;
        if (row.at("name") == "chip.run.proof.stream_wait") waits.push_back(row);
        if (row.at("name") == "chip.run.proof.observe_return") result = row;
    }
    assert(!result.empty());
    assert(result.at("ep") == std::to_string(identity.run_epoch));
    assert(result.at("gen") == std::to_string(identity.generation));
    assert(result.at("disp") == std::to_string(identity.dispatch_id));
    assert(result.at("slot") == std::to_string(identity.pipeline_slot));
    assert(result.at("rc") == std::to_string(rc));
    assert(result.at("cached") == (cached ? "1" : "0"));
    assert(result.at("physical") == (cached || rc == 0 ? "1" : "0"));
    assert(result.at("polled") == (!cached && rc != 0 ? "1" : "0"));
    assert(waits.size() == (sync_error != 0 ? 1 : 2));
    for (size_t index = 0; index != waits.size(); ++index) {
        const auto &wait = waits[index];
        std::ostringstream stream;
        stream << (index == 0 ? aicpu : aicore);
        assert(wait.at("s") == stream.str());
        assert(wait.at("role") == (index == 0 ? "AICPU" : "AICore"));
        assert(wait.at("rc") == std::to_string(index == 0 ? sync_error : aicore_sync_error));
        assert(wait.at("fresh") == "1" && std::stol(wait.at("timeout_ms")) > 0);
        assert(wait.at("pid") == result.at("pid") && wait.at("tid") == result.at("tid"));
    }
    for (const auto &record : runtime_diagnostics)
        if (record.find("name=chip.run.proof.") != std::string::npos)
            std::fwrite(record.data(), 1, record.size(), stderr);
}

static void check_proof_identity_widths() {
    const size_t first_record = runtime_diagnostics.size();
    size_t event = 0;
    RunCompletionFence::DeviceEventOps ops;
    ops.create = [&](void **out) {
        *out = reinterpret_cast<void *>(UINTPTR_MAX - event++);
        return 0;
    };
    ops.record = [](void *, void *) {
        return 0;
    };
    ops.query = [](void *, bool *complete) {
        *complete = true;
        return 0;
    };
    ops.destroy = [](void *) {
        return 0;
    };
    RunCompletionFence fence(std::move(ops));
    const NativeRunIdentity identity{UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT32_MAX};
    assert(fence.arm(identity) == 0);
    for (auto role : {RunCompletionFence::StreamRole::Aicore, RunCompletionFence::StreamRole::Aicpu}) {
        fence.note_kernel_submitted(identity, role);
        assert(fence.record(identity, role, reinterpret_cast<void *>(UINTPTR_MAX)) == 0);
    }
    assert(fence.poll(identity) == RunCompletionFence::Completion::Complete);
    assert(runtime_diagnostics.size() - first_record == 4);
    for (size_t index = first_record; index < runtime_diagnostics.size(); ++index) {
        const auto row = proof_fields(runtime_diagnostics[index]);
        assert(row.at("name") != "chip.run.proof.invalid");
        assert(std::stoul(row.at("depth")) > 0);
        assert(row.at("ep") == "18446744073709551615");
        assert(row.at("gen") == "18446744073709551615");
        assert(row.at("disp") == "18446744073709551615");
        assert(row.at("slot") == "4294967295");
    }
    std::fprintf(stderr, "proof maximum_width_formatter_records=4 untruncated=1 SDK_calls=0\n");
}
static DeviceRunResultRegion observation_region;
static bool owned_close_active, completion_events_ready = true;
static int force_resets, stale_resource_calls;
static bool fail_force_reset, fail_reset_probe;
static bool acl_initialized, fail_acl_finalize_once;
static int acl_finalize_calls;
static std::vector<void *> reset_allocations, reset_streams, reset_events;
static std::vector<void *> destroyed_streams, observed_streams;
static void *timing_target;
static const void *timing_source;
static size_t timing_bytes;
static void *result_target;
static const void *result_source;
bool device_phase_capture_enabled() { return true; }
extern "C" const char *get_platform();
extern "C" bool publish_runtime_chip_swimlane_extensions(Runtime *) noexcept {
    ++metadata_extensions;
    return true;
}
static void *failed_destroy_event;
static void *caller = &tokens[60];
static std::vector<std::string> trace;
static std::vector<void *> stream_trace, event_trace;
static int step(const char *op, void *stream, void *event = nullptr) {
    trace.emplace_back(op);
    stream_trace.push_back(stream);
    event_trace.push_back(event);
    return ++submits == fail_at ? -777 : 0;
}
static bool old_device_address(const void *address) {
    const auto target = reinterpret_cast<uintptr_t>(address);
    for (void *base : reset_allocations) {
        const auto found = live_allocations.find(base);
        if (found != live_allocations.end()) {
            const auto begin = reinterpret_cast<uintptr_t>(base);
            if (target >= begin && target - begin < found->second) return true;
        }
    }
    return false;
}
static bool stale_stream(void *stream) {
    return std::find(reset_streams.begin(), reset_streams.end(), stream) != reset_streams.end();
}
static bool stale_event(void *event) {
    return std::find(reset_events.begin(), reset_events.end(), event) != reset_events.end();
}
extern "C" rtError_t rtMalloc(void **out, uint64_t bytes, uint32_t, uint16_t) {
    ++allocation_attempts;
    if (allocation_attempts == fail_allocation) {
        *out = nullptr;
        return -777;
    }
    *out = std::malloc(bytes);
    if (*out != nullptr) {
        live_allocations.emplace(*out, bytes);
        std::memset(*out, 0xa5, bytes);
    }
    return *out ? 0 : -1;
}
extern "C" rtError_t rtFree(void *ptr) {
    if (std::find(reset_allocations.begin(), reset_allocations.end(), ptr) != reset_allocations.end()) {
        ++stale_resource_calls;
        return -983;
    }
    assert(live_allocations.erase(ptr) == 1);
    std::free(ptr);
    return 0;
}
extern "C" rtError_t rtMallocHost(void **out, uint64_t bytes, uint16_t) {
    *out = std::malloc(bytes);
    return *out ? 0 : -1;
}
extern "C" rtError_t rtFreeHost(void *ptr) {
    std::free(ptr);
    return 0;
}
extern "C" rtError_t rtMemcpy(void *dst, uint64_t, const void *src, uint64_t bytes, rtMemcpyKind_t kind) {
    if (old_device_address(dst) || old_device_address(src)) {
        ++stale_resource_calls;
        return -983;
    }
    ++copies;
    if (observation_probe && kind == RT_MEMCPY_DEVICE_TO_HOST) {
        if (src == observation_record) {
            ++private_reads;
            observation_steps.emplace_back("private-read");
            if (private_copy_error != 0) return private_copy_error;
        }
        if (dst == observation_output_host) {
            ++output_reads;
            observation_steps.emplace_back("output-read");
            if (output_copy_error != 0) return output_copy_error;
        }
    }
    std::memcpy(dst, src, bytes);
    return 0;
}
extern "C" rtError_t
rtMemcpyAsync(void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t kind, rtStream_t stream) {
    const int rc = step("publish", stream);
    if (rc == 0 && dst == result_target) {
        assert(bytes == sizeof(uint64_t));
        result_source = src;
        return 0;
    }
    if (rc == 0 && dst == timing_target) {
        timing_source = src;
        timing_bytes = bytes;
        return 0;
    }
    return rc == 0 ? rtMemcpy(dst, capacity, src, bytes, kind) : rc;
}
extern "C" rtError_t rtStreamCreate(rtStream_t *out, int32_t) {
    *out = &tokens[next_stream++];
    return 0;
}
extern "C" rtError_t rtStreamDestroy(rtStream_t stream) {
    if (std::find(reset_streams.begin(), reset_streams.end(), stream) != reset_streams.end()) {
        ++stale_resource_calls;
        return -983;
    }
    assert(stream != caller);
    destroyed_streams.push_back(stream);
    return 0;
}
extern "C" aclError aclrtCreateEventExWithFlag(aclrtEvent *out, uint32_t) {
    *out = &tokens[16 + next_event++];
    return 0;
}
extern "C" aclError aclrtDestroyEvent(aclrtEvent event) {
    if (stale_event(event)) {
        ++stale_resource_calls;
        return -983;
    }
    if (event == failed_destroy_event) {
        failed_destroy_event = nullptr;
        return -778;
    }
    return 0;
}
extern "C" aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
    if (stale_event(event) || stale_stream(stream)) {
        ++stale_resource_calls;
        return -983;
    }
    return step("record", stream, event);
}
extern "C" aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
    if (stale_event(event) || stale_stream(stream)) {
        ++stale_resource_calls;
        return -983;
    }
    return step("wait", stream, event);
}
extern "C" aclError aclrtEventGetTimestamp(aclrtEvent, uint64_t *timestamp) {
    *timestamp = 1000;
    return 0;
}
extern "C" aclError aclrtStreamGetId(aclrtStream, int32_t *out) {
    *out = 1;
    return 0;
}
extern "C" aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t) {
    if (stale_stream(stream)) {
        ++stale_resource_calls;
        return -983;
    }
    observed_streams.push_back(stream);
    ++syncs;
    if (observation_probe) observation_steps.emplace_back("stream-observer");
    if (sync_error != 0 && (sync_error_stream == nullptr || sync_error_stream == stream)) return sync_error;
    return stream == observation_core_stream ? aicore_sync_error : 0;
}
extern "C" aclError aclrtSynchronizeEventWithTimeout(aclrtEvent event, int32_t) {
    if (stale_event(event)) {
        ++stale_resource_calls;
        return -983;
    }
    return sync_error;
}
extern "C" aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status) {
    ++event_queries;
    if (stale_event(event)) {
        ++stale_resource_calls;
        return -983;
    }
    if (event_query_error != 0 && event == event_query_error_event) return event_query_error;
    *status = completion_events_ready ? ACL_EVENT_RECORDED_STATUS_COMPLETE : ACL_EVENT_RECORDED_STATUS_NOT_READY;
    return 0;
}
extern "C" aclError aclrtSynchronizeDeviceWithTimeout(int32_t) {
    assert(owned_close_active);
    ++device_drains;
    return sync_error;
}
extern "C" rtError_t rtSetDevice(int32_t) {
    assert(owned_close_active);
    return 0;
}
extern "C" rtError_t rtDeviceReset(int32_t) {
    assert(owned_close_active);
    return 0;
}
extern "C" aclError aclInit(const char *) {
    assert(owned_close_active);
    if (acl_initialized) return 100002;
    acl_initialized = true;
    return 0;
}
extern "C" aclError aclFinalize() {
    assert(owned_close_active);
    ++acl_finalize_calls;
    if (fail_acl_finalize_once) {
        fail_acl_finalize_once = false;
        return -982;
    }
    acl_initialized = false;
    return 0;
}
extern "C" aclError aclrtSetDevice(int32_t) {
    assert(owned_close_active);
    return 0;
}
extern "C" aclError aclrtResetDevice(int32_t) {
    assert(owned_close_active);
    return 0;
}
extern "C" aclError aclrtResetDeviceForce(int32_t) {
    assert(owned_close_active);
    ++force_resets;
    if (fail_force_reset) return -980;
    for (const auto &allocation : live_allocations)
        reset_allocations.push_back(allocation.first);
    for (int index = 0; index != next_stream; ++index)
        reset_streams.push_back(&tokens[index]);
    for (int index = 0; index != next_event; ++index)
        reset_events.push_back(&tokens[16 + index]);
    return 0;
}
extern "C" aclError aclrtCreateStream(aclrtStream *stream) {
    if (owned_close_active && fail_reset_probe && force_resets != 0) return -981;
    return rtStreamCreate(stream, 0);
}
extern "C" aclError aclrtDestroyStream(aclrtStream stream) { return rtStreamDestroy(stream); }
extern "C" aclError aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus *status, aclmdlRI *model) {
    *status = capture ? ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE : ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    *model = nullptr;
    return 0;
}
// Kernel submission, recovery and scheduler-export leaves are test transports.
// The production runner performs publication, arming, launch/drain decisions,
// both callbacks, and every event/reference step.
int DeviceRunnerBase::launch_aicore_kernel(rtStream_t stream, const KernelArgs &) {
    if (observation_probe) observation_core_stream = stream;
    return step("core", stream);
}
int host::LoadAicpuOp::LaunchBuiltInOp(rtStream_t stream, void *, size_t, int, const std::string &) {
    const int rc = step("cpu", stream);
    if (rc == 0 && observation_probe) {
        for (const auto &[address, bytes] : observation_outputs)
            std::memset(address, 0x42, bytes);
        std::memcpy(observation_record, &observation_region, sizeof(observation_region));
    }
    return rc;
}
static void initialize_borrowed_context(DeviceRunnerBase &runner) {
    assert(runner.execution_mode_latch().latch(SIMPLER_MODE_KERNEL) == 0);
    KernelContextOps ops;
    ops.get_current_device = [](void *, int *out) noexcept {
        *out = 0;
        return 0;
    };
    ops.get_current_context = [](void *, void **out) noexcept {
        *out = &tokens[61];
        return 0;
    };
    ops.create_hidden_stream = [](void *, void **out) noexcept {
        *out = &tokens[next_stream++];
        return 0;
    };
    ops.destroy_hidden_stream = [](void *, void *) noexcept {
        return 0;
    };
    ops.create_event = [](void *, void **out) noexcept {
        *out = &tokens[16 + next_event++];
        return 0;
    };
    ops.destroy_event = [](void *, void *) noexcept {
        return 0;
    };
    assert(runner.kernel_execution_state().initialize(0, ops) == 0);
}
class ProbeRunner : public DeviceRunner {
public:
    // Partial ownership is real; only the late-health case exercises recovery.
    void recover_device_or_mark_unusable(int rc) override {
        if (check_late_health) {
            ++recovery_calls;
            recovery_rc = rc;
            // The exported backend symbol keeps recovery policy in the runtime
            // DSO without widening its private C++ interface.
            using RecoverFn = void (*)(DeviceRunner *, int);
            Dl_info recovery_owner{}, runtime_owner{};
            // The executable links one runtime DSO with this SONAME. Handle-scoped
            // lookups avoid both canonical PLT addresses and its A5 backend copy.
            void *runtime_dso = dlopen("libhost_runtime.so", RTLD_NOW | RTLD_NOLOAD);
            assert(runtime_dso != nullptr);
            void *runtime_api = dlsym(runtime_dso, "get_runtime_size");
            assert(runtime_api != nullptr && dladdr(runtime_api, &runtime_owner) != 0);
            auto recover =
                reinterpret_cast<RecoverFn>(dlsym(runtime_dso, "_ZN12DeviceRunner31recover_device_or_mark_unusableEi"));
            assert(recover != nullptr && dladdr(reinterpret_cast<void *>(recover), &recovery_owner) != 0);
            assert(recovery_owner.dli_fbase == runtime_owner.dli_fbase);
            std::fprintf(stderr, "RECOVERY_RUNTIME path=%s\n", runtime_owner.dli_fname);
            recover(this, rc);
            assert(dlclose(runtime_dso) == 0);
        }
    }
    bool check_late_health{false};
    int recovery_calls{0};
    int recovery_rc{0};
    void initialize(bool borrowed = true) {
        device_id_ = 0;
        if (borrowed) {
            initialize_borrowed_context(*this);
        } else {
            assert(execution_mode_latch().latch(SIMPLER_MODE_PROGRAM) == 0);
            assert(rtStreamCreate(&stream_aicpu_, 0) == 0);
            assert(rtStreamCreate(&stream_aicore_, 0) == 0);
        }
    }
    using DeviceRunnerBase::ensure_device_run_result_region;
    using DeviceRunnerBase::ensure_device_wall_buffer;
    bool result_initialized() const { return device_run_result_initialized_[0]; }
    void release_result(uint64_t epoch = 1) { release_device_run_result(0, epoch, true); }
    using DeviceRunnerBase::discharge_boundary_waits;
    using DeviceRunnerBase::slot_persistent_args;
    int finish_common() { return finalize_common(); }
    void *whole_event(const NativeRunIdentity &identity) {
        return run_fence(identity.pipeline_slot).boundary_event(identity, RunCompletionFence::StreamRole::Aicpu);
    }
    int retire_boundary(const NativeRunIdentity &identity) {
        return run_fence(identity.pipeline_slot).retire(identity);
    }
    void prepare_diagnostics(Runtime &runtime, int level, int pmu_event_type, int dump_level = 0) {
        max_block_dim_ = 1;
        CallConfig config;
        config.aicpu_thread_num = 2;
        config.enable_chip_swimlane = level;
        config.enable_pmu = pmu_event_type;
        config.enable_dump_args = dump_level;
        assert(prepare_launch_shape(runtime, config) == 0);
        PreparedExecution execution(NativeRunIdentity{}, runtime, config, 0);
        assert(
            runtime.dev.pmu_event_type ==
            (execution.dfx.pmu_enabled ? static_cast<uint32_t>(execution.dfx.pmu_event_type) : 0)
        );
    }
    bool timing_armed() const { return device_timing_armed_[0]; }
    void consume_timing() { device_timing_armed_[0] = false; }
    void release_timing() {
        free_tensor(device_wall_dev_ptrs_[0]);
        device_wall_dev_ptrs_[0] = nullptr;
    }
    MemoryAllocator &allocator() { return mem_alloc_; }
    void check_failed_event_close() {
        Runtime runtime;
        runtime.dev = {};
        const NativeRunIdentity id{1, 1, 1, 0};
        PreparedExecution prepared(id, runtime, CallConfig{}, 0);
        prepared.caller_stream = caller;
        prepared.streams = borrowed_run_streams();
        assert(arm_run_fence(prepared) == 0);
        retire_run_fence(prepared);
        // The last created event is the caller tail, not a context event.
        failed_destroy_event = &tokens[16 + next_event - 1];
        void *control_stream = kernel_execution_state().hidden_stream(KernelStreamKind::Aicpu);
        assert(finalize_common() == -778);
        assert(kernel_execution_state().has_live_resources());
        assert(kernel_execution_state().phase() == KernelContextPhase::Closing);
        assert(!kernel_execution_state().accepts_dispatch());
        assert(kernel_execution_state().hidden_stream(KernelStreamKind::Aicpu) == control_stream);
        assert(kernel_resources_live());
        assert(finalize_common() == 0);
        assert(!kernel_execution_state().has_live_resources());
    }
    void finish() {
        // This probe owns only host-backed stub objects. No device task can
        // retain the partial launches deliberately left uncompleted here.
        (void)abandon_common_after_device_failure();
        kernel_execution_state().close();
        device_id_ = -1;
    }
};
static void check(int fault, bool capturing) {
    fail_at = fault;
    capture = capturing;
    next_stream = next_event = submits = syncs = copies = 0;
    trace.clear();
    stream_trace.clear();
    event_trace.clear();
    ProbeRunner runner;
    runner.initialize();
    Runtime runtime;
    runtime.dev = {};
    MemoryAllocator allocator;
    SlotPersistentArgs slot;
    NativeRunIdentity id{1, 1, 1, 0};
    auto prepared = std::make_unique<DeviceRunnerBase::PreparedExecution>(id, runtime, CallConfig{}, 0);
    prepared->caller_stream = caller;
    assert(prepared->kernel_args.prepare_runtime_args(runtime, allocator, slot) == 0);
    const int prior_copies = copies;
    assert(simpler_validate_execution_context(nullptr) == PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    assert(simpler_validate_execution_context(&runner) == 0);
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(id));
    assert(syncs == 0);
    if (capturing) {
        assert(outcome.rc == PTO_RUNTIME_ERR_UNSUPPORTED);
        assert(outcome.progress == LaunchProgress::NotStarted);
        assert(trace.empty() && copies == prior_copies);
    } else if (fault != 0) {
        assert(outcome.rc == -777);
        assert(outcome.progress == LaunchProgress::Partial);
        assert(submits == fault);
        assert(outcome.active != nullptr && outcome.prepared == nullptr);
    } else {
        assert(outcome.rc == 0 && outcome.progress == LaunchProgress::Complete);
        const auto streams = outcome.active->prepared->streams;
        assert(streams.aicpu == caller && streams.aicore != caller);
        assert(
            (trace ==
             std::vector<std::string>{
                 "publish", "record", "wait", "record", "core", "record", "cpu", "wait", "record", "record", "record"
             })
        );
        assert(
            (stream_trace == std::vector<void *>{
                                 caller, caller, streams.aicore, streams.aicore, streams.aicore, streams.aicore, caller,
                                 caller, caller, caller, caller
                             })
        );
    }
    if (!capturing && fault == 0) {
        // Each wait names its own record, not just the expected stream.
        for (const auto edge : {std::pair{1, 2}, std::pair{5, 7}}) {
            assert(event_trace[edge.first] != nullptr);
            assert(event_trace[edge.first] == event_trace[edge.second]);
        }
        assert(event_trace[1] != event_trace[5]);
        assert(event_trace[10] != event_trace[9]);
    }
    runner.finish();
    assert(release_slot_persistent_args(slot, allocator) == 0);
}
static void check_joined_callers(void *first_caller, void *second_caller, bool fail_proof = false) {
    fail_at = 0;
    capture = false;
    next_stream = next_event = submits = syncs = copies = 0;
    trace.clear();
    stream_trace.clear();
    event_trace.clear();
    ProbeRunner runner;
    runner.initialize();
    Runtime first_runtime, second_runtime;
    first_runtime.dev = second_runtime.dev = {};
    MemoryAllocator allocator;
    SlotPersistentArgs first_slot, second_slot;
    const NativeRunIdentity first_id{1, 1, 1, 0}, second_id{2, 1, 2, 1};
    auto first = std::make_unique<DeviceRunnerBase::PreparedExecution>(first_id, first_runtime, CallConfig{}, 0);
    first->caller_stream = first_caller;
    first->joinable_boundary = true;
    assert(first->kernel_args.prepare_runtime_args(first_runtime, allocator, first_slot) == 0);
    auto preceding = runner.launch_execution(std::move(first), NativeRunExecutionTestPeer::mint(first_id));
    assert(preceding.rc == 0 && preceding.active != nullptr);
    void *whole_done = runner.whole_event(first_id);
    assert(whole_done != nullptr);
    auto second = std::make_unique<DeviceRunnerBase::PreparedExecution>(second_id, second_runtime, CallConfig{}, 1);
    second->caller_stream = second_caller;
    second->joinable_boundary = true;
    second->join = NativeRunJoin{preceding.active->prepared.get(), first_id};
    assert(second->kernel_args.prepare_runtime_args(second_runtime, allocator, second_slot) == 0);
    submits = 0;
    trace.clear();
    stream_trace.clear();
    event_trace.clear();
    fail_at = fail_proof ? 2 : 0;
    auto following = runner.launch_execution(std::move(second), NativeRunExecutionTestPeer::mint(second_id));
    assert(syncs == 0);
    assert(following.active != nullptr);
    const auto streams = following.active->prepared->streams;
    if (second_caller != nullptr) assert(streams.aicpu == second_caller);
    assert(trace[0] == "wait" && trace[1] == "record");
    assert(stream_trace[0] == streams.aicpu && stream_trace[1] == streams.aicpu);
    assert(event_trace[0] == whole_done);
    if (fail_proof) {
        assert(following.rc == -777 && following.progress == LaunchProgress::Partial);
        assert(trace.size() == 2);
        const size_t pending = runner.queued_boundary_wait_count();
        assert(pending == 2);
        for (int retry = 0; retry != 2; ++retry) {
            assert(runner.discharge_boundary_waits(*preceding.active->prepared, true) != 0);
            assert(runner.queued_boundary_wait_count() == 1);
            assert(runner.finish_common() != 0);
            assert(runner.queued_boundary_wait_count() == 1);
        }
    } else {
        assert(following.rc == 0 && following.progress == LaunchProgress::Complete);
        assert(trace[2] == "publish" && stream_trace[2] == streams.aicpu);
        assert(trace[3] == "record" && stream_trace[3] == streams.aicpu);
        assert(trace[4] == "wait" && stream_trace[4] == streams.aicore);
        assert(event_trace[3] == event_trace[4]);
        assert(runner.retire_boundary(first_id) == PTO_RUNTIME_ERR_INTERNAL);
        assert(runner.discharge_boundary_waits(*preceding.active->prepared, true) == 0);
        assert(runner.retire_boundary(first_id) == 0);
    }
    preceding.active->prepared->kernel_args.release_run_view();
    following.active->prepared->kernel_args.release_run_view();
}

struct ChipTaskArgs;
struct OrchContext;

class OwnedCloseRunner : public DeviceRunner {
public:
    void initialize(bool borrowed) {
        if (borrowed) initialize_borrowed_context(*this);
        device_id_ = 0;
        timeout_configuration_rc_ = 0;
        max_block_dim_ = block_dim_ = 1;
        if (!borrowed) {
            assert(execution_mode_latch().latch(SIMPLER_MODE_PROGRAM) == 0);
            // Program's initializer owns its persistent pair, including A5's fixed pair.
            assert(rtStreamCreate(&stream_aicpu_, 0) == 0);
            assert(rtStreamCreate(&stream_aicore_, 0) == 0);
        }
        assert(install_staged_workspace() == 0);
        assert(set_run_resource_grant(2));
        enable_automatic_admission();
        struct EntryPoints {
            void (*entry)(const ChipTaskArgs &);
            void (*bind)(OrchContext *);
            void (*finish)();
        };
        static EntryPoints empty_orchestration{[](const ChipTaskArgs &) {}, [](OrchContext *) {}, nullptr};
        CallableState callable;
        callable.hash = callable.chip_buffer_hash = 1;
        callable.signature = {ArgDirection::INOUT};
        if (std::strcmp(SIMPLER_RUNTIME_NAME, "host_build_graph") == 0) {
            callable.host_orch_func_ptr = &empty_orchestration;
        }
        callables_[1] = std::move(callable);
        chip_callable_buffers_[1] = ChipCallableBuffer{};
    }
    void initialize_scope_stats() {
        assert(
            scope_stats_collector_.init(
                1,
                [this](size_t bytes) {
                    return mem_alloc_.alloc(bytes);
                },
                nullptr,
                [this](void *ptr) {
                    return mem_alloc_.free(ptr);
                },
                device_id_
            ) == 0
        );
        assert(scope_stats_collector_.get_scope_stats_shm_device_ptr() != nullptr);
    }
    int fill_persistent_arch_fields(KernelArgs *, uint64_t) override { return 0; }
    int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *out
    ) override {
        *out = std::make_unique<PreparedExecution>(identity, runtime, config, slot);
        (*out)->resources_owned = true;
        int rc = ensure_device_run_result_region(slot, identity.run_epoch, (*out)->kernel_args);
        if (observation_probe) {
            (*out)->scheduler_runtime_mode = observation_mode;
            observation_record = reinterpret_cast<void *>((*out)->kernel_args.args.run_result_data_base);
            for (const auto &lease : runtime.tensor_leases())
                if (lease.needs_copy_back) observation_outputs.emplace_back(lease.dev_ptr, lease.size);
        }
        return rc == 0 ? (*out)->kernel_args.prepare_runtime_args(runtime, mem_alloc_, slot_persistent_args(slot)) : rc;
    }
};

class ObservationRunner : public OwnedCloseRunner {
public:
    void recover_device_or_mark_unusable(int) override {}
    bool clock_timing{false};
    int retire_completed_execution(ActiveExecution &active) override {
        const int rc = OwnedCloseRunner::retire_completed_execution(active);
        if (rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE && clock_timing) {
            // Device timing is the controlled transport leaf; emission/export stay real.
            device_run_timing_[active.prepared->pipeline_slot].phase_ns[static_cast<int>(AicpuPhase::RunWall)] = 1000;
        }
        return rc;
    }
};

static void check_original_core_observation(uint32_t mode = 0, bool stream_fails = false, bool borrowed = true);

template <class Header>
static auto write_diagnostic_error_tail(DeviceRunResultRegion &region, int)
    -> decltype(&Header::orch_error_code, void()) {
    Header header{};
    header.sched_error_code.store(100, std::memory_order_relaxed);
    constexpr size_t offset = offsetof(Header, orch_error_code);
    constexpr size_t bytes = sizeof(Header) - offset;
    static_assert(bytes <= sizeof(region.payload));
    region.payload_bytes = bytes;
    std::memcpy(region.payload, reinterpret_cast<const uint8_t *>(&header) + offset, bytes);
}
template <class Header>
static void write_diagnostic_error_tail(DeviceRunResultRegion &, long) {}

static void check_clock_export(uint32_t mode, const std::filesystem::path &directory) {
    assert(std::filesystem::create_directories(directory));
    bind_runtime_log(directory.string());
    observation_probe = true;
    observation_mode = mode;
    owned_close_active = true;
    ObservationRunner runner;
    runner.initialize(false);
    runner.clock_timing = true;
    void *original = std::aligned_alloc(get_runtime_alignment(), get_runtime_size());
    void *successor = std::aligned_alloc(get_runtime_alignment(), get_runtime_size());
    assert(original != nullptr && successor != nullptr);
    std::memset(original, 0, get_runtime_size());
    std::memset(successor, 0, get_runtime_size());
    CallConfig config;
    config.aicpu_thread_num = 2;
    for (int ring = 0; ring != RUNTIME_ENV_RING_COUNT; ++ring) {
        config.runtime_env.ring_task_window[ring] = 4;
        config.runtime_env.ring_heap[ring] = 1024;
        config.runtime_env.ring_dep_pool[ring] = 4;
    }
    const std::string prefix = directory.string();
    assert(prefix.size() < sizeof(config.output_prefix));
    std::memcpy(config.output_prefix, prefix.c_str(), prefix.size() + 1);
    std::vector<uint8_t> output(16, 0x71);
    observation_output_host = output.data();
    const uint32_t shape[]{16};
    ChipTensor tensor;
    tensor.init_external(output.data(), output.size(), shape, 1, DataType::UINT8, AddressSpace::HOST);
    ChipStorageTaskArgs args;
    args.add_tensor(tensor);
    NativeRunDescriptor descriptor{0, 0, 1, 1, 1, 1};
    assert(simpler_prepare_run(&runner, original, 1, &args, &config, &descriptor) == 0);
    const auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(original);
    const uint64_t invocation = state->trace_inv;
    const auto runner_records = [&] {
        return std::count_if(runtime_diagnostics.begin(), runtime_diagnostics.end(), [&](const auto &record) {
            return record.find(" name=chip.run.runner_run ") != std::string::npos &&
                   record.find(" inv=" + std::to_string(invocation) + " ") != std::string::npos;
        });
    };
    DfxRunConfig dfx;
    dfx.chip_swimlane_level = ChipSwimlaneLevel::SCHED_PHASES;
    dfx.output_prefix = prefix;
    (void)runner.host_phase_pool_arm(0, false);
    runner.begin_host_phase_run(0, dfx);
    assert(!runner.host_clock_alignment_log_required(0));
    assert(runner.host_phase_pool_arm(0, true) != nullptr);
    assert(!runner.host_clock_alignment_log_required(0));
    runner.host_phase_pool_finish(0, 0, invocation);
    assert(runner.host_clock_alignment_log_required(0));
    assert(simpler_launch_run(&runner, original) == 0);
    assert(simpler_poll_run(&runner, original) == SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    assert(state->active_execution->prepared->retirement_diagnostics_rc == 0);
    assert(runner_records() == 1);
    const auto target = directory / ("host_clock_alignment." + std::to_string(getpid()) + ".log");
    assert(!std::filesystem::exists(target));

    descriptor.run_epoch = descriptor.run_id = descriptor.dispatch_id = 2;
    assert(simpler_prepare_run(&runner, successor, 1, &args, &config, &descriptor) == 0);
    assert(!runner.host_clock_alignment_log_required(0));
    const uint64_t successor_invocation = static_cast<NativeRunContext<DeviceRunnerBase> *>(successor)->trace_inv;
    assert(simpler_finalize_run(&runner, original) == 0);
    assert(runner_records() == 1);
    assert(output == std::vector<uint8_t>(output.size(), 0x42));
    std::ifstream file(target);
    assert(file);
    bool root = false, execution = false, wall = false;
    std::string exported;
    for (std::string line; std::getline(file, line);) {
        assert(line.find(" inv=" + std::to_string(invocation) + " ") != std::string::npos);
        assert(line.find(" inv=" + std::to_string(successor_invocation) + " ") == std::string::npos);
        root |= line.find(" name=chip.run ") != std::string::npos;
        execution |= line.find(" name=chip.run.runner_run ") != std::string::npos;
        wall |= line.find(" name=chip.run.runner_run.device_wall ") != std::string::npos;
        exported += line + '\n';
    }
    assert(root && execution && wall);
    assert(simpler_finalize_run(&runner, successor) == 0);
    std::ifstream unchanged(target);
    assert(std::string(std::istreambuf_iterator<char>(unchanged), {}) == exported);
    std::fprintf(
        stderr, "CLOCK_EXPORT path=%s inv=%llu successor_inv=%llu root=1 runner=1 wall=1 slot_reused=1\n",
        target.c_str(), static_cast<unsigned long long>(invocation),
        static_cast<unsigned long long>(successor_invocation)
    );
    std::free(original);
    std::free(successor);
    std::puts("PASS: original clock export follows root span and survives slot reuse");
}

// Each observer case runs in a fresh process. RTS leaves model transfer and
// stream status; prepare, bind, progress, wait, output delivery and finalize
// execute through the real host runtime.
static void check_observation(const std::string &scenario, uint32_t mode) {
    if (scenario == "late-stream-error") {
        for (bool borrowed : {true, false}) {
            check_original_core_observation(mode, false, borrowed);
            check_original_core_observation(mode, true, borrowed);
        }
        std::puts("PASS: late error uses original retained streams after slot retirement");
        return;
    }
    bind_runtime_log();
    observation_probe = true;
    observation_mode = mode;
    const bool borrowed = scenario == "kernel-no-copy";
    owned_close_active = true;
    completion_events_ready = scenario != "unknown-stop";
    auto *runner = new ObservationRunner;
    runner->initialize(borrowed);
    void *storage = std::aligned_alloc(get_runtime_alignment(), get_runtime_size());
    assert(storage != nullptr);
    std::memset(storage, 0, get_runtime_size());
    std::vector<uint8_t> host_output(16, 0x71);
    observation_output_host = host_output.data();
    const uint32_t shape[]{16};
    ChipTensor tensor;
    tensor.init_external(
        host_output.data(), host_output.size(), shape, 1, DataType::UINT8,
        borrowed ? AddressSpace::DEVICE : AddressSpace::HOST
    );
    ChipStorageTaskArgs args;
    args.add_tensor(tensor);
    CallConfig config;
    config.aicpu_thread_num = 2;
    for (int ring = 0; ring != RUNTIME_ENV_RING_COUNT; ++ring) {
        config.runtime_env.ring_task_window[ring] = 4;
        config.runtime_env.ring_heap[ring] = 1024;
        config.runtime_env.ring_dep_pool[ring] = 4;
    }
    const NativeRunDescriptor descriptor{0, 0, 1, 1, 1, 1};
    assert(simpler_prepare_run(runner, storage, 1, &args, &config, &descriptor) == 0);
    const auto *prepared_state = static_cast<NativeRunContext<DeviceRunnerBase> *>(storage);
    const uint64_t trace_inv = prepared_state->trace_inv;
    const uint64_t trace_hid = prepared_state->trace_hid;
    assert(trace_inv != 0);
    observation_region.published = descriptor.run_epoch;
    observation_region.verdict = static_cast<uint32_t>(DeviceRunVerdict::Ok);
    if (scenario == "failed-record-success" || scenario == "error-record-error") {
        observation_region.verdict = static_cast<uint32_t>(DeviceRunVerdict::Error);
        observation_region.completion_code = -100;
        observation_region.code_source = static_cast<uint32_t>(DeviceRunCodeSource::Header);
        write_diagnostic_error_tail<SharedMemoryHeader>(observation_region, 0);
    } else if (scenario == "missing-record-success" || scenario == "program-output" || scenario == "unknown-stop") {
        observation_region = DeviceRunResultRegion{};
    } else if (scenario == "stale-record-success") {
        observation_region.published = descriptor.run_epoch + 1;
    } else if (scenario == "malformed-record-success") {
        observation_region.completion_code = -100;
    }
    assert(
        (borrowed ? simpler_launch_run_on_stream(runner, storage, caller, nullptr) :
                    simpler_launch_run(runner, storage)) == 0
    );
    const auto original_streams = prepared_state->active_execution->prepared->streams;
    const NativeRunIdentity original_identity{
        descriptor.run_epoch, descriptor.generation, descriptor.dispatch_id, descriptor.pipeline_slot
    };
    if (!borrowed) assert(host_output == std::vector<uint8_t>(host_output.size(), 0x71));
    assert(syncs == 0 && private_reads == 0 && output_reads == 0);
    if (completion_events_ready) {
        const bool waited = scenario == "missing-record-success";
        if (waited) assert(simpler_wait_run_completion(runner, storage) == 0);
        else assert(simpler_poll_run(runner, storage) == SIMPLER_NATIVE_RUN_POLL_COMPLETE);
        check_completion_proof_records(
            original_identity, trace_inv, trace_hid, original_streams.aicore, original_streams.aicpu, waited
        );
        const auto proof_size = runtime_diagnostics.size();
        assert(simpler_poll_run(runner, storage) == SIMPLER_NATIVE_RUN_POLL_COMPLETE);
        assert(runtime_diagnostics.size() == proof_size);
        check_claim_release(trace_inv, trace_hid, true);
        if (!borrowed) assert(host_output == std::vector<uint8_t>(host_output.size(), 0x71));
        assert(syncs == 0 && private_reads == 0 && output_reads == 0);
    } else {
        const auto pending_size = runtime_diagnostics.size();
        for (int poll = 0; poll != 64; ++poll)
            assert(simpler_poll_run(runner, storage) == SIMPLER_NATIVE_RUN_POLL_NOT_READY);
        assert(runtime_diagnostics.size() == pending_size);
        for (const auto &record : runtime_diagnostics) {
            const auto row = proof_fields(record);
            if (!row.count("name") || row.at("name") != "chip.run.proof.record") continue;
            event_query_error_event = reinterpret_cast<void *>(std::stoull(row.at("e"), nullptr, 16));
            break;
        }
        assert(event_query_error_event != nullptr);
        event_query_error = 507018;
        assert(simpler_poll_run(runner, storage) == SIMPLER_NATIVE_RUN_POLL_ERROR);
        const int errored_queries = event_queries;
        const auto error_size = runtime_diagnostics.size();
        assert(simpler_poll_run(runner, storage) == SIMPLER_NATIVE_RUN_POLL_ERROR);
        assert(event_queries == errored_queries && runtime_diagnostics.size() == error_size);
        assert(std::any_of(runtime_diagnostics.begin(), runtime_diagnostics.end(), [](const auto &record) {
            const auto row = proof_fields(record);
            return row.count("name") && row.at("name") == "chip.run.proof.event_query" && row.at("rc") == "507018" &&
                   row.at("status_valid") == "0";
        }));
        event_query_error = 0;
        assert(!complete_proof(
            runtime_diagnostics, original_identity, trace_inv, trace_hid, original_streams.aicore,
            original_streams.aicpu
        ));
        std::fprintf(stderr, "proof pending_polls=64 added_records=0 cached_error_queries=0\n");
    }
    const bool stream_fails = scenario == "success-record-error" || scenario == "error-record-error" ||
                              scenario == "copy-error-error" || scenario == "unknown-stop";
    sync_error = stream_fails ? 507018 : 0;
    aicore_sync_error = scenario == "core-error" ? 507018 : 0;
    private_copy_error = (scenario == "copy-error-success" || scenario == "copy-error-error") ? -777 : 0;
    output_copy_error = scenario == "output-copy-error" ? -788 : 0;
    const int expected = stream_fails || aicore_sync_error != 0 ? 507018 : 0;
    const int wait_rc = simpler_wait_run(runner, storage);
    std::fprintf(
        stderr, "observer case=%s mode=%u wait=%d expected=%d private_reads=%d syncs=%d\n", scenario.c_str(), mode,
        wait_rc, expected, private_reads, syncs
    );
    assert(wait_rc == expected);
    check_stream_proof(
        original_identity, original_streams.aicore, original_streams.aicpu, expected, completion_events_ready
    );
    if (completion_events_ready) assert(syncs == (sync_error != 0 ? 1 : 2));
    else assert(syncs >= 1);  // Unknown-stop cleanup can attempt additional drains.
    assert(private_reads == (expected != 0 ? 1 : 0));
    assert(output_reads == 0);
    assert(!observation_steps.empty() && observation_steps.front() == "stream-observer");
    assert(simpler_wait_run(runner, storage) == expected);
    const int syncs_before_finalize = syncs;
    const int final_rc = simpler_finalize_run(runner, storage);
    const int expected_finalize = output_copy_error != 0 && expected == 0 ? output_copy_error : expected;
    std::fprintf(
        stderr, "observer finalize=%d expected=%d output_reads=%d\n", final_rc, expected_finalize, output_reads
    );
    assert(final_rc == expected_finalize);
    assert(syncs == syncs_before_finalize);
    if (completion_events_ready) {
        check_claim_release(trace_inv, trace_hid);
        std::fprintf(
            stderr, "claim_release inv=%llu hid=%llu depth=1 count=1 before-finalize\n",
            static_cast<unsigned long long>(trace_inv), static_cast<unsigned long long>(trace_hid)
        );
    }
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "host_build_graph") == 0) {
        if (scenario == "error-record-error") {
            assert(std::any_of(runtime_diagnostics.begin(), runtime_diagnostics.end(), [](const auto &message) {
                return message.find("Observed runtime status: -100 (shared scheduler header not read)") !=
                       std::string::npos;
            }));
            assert(std::none_of(runtime_diagnostics.begin(), runtime_diagnostics.end(), [](const auto &message) {
                return message.find("sched_error_code=0") != std::string::npos;
            }));
            for (const auto &message : runtime_diagnostics) {
                if (message.find("Observed runtime status:") != std::string::npos)
                    std::fwrite(message.data(), 1, message.size(), stderr);
            }
        }
        assert(std::none_of(runtime_diagnostics.begin(), runtime_diagnostics.end(), [](const auto &message) {
            return message.find("outside the retained temporary block this slot publishes") != std::string::npos;
        }));
    }
    if (scenario == "unknown-stop") {
        assert(runner->native_runs_outstanding());
        assert(host_output == std::vector<uint8_t>(host_output.size(), 0x71));
        assert(output_reads == 0);
    } else if (!borrowed) {
        assert(output_reads == (expected == 0 ? 1 : 0));
        assert(
            host_output ==
            std::vector<uint8_t>(host_output.size(), expected == 0 && output_copy_error == 0 ? 0x42 : 0x71)
        );
    } else {
        assert(output_reads == 0);
    }
    if (scenario == "success-record-error" && mode == 0) check_proof_identity_widths();
    std::puts("PASS: real resident observer and Program delivery contract");
}

static void check_owned_unknown_close(const std::string &scenario) {
    const bool borrowed = scenario == "borrowed";
    fail_force_reset = scenario == "reset-failure";
    fail_reset_probe = scenario == "probe-failure";
    const bool confirmed = !borrowed && !fail_force_reset && !fail_reset_probe;
    const bool retry_acl_close = scenario == "acl-finalize-retry";
    acl_initialized = false;
    fail_acl_finalize_once = retry_acl_close;
    acl_finalize_calls = 0;
    owned_close_active = true;
    completion_events_ready = false;
    fail_at = force_resets = stale_resource_calls = 0;
    sync_error = -739;
    next_stream = next_event = submits = syncs = copies = 0;
    reset_allocations.clear();
    reset_streams.clear();
    reset_events.clear();
    // Unknown-stop carrier bytes are deliberately retained, never destroyed.
    auto *runner = new OwnedCloseRunner;
    runner->initialize(borrowed);
    if (retry_acl_close) assert(runner->ensure_acl_ready(0) == 0);
    runner->initialize_scope_stats();
    void *storage = std::aligned_alloc(get_runtime_alignment(), get_runtime_size());
    assert(storage != nullptr);
    std::memset(storage, 0, get_runtime_size());
    auto *host_output = new std::vector<uint8_t>(16, 0x71);
    const uint32_t shape[]{16};
    ChipTensor tensor;
    tensor.init_external(
        host_output->data(), host_output->size(), shape, 1, DataType::UINT8,
        borrowed ? AddressSpace::DEVICE : AddressSpace::HOST
    );
    ChipStorageTaskArgs args;
    args.add_tensor(tensor);
    CallConfig config;
    config.aicpu_thread_num = 2;
    for (int ring = 0; ring != RUNTIME_ENV_RING_COUNT; ++ring) {
        config.runtime_env.ring_task_window[ring] = 4;
        config.runtime_env.ring_heap[ring] = 1024;
        config.runtime_env.ring_dep_pool[ring] = 4;
    }
    const NativeRunDescriptor descriptor{0, 0, 1, 1, 1, 1};
    assert(simpler_prepare_run(runner, storage, 1, &args, &config, &descriptor) == 0);
    assert(
        (borrowed ? simpler_launch_run_on_stream(runner, storage, caller, nullptr) :
                    simpler_launch_run(runner, storage)) == 0
    );
    assert(finalize_device(runner) != 0 && force_resets == 0);  // Healthy live owners still refuse.
    const int original_error = simpler_wait_run(runner, storage);
    assert(original_error == -739);
    assert(simpler_poll_run(runner, storage) != SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    assert(simpler_finalize_run(runner, storage) == original_error);
    assert(runner->native_runs_outstanding());
    const uint64_t generation_before = runner->device_health().generation();
    const auto allocations_before_close = live_allocations;
    int close_rc = finalize_device(runner);
    if (!confirmed) assert(live_allocations == allocations_before_close);
    if (retry_acl_close) {
        assert(close_rc == -982 && runner->device_id() == 0);
        assert(runner->device_health().generation() != generation_before);
        assert(force_resets == 1 && acl_finalize_calls == 1 && acl_initialized);
        const uint64_t reset_generation = runner->device_health().generation();
        const auto operations = std::make_tuple(submits, syncs, copies, stale_resource_calls);
        close_rc = finalize_device(runner);
        assert(close_rc == 0 && force_resets == 1 && acl_finalize_calls == 2 && !acl_initialized);
        assert(runner->device_health().generation() == reset_generation);
        assert(operations == std::make_tuple(submits, syncs, copies, stale_resource_calls));
        std::fprintf(
            stderr, "standalone collector acl-finalize-retry: confirmed reset reused; original error=%d\n",
            original_error
        );
    }
    assert((close_rc == 0) == confirmed);
    assert((runner->device_health().generation() != generation_before) == confirmed);
    if (borrowed) assert(force_resets == 0);
    else if (confirmed) assert(force_resets == 1);
    else assert(force_resets != 0);
    assert(stale_resource_calls == 0);
    assert(*host_output == std::vector<uint8_t>(host_output->size(), 0x71));
    if (borrowed) {
        assert(runner->native_runs_outstanding());
        assert(force_resets == 0 && runner->device_id() == 0);
        assert(!live_allocations.empty());
        // The context, canonical carrier and caller backing stay live until
        // this child exits: no reset or physical completion was proved.
        std::fprintf(stderr, "standalone collector borrowed: context retained, no reset\n");
        return;
    }
    if (!confirmed) {
        assert(runner->native_runs_outstanding());
        assert(runner->device_id() == 0);
        std::fprintf(stderr, "standalone collector %s: unconfirmed context retained\n", scenario.c_str());
        const int failed_reset_count = force_resets;
        const auto before_retry = std::make_tuple(submits, syncs, copies, stale_resource_calls);
        fail_force_reset = fail_reset_probe = false;
        assert(finalize_device(runner) == 0);
        assert(force_resets == failed_reset_count + 1);
        assert(runner->device_health().generation() != generation_before);
        assert(stale_resource_calls == 0);
        assert(before_retry == std::make_tuple(submits, syncs, copies, stale_resource_calls));
        assert(*host_output == std::vector<uint8_t>(host_output->size(), 0x71));
        std::fprintf(
            stderr, "standalone collector %s: same-runner retry confirmed; original error=%d\n", scenario.c_str(),
            original_error
        );
    }
    const auto operations_after_close = std::make_tuple(submits, syncs, copies, stale_resource_calls);
    assert(simpler_poll_run(runner, storage) < 0);
    assert(simpler_wait_run(runner, storage) != 0);
    assert(simpler_finalize_run(runner, storage) != 0);
    assert(operations_after_close == std::make_tuple(submits, syncs, copies, stale_resource_calls));
    delete runner;
    delete host_output;
    assert(stale_resource_calls == 0);
    std::fprintf(stderr, "owned-close confirmed: reset proved, old raw handles refused without RTS\n");
    // CPU provider teardown is external to all old-generation RTS leaves.
    for (const auto &allocation : live_allocations)
        std::free(allocation.first);
    live_allocations.clear();
    reset_allocations.clear();
    reset_streams.clear();
    reset_events.clear();
    completion_events_ready = true;
    sync_error = 0;
    owned_close_active = false;
}

// The internal stale mark models code publication after physical retirement.
// A public registration may still refuse while old observation owners exist.
static void check_original_core_observation(uint32_t mode, bool stream_fails, bool borrowed) {
    const size_t proof_start = runtime_diagnostics.size();
    const auto current_proof_records = [&]() {
        return std::vector<std::string>(runtime_diagnostics.begin() + proof_start, runtime_diagnostics.end());
    };
    if (stream_fails) bind_runtime_log();
    fail_at = 0;
    capture = false;
    next_stream = next_event = submits = syncs = copies = 0;
    destroyed_streams.clear();
    observed_streams.clear();
    if (!borrowed) owned_close_active = true;
    ProbeRunner runner;
    runner.initialize(borrowed);
    runner.check_late_health = true;
    Runtime first_runtime, second_runtime;
    first_runtime.dev = second_runtime.dev = {};
    const NativeRunIdentity first_id{1, 1, 1, 0}, second_id{2, 1, 2, 1};
    void *first_record = nullptr;
    auto launch = [&](Runtime &runtime, const NativeRunIdentity &identity) {
        auto prepared = std::make_unique<DeviceRunnerBase::PreparedExecution>(
            identity, runtime, CallConfig{}, identity.pipeline_slot
        );
        prepared->caller_stream = borrowed ? caller : nullptr;
        prepared->resources_owned = true;
        if (identity.run_epoch == first_id.run_epoch && stream_fails) {
            prepared->scheduler_runtime_mode = mode;
            assert(runner.ensure_device_run_result_region(0, identity.run_epoch, prepared->kernel_args) == 0);
            first_record = reinterpret_cast<void *>(prepared->kernel_args.args.run_result_data_base);
        }
        assert(
            prepared->kernel_args.prepare_runtime_args(
                runtime, runner.allocator(), runner.slot_persistent_args(identity.pipeline_slot)
            ) == 0
        );
        return runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(identity));
    };
    auto first = launch(first_runtime, first_id);
    assert(first.rc == 0 && first.active != nullptr);
    if (stream_fails) {
        DeviceRunResultRegion region{};
        region.published = first_id.run_epoch;
        region.verdict = static_cast<uint32_t>(DeviceRunVerdict::Ok);
        std::memcpy(first_record, &region, sizeof(region));
    }
    const auto original = first.active->prepared->streams;
    const int reads_after_launch = copies;
    assert(runner.retire_completed_execution(*first.active) == SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    assert(syncs == 0 && copies == reads_after_launch);
    runner.mark_run_streams_stale();
    auto second = launch(second_runtime, second_id);
    assert(second.rc == 0 && second.active != nullptr);
    const auto following = second.active->prepared->streams;
    if (std::strcmp(get_platform(), "a2a3") == 0) assert(original.aicore != following.aicore);
    else assert(original.aicore == following.aicore);
    assert(std::find(destroyed_streams.begin(), destroyed_streams.end(), original.aicore) == destroyed_streams.end());
    if (!stream_fails) assert(runner.retire_completed_execution(*second.active) == SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    assert(syncs == 0);
    if (stream_fails) {
        Runtime reused_runtime;
        reused_runtime.dev = {};
        const NativeRunIdentity reused_id{3, 2, 3, first_id.pipeline_slot};
        auto reused = launch(reused_runtime, reused_id);
        assert(reused.rc == 0 && reused.active != nullptr);
        assert(runner.retire_completed_execution(*reused.active) == SIMPLER_NATIVE_RUN_POLL_COMPLETE);
        reused.active.reset();
        assert(complete_proof(current_proof_records(), first_id, 0, 0, original.aicore, original.aicpu));
    }
    const auto successor_allocations = live_allocations;
    const auto *successor_owner = second.active->prepared.get();
    void *successor_args = second.active->prepared->kernel_args.args.runtime_args;
    const int resets_before_observation = force_resets;
    const int drains_before_observation = device_drains;
    assert(runner.can_accept_run());
    assert(runner.accepts_new_run());
    sync_error = stream_fails ? 507018 : 0;
    sync_error_stream = stream_fails ? original.aicore : nullptr;
    assert(runner.drain_execution(*first.active).combined() == sync_error);
    if (stream_fails) {
        std::fprintf(
            stderr, "LATE_HEALTH borrowed=%d rc=%d recovery_calls=%d admission=%d phase=%d successor_owned=%d\n",
            borrowed ? 1 : 0, sync_error, runner.recovery_calls, runner.can_accept_run() ? 1 : 0,
            static_cast<int>(runner.kernel_execution_state().phase()), second.active->prepared->resources_owned ? 1 : 0
        );
        assert(runner.recovery_calls == 1 && runner.recovery_rc == sync_error);
        assert(!runner.can_accept_run());
        assert(!runner.accepts_new_run());
        if (borrowed) assert(runner.kernel_execution_state().phase() == KernelContextPhase::Poisoned);
        assert(device_drains == drains_before_observation + (borrowed ? 0 : 1));
        assert(force_resets == resets_before_observation);
        assert(second.active->prepared.get() == successor_owner);
        assert(second.active->prepared->resources_owned && !second.active->prepared->physical_complete);
        assert(second.active->prepared->kernel_args.args.runtime_args == successor_args);
        assert(live_allocations == successor_allocations);
        assert((observed_streams == std::vector<void *>{original.aicpu, original.aicore}));
        assert(copies == reads_after_launch + 3);  // Two successor descriptors and error-only private read.
        const auto own_proof = current_proof_records();
        assert(std::any_of(own_proof.begin(), own_proof.end(), [](const auto &line) {
            const auto row = proof_fields(line);
            return row.count("name") && row.at("name") == "chip.run.proof.observe_return" && row.at("ep") == "1" &&
                   row.at("gen") == "1" && row.at("disp") == "1" && row.at("slot") == "0" && row.at("cached") == "1" &&
                   row.at("physical") == "1" && row.at("retained") == "0" && row.at("rc") == "507018";
        }));
        auto missing_old = own_proof;
        missing_old.erase(
            std::remove_if(
                missing_old.begin(), missing_old.end(),
                [](const auto &line) {
                    const auto row = proof_fields(line);
                    return row.count("name") && row.at("name") == "chip.run.proof.consume" && row.at("ep") == "1";
                }
            ),
            missing_old.end()
        );
        assert(!complete_proof(missing_old, first_id, 0, 0, original.aicore, original.aicpu));
        std::fprintf(
            stderr, "proof same_slot_reuse=1 old_fresh_proof=1 stale_slot_ledger=UNKNOWN missing_old=UNKNOWN\n"
        );
    } else {
        assert((observed_streams == std::vector<void *>{original.aicpu, original.aicore}));
        assert(copies == reads_after_launch + 1);  // Only the second runtime descriptor publication.
    }
    sync_error = 0;
    sync_error_stream = nullptr;
    if (stream_fails) assert(runner.retire_completed_execution(*second.active) == SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    assert(runner.drain_execution(*second.active).combined() == 0);
    assert(runner.recovery_calls == (stream_fails ? 1 : 0));
    if (stream_fails) assert(!runner.can_accept_run());
    first.active.reset();
    second.active.reset();
    runner.finish();
}

static void check_metadata_publication(int fault, bool drain_fails = false) {
    metadata_extensions = 0;
    fail_at = fault;
    capture = false;
    next_stream = next_event = submits = syncs = copies = 0;
    trace.clear();
    stream_trace.clear();
    event_trace.clear();
    ProbeRunner runner;
    runner.initialize();
    Runtime runtime;
    runtime.dev = {};
    MemoryAllocator allocator;
    SlotPersistentArgs slot;
    uint64_t target = 0, source = 37;
    HostMetadataRegion region{&target, &source, sizeof(source)};
    assert(slot.staged_launch_metadata.prepare(&region, 1) == 0);
    source = 99;
    NativeRunIdentity id{1, 1, 1, 0};
    auto prepared = std::make_unique<DeviceRunnerBase::PreparedExecution>(id, runtime, CallConfig{}, 0);
    prepared->caller_stream = caller;
    assert(prepared->kernel_args.prepare_runtime_args(runtime, allocator, slot) == 0);
    assert(target == 0 && copies == 0);
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(id));
    assert(syncs == 0);
    if (fault != 0) {
        assert(outcome.rc == -777 && outcome.progress == LaunchProgress::Partial);
        assert(submits == fault && outcome.active != nullptr && outcome.prepared == nullptr);
        assert(trace.back() == "publish");
        for (const auto &step : trace)
            assert(step != "core" && step != "cpu");
        assert(target == (fault == 1 ? 0u : 37u));
    } else {
        assert(outcome.rc == 0 && outcome.progress == LaunchProgress::Complete);
        assert(target == 37);
        assert(
            (trace == std::vector<std::string>{
                          "publish", "publish", "record", "wait", "record", "core", "record", "cpu", "wait", "record",
                          "record", "record"
                      })
        );
        const auto streams = outcome.active->prepared->streams;
        for (size_t i : {0u, 1u, 2u})
            assert(stream_trace[i] == streams.aicpu);
        assert(stream_trace[3] == streams.aicore);
        assert(event_trace[2] == event_trace[3] && event_trace[2] != nullptr);
    }
    // A5's production backend is linked into this probe so its private runtime
    // exporter can be observed without replacing the drain's readiness check.
    if (std::strcmp(get_platform(), "a5") == 0) {
        outcome.active->prepared->resources_owned = true;
        outcome.active->prepared->dfx.chip_swimlane_level = ChipSwimlaneLevel::TASK_TIMING;
        sync_error = drain_fails ? -779 : 0;
        (void)runner.drain_execution(*outcome.active);
        sync_error = 0;
        assert(metadata_extensions == (fault == 0 ? 1 : 0));
    }
    runner.finish();
    assert(release_slot_persistent_args(slot, allocator) == 0);
}
static void check_level_publication(bool entry_route) {
    unsetenv("SIMPLER_PMU_EVENT_TYPE");
    next_stream = next_event = submits = copies = 0;
    fail_at = 0;
    ProbeRunner runner;
    runner.initialize();
    Runtime runtime;
    auto &slot = runner.slot_persistent_args(0);
    for (const auto [level, pmu, dump_level] :
         {std::tuple{4, 0, 0},
          {1, 0, 0},
          {0, 0, 0},
          {3, 0, 0},
          {2, 0, 0},
          {2, 2, 0},
          {2, 4, 0},
          {2, 0, 0},
          {2, 8, 0},
          {0, 0, 3},
          {0, 0, 1},
          {0, 0, 0},
          {0, 0, 2},
          {0, 0, 3}}) {
        runner.prepare_diagnostics(runtime, level, pmu, dump_level);
        assert(runtime.dev.dump_args_level == static_cast<uint32_t>(dump_level));
        assert(runtime.dev.chip_swimlane_level == static_cast<uint32_t>(level));
        KernelArgsHelper helper;
        assert(helper.prepare_runtime_args(runtime, runner.allocator(), slot) == 0);
        runtime.dev.chip_swimlane_level = 99;
        runtime.dev.pmu_event_type = 99;
        runtime.dev.dump_args_level = 99;
        LaunchProgressSink sink;
        assert(helper.publish_runtime_args(entry_route, caller, sink) == 0);
        assert(sink.submitted());
        uint32_t published;
        const auto *device = reinterpret_cast<const std::byte *>(slot.runtime_args);
        std::memcpy(&published, device + offsetof(DeviceRuntimeLaunchDesc, chip_swimlane_level), sizeof(published));
        assert(published == static_cast<uint32_t>(level));
        std::memcpy(&published, device + offsetof(DeviceRuntimeLaunchDesc, pmu_event_type), sizeof(published));
        assert(published == static_cast<uint32_t>(pmu));
        std::memcpy(&published, device + offsetof(DeviceRuntimeLaunchDesc, dump_args_level), sizeof(published));
        assert(published == static_cast<uint32_t>(dump_level));
    }
    assert(setenv("SIMPLER_PMU_EVENT_TYPE", "4", 1) == 0);
    runner.prepare_diagnostics(runtime, 0, 2);
    assert(runtime.dev.pmu_event_type == 4);
    KernelArgsHelper overridden;
    assert(overridden.prepare_runtime_args(runtime, runner.allocator(), slot) == 0);
    assert(setenv("SIMPLER_PMU_EVENT_TYPE", "1", 1) == 0);
    LaunchProgressSink sink;
    assert(overridden.publish_runtime_args(entry_route, caller, sink) == 0);
    uint32_t published;
    const auto *device = reinterpret_cast<const std::byte *>(slot.runtime_args);
    std::memcpy(&published, device + offsetof(DeviceRuntimeLaunchDesc, pmu_event_type), sizeof(published));
    assert(published == 4);
    unsetenv("SIMPLER_PMU_EVENT_TYPE");
    assert(release_slot_persistent_args(slot, runner.allocator()) == 0);
    runner.finish();
}
static void check_result_prepare_refusal() {
    next_stream = next_event = 0;
    ProbeRunner runner;
    runner.initialize();
    Runtime runtime;
    const NativeRunIdentity id{1, 1, 1, 0};
    DeviceRunnerBase::PreparedExecution prepared(id, runtime, CallConfig{}, 0);
    const int before = allocation_attempts;
    assert(runner.ensure_device_run_result_region(0, 0, prepared.kernel_args) != 0);
    assert(allocation_attempts == before);
    fail_allocation = allocation_attempts + 1;
    assert(runner.ensure_device_run_result_region(0, 1, prepared.kernel_args) != 0);
    assert(prepared.kernel_args.args.run_result_data_base == 0);
    fail_allocation = 0;
    for (int retry = 0; retry < 2; ++retry) {
        prepared.resources_owned = true;
        assert(runner.ensure_device_run_result_region(0, 1, prepared.kernel_args) == 0);
        assert(!runner.result_initialized());
        assert(runner.slot_persistent_args(0).staged_launch_metadata.size() == sizeof(uint64_t));
        runner.abandon_prepared_execution(prepared);
        assert(runner.slot_persistent_args(0).staged_launch_metadata.size() == 0);
    }
    runner.release_result();
    runner.finish();
}
static void check_result_publication(bool has_caller, int failure_step) {
    capture = false;
    next_stream = next_event = submits = syncs = copies = 0;
    trace.clear();
    stream_trace.clear();
    event_trace.clear();
    fail_at = failure_step == 0 ? 0 : failure_step;
    result_source = nullptr;
    ProbeRunner runner;
    runner.initialize();
    Runtime runtime;
    runtime.dev = {};
    const NativeRunIdentity id{1, 1, 1, 0};
    auto prepared = std::make_unique<DeviceRunnerBase::PreparedExecution>(id, runtime, CallConfig{}, 0);
    if (has_caller) prepared->caller_stream = caller;
    auto &slot = runner.slot_persistent_args(0);
    assert(runner.ensure_device_run_result_region(0, id.run_epoch, prepared->kernel_args) == 0);
    result_target = reinterpret_cast<void *>(prepared->kernel_args.args.run_result_data_base);
    assert(result_target != nullptr);
    assert(!runner.result_initialized() && copies == 0);
    assert(slot.staged_launch_metadata.size() == sizeof(uint64_t));
    // Allocator bytes can resemble a fully valid success for the requested epoch.
    DeviceRunResultRegion forged{};
    forged.published = id.run_epoch;
    forged.verdict = static_cast<uint32_t>(DeviceRunVerdict::Ok);
    assert(device_run_result_terminal(forged, id.run_epoch).state == DeviceRunTerminalState::Succeeded);
    std::memcpy(result_target, &forged, sizeof(forged));
    assert(runner.read_device_run_result(0, id.run_epoch) == 0);
    assert(runner.device_run_result_read_status(0, id.run_epoch) == RunRecordRead::NotAttempted);
    assert(runner.device_run_terminal(0, id.run_epoch).state == DeviceRunTerminalState::Undecided);
    assert(copies == 0);
    assert(prepared->kernel_args.prepare_runtime_args(runtime, runner.allocator(), slot) == 0);
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(id));
    assert(syncs == 0);
    if (failure_step != 0) {
        assert(outcome.rc == -777 && outcome.progress == LaunchProgress::Partial);
        assert(runner.result_initialized() == (failure_step == 3));
        for (const auto &op : trace)
            assert(op != "core" && op != "cpu");
        if (!runner.result_initialized()) {
            const int before = copies;
            assert(runner.read_device_run_result(0, id.run_epoch) == 0);
            assert(copies == before);
            assert(runner.device_run_result_read_status(0, id.run_epoch) == RunRecordRead::NotAttempted);
        }
    } else {
        assert(outcome.rc == 0 && outcome.progress == LaunchProgress::Complete);
        assert(runner.result_initialized());
        const size_t publication = 0;
        assert(trace[publication] == "publish" && trace[publication + 1] == "publish");
        assert(stream_trace[publication] == outcome.active->prepared->streams.aicpu);
        assert(stream_trace[publication + 1] == outcome.active->prepared->streams.aicpu);
    }
    if (outcome.active != nullptr) outcome.active->prepared->kernel_args.release_run_view();
    // Complete the accepted DMA after the per-run snapshot has been released.
    if (result_source != nullptr) {
        uint64_t before;
        std::memcpy(&before, result_target, sizeof(before));
        assert(before == id.run_epoch);
        std::memcpy(result_target, result_source, sizeof(uint64_t));
        uint64_t after;
        std::memcpy(&after, result_target, sizeof(after));
        assert(after == 0);
    }
    if (failure_step == 0) {
        assert(runner.read_device_run_result(0, id.run_epoch) == 0);
        assert(runner.device_run_result_read_status(0, id.run_epoch) == RunRecordRead::Ok);
        assert(runner.device_run_terminal(0, id.run_epoch).state == DeviceRunTerminalState::Undecided);
        const int before = allocation_attempts;
        KernelArgsHelper warm;
        assert(runner.ensure_device_run_result_region(0, 2, warm) == 0);
        assert(allocation_attempts == before + 1);
        assert(warm.args.run_result_data_base != reinterpret_cast<uint64_t>(result_target));
        assert(slot.staged_launch_metadata.size() == sizeof(uint64_t));
        runner.release_result(2);
    }
    result_target = nullptr;
    runner.release_result();
    runner.finish();
}
static void check_timing_prepare_refusal() {
    next_stream = next_event = 0;
    ProbeRunner runner;
    runner.initialize();
    Runtime runtime;
    const NativeRunIdentity id{1, 1, 1, 0};
    DeviceRunnerBase::PreparedExecution prepared(id, runtime, CallConfig{}, 0);
    fail_allocation = allocation_attempts + 1;
    assert(runner.ensure_device_wall_buffer(0, prepared.kernel_args) == 0);
    assert(prepared.kernel_args.args.device_wall_data_base == 0);
    assert(runner.slot_persistent_args(0).staged_launch_metadata.size() == 0);
    fail_allocation = 0;
    prepared.resources_owned = true;
    assert(runner.ensure_device_wall_buffer(0, prepared.kernel_args) == 0);
    const size_t reset_bytes = runner.slot_persistent_args(0).staged_launch_metadata.size();
    assert(reset_bytes != 0 && !runner.timing_armed());
    runner.abandon_prepared_execution(prepared);
    assert(runner.slot_persistent_args(0).staged_launch_metadata.size() == 0);
    assert(!runner.timing_armed());
    prepared.resources_owned = true;
    assert(runner.ensure_device_wall_buffer(0, prepared.kernel_args) == 0);
    assert(runner.slot_persistent_args(0).staged_launch_metadata.size() == reset_bytes);
    runner.abandon_prepared_execution(prepared);
    runner.release_timing();
    runner.finish();
}
static void check_timing_publication(int fault) {
    fail_at = fault;
    capture = false;
    next_stream = next_event = submits = syncs = copies = 0;
    trace.clear();
    stream_trace.clear();
    event_trace.clear();
    timing_source = nullptr;
    ProbeRunner runner;
    runner.initialize();
    Runtime runtime;
    runtime.dev = {};
    const NativeRunIdentity id{1, 1, 1, 0};
    auto prepared = std::make_unique<DeviceRunnerBase::PreparedExecution>(id, runtime, CallConfig{}, 0);
    prepared->caller_stream = caller;
    auto &slot = runner.slot_persistent_args(0);
    uint64_t scheduler_target = 0, scheduler_source = 37;
    const HostMetadataRegion region{&scheduler_target, &scheduler_source, sizeof(scheduler_source)};
    assert(slot.staged_launch_metadata.prepare(&region, 1) == 0);
    assert(runner.ensure_device_wall_buffer(0, prepared->kernel_args) == 0);
    timing_target = reinterpret_cast<void *>(prepared->kernel_args.args.device_wall_data_base);
    assert(timing_target != nullptr);
    constexpr int threads = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    constexpr size_t bytes = device_phase_buffer_bytes(threads);
    assert(slot.staged_launch_metadata.size() == sizeof(scheduler_source) + bytes);
    std::memset(timing_target, 0xa5, bytes);
    assert(copies == 0 && !runner.timing_armed());
    assert(prepared->kernel_args.prepare_runtime_args(runtime, runner.allocator(), slot) == 0);
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(id));
    assert(syncs == 0);
    if (fault) {
        assert(outcome.rc == -777 && outcome.progress == LaunchProgress::Partial);
        assert(!runner.timing_armed());
        for (const auto &op : trace)
            assert(op != "core" && op != "cpu");
    } else {
        assert(outcome.rc == 0 && outcome.progress == LaunchProgress::Complete);
        assert(runner.timing_armed());
        assert(scheduler_target == scheduler_source);
        assert(trace[0] == "publish" && trace[1] == "publish" && trace[2] == "publish");
        for (int i : {0, 1, 2})
            assert(stream_trace[i] == outcome.active->prepared->streams.aicpu);
        // The DMA reader outlives the per-run host snapshot, with no intervening wait.
        outcome.active->prepared->kernel_args.release_run_view();
        assert(timing_source != nullptr && timing_bytes == bytes);
        assert(static_cast<unsigned char *>(timing_target)[0] == 0xa5);
        std::memcpy(timing_target, timing_source, timing_bytes);
        DevicePhaseBufferStorage<threads> expected{};
        reset_device_phase_buffer(&expected, threads);
        assert(std::memcmp(timing_target, &expected, bytes) == 0);
        KernelArgsHelper unread;
        assert(runner.ensure_device_wall_buffer(0, unread) == 0);
        assert(unread.args.device_wall_data_base == 0);
        assert(slot.staged_launch_metadata.size() == 0);
    }
    runner.consume_timing();
    runner.release_timing();
    timing_target = nullptr;
    runner.finish();
}
static void check_arena_growth(SimplerExecutionMode mode) {
    const size_t before = live_allocations.size();
    {
        DeviceRunner runner;
        assert(runner.execution_mode_latch().latch(mode) == 0);
        assert(runner.setup_static_arena(0, 4096, 2048, 1024) == 0);
        assert(runner.setup_static_arena(1, 4096, 2048, 1024) == 0);
        void *original[] = {
            runner.acquire_pooled_gm_heap(0), runner.acquire_pooled_gm_sm(0), runner.acquire_pooled_runtime_arena(0)
        };
        void *other_bank[] = {
            runner.acquire_pooled_gm_heap(1), runner.acquire_pooled_gm_sm(1), runner.acquire_pooled_runtime_arena(1)
        };
        for (void *base : original)
            std::memset(base, 0xa5, 1024);
        for (void *base : other_bank)
            std::memset(base, 0x5a, 1024);
        const size_t committed = live_allocations.size();
        for (int failure = 1; failure <= 3; ++failure) {
            fail_allocation = allocation_attempts + failure;
            assert(runner.setup_static_arena(0, 8192, 4096, 2048) != 0);
            assert(allocation_attempts == fail_allocation);
            fail_allocation = 0;
            assert(live_allocations.size() == committed);
            assert(runner.acquire_pooled_gm_heap(0) == original[0]);
            assert(runner.acquire_pooled_gm_sm(0) == original[1]);
            assert(runner.acquire_pooled_runtime_arena(0) == original[2]);
            for (void *base : original)
                for (size_t i = 0; i < 1024; ++i)
                    assert(static_cast<unsigned char *>(base)[i] == 0xa5);
            const int attempts = allocation_attempts;
            assert(runner.setup_static_arena(0, 4096, 2048, 1024) == 0);
            assert(allocation_attempts == attempts);
        }
        assert(runner.setup_static_arena(0, 8192, 4096, 2048) == 0);
        void *grown[] = {
            runner.acquire_pooled_gm_heap(0), runner.acquire_pooled_gm_sm(0), runner.acquire_pooled_runtime_arena(0)
        };
        for (size_t i = 0; i < 3; ++i)
            assert(grown[i] != original[i]);
        const int attempts = allocation_attempts;
        assert(runner.setup_static_arena(0, 4096, 2048, 1024) == 0);
        assert(allocation_attempts == attempts);
        assert(runner.acquire_pooled_gm_heap(0) == grown[0]);
        assert(runner.acquire_pooled_gm_sm(0) == grown[1]);
        assert(runner.acquire_pooled_runtime_arena(0) == grown[2]);
        assert(runner.acquire_pooled_gm_heap(1) == other_bank[0]);
        assert(runner.acquire_pooled_gm_sm(1) == other_bank[1]);
        assert(runner.acquire_pooled_runtime_arena(1) == other_bank[2]);
        for (void *base : other_bank)
            for (size_t i = 0; i < 1024; ++i)
                assert(static_cast<unsigned char *>(base)[i] == 0x5a);
    }
    assert(live_allocations.size() == before);
}
int main(int argc, char **argv) {
    if (argc == 4 && std::strcmp(argv[1], "--observer") == 0) {
        if (std::strcmp(argv[2], "clock-export") == 0) {
            check_clock_export(
                static_cast<uint32_t>(std::stoul(argv[3])),
                std::filesystem::absolute(argv[0]).parent_path() / ("clock-export-" + std::string(argv[3]))
            );
            return 0;
        }
        check_observation(argv[2], static_cast<uint32_t>(std::stoul(argv[3])));
        return 0;
    }
    if (argc == 2) {
        check_owned_unknown_close(argv[1]);
        return 0;
    }
    check_owned_unknown_close("confirmed-dfx");
    check_joined_callers(nullptr, nullptr);
    check_joined_callers(nullptr, caller);
    check_joined_callers(caller, nullptr);
    check_joined_callers(caller, caller);
    check_joined_callers(caller, &tokens[62]);
    check_joined_callers(caller, &tokens[62], true);
    check_original_core_observation();
    check_level_publication(false);
    check_level_publication(true);
    check_result_prepare_refusal();
    for (bool has_caller : {false, true})
        for (int failure_step : {0, 1, 2, 3})
            check_result_publication(has_caller, failure_step);
    check_timing_prepare_refusal();
    check_timing_publication(0);
    check_timing_publication(2);
    check_timing_publication(3);
    check_metadata_publication(0);
    check_metadata_publication(1);  // Metadata enqueue failure.
    check_metadata_publication(2);  // Descriptor enqueue after accepted metadata.

    if (std::strcmp(get_platform(), "a5") == 0) {
        check_metadata_publication(0, true);
        check_metadata_publication(1, true);
        check_metadata_publication(2, true);
    }
    check_arena_growth(SIMPLER_MODE_PROGRAM);
    check_arena_growth(SIMPLER_MODE_KERNEL);
    check(0, false);
    // Positions 4 and 9 are optional timing marks; their failure is deliberately non-fatal.
    for (int fault : {1, 2, 3, 5, 6, 7, 8, 10, 11})
        check(fault, false);
    check(0, true);
    next_stream = next_event = 0;
    ProbeRunner runner;
    runner.initialize();
    const int prior_copies = copies;
    assert(simpler_validate_caller_stream(nullptr, caller) == PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    assert(simpler_validate_caller_stream(&runner, nullptr) == PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    assert(simpler_validate_caller_stream(&runner, caller) == PTO_RUNTIME_ERR_UNSUPPORTED);
    capture = false;
    assert(simpler_validate_caller_stream(&runner, caller) == 0);
    assert(copies == prior_copies);
    runner.check_failed_event_close();
    runner.finish();
    std::puts("PASS: real launch/drain ordering, publication failures, and pre-publication capture refusal");
}

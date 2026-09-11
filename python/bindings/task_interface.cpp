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
/**
 * Nanobind Python extension for task_interface headers.
 *
 * Wraps DataType, ChipTensor, ChipStorageTaskArgs, TaskArgs (unified
 * vector-backed builder with per-tensor TensorArgType tags), TensorArgType,
 * ArgDirection, CoreCallable, ChipCallable, and helper functions from
 * data_type.h / tensor.h / task_args.h / task_args_wire.h / arg_direction.h /
 * callable.h.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cerrno>
#include <cstddef>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <exception>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "arg_direction.h"
#include "callable.h"
#include "callable_protocol.h"
#include "chip_run_lane.h"
#include "chip_worker.h"
#include "common/host_span_names.h"
#include "common/host_span_scope.h"
#include "common/log_clock.h"
#include "host_log.h"
#include "data_type.h"
#include "worker_chip_orch_comm.h"
#include "worker_bind.h"
#include "task_args_wire.h"
#include "tensor.h"

namespace nb = nanobind;

namespace {

struct LocalAclMemLocation {
    uint32_t id{0};
    int type{0};
};

struct LocalAclPhysicalMemProp {
    int handleType{0};
    int allocationType{0};
    int memAttr{0};
    LocalAclMemLocation location{};
    uint64_t reserve{0};
};

struct LocalAclMemAccessDesc {
    int flags{0};
    LocalAclMemLocation location{};
    uint8_t rsv[12]{};
};

void append_cleanup_error(std::string &cleanup_error, const std::string &message);

class AclRuntimeApi {
public:
    AclRuntimeApi() = default;
    // Intentionally no aclFinalize/dlclose here: finalizing ACL during static
    // destruction triggers std::bad_alloc in onboard teardown. The ACL context
    // and library handle are deliberately leaked at process exit.
    ~AclRuntimeApi() = default;

    AclRuntimeApi(const AclRuntimeApi &) = delete;
    AclRuntimeApi &operator=(const AclRuntimeApi &) = delete;

    int (*aclInit)(const char *){nullptr};
    int (*aclrtSetDevice)(int){nullptr};
    int (*aclrtGetDevice)(int *){nullptr};
    int (*aclrtMemcpy)(void *, size_t, const void *, size_t, int){nullptr};
    int (*aclrtMemGetAllocationGranularity)(LocalAclPhysicalMemProp *, int, size_t *){nullptr};
    int (*aclrtMallocPhysical)(void **, size_t, const LocalAclPhysicalMemProp *, uint64_t){nullptr};
    int (*aclrtFreePhysical)(void *){nullptr};
    int (*aclrtReserveMemAddress)(void **, size_t, size_t, void *, uint64_t){nullptr};
    int (*aclrtReleaseMemAddress)(void *){nullptr};
    int (*aclrtMapMem)(void *, size_t, size_t, void *, uint64_t){nullptr};
    int (*aclrtUnmapMem)(void *){nullptr};
    int (*aclrtMemSetAccess)(void *, size_t, LocalAclMemAccessDesc *, size_t){nullptr};
    int (*aclrtMemExportToShareableHandle)(void *, int, uint64_t, uint64_t *){nullptr};
    int (*aclrtMemImportFromShareableHandle)(uint64_t, int32_t, void **){nullptr};

    void load() {
        lib_ = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
        if (lib_ == nullptr) {
            lib_ = dlopen("libascendcl.so.1", RTLD_NOW | RTLD_LOCAL);
        }
        if (lib_ == nullptr && dlsym(RTLD_DEFAULT, "aclrtMemcpy") == nullptr) {
            throw std::runtime_error(std::string("failed to load libascendcl.so: ") + dlerror());
        }
        aclInit = reinterpret_cast<int (*)(const char *)>(resolve_symbol("aclInit"));
        aclrtSetDevice = reinterpret_cast<int (*)(int)>(resolve_symbol("aclrtSetDevice"));
        aclrtGetDevice = reinterpret_cast<int (*)(int *)>(resolve_symbol("aclrtGetDevice"));
        aclrtMemcpy =
            reinterpret_cast<int (*)(void *, size_t, const void *, size_t, int)>(resolve_symbol("aclrtMemcpy"));
        aclrtMemGetAllocationGranularity = reinterpret_cast<int (*)(LocalAclPhysicalMemProp *, int, size_t *)>(
            resolve_symbol("aclrtMemGetAllocationGranularity")
        );
        aclrtMallocPhysical = reinterpret_cast<int (*)(void **, size_t, const LocalAclPhysicalMemProp *, uint64_t)>(
            resolve_symbol("aclrtMallocPhysical")
        );
        aclrtFreePhysical = reinterpret_cast<int (*)(void *)>(resolve_symbol("aclrtFreePhysical"));
        aclrtReserveMemAddress = reinterpret_cast<int (*)(void **, size_t, size_t, void *, uint64_t)>(
            resolve_symbol("aclrtReserveMemAddress")
        );
        aclrtReleaseMemAddress = reinterpret_cast<int (*)(void *)>(resolve_symbol("aclrtReleaseMemAddress"));
        aclrtMapMem =
            reinterpret_cast<int (*)(void *, size_t, size_t, void *, uint64_t)>(resolve_symbol("aclrtMapMem"));
        aclrtUnmapMem = reinterpret_cast<int (*)(void *)>(resolve_symbol("aclrtUnmapMem"));
        aclrtMemSetAccess = reinterpret_cast<int (*)(void *, size_t, LocalAclMemAccessDesc *, size_t)>(
            resolve_symbol("aclrtMemSetAccess")
        );
        aclrtMemExportToShareableHandle = reinterpret_cast<int (*)(void *, int, uint64_t, uint64_t *)>(
            resolve_symbol("aclrtMemExportToShareableHandle")
        );
        aclrtMemImportFromShareableHandle =
            reinterpret_cast<int (*)(uint64_t, int32_t, void **)>(resolve_symbol("aclrtMemImportFromShareableHandle"));
    }

    void init() {
        // load() throws when aclInit cannot be resolved, so it is always set here.
        if (initialized_) {
            return;
        }
        int rc = aclInit(nullptr);
        if (rc != kAclSuccess) {
            throw std::runtime_error("aclInit failed with code " + std::to_string(rc));
        }
        initialized_ = true;
    }

    void bind_device_with_check(int device_id) const { acl_check(aclrtSetDevice(device_id), "aclrtSetDevice"); }

    int current_device_with_check() const {
        int device_id = -1;
        acl_check(aclrtGetDevice(&device_id), "aclrtGetDevice");
        return device_id;
    }

    void memcpy_h2d_with_check(void *dst, size_t dst_size, const void *src, size_t count) const {
        acl_check(aclrtMemcpy(dst, dst_size, src, count, kAclMemcpyHostToDevice), "aclrtMemcpy H2D");
    }

    void memcpy_d2h_with_check(void *dst, size_t dst_size, const void *src, size_t count) const {
        acl_check(aclrtMemcpy(dst, dst_size, src, count, kAclMemcpyDeviceToHost), "aclrtMemcpy D2H");
    }

    uint64_t vmm_granularity_with_check(int device_id) const {
        LocalAclPhysicalMemProp prop{};
        prop.handleType = kAclMemHandleTypeNone;
        prop.allocationType = kAclMemAllocationTypePinned;
        prop.memAttr = kAclHbmMemNormal;
        prop.location.id = static_cast<uint32_t>(device_id);
        prop.location.type = kAclMemLocationTypeDevice;
        size_t granularity = 0;
        acl_check(
            aclrtMemGetAllocationGranularity(&prop, kAclRtMemAllocGranularityMinimum, &granularity),
            "aclrtMemGetAllocationGranularity"
        );
        return static_cast<uint64_t>(granularity);
    }

    void *vmm_malloc_physical_with_check(uint64_t bytes, int device_id) const {
        LocalAclPhysicalMemProp prop{};
        prop.handleType = kAclMemHandleTypeNone;
        prop.allocationType = kAclMemAllocationTypePinned;
        prop.memAttr = kAclHbmMemNormal;
        prop.location.id = static_cast<uint32_t>(device_id);
        prop.location.type = kAclMemLocationTypeDevice;
        void *handle = nullptr;
        acl_check(aclrtMallocPhysical(&handle, static_cast<size_t>(bytes), &prop, 0), "aclrtMallocPhysical");
        return handle;
    }

    void *vmm_reserve_with_check(uint64_t bytes) const {
        void *va = nullptr;
        acl_check(aclrtReserveMemAddress(&va, static_cast<size_t>(bytes), 0, nullptr, 0), "aclrtReserveMemAddress");
        return va;
    }

    void vmm_map_with_check(void *va, uint64_t bytes, void *handle) const {
        acl_check(aclrtMapMem(va, static_cast<size_t>(bytes), 0, handle, 0), "aclrtMapMem");
    }

    void vmm_set_access_with_check(void *va, uint64_t bytes, int device_id) const {
        LocalAclMemAccessDesc access{};
        access.flags = kAclRtMemAccessReadwrite;
        access.location.type = kAclMemLocationTypeDevice;
        access.location.id = static_cast<uint32_t>(device_id);
        acl_check(aclrtMemSetAccess(va, static_cast<size_t>(bytes), &access, 1), "aclrtMemSetAccess");
    }

    uint64_t vmm_export_shareable_with_check(void *handle) const {
        uint64_t shareable = 0;
        acl_check(
            aclrtMemExportToShareableHandle(
                handle, kAclMemHandleTypeNone, kAclRtVmmExportFlagDisablePidValidation, &shareable
            ),
            "aclrtMemExportToShareableHandle"
        );
        return shareable;
    }

    void *vmm_import_shareable_with_check(uint64_t shareable, int device_id) const {
        void *handle = nullptr;
        acl_check(
            aclrtMemImportFromShareableHandle(shareable, device_id, &handle), "aclrtMemImportFromShareableHandle"
        );
        return handle;
    }

    int vmm_unmap(void *va) const { return aclrtUnmapMem(va); }

    int vmm_release_address(void *va) const { return aclrtReleaseMemAddress(va); }

    int vmm_free_physical(void *handle) const { return aclrtFreePhysical(handle); }

    void vmm_zero_bytes(void *va, uint64_t offset, uint64_t nbytes) const {
        std::vector<uint8_t> zeros(static_cast<size_t>(nbytes), 0);
        void *dst = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(va) + static_cast<uintptr_t>(offset));
        memcpy_h2d_with_check(dst, static_cast<size_t>(nbytes), zeros.data(), static_cast<size_t>(nbytes));
    }

    void vmm_release_collecting(void *va, void *handle, std::string &cleanup_error) const {
        if (va != nullptr) {
            int rc = aclrtUnmapMem(va);
            if (rc != kAclSuccess) {
                append_cleanup_error(cleanup_error, "aclrtUnmapMem failed with code " + std::to_string(rc));
            }
            rc = aclrtReleaseMemAddress(va);
            if (rc != kAclSuccess) {
                append_cleanup_error(cleanup_error, "aclrtReleaseMemAddress failed with code " + std::to_string(rc));
            }
        }
        if (handle != nullptr) {
            int rc = aclrtFreePhysical(handle);
            if (rc != kAclSuccess) {
                append_cleanup_error(cleanup_error, "aclrtFreePhysical failed with code " + std::to_string(rc));
            }
        }
    }

private:
    static constexpr int kAclSuccess = 0;
    static constexpr int kAclMemcpyHostToDevice = 1;
    static constexpr int kAclMemcpyDeviceToHost = 2;
    static constexpr int kAclMemHandleTypeNone = 0;
    static constexpr int kAclMemAllocationTypePinned = 0;
    static constexpr int kAclHbmMemNormal = 5;
    static constexpr int kAclMemLocationTypeDevice = 1;
    static constexpr int kAclRtMemAllocGranularityMinimum = 0;
    static constexpr int kAclRtMemAccessReadwrite = 0x3;
    static constexpr uint64_t kAclRtVmmExportFlagDisablePidValidation = 0x1ULL;

    void *lib_{nullptr};
    bool initialized_{false};

    void *resolve_symbol(const char *name) const {
        void *sym = dlsym(RTLD_DEFAULT, name);
        if (sym == nullptr && lib_ != nullptr) {
            sym = dlsym(lib_, name);
        }
        if (sym == nullptr) {
            throw std::runtime_error(std::string("CANN ACL symbol not found: ") + name);
        }
        return sym;
    }

    static void acl_check(int rc, const char *op) {
        if (rc != kAclSuccess) {
            throw std::runtime_error(std::string(op) + " failed with code " + std::to_string(rc));
        }
    }
};

AclRuntimeApi &acl_api() {
    static std::once_flag once;
    // Intentionally process-lifetime: late Python finalizers may still need
    // the initialized ACL dispatch table after ordinary static destruction
    // begins, so deleting it would reintroduce a destruction-order use-after-free.
    static AclRuntimeApi *api{nullptr};
    std::call_once(once, []() {
        auto candidate = std::make_unique<AclRuntimeApi>();
        candidate->load();
        candidate->init();
        api = candidate.release();
    });
    return *api;
}

std::string region_shm_name_for_open(const std::string &token) {
    if (token.empty()) {
        throw std::invalid_argument("mapped-region sim backing shm token must be non-empty");
    }
    if (token[0] == '/') {
        return token;
    }
    return "/" + token;
}

WorkerChipOrchNotifyOp checked_region_notify_op(int op) {
    auto typed = static_cast<WorkerChipOrchNotifyOp>(op);
    if (!worker_chip_orch_comm::valid_notify_op(typed)) {
        throw std::invalid_argument("region counter notify op is invalid");
    }
    return typed;
}

WorkerChipOrchWaitCmp checked_region_wait_cmp(int cmp) {
    auto typed = static_cast<WorkerChipOrchWaitCmp>(cmp);
    if (!worker_chip_orch_comm::valid_wait_cmp(typed)) {
        throw std::invalid_argument("region counter wait comparison is invalid");
    }
    return typed;
}

class RegionCleanupErrors {
public:
    void record(const std::string &owner_token, const std::string &message) noexcept {
        try {
            std::lock_guard<std::mutex> lk(mu_);
            append_cleanup_error(errors_[owner_token], message);
        } catch (...) {}
    }

    std::string take(const std::string &owner_token) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = errors_.find(owner_token);
        if (it == errors_.end()) {
            return {};
        }
        std::string error = std::move(it->second);
        errors_.erase(it);
        return error;
    }

    std::string peek(const std::string &owner_token) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = errors_.find(owner_token);
        return it == errors_.end() ? std::string{} : it->second;
    }

    void acknowledge(const std::string &owner_token, const std::string &observed) {
        if (observed.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lk(mu_);
        auto it = errors_.find(owner_token);
        if (it == errors_.end()) {
            return;
        }
        if (it->second == observed) {
            errors_.erase(it);
            return;
        }
        if (it->second.size() > observed.size() + 2 && it->second.compare(0, observed.size(), observed) == 0 &&
            it->second.compare(observed.size(), 2, "; ") == 0) {
            it->second.erase(0, observed.size() + 2);
        }
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<std::string, std::string> errors_;
};

RegionCleanupErrors &region_cleanup_errors() {
    static auto *errors = new RegionCleanupErrors();
    return *errors;
}

enum class RegionMappingProfile { SimPosixShm, OnboardVmm };

class RegionMapping {
public:
    RegionMapping() = default;
    RegionMapping(const RegionMapping &) = delete;
    RegionMapping &operator=(const RegionMapping &) = delete;

    ~RegionMapping() noexcept {
        try {
            std::string cleanup_error;
            close_collecting(cleanup_error);
            if (!cleanup_error.empty()) {
                region_cleanup_errors().record(owner_token, cleanup_error);
            }
        } catch (...) {
            region_cleanup_errors().record(owner_token, "mapped-region cleanup failed with an unknown error");
        }
    }

    std::string owner_token;
    RegionMappingProfile profile{RegionMappingProfile::SimPosixShm};
    int fd{-1};
    uint64_t device_addr{0};
    int device_id{-1};
    uint64_t shareable_handle{0};
    void *vmm_handle{nullptr};
    uint64_t mapping_bytes{0};

    void close() {
        std::string cleanup_error;
        close_collecting(cleanup_error);
        if (!cleanup_error.empty()) {
            throw std::runtime_error(cleanup_error);
        }
    }

    void bind_acl_device() const {
        if (device_id < 0) {
            throw std::runtime_error("onboard mapped-region handle has no device id");
        }
        acl_api().bind_device_with_check(device_id);
    }

    void validate_mapping_range_or_throw(uint64_t offset, uint64_t nbytes) const {
        if (nbytes == 0 || offset > mapping_bytes || nbytes > mapping_bytes - offset) {
            throw std::out_of_range("mapped-region byte access is out of range");
        }
    }

    void copy_to(uint64_t offset, const void *host_ptr, uint64_t nbytes) const {
        validate_mapping_range_or_throw(offset, nbytes);
        if (profile == RegionMappingProfile::SimPosixShm) {
            auto *dst = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(device_addr));
            std::memcpy(dst + offset, host_ptr, static_cast<size_t>(nbytes));
            return;
        }
        bind_acl_device();
        void *dst = reinterpret_cast<void *>(static_cast<uintptr_t>(device_addr + offset));
        acl_api().memcpy_h2d_with_check(dst, static_cast<size_t>(nbytes), host_ptr, static_cast<size_t>(nbytes));
    }

    void copy_from(void *host_ptr, uint64_t offset, uint64_t nbytes) const {
        validate_mapping_range_or_throw(offset, nbytes);
        if (profile == RegionMappingProfile::SimPosixShm) {
            const auto *src = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(device_addr));
            std::memcpy(host_ptr, src + offset, static_cast<size_t>(nbytes));
            return;
        }
        bind_acl_device();
        const void *src = reinterpret_cast<const void *>(static_cast<uintptr_t>(device_addr + offset));
        acl_api().memcpy_d2h_with_check(host_ptr, static_cast<size_t>(nbytes), src, static_cast<size_t>(nbytes));
    }

    int32_t load_counter(uint64_t offset) const {
        int32_t value = 0;
        copy_from(&value, offset, sizeof(value));
        return value;
    }

    void store_counter(uint64_t offset, int32_t value) const { copy_to(offset, &value, sizeof(value)); }

    void notify_counter(uint64_t offset, int32_t value, WorkerChipOrchNotifyOp op) const {
        if (offset % sizeof(int32_t) != 0) {
            throw std::invalid_argument("region counter offset must be 4-byte aligned");
        }
        if (!worker_chip_orch_comm::valid_notify_op(op)) {
            throw std::invalid_argument("region counter notify op is invalid");
        }
        if (op == WorkerChipOrchNotifyOp::Add) {
            value = load_counter(offset) + value;
        }
        store_counter(offset, value);
    }

    std::tuple<bool, int32_t> test_counter(uint64_t offset, int32_t operand, WorkerChipOrchWaitCmp cmp) const {
        if (offset % sizeof(int32_t) != 0) {
            throw std::invalid_argument("region counter offset must be 4-byte aligned");
        }
        if (!worker_chip_orch_comm::valid_wait_cmp(cmp)) {
            throw std::invalid_argument("region counter wait comparison is invalid");
        }
        int32_t observed = load_counter(offset);
        return std::make_tuple(worker_chip_orch_comm::compare_counter(observed, operand, cmp), observed);
    }

    std::tuple<int, int, int32_t, bool, std::string>
    wait_counter(uint64_t offset, int32_t operand, WorkerChipOrchWaitCmp cmp, uint64_t timeout_ns) const {
        if (offset % sizeof(int32_t) != 0) {
            throw std::invalid_argument("region counter offset must be 4-byte aligned");
        }
        if (!worker_chip_orch_comm::valid_wait_cmp(cmp)) {
            throw std::invalid_argument("region counter wait comparison is invalid");
        }
        auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(timeout_ns);
        while (true) {
            int32_t observed = load_counter(offset);
            bool matched = worker_chip_orch_comm::compare_counter(observed, operand, cmp);
            if (matched) {
                return std::make_tuple(kWaitStatusOk, kWaitErrorNone, observed, true, std::string{});
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                return std::make_tuple(
                    kWaitStatusTimeout, kWaitErrorSignalTimeout, observed, false, std::string{"SIGNAL_WAIT timed out"}
                );
            }
            std::this_thread::sleep_for(std::chrono::nanoseconds(kWaitPollIntervalNs));
        }
    }

private:
    void close_collecting(std::string &cleanup_error) {
        uint64_t mapped_addr = std::exchange(device_addr, 0);
        uint64_t mapped_bytes = std::exchange(mapping_bytes, 0);
        void *physical_handle = std::exchange(vmm_handle, nullptr);
        int mapped_device_id = std::exchange(device_id, -1);
        int mapped_fd = std::exchange(fd, -1);

        if (profile == RegionMappingProfile::OnboardVmm) {
            if (mapped_addr == 0 && physical_handle == nullptr) {
                return;
            }
            try {
                if (mapped_device_id < 0) {
                    throw std::runtime_error("onboard mapped-region handle has no device id");
                }
                AclRuntimeApi &api = acl_api();
                api.bind_device_with_check(mapped_device_id);
                api.vmm_release_collecting(
                    reinterpret_cast<void *>(static_cast<uintptr_t>(mapped_addr)), physical_handle, cleanup_error
                );
            } catch (const std::exception &exc) {
                append_cleanup_error(cleanup_error, exc.what());
            } catch (...) {
                append_cleanup_error(cleanup_error, "onboard mapped-region cleanup failed");
            }
            return;
        }

        if (mapped_addr != 0 &&
            munmap(reinterpret_cast<void *>(static_cast<uintptr_t>(mapped_addr)), mapped_bytes) != 0) {
            int err = errno;
            append_cleanup_error(cleanup_error, std::string("sim mapped-region munmap failed: ") + std::strerror(err));
        }
        if (mapped_fd >= 0 && ::close(mapped_fd) != 0) {
            int err = errno;
            append_cleanup_error(cleanup_error, std::string("sim mapped-region close failed: ") + std::strerror(err));
        }
    }

    static constexpr int kWaitStatusOk = 0;
    static constexpr int kWaitStatusTimeout = -1;
    static constexpr int kWaitErrorNone = 0;
    static constexpr int kWaitErrorSignalTimeout = 7;
    static constexpr int64_t kWaitPollIntervalNs = 50000;
};

class RegionEntry {
public:
    explicit RegionEntry(std::unique_ptr<RegionMapping> mapping) :
        mapping_(std::move(mapping)) {}

    void acquire() {
        std::lock_guard<std::mutex> lk(mu_);
        if (state_ != State::OPEN) {
            throw std::runtime_error("mapped-region handle is closed or unknown");
        }
        active_leases_ += 1;
    }

    void release() noexcept {
        std::lock_guard<std::mutex> lk(mu_);
        if (active_leases_ == 0) {
            return;
        }
        active_leases_ -= 1;
        if (active_leases_ == 0) {
            idle_.notify_all();
        }
    }

    RegionMapping &mapping() { return *mapping_; }

    size_t active_leases() const {
        std::lock_guard<std::mutex> lk(mu_);
        return active_leases_;
    }

    bool closing() const {
        std::lock_guard<std::mutex> lk(mu_);
        return state_ == State::CLOSING;
    }

    void close() {
        std::unique_ptr<RegionMapping> mapping;
        std::exception_ptr close_error;
        {
            std::unique_lock<std::mutex> lk(mu_);
            if (state_ != State::OPEN) {
                idle_.wait(lk, [this]() {
                    return state_ == State::CLOSED;
                });
                close_error = close_error_;
                lk.unlock();
                if (close_error != nullptr) {
                    std::rethrow_exception(close_error);
                }
                return;
            }
            state_ = State::CLOSING;
            idle_.wait(lk, [this]() {
                return active_leases_ == 0;
            });
            mapping = std::move(mapping_);
        }

        try {
            if (mapping != nullptr) {
                mapping->close();
            }
        } catch (...) {
            close_error = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            close_error_ = close_error;
            state_ = State::CLOSED;
        }
        idle_.notify_all();
        if (close_error != nullptr) {
            std::rethrow_exception(close_error);
        }
    }

private:
    enum class State { OPEN, CLOSING, CLOSED };

    std::unique_ptr<RegionMapping> mapping_;
    mutable std::mutex mu_;
    std::condition_variable idle_;
    size_t active_leases_{0};
    State state_{State::OPEN};
    std::exception_ptr close_error_;
};

class RegionLease {
public:
    explicit RegionLease(std::shared_ptr<RegionEntry> entry) :
        entry_(std::move(entry)) {
        entry_->acquire();
    }
    RegionLease(const RegionLease &) = delete;
    RegionLease &operator=(const RegionLease &) = delete;
    RegionLease(RegionLease &&) noexcept = default;
    RegionLease &operator=(RegionLease &&) = delete;
    ~RegionLease() {
        if (entry_ != nullptr) {
            entry_->release();
        }
    }

    RegionMapping *operator->() { return &entry_->mapping(); }

private:
    std::shared_ptr<RegionEntry> entry_;
};

class RegionRegistry {
public:
    uint64_t emplace(std::unique_ptr<RegionMapping> mapping) {
        auto entry = std::make_shared<RegionEntry>(std::move(mapping));
        std::lock_guard<std::mutex> lk(mu_);
        if (std::exchange(fail_next_insert_for_test_, false)) {
            throw std::runtime_error("injected mapped-region registry insertion failure");
        }
        uint64_t handle = next_handle_;
        auto result = regions_.emplace(handle, std::move(entry));
        if (!result.second) {
            throw std::overflow_error("mapped-region handle space is exhausted");
        }
        next_handle_ += 1;
        if (next_handle_ == 0) {
            next_handle_ = 1;
        }
        return handle;
    }

    RegionLease lease(uint64_t handle) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = regions_.find(handle);
        if (it == regions_.end()) {
            throw std::runtime_error("mapped-region handle is closed or unknown");
        }
        return RegionLease(it->second);
    }

    size_t active_leases(uint64_t handle) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = regions_.find(handle);
        if (it == regions_.end()) {
            throw std::runtime_error("mapped-region handle is closed or unknown");
        }
        return it->second->active_leases();
    }

    bool closing(uint64_t handle) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = regions_.find(handle);
        if (it == regions_.end()) {
            throw std::runtime_error("mapped-region handle is closed or unknown");
        }
        return it->second->closing();
    }

    void close(uint64_t handle) {
        std::shared_ptr<RegionEntry> entry;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = regions_.find(handle);
            if (it == regions_.end()) {
                return;
            }
            entry = it->second;
        }

        std::exception_ptr close_error;
        try {
            entry->close();
        } catch (...) {
            close_error = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = regions_.find(handle);
            if (it != regions_.end() && it->second == entry) {
                regions_.erase(it);
            }
        }
        if (close_error != nullptr) {
            std::rethrow_exception(close_error);
        }
    }

    void fail_next_insert_for_test() {
        std::lock_guard<std::mutex> lk(mu_);
        fail_next_insert_for_test_ = true;
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<uint64_t, std::shared_ptr<RegionEntry>> regions_;
    uint64_t next_handle_{1};
    bool fail_next_insert_for_test_{false};
};

RegionRegistry &region_registry() {
    static auto *registry = new RegionRegistry();
    return *registry;
}

void close_region(uint64_t handle) { region_registry().close(handle); }

class RegionHandle {
public:
    explicit RegionHandle(uint64_t handle, std::string owner_token) :
        handle_(handle),
        owner_token_(std::move(owner_token)) {}
    RegionHandle(const RegionHandle &) = delete;
    RegionHandle &operator=(const RegionHandle &) = delete;
    RegionHandle(RegionHandle &&other) noexcept :
        handle_(std::exchange(other.handle_, 0)),
        owner_token_(std::move(other.owner_token_)) {}
    RegionHandle &operator=(RegionHandle &&) = delete;

    ~RegionHandle() noexcept {
        if (handle_ == 0) {
            return;
        }
        try {
            close_region(handle_);
        } catch (const std::exception &exc) {
            region_cleanup_errors().record(owner_token_, exc.what());
        } catch (...) {
            region_cleanup_errors().record(owner_token_, "mapped-region owner cleanup failed with an unknown error");
        }
    }

    uint64_t value() const { return handle_; }

private:
    uint64_t handle_{0};
    std::string owner_token_;
};

RegionHandle import_sim_region(
    const std::string &token, uint64_t mapping_bytes, const std::string &owner_token, const char *size_error,
    const char *owner_error, const char *open_error, const char *mmap_error
) {
    if (mapping_bytes == 0 || mapping_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::invalid_argument(size_error);
    }
    if (owner_token.empty()) {
        throw std::invalid_argument(owner_error);
    }
    std::string handle_owner_token = owner_token;
    std::string name = region_shm_name_for_open(token);
    auto mapping = std::make_unique<RegionMapping>();
    mapping->owner_token = owner_token;
    mapping->fd = shm_open(name.c_str(), O_RDWR, 0);
    if (mapping->fd < 0) {
        throw std::runtime_error(open_error);
    }
    void *base = mmap(nullptr, static_cast<size_t>(mapping_bytes), PROT_READ | PROT_WRITE, MAP_SHARED, mapping->fd, 0);
    if (base == MAP_FAILED) {
        int err = errno;
        throw std::runtime_error(std::string(mmap_error) + std::strerror(err));
    }
    mapping->profile = RegionMappingProfile::SimPosixShm;
    mapping->device_addr = reinterpret_cast<uint64_t>(base);
    mapping->mapping_bytes = mapping_bytes;
    uint64_t handle = region_registry().emplace(std::move(mapping));
    return RegionHandle(handle, std::move(handle_owner_token));
}

RegionHandle import_onboard_region(
    int device_id, uint64_t shareable_handle, uint64_t mapping_bytes, const std::string &owner_token,
    const char *device_error, const char *size_error, const char *owner_error
) {
    if (device_id < 0) {
        throw std::invalid_argument(device_error);
    }
    if (mapping_bytes == 0 || mapping_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::invalid_argument(size_error);
    }
    if (owner_token.empty()) {
        throw std::invalid_argument(owner_error);
    }
    std::string handle_owner_token = owner_token;
    auto mapping = std::make_unique<RegionMapping>();
    mapping->owner_token = owner_token;
    mapping->profile = RegionMappingProfile::OnboardVmm;
    mapping->device_id = device_id;
    mapping->mapping_bytes = mapping_bytes;
    mapping->bind_acl_device();
    AclRuntimeApi &api = acl_api();
    mapping->shareable_handle = shareable_handle;
    mapping->vmm_handle = api.vmm_import_shareable_with_check(shareable_handle, device_id);
    void *mapped_addr = api.vmm_reserve_with_check(mapping_bytes);
    mapping->device_addr = reinterpret_cast<uint64_t>(mapped_addr);
    api.vmm_map_with_check(mapped_addr, mapping_bytes, mapping->vmm_handle);
    api.vmm_set_access_with_check(mapped_addr, mapping_bytes, device_id);
    uint64_t handle = region_registry().emplace(std::move(mapping));
    return RegionHandle(handle, std::move(handle_owner_token));
}

uint64_t align_vmm_bytes(uint64_t bytes, uint64_t granularity) {
    if (bytes == 0 || bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::invalid_argument("L3-L2 onboard VMM region requires a positive mapping size");
    }
    if (granularity == 0) {
        return bytes;
    }
    uint64_t remainder = bytes % granularity;
    if (remainder == 0) {
        return bytes;
    }
    uint64_t bump = granularity - remainder;
    if (bytes > std::numeric_limits<uint64_t>::max() - bump) {
        throw std::overflow_error("L3-L2 onboard VMM mapping size overflowed");
    }
    return bytes + bump;
}

void append_cleanup_error(std::string &cleanup_error, const std::string &message) {
    if (!cleanup_error.empty()) {
        cleanup_error += "; ";
    }
    cleanup_error += message;
}

enum class RegionVmmStage {
    None = 0,
    PhysicalAlloc,
    VaReserve,
    Map,
    SetAccess,
    Export,
    Unmap,
    VaRelease,
    PhysicalFree,
    ZeroBytes,
    BindDevice,
};

enum class RegionVmmFailWhen { Before, After };

enum class RegionVmmUnmapClass { Success, AlreadyGone, Failed };

struct RegionVmmExport {
    uint64_t device_addr{0};
    uint64_t mapping_bytes{0};
    uint64_t shareable_handle{0};
    uint64_t registry_handle{0};
};

struct RegionVmmInspect {
    bool present{false};
    int device_id{-1};
    uint64_t device_addr{0};
    uint64_t mapping_bytes{0};
    uint64_t shareable_handle{0};
    uint64_t physical_handle{0};
    bool physical_allocated{false};
    bool va_reserved{false};
    bool mapped{false};
    bool access_set{false};
    bool exported{false};
    bool unmap_attempted{false};
    bool unmap_complete{false};
    bool va_release_attempted{false};
    bool va_release_complete{false};
    bool physical_free_attempted{false};
    bool physical_free_complete{false};
    bool release_once_done{false};
    std::string first_cleanup_failure;
    std::vector<std::string> local_cleanup_details;
};

struct RegionVmmAllocation {
    int device_id{-1};
    void *physical_handle{nullptr};
    void *va{nullptr};
    uint64_t mapping_bytes{0};
    uint64_t shareable_handle{0};
    bool physical_allocated{false};
    bool va_reserved{false};
    bool mapped{false};
    bool access_set{false};
    bool exported{false};
    bool unmap_attempted{false};
    bool unmap_complete{false};
    bool va_release_attempted{false};
    bool va_release_complete{false};
    bool physical_free_attempted{false};
    bool physical_free_complete{false};
    bool release_once_done{false};
    std::string first_cleanup_failure;
    std::vector<std::string> local_cleanup_details;
};

struct RegionVmmTestHooks {
    bool use_fake{false};
    std::optional<RegionVmmStage> fail_stage;
    RegionVmmFailWhen fail_when{RegionVmmFailWhen::Before};
    bool unmap_already_gone{false};
    std::vector<std::string> issued_ops;
    uint64_t fake_granularity{64};
    uint64_t next_fake_id{1};
    uint64_t next_shareable{1};
    std::unordered_map<uint64_t, std::vector<uint8_t>> fake_va_bytes;
};

std::mutex &region_vmm_mu() {
    static auto *mu = new std::mutex();
    return *mu;
}

RegionVmmTestHooks &region_vmm_test_hooks() {
    static auto *hooks = new RegionVmmTestHooks();
    return *hooks;
}

std::unordered_map<uint64_t, RegionVmmAllocation> &g_region_vmm_allocations() {
    static auto *allocations = new std::unordered_map<uint64_t, RegionVmmAllocation>();
    return *allocations;
}

uint64_t &region_vmm_next_handle() {
    static auto *next = new uint64_t(1);
    return *next;
}

const char *region_vmm_stage_name(RegionVmmStage stage) {
    switch (stage) {
    case RegionVmmStage::PhysicalAlloc:
        return "physical_alloc";
    case RegionVmmStage::VaReserve:
        return "va_reserve";
    case RegionVmmStage::Map:
        return "map";
    case RegionVmmStage::SetAccess:
        return "set_access";
    case RegionVmmStage::Export:
        return "export";
    case RegionVmmStage::Unmap:
        return "unmap";
    case RegionVmmStage::VaRelease:
        return "va_release";
    case RegionVmmStage::PhysicalFree:
        return "physical_free";
    case RegionVmmStage::ZeroBytes:
        return "zero_bytes";
    case RegionVmmStage::BindDevice:
        return "bind_device";
    case RegionVmmStage::None:
        return "none";
    }
    return "none";
}

RegionVmmStage region_vmm_parse_stage(const std::string &name) {
    if (name == "physical_alloc") {
        return RegionVmmStage::PhysicalAlloc;
    }
    if (name == "va_reserve") {
        return RegionVmmStage::VaReserve;
    }
    if (name == "map") {
        return RegionVmmStage::Map;
    }
    if (name == "set_access") {
        return RegionVmmStage::SetAccess;
    }
    if (name == "export") {
        return RegionVmmStage::Export;
    }
    if (name == "unmap") {
        return RegionVmmStage::Unmap;
    }
    if (name == "va_release") {
        return RegionVmmStage::VaRelease;
    }
    if (name == "physical_free") {
        return RegionVmmStage::PhysicalFree;
    }
    if (name == "zero_bytes") {
        return RegionVmmStage::ZeroBytes;
    }
    if (name == "bind_device") {
        return RegionVmmStage::BindDevice;
    }
    throw std::invalid_argument("unknown region VMM stage: " + name);
}

void region_vmm_record_issued(const char *op) { region_vmm_test_hooks().issued_ops.emplace_back(op); }

bool region_vmm_consume_fail(RegionVmmStage stage, RegionVmmFailWhen when) {
    RegionVmmTestHooks &hooks = region_vmm_test_hooks();
    if (!hooks.fail_stage.has_value() || *hooks.fail_stage != stage || hooks.fail_when != when) {
        return false;
    }
    hooks.fail_stage.reset();
    return true;
}

void region_vmm_throw_stage_failure(RegionVmmStage stage, const char *when) {
    throw std::runtime_error(
        std::string("injected region VMM ") + when + " " + region_vmm_stage_name(stage) + " failure"
    );
}

void *region_vmm_next_fake_ptr() {
    RegionVmmTestHooks &hooks = region_vmm_test_hooks();
    uint64_t id = hooks.next_fake_id++;
    return reinterpret_cast<void *>(static_cast<uintptr_t>(id) << 12);
}

void region_vmm_bind_device(int device_id) {
    region_vmm_record_issued("bind_device");
    if (region_vmm_consume_fail(RegionVmmStage::BindDevice, RegionVmmFailWhen::Before)) {
        region_vmm_throw_stage_failure(RegionVmmStage::BindDevice, "before");
    }
    if (region_vmm_test_hooks().use_fake) {
        if (device_id < 0) {
            throw std::runtime_error("region VMM fake driver requires a non-negative device id");
        }
        return;
    }
    acl_api().bind_device_with_check(device_id);
}

uint64_t region_vmm_granularity(int device_id) {
    region_vmm_record_issued("granularity");
    if (region_vmm_test_hooks().use_fake) {
        return region_vmm_test_hooks().fake_granularity;
    }
    return acl_api().vmm_granularity_with_check(device_id);
}

void *region_vmm_malloc_physical(uint64_t bytes, int device_id) {
    region_vmm_record_issued("physical_alloc");
    if (region_vmm_test_hooks().use_fake) {
        (void)bytes;
        (void)device_id;
        return region_vmm_next_fake_ptr();
    }
    return acl_api().vmm_malloc_physical_with_check(bytes, device_id);
}

void *region_vmm_reserve(uint64_t bytes) {
    region_vmm_record_issued("va_reserve");
    if (region_vmm_test_hooks().use_fake) {
        void *va = region_vmm_next_fake_ptr();
        region_vmm_test_hooks().fake_va_bytes[reinterpret_cast<uint64_t>(va)] =
            std::vector<uint8_t>(static_cast<size_t>(bytes), 0xff);
        return va;
    }
    return acl_api().vmm_reserve_with_check(bytes);
}

void region_vmm_map(void *va, uint64_t bytes, void *handle) {
    region_vmm_record_issued("map");
    if (region_vmm_test_hooks().use_fake) {
        (void)va;
        (void)bytes;
        (void)handle;
        return;
    }
    acl_api().vmm_map_with_check(va, bytes, handle);
}

void region_vmm_set_access(void *va, uint64_t bytes, int device_id) {
    region_vmm_record_issued("set_access");
    if (region_vmm_test_hooks().use_fake) {
        (void)va;
        (void)bytes;
        (void)device_id;
        return;
    }
    acl_api().vmm_set_access_with_check(va, bytes, device_id);
}

uint64_t region_vmm_export_shareable(void *handle) {
    region_vmm_record_issued("export");
    if (region_vmm_test_hooks().use_fake) {
        (void)handle;
        return region_vmm_test_hooks().next_shareable++;
    }
    return acl_api().vmm_export_shareable_with_check(handle);
}

int region_vmm_unmap(void *va) {
    region_vmm_record_issued("unmap");
    RegionVmmTestHooks &hooks = region_vmm_test_hooks();
    if (hooks.use_fake) {
        (void)va;
        if (std::exchange(hooks.unmap_already_gone, false)) {
            return 2;
        }
        return 0;
    }
    return acl_api().vmm_unmap(va);
}

int region_vmm_release_address(void *va) {
    region_vmm_record_issued("va_release");
    if (region_vmm_test_hooks().use_fake) {
        region_vmm_test_hooks().fake_va_bytes.erase(reinterpret_cast<uint64_t>(va));
        return 0;
    }
    return acl_api().vmm_release_address(va);
}

int region_vmm_free_physical(void *handle) {
    region_vmm_record_issued("physical_free");
    if (region_vmm_test_hooks().use_fake) {
        (void)handle;
        return 0;
    }
    return acl_api().vmm_free_physical(handle);
}

void region_vmm_zero_bytes(void *va, uint64_t offset, uint64_t nbytes) {
    region_vmm_record_issued("zero_bytes");
    if (region_vmm_test_hooks().use_fake) {
        auto &backing = region_vmm_test_hooks().fake_va_bytes;
        auto it = backing.find(reinterpret_cast<uint64_t>(va));
        if (it == backing.end()) {
            throw std::runtime_error("region VMM fake zero_bytes has no reserved VA");
        }
        uint64_t end = offset + nbytes;
        if (end < offset || end > static_cast<uint64_t>(it->second.size())) {
            throw std::invalid_argument("region VMM fake zero_bytes range exceeds mapping");
        }
        std::fill(
            it->second.begin() + static_cast<std::ptrdiff_t>(offset),
            it->second.begin() + static_cast<std::ptrdiff_t>(end), static_cast<uint8_t>(0)
        );
        return;
    }
    acl_api().vmm_zero_bytes(va, offset, nbytes);
}

RegionVmmUnmapClass region_vmm_classify_unmap(int rc) {
    if (rc == 0) {
        return RegionVmmUnmapClass::Success;
    }
    if (region_vmm_test_hooks().use_fake && rc == 2) {
        return RegionVmmUnmapClass::AlreadyGone;
    }
    return RegionVmmUnmapClass::Failed;
}

void region_vmm_note_cleanup_failure(RegionVmmAllocation &record, const std::string &detail) {
    record.local_cleanup_details.push_back(detail);
    if (record.first_cleanup_failure.empty()) {
        record.first_cleanup_failure = detail;
    }
}

bool region_vmm_cleanup_complete(const RegionVmmAllocation &record) {
    if (record.mapped && !record.unmap_complete) {
        return false;
    }
    if (record.va_reserved && !record.va_release_complete) {
        return false;
    }
    if (record.physical_allocated && !record.physical_free_complete) {
        return false;
    }
    return record.first_cleanup_failure.empty();
}

RegionVmmInspect region_vmm_inspect_record(uint64_t handle, const RegionVmmAllocation *record) {
    RegionVmmInspect inspect{};
    if (record == nullptr) {
        return inspect;
    }
    inspect.present = true;
    inspect.device_id = record->device_id;
    inspect.device_addr = reinterpret_cast<uint64_t>(record->va);
    inspect.mapping_bytes = record->mapping_bytes;
    inspect.shareable_handle = record->shareable_handle;
    inspect.physical_handle = reinterpret_cast<uint64_t>(record->physical_handle);
    inspect.physical_allocated = record->physical_allocated;
    inspect.va_reserved = record->va_reserved;
    inspect.mapped = record->mapped;
    inspect.access_set = record->access_set;
    inspect.exported = record->exported;
    inspect.unmap_attempted = record->unmap_attempted;
    inspect.unmap_complete = record->unmap_complete;
    inspect.va_release_attempted = record->va_release_attempted;
    inspect.va_release_complete = record->va_release_complete;
    inspect.physical_free_attempted = record->physical_free_attempted;
    inspect.physical_free_complete = record->physical_free_complete;
    inspect.release_once_done = record->release_once_done;
    inspect.first_cleanup_failure = record->first_cleanup_failure;
    inspect.local_cleanup_details = record->local_cleanup_details;
    (void)handle;
    return inspect;
}

uint64_t region_vmm_begin(int device_id) {
    if (device_id < 0) {
        throw std::invalid_argument("region VMM begin requires a non-negative device id");
    }
    std::lock_guard<std::mutex> lk(region_vmm_mu());
    uint64_t handle = region_vmm_next_handle();
    if (handle == 0) {
        throw std::overflow_error("region VMM handle space is exhausted");
    }
    RegionVmmAllocation record{};
    record.device_id = device_id;
    auto inserted = g_region_vmm_allocations().emplace(handle, record);
    if (!inserted.second) {
        throw std::overflow_error("region VMM handle space is exhausted");
    }
    region_vmm_next_handle() = handle + 1;
    return handle;
}

void region_vmm_run_allocate_stage(
    RegionVmmStage stage, const std::function<void()> &issue, const std::function<void()> &commit
) {
    if (region_vmm_consume_fail(stage, RegionVmmFailWhen::Before)) {
        region_vmm_throw_stage_failure(stage, "before");
    }
    issue();
    commit();
    if (region_vmm_consume_fail(stage, RegionVmmFailWhen::After)) {
        region_vmm_throw_stage_failure(stage, "after");
    }
}

RegionVmmExport region_vmm_allocate_export(uint64_t handle, uint64_t logical_bytes) {
    std::lock_guard<std::mutex> lk(region_vmm_mu());
    auto it = g_region_vmm_allocations().find(handle);
    if (it == g_region_vmm_allocations().end()) {
        throw std::runtime_error("region VMM allocate_export handle is unknown");
    }
    RegionVmmAllocation &record = it->second;
    if (record.physical_allocated || record.va_reserved || record.mapped || record.exported) {
        throw std::runtime_error("region VMM allocate_export already has partial or complete state");
    }
    region_vmm_bind_device(record.device_id);
    uint64_t mapping_bytes = align_vmm_bytes(logical_bytes, region_vmm_granularity(record.device_id));
    record.mapping_bytes = mapping_bytes;

    region_vmm_run_allocate_stage(
        RegionVmmStage::PhysicalAlloc,
        [&]() {
            record.physical_handle = region_vmm_malloc_physical(mapping_bytes, record.device_id);
        },
        [&]() {
            record.physical_allocated = true;
        }
    );
    region_vmm_run_allocate_stage(
        RegionVmmStage::VaReserve,
        [&]() {
            record.va = region_vmm_reserve(mapping_bytes);
        },
        [&]() {
            record.va_reserved = true;
        }
    );
    region_vmm_run_allocate_stage(
        RegionVmmStage::Map,
        [&]() {
            region_vmm_map(record.va, mapping_bytes, record.physical_handle);
        },
        [&]() {
            record.mapped = true;
        }
    );
    region_vmm_run_allocate_stage(
        RegionVmmStage::SetAccess,
        [&]() {
            region_vmm_set_access(record.va, mapping_bytes, record.device_id);
        },
        [&]() {
            record.access_set = true;
        }
    );
    region_vmm_run_allocate_stage(
        RegionVmmStage::Export,
        [&]() {
            record.shareable_handle = region_vmm_export_shareable(record.physical_handle);
        },
        [&]() {
            record.exported = true;
        }
    );

    RegionVmmExport exported{};
    exported.device_addr = reinterpret_cast<uint64_t>(record.va);
    exported.mapping_bytes = record.mapping_bytes;
    exported.shareable_handle = record.shareable_handle;
    exported.registry_handle = handle;
    return exported;
}

bool region_vmm_run_cleanup_stage(
    RegionVmmAllocation &record, RegionVmmStage stage, bool &attempted, bool &complete,
    const std::function<int()> &issue, const std::function<void()> &commit_success
) {
    attempted = true;
    if (region_vmm_consume_fail(stage, RegionVmmFailWhen::Before)) {
        region_vmm_note_cleanup_failure(
            record, std::string("injected region VMM before ") + region_vmm_stage_name(stage) + " failure"
        );
        return false;
    }
    int rc = issue();
    if (stage == RegionVmmStage::Unmap) {
        RegionVmmUnmapClass classified = region_vmm_classify_unmap(rc);
        if (classified == RegionVmmUnmapClass::Failed) {
            region_vmm_note_cleanup_failure(record, std::string("unmap failed with code ") + std::to_string(rc));
            return false;
        }
        commit_success();
        complete = true;
        if (region_vmm_consume_fail(stage, RegionVmmFailWhen::After)) {
            region_vmm_note_cleanup_failure(
                record, std::string("injected region VMM after ") + region_vmm_stage_name(stage) + " failure"
            );
        }
        return true;
    }
    if (rc != 0) {
        region_vmm_note_cleanup_failure(
            record, std::string(region_vmm_stage_name(stage)) + " failed with code " + std::to_string(rc)
        );
        return false;
    }
    commit_success();
    complete = true;
    if (region_vmm_consume_fail(stage, RegionVmmFailWhen::After)) {
        region_vmm_note_cleanup_failure(
            record, std::string("injected region VMM after ") + region_vmm_stage_name(stage) + " failure"
        );
        return false;
    }
    return true;
}

void region_vmm_release(uint64_t handle) {
    std::lock_guard<std::mutex> lk(region_vmm_mu());
    auto it = g_region_vmm_allocations().find(handle);
    if (it == g_region_vmm_allocations().end()) {
        return;
    }
    RegionVmmAllocation &record = it->second;
    if (record.release_once_done) {
        if (!record.first_cleanup_failure.empty()) {
            throw std::runtime_error(record.first_cleanup_failure);
        }
        return;
    }
    record.release_once_done = true;
    try {
        region_vmm_bind_device(record.device_id);
    } catch (const std::exception &exc) {
        region_vmm_note_cleanup_failure(record, std::string("bind_device failed: ") + exc.what());
        throw std::runtime_error(record.first_cleanup_failure);
    }

    bool unmap_ok = true;
    if (record.mapped) {
        unmap_ok = region_vmm_run_cleanup_stage(
            record, RegionVmmStage::Unmap, record.unmap_attempted, record.unmap_complete,
            [&]() {
                return region_vmm_unmap(record.va);
            },
            [&]() {
                record.mapped = false;
            }
        );
    }

    if (record.mapped && !unmap_ok) {
        throw std::runtime_error(record.first_cleanup_failure);
    }

    if (record.va_reserved && !record.va_release_attempted) {
        region_vmm_run_cleanup_stage(
            record, RegionVmmStage::VaRelease, record.va_release_attempted, record.va_release_complete,
            [&]() {
                return region_vmm_release_address(record.va);
            },
            [&]() {
                record.va_reserved = false;
                record.va = nullptr;
            }
        );
    }
    if (record.physical_allocated && !record.physical_free_attempted) {
        region_vmm_run_cleanup_stage(
            record, RegionVmmStage::PhysicalFree, record.physical_free_attempted, record.physical_free_complete,
            [&]() {
                return region_vmm_free_physical(record.physical_handle);
            },
            [&]() {
                record.physical_allocated = false;
                record.physical_handle = nullptr;
            }
        );
    }

    if (region_vmm_cleanup_complete(record)) {
        g_region_vmm_allocations().erase(it);
        return;
    }
    throw std::runtime_error(
        record.first_cleanup_failure.empty() ? "region VMM release left uncleared ledger state" :
                                               record.first_cleanup_failure
    );
}

void region_vmm_zero_range(uint64_t handle, uint64_t offset, uint64_t nbytes) {
    std::lock_guard<std::mutex> lk(region_vmm_mu());
    auto it = g_region_vmm_allocations().find(handle);
    if (it == g_region_vmm_allocations().end()) {
        throw std::runtime_error("region VMM zero_bytes handle is unknown");
    }
    RegionVmmAllocation &record = it->second;
    if (!record.mapped || record.va == nullptr) {
        throw std::runtime_error("region VMM zero_bytes requires a mapped VA");
    }
    uint64_t end = offset + nbytes;
    if (end < offset || end > record.mapping_bytes) {
        throw std::invalid_argument("region VMM zero_bytes range exceeds mapping_bytes");
    }
    region_vmm_bind_device(record.device_id);
    if (region_vmm_consume_fail(RegionVmmStage::ZeroBytes, RegionVmmFailWhen::Before)) {
        region_vmm_throw_stage_failure(RegionVmmStage::ZeroBytes, "before");
    }
    region_vmm_zero_bytes(record.va, offset, nbytes);
    if (region_vmm_consume_fail(RegionVmmStage::ZeroBytes, RegionVmmFailWhen::After)) {
        region_vmm_throw_stage_failure(RegionVmmStage::ZeroBytes, "after");
    }
}

RegionVmmInspect region_vmm_inspect(uint64_t handle) {
    std::lock_guard<std::mutex> lk(region_vmm_mu());
    auto it = g_region_vmm_allocations().find(handle);
    if (it == g_region_vmm_allocations().end()) {
        return RegionVmmInspect{};
    }
    return region_vmm_inspect_record(handle, &it->second);
}

std::vector<uint64_t> region_vmm_live_handles() {
    std::lock_guard<std::mutex> lk(region_vmm_mu());
    std::vector<uint64_t> handles;
    handles.reserve(g_region_vmm_allocations().size());
    for (const auto &entry : g_region_vmm_allocations()) {
        handles.push_back(entry.first);
    }
    return handles;
}

std::vector<uint8_t> region_vmm_test_fake_bytes(uint64_t handle) {
    std::lock_guard<std::mutex> lk(region_vmm_mu());
    if (!region_vmm_test_hooks().use_fake) {
        throw std::runtime_error("region VMM fake bytes require the fake driver");
    }
    auto it = g_region_vmm_allocations().find(handle);
    if (it == g_region_vmm_allocations().end() || it->second.va == nullptr) {
        throw std::runtime_error("region VMM fake bytes require a reserved VA");
    }
    auto bytes_it = region_vmm_test_hooks().fake_va_bytes.find(reinterpret_cast<uint64_t>(it->second.va));
    if (bytes_it == region_vmm_test_hooks().fake_va_bytes.end()) {
        throw std::runtime_error("region VMM fake bytes have no reserved VA backing");
    }
    return bytes_it->second;
}

std::vector<std::string> region_vmm_issued_ops() {
    std::lock_guard<std::mutex> lk(region_vmm_mu());
    return region_vmm_test_hooks().issued_ops;
}

void region_vmm_test_use_fake_driver() {
    std::lock_guard<std::mutex> lk(region_vmm_mu());
    region_vmm_test_hooks().use_fake = true;
}

void region_vmm_test_reset_hooks() {
    std::lock_guard<std::mutex> lk(region_vmm_mu());
    RegionVmmTestHooks &hooks = region_vmm_test_hooks();
    hooks.fail_stage.reset();
    hooks.fail_when = RegionVmmFailWhen::Before;
    hooks.unmap_already_gone = false;
    hooks.issued_ops.clear();
}

void region_vmm_test_fail_stage(const std::string &stage, const std::string &when) {
    std::lock_guard<std::mutex> lk(region_vmm_mu());
    RegionVmmTestHooks &hooks = region_vmm_test_hooks();
    hooks.fail_stage = region_vmm_parse_stage(stage);
    if (when == "before") {
        hooks.fail_when = RegionVmmFailWhen::Before;
    } else if (when == "after") {
        hooks.fail_when = RegionVmmFailWhen::After;
    } else {
        throw std::invalid_argument("region VMM fail when must be before or after");
    }
}

void region_vmm_test_set_unmap_already_gone() {
    std::lock_guard<std::mutex> lk(region_vmm_mu());
    region_vmm_test_hooks().unmap_already_gone = true;
}

// The int wire value of a dtype given either a DataType enumerator or its int value. The nanobind
// DataType enum is not arithmetic, so a caller holding one has only `.value`; accept both forms.
// The enumerator is tried first as a type check: `hasattr` costs an attribute lookup, and on an int
// it costs a raised-and-cleared AttributeError as well.
uint8_t datatype_wire_value(nb::handle dtype) {
    if (nb::isinstance<DataType>(dtype)) return static_cast<uint8_t>(nb::cast<DataType>(dtype));
    if (nb::hasattr(dtype, "value")) return nb::cast<uint8_t>(nb::getattr(dtype, "value"));
    return nb::cast<uint8_t>(dtype);
}

// Write a view's `ndims` / `shapes` / `strides` into `t`. A `strides` of None is contiguous
// (row-major): strides[i] = prod(shapes[i+1:]). The two ways to build a Tensor from Python — the
// constructor and `BufferDescriptor.tensor` — share this so a view means the same thing in both.
// Bounds only; `validate_tensor` is the gate on the finished Tensor.
void fill_view(Tensor *t, nb::handle shapes, nb::handle strides) {
    // PySequence_Fast hands back a tuple or list unchanged, so the common call reads its elements
    // straight out of the caller's own object.
    PyObject *raw_shapes = PySequence_Fast(shapes.ptr(), "Tensor shapes must be a sequence");
    if (raw_shapes == nullptr) throw nb::python_error();
    nb::object shapes_fast = nb::steal(raw_shapes);
    const Py_ssize_t ndims = PySequence_Fast_GET_SIZE(raw_shapes);
    if (ndims == 0 || ndims > static_cast<Py_ssize_t>(MAX_TENSOR_DIMS)) {
        throw std::invalid_argument(
            "Tensor ndims must be in [1, " + std::to_string(MAX_TENSOR_DIMS) + "], got " + std::to_string(ndims)
        );
    }
    PyObject **shape_items = PySequence_Fast_ITEMS(raw_shapes);
    t->ndims = static_cast<uint32_t>(ndims);
    for (Py_ssize_t i = 0; i < ndims; ++i)
        t->shapes[i] = nb::cast<uint32_t>(nb::handle(shape_items[i]));

    if (strides.is_none()) {
        // A stride is a u32 field, so a suffix product that does not fit one names no representable
        // view: rejecting is what keeps a wrapped-small stride from passing validate_tensor's extent
        // check while the view really spans past the backing. The accumulator is u64 and bounded
        // before every further multiply, so it cannot wrap either.
        uint64_t acc = 1;
        for (Py_ssize_t i = ndims; i-- > 0;) {
            t->strides[i] = static_cast<uint32_t>(acc);
            acc *= t->shapes[i];
            if (i > 0 && acc > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                throw std::invalid_argument(
                    "Tensor contiguous stride does not fit in uint32: prod(shapes[" + std::to_string(i) + ":]) is " +
                    std::to_string(acc)
                );
            }
        }
        return;
    }
    PyObject *raw_strides = PySequence_Fast(strides.ptr(), "Tensor strides must be a sequence");
    if (raw_strides == nullptr) throw nb::python_error();
    nb::object strides_fast = nb::steal(raw_strides);
    if (PySequence_Fast_GET_SIZE(raw_strides) != ndims) {
        throw std::invalid_argument("Tensor shapes and strides must have equal length");
    }
    PyObject **stride_items = PySequence_Fast_ITEMS(raw_strides);
    for (Py_ssize_t i = 0; i < ndims; ++i)
        t->strides[i] = nb::cast<uint32_t>(nb::handle(stride_items[i]));
}

// The leading `ndims` entries of a wire shapes[] / strides[] array as a Python tuple. The trailing
// entries are unused padding, so exposing them would invent dimensions the tensor does not have.
nb::tuple dims_tuple(const uint32_t *dims, uint32_t ndims) {
    if (ndims > static_cast<uint32_t>(MAX_TENSOR_DIMS)) ndims = static_cast<uint32_t>(MAX_TENSOR_DIMS);
    nb::list out;
    for (uint32_t i = 0; i < ndims; ++i)
        out.append(dims[i]);
    return nb::tuple(out);
}

// Resolve one wire tensor onto a local base and build the address-bearing device POD.
// `resolved` maps CanonicalIdentity -> (local_base, address_space); the caller populates it by
// materializing each embedded descriptor.
ChipTensor materialize_one(const Tensor &r, nb::dict resolved) {
    uint64_t elem = get_element_size(r.dtype);
    if (elem == 0) {
        throw std::runtime_error("materialize: unknown dtype");
    }
    if (r.byte_offset % elem != 0) {
        throw std::runtime_error("materialize: byte_offset is not a multiple of dtype size");
    }
    nb::object key = nb::cast(r.buffer.identity);
    if (!resolved.contains(key)) {
        throw std::runtime_error("materialize: canonical identity not in the import registry");
    }
    nb::tuple val = nb::cast<nb::tuple>(resolved[key]);
    auto base = nb::cast<uint64_t>(val[0]);
    auto addr_space = nb::cast<int>(val[1]);
    // The view origin is base + byte_offset (start_offset folded into addr); strides carry any
    // non-row-major layout (transpose / permute / step-slice), which ChipTensor expresses natively.
    return make_tensor_strided(
        reinterpret_cast<void *>(static_cast<uintptr_t>(base + r.byte_offset)), r.shapes, r.strides, r.ndims, r.dtype,
        static_cast<AddressSpace>(addr_space)
    );
}

// Which device allocations a dispatch's operands may name: the private snapshot of every live
// child allocation, keyed by the identity that resolves it. The owner-side Python registry is the
// source of truth for lifetime; this mirror exists so the per-argument dispatch check reads the
// registered descriptor without materializing one Python object per argument.
struct ProvenanceEntry {
    BufferDescriptor descriptor;
    int32_t owner_worker_id;
};

struct ProvenanceTable {
    std::unordered_map<CanonicalIdentity, ProvenanceEntry, CanonicalIdentityHash> entries;
};

// Why one argument failed the dispatch check. The caller names the argument and raises: the
// identity's rendering and the api name live on the Python side, and the failure path is cold.
constexpr int PROV_NOT_LIVE = 0;
constexpr int PROV_DESCRIPTOR_MISMATCH = 1;

// The same rule the submit point enforces, applied early so a mistake surfaces at the offending
// add_tensor call rather than at submit. A tag can change afterwards, which is why submit re-checks.
void check_access_subset(uint8_t granted, TensorArgType tag) {
    if (!access_permits(granted, tag)) {
        throw std::invalid_argument("TaskArgs.add_tensor: arg TensorArgType requires access not granted by the buffer");
    }
}

}  // namespace

// ============================================================================
// Module definition
// ============================================================================

#ifndef SIMPLER_BUILD_COMMIT
#define SIMPLER_BUILD_COMMIT ""
#endif

NB_MODULE(_task_interface, m) {
    m.doc() = "Nanobind bindings for task_interface (DataType, Buffer/Tensor wire ABI, ChipTensor, TaskArgs variants)";

    // Source commit this extension was compiled from; "" when git was
    // unavailable at build time. simpler.task_interface compares it against the
    // working tree so a binding built from other sources cannot be used
    // silently — struct layouts differ and fields read as garbage.
    m.attr("__build_commit__") = SIMPLER_BUILD_COMMIT;

    // --- DataType enum ---
    nb::enum_<DataType>(m, "DataType")
        .value("FLOAT32", DataType::FLOAT32)
        .value("FLOAT16", DataType::FLOAT16)
        .value("INT32", DataType::INT32)
        .value("INT16", DataType::INT16)
        .value("INT8", DataType::INT8)
        .value("UINT8", DataType::UINT8)
        .value("BFLOAT16", DataType::BFLOAT16)
        .value("INT64", DataType::INT64)
        .value("UINT64", DataType::UINT64)
        .value("UINT16", DataType::UINT16)
        .value("UINT32", DataType::UINT32)
        .value("FP8E4M3FN", DataType::FP8E4M3FN)  // A5 only
        .value("FP8E8M0", DataType::FP8E8M0)      // A5 only
        .value("FP4E2M1", DataType::FP4E2M1);     // A5 only

    // --- Free functions ---
    m.def(
        "get_element_size", &get_element_size, nb::arg("dtype"),
        "Return the byte size of a single element of the given DataType."
    );

    m.def(
        "get_dtype_name",
        [](DataType dt) -> std::string {
            return get_dtype_name(dt);
        },
        nb::arg("dtype"), "Return the string name of a DataType."
    );

    // --- Constants ---
    m.attr("MAX_TENSOR_DIMS") = MAX_TENSOR_DIMS;
    m.attr("MAX_REGISTERED_CALLABLE_IDS") = MAX_REGISTERED_CALLABLE_IDS;
    m.attr("RUNTIME_ENV_RING_COUNT") = RUNTIME_ENV_RING_COUNT;
#if SIMPLER_HOST_STRACE
    m.attr("HOST_STRACE_ENABLED") = true;
#else
    m.attr("HOST_STRACE_ENABLED") = false;
#endif
    m.def(
        "_host_spans_active",
        [] {
            return simpler::host_trace::enabled();
        },
        "Return whether this extension currently emits TIMING-level host spans."
    );
    m.def(
        "_monotonic_now_ns",
        [] {
            return simpler::log::monotonic_now_ns();
        },
        "Read the clock every host record is stamped with, so a Python-timed span shares one clock with the C++ "
        "spans by construction rather than by both platforms happening to map their monotonic clock the same way."
    );
    m.def(
        "_host_log_directory",
        [] {
            const char *bound = HostLogger::get_instance().log_directory();
            return bound == nullptr ? std::string() : std::string(bound);
        },
        "The directory this process's host log is written to, or an empty string while it writes to stderr."
    );
    m.def(
        "_emit_host_log",
        [](int level, const std::string &func, const std::string &message) {
            if (!simpler::log::is_valid_level(level)) return false;
            HostLogger::get_instance().log(
                static_cast<simpler::log::LogLevel>(level), func.c_str(), "%s", message.c_str()
            );
            return true;
        },
        nb::arg("level"), nb::arg("func"), nb::arg("message"),
        "Emit one already-formatted message through the host logger, so a Python record carries the same envelope, "
        "clock and destination as a C++ one. Returns false for a level outside the ladder."
    );
    m.def(
        "_initialize_host_log",
        [](int level, bool defer_writer) {
            if (!simpler::log::is_valid_level(level)) return false;
            HostLogger &logger = HostLogger::get_instance();
            if (defer_writer && !logger.prepare_to_fork()) return false;
            logger.set_level(static_cast<simpler::log::LogLevel>(level), /*defer_writer=*/true);
            return defer_writer || logger.start_writer();
        },
        nb::arg("level"), nb::arg("defer_writer") = false,
        "Seed the process-owned host-log state. A hierarchical worker defers its writer until after its last fork."
    );
    m.def(
        "_start_host_log_writer",
        [] {
            return HostLogger::get_instance().start_writer();
        },
        "Start this process's bounded host-log writer after its final local fork."
    );
    m.def(
        "_flush_host_log",
        [](uint32_t timeout_ms) {
            return HostLogger::get_instance().flush(timeout_ms);
        },
        nb::arg("timeout_ms") = 1000, nb::call_guard<nb::gil_scoped_release>(),
        "Wait boundedly for all host-log records accepted by this process to be written."
    );
    m.def(
        "_host_log_dropped_records",
        [] {
            return HostLogger::get_instance().dropped_records();
        },
        "Return the number of host-log records rejected by or lost from this process sink."
    );
    m.def(
        "_host_log_dropped_records_by_reason",
        [] {
            const HostLogger &logger = HostLogger::get_instance();
            nb::dict counts;
            counts["queue_full"] = logger.dropped_records(SIMPLER_HOST_LOG_DROP_QUEUE_FULL);
            counts["claim_exhausted"] = logger.dropped_records(SIMPLER_HOST_LOG_DROP_CLAIM_EXHAUSTED);
            counts["output_failed"] = logger.dropped_records(SIMPLER_HOST_LOG_DROP_OUTPUT_FAILED);
            counts["not_admitted"] = logger.dropped_records(SIMPLER_HOST_LOG_DROP_NOT_ADMITTED);
            return counts;
        },
        "Return this process's host-log drops attributed by cause. The keys sum to "
        "_host_log_dropped_records(). `queue_full` means the queue is too small for the burst; "
        "`claim_exhausted` means the lock-free claim budget lost to contention with room still "
        "in the queue; `output_failed` means the destination rejected the write; `not_admitted` "
        "means there was no sink to submit to. They call for different fixes, which is why the "
        "total alone is not actionable."
    );
    m.def(
        "_host_log_pending_records",
        [] {
            return HostLogger::get_instance().pending_records();
        },
        "Return the number of accepted host-log records not yet written by this process sink."
    );
    m.def(
        "_set_host_log_directory",
        [](const std::string &path) {
            HostLogger::get_instance().set_log_directory(path.c_str());
            const char *bound = HostLogger::get_instance().log_directory();
            return bound == nullptr ? std::string() : std::string(bound);
        },
        nb::arg("path"),
        "Write this process's host log to <path>/host.<pid>.log instead of stderr, and return the directory "
        "actually in effect. Applies to every record this logger writes, including the host spans Python emits. "
        "The first non-empty path in a process wins; an empty path leaves the logger on stderr."
    );
    m.def(
        "_set_host_span_level_prefix",
        [](const std::string &word) {
            simpler::host_trace::set_level_prefix(word.c_str());
#if SIMPLER_HOST_STRACE
            return std::string(simpler::host_trace::level_prefix());
#else
            return word;
#endif
        },
        nb::arg("word"),
        "Bind this process's level word for host-scheduler span names, and return what is now bound. "
        "Python owns the level -> word mapping (simpler.worker_level); this pushes the resolved word so "
        "the C++ emit sites and Python agree on one derivation."
    );
    m.def(
        "_emit_host_span",
        [](const std::string &name, uint64_t invocation_id, uint64_t callable_hash, int32_t depth, int64_t timestamp_ns,
           int64_t duration_ns, const std::string &attributes) {
            simpler::host_trace::emit(
                name.c_str(), invocation_id, callable_hash, depth, timestamp_ns, duration_ns, attributes.c_str()
            );
        },
        nb::arg("name"), nb::arg("invocation_id"), nb::arg("callable_hash"), nb::arg("depth"), nb::arg("timestamp_ns"),
        nb::arg("duration_ns"), nb::arg("attributes") = "",
        "Emit one explicitly timed host span through this extension's bound logger sink."
    );
    // Byte size of a ChipTensor and the offset of its address_space field within it.
    // A task-args blob stores ChipTensors as a raw memcpy array, so a Python-side
    // blob walker locates tensor i's fields at i * CHIP_TENSOR_STRIDE_BYTES without
    // reimplementing the struct layout.
    m.attr("CHIP_TENSOR_STRIDE_BYTES") = static_cast<int>(sizeof(ChipTensor));
    m.attr("CHIP_TENSOR_ADDRESS_SPACE_OFFSET") = static_cast<int>(offsetof(ChipTensor, address_space));

    // Width of the opaque per-incarnation nonce, so the owner can draw one of the right size.
    // The struct sizes stay unexported: no Python path turns these types into bytes or back, so a
    // byte count here would have no reader — the layout is pinned by buffer.h's static_asserts.
    m.attr("OWNER_INSTANCE_ID_BYTES") = static_cast<int>(OWNER_INSTANCE_ID_BYTES);

    // --- Buffer ABI enums ---
    // nb::is_arithmetic makes these IntEnums, so a wire value and an enumerator compare and
    // int()-convert interchangeably — the descriptor stores them as raw u8.
    nb::enum_<AddressSpace>(m, "AddressSpace", nb::is_arithmetic())
        .value("HOST", AddressSpace::HOST)
        .value("DEVICE", AddressSpace::DEVICE);

    nb::enum_<AccessMode>(m, "AccessMode", nb::is_arithmetic())
        .value("READ", AccessMode::READ)
        .value("WRITE", AccessMode::WRITE)
        .value("READWRITE", AccessMode::READWRITE);

    nb::enum_<BackendKind>(m, "BackendKind", nb::is_arithmetic())
        .value("FORK_SHM", BackendKind::FORK_SHM)
        .value("POSIX_SHM", BackendKind::POSIX_SHM)
        .value("VMM_WINDOW", BackendKind::VMM_WINDOW)
        .value("REMOTE_SIDECAR", BackendKind::REMOTE_SIDECAR)
        .value("DEVICE_MALLOC", BackendKind::DEVICE_MALLOC)
        .value("FORK_COW", BackendKind::FORK_COW);

    // --- CanonicalIdentity ---
    // Equality and hashing fold exactly the three meaningful fields, so a decode with dirty wire
    // padding keys identically to a clean one — two views of a backing never split across buckets.
    // There is deliberately no `pack()`: the only correct key for this type is the value itself, and
    // exposing its bytes is what once let a registry key on padding and split one backing in two.
    nb::class_<CanonicalIdentity>(m, "CanonicalIdentity")
        .def(
            "__init__",
            [](CanonicalIdentity *self, nb::bytes owner_instance_id, uint64_t buffer_id, uint32_t generation) {
                if (owner_instance_id.size() != OWNER_INSTANCE_ID_BYTES) {
                    throw std::invalid_argument(
                        "owner_instance_id must be " + std::to_string(OWNER_INSTANCE_ID_BYTES) + " bytes, got " +
                        std::to_string(owner_instance_id.size())
                    );
                }
                new (self) CanonicalIdentity{};
                std::memcpy(self->owner_instance_id, owner_instance_id.c_str(), OWNER_INSTANCE_ID_BYTES);
                self->buffer_id = buffer_id;
                self->generation = generation;
            },
            nb::arg("owner_instance_id"), nb::arg("buffer_id"), nb::arg("generation") = 1
        )

        .def_prop_ro(
            "owner_instance_id",
            [](const CanonicalIdentity &self) {
                return nb::bytes(reinterpret_cast<const char *>(self.owner_instance_id), OWNER_INSTANCE_ID_BYTES);
            },
            "The opaque per-incarnation nonce (bytewise-compared, no integer meaning)."
        )
        .def_ro("buffer_id", &CanonicalIdentity::buffer_id)
        .def_ro("generation", &CanonicalIdentity::generation)

        .def(
            "__eq__",
            [](const CanonicalIdentity &a, const CanonicalIdentity &b) {
                return a == b;
            }
        )
        .def(
            "__ne__",
            [](const CanonicalIdentity &a, const CanonicalIdentity &b) {
                return a != b;
            }
        )
        .def(
            "__hash__",
            [](const CanonicalIdentity &self) {
                return CanonicalIdentityHash{}(self);
            }
        )
        .def("__repr__", [](const CanonicalIdentity &self) -> std::string {
            std::ostringstream os;
            os << "CanonicalIdentity(owner_instance_id=0x" << std::hex;
            for (uint32_t i = 0; i < OWNER_INSTANCE_ID_BYTES; ++i) {
                os << std::setw(2) << std::setfill('0') << static_cast<int>(self.owner_instance_id[i]);
            }
            os << std::dec << ", buffer_id=" << self.buffer_id << ", generation=" << self.generation << ")";
            return os.str();
        });

    // --- BufferDescriptor ---
    // Construction runs validate_buffer_descriptor, so an unsupported address_space x backend_kind
    // or an over-long backend body is refused before the descriptor can reach a Tensor or the wire.
    nb::class_<BufferDescriptor>(m, "BufferDescriptor")
        .def(
            "__init__",
            [](BufferDescriptor *self, const CanonicalIdentity &identity, AddressSpace address_space, AccessMode access,
               BackendKind backend_kind, uint64_t nbytes, nb::bytes body, uint32_t owner_worker_path_id) {
                if (body.size() > DESC_MAX_BYTES) {
                    throw std::invalid_argument(
                        "backend body exceeds DESC_MAX_BYTES (" + std::to_string(DESC_MAX_BYTES) + "), got " +
                        std::to_string(body.size())
                    );
                }
                new (self) BufferDescriptor{};
                self->magic = BUFFER_DESCRIPTOR_MAGIC;
                self->address_space = static_cast<uint8_t>(address_space);
                self->access = static_cast<uint8_t>(access);
                self->backend_kind = static_cast<uint8_t>(backend_kind);
                self->identity = identity;
                self->nbytes = nbytes;
                self->owner_worker_path_id = owner_worker_path_id;
                self->body_len = static_cast<uint16_t>(body.size());
                std::memcpy(self->body, body.c_str(), body.size());
                validate_buffer_descriptor(*self);
            },
            nb::arg("identity"), nb::arg("address_space"), nb::arg("access"), nb::arg("backend_kind"),
            nb::arg("nbytes"), nb::arg("body") = nb::bytes(""), nb::arg("owner_worker_path_id") = 0
        )

        .def_ro("identity", &BufferDescriptor::identity)
        .def_prop_ro(
            "address_space",
            [](const BufferDescriptor &self) {
                return static_cast<AddressSpace>(self.address_space);
            }
        )
        .def_prop_ro(
            "access",
            [](const BufferDescriptor &self) {
                return static_cast<AccessMode>(self.access);
            }
        )
        .def_prop_ro(
            "backend_kind",
            [](const BufferDescriptor &self) {
                return static_cast<BackendKind>(self.backend_kind);
            }
        )
        .def_ro("nbytes", &BufferDescriptor::nbytes)
        .def_ro("owner_worker_path_id", &BufferDescriptor::owner_worker_path_id)
        .def_prop_ro(
            "body",
            [](const BufferDescriptor &self) {
                return nb::bytes(self.body, self.body_len);
            },
            "The per-backend materialization payload (shm name, base VA, ...), body_len bytes."
        )

        .def(
            "__eq__",
            [](const BufferDescriptor &a, const BufferDescriptor &b) {
                return a == b;
            }
        )
        .def(
            "__ne__",
            [](const BufferDescriptor &a, const BufferDescriptor &b) {
                return a != b;
            }
        )
        .def(
            "tensor",
            [](const BufferDescriptor &self, nb::object shapes, nb::object dtype, nb::object strides,
               uint64_t byte_offset) -> Tensor {
                Tensor t{};
                t.buffer = self;
                t.byte_offset = byte_offset;
                t.dtype = static_cast<DataType>(datatype_wire_value(dtype));
                fill_view(&t, shapes, strides);
                validate_tensor(t);
                return t;
            },
            nb::arg("shapes"), nb::arg("dtype"), nb::arg("strides") = nb::none(), nb::arg("byte_offset") = 0,
            "A Tensor viewing this backing. `strides` default to contiguous (row-major) element strides."
        )

        .def("__repr__", [](const BufferDescriptor &self) -> std::string {
            std::ostringstream os;
            os << "BufferDescriptor(buffer_id=" << self.identity.buffer_id
               << ", generation=" << self.identity.generation << ", address_space=" << int(self.address_space)
               << ", access=" << int(self.access) << ", backend_kind=" << int(self.backend_kind)
               << ", nbytes=" << self.nbytes << ", body_len=" << self.body_len << ")";
            return os.str();
        });

    // --- Tensor ---
    // The L3+ task argument: a strided view over a buffer, carrying that buffer's descriptor whole
    // and no address. Construction runs validate_tensor, the same gate blob decode runs, so a view
    // that does not fit its backing cannot be built in the first place.
    //
    // No bytes cross this binding in either direction. Python builds a Tensor from its fields and
    // receives one already decoded; turning mailbox bytes into a Tensor is task_args_wire.h's job,
    // and keeping that the only decode path is what makes validate_tensor a gate rather than a habit.
    nb::class_<Tensor>(m, "Tensor")
        .def(
            "__init__",
            [](Tensor *self, const BufferDescriptor &buffer, uint64_t byte_offset, nb::sequence shapes,
               nb::sequence strides, nb::object dtype) {
                new (self) Tensor{};
                self->buffer = buffer;
                self->byte_offset = byte_offset;
                self->dtype = static_cast<DataType>(datatype_wire_value(dtype));
                fill_view(self, shapes, strides);
                validate_tensor(*self);
            },
            nb::arg("buffer"), nb::arg("byte_offset"), nb::arg("shapes"), nb::arg("strides"), nb::arg("dtype")
        )

        .def_ro("buffer", &Tensor::buffer)
        .def_ro("byte_offset", &Tensor::byte_offset)
        .def_ro("ndims", &Tensor::ndims)
        .def_prop_ro(
            "shapes",
            [](const Tensor &self) {
                return dims_tuple(self.shapes, self.ndims);
            }
        )
        .def_prop_ro(
            "strides",
            [](const Tensor &self) {
                return dims_tuple(self.strides, self.ndims);
            }
        )
        .def_prop_ro(
            "dtype",
            [](const Tensor &self) {
                return static_cast<int>(self.dtype);
            },
            "The dtype's int wire value (a DataType enumerator's .value)."
        )

        .def(
            "__eq__",
            [](const Tensor &a, const Tensor &b) {
                if (!(a.buffer == b.buffer) || a.byte_offset != b.byte_offset || a.ndims != b.ndims ||
                    a.dtype != b.dtype) {
                    return false;
                }
                for (uint32_t i = 0; i < a.ndims; ++i) {
                    if (a.shapes[i] != b.shapes[i] || a.strides[i] != b.strides[i]) return false;
                }
                return true;
            }
        )
        .def("__repr__", [](const Tensor &self) -> std::string {
            std::ostringstream os;
            os << "Tensor(buffer_id=" << self.buffer.identity.buffer_id << ", byte_offset=" << self.byte_offset
               << ", shapes=(";
            for (uint32_t i = 0; i < self.ndims; ++i) {
                if (i) os << ", ";
                os << self.shapes[i];
            }
            os << "), strides=(";
            for (uint32_t i = 0; i < self.ndims; ++i) {
                if (i) os << ", ";
                os << self.strides[i];
            }
            os << "), dtype=" << get_dtype_name(self.dtype) << ")";
            return os.str();
        });

    // --- ChipTensor ---
    // The unified strided tensor descriptor. Constructed contiguous via make()
    // (row-major strides, start_offset == 0); see src/common/task_interface/tensor.h.
    nb::class_<ChipTensor>(m, "ChipTensor")
        .def(nb::init<>())

        .def_static(
            "make",
            [](uint64_t data, nb::tuple shapes, DataType dtype, bool child_memory) -> ChipTensor {
                size_t n = nb::len(shapes);
                if (n == 0 || n > MAX_TENSOR_DIMS)
                    throw std::invalid_argument("ChipTensor.make: shapes length must be in [1, MAX_TENSOR_DIMS]");
                uint32_t shp[MAX_TENSOR_DIMS];
                for (size_t i = 0; i < n; ++i)
                    shp[i] = nb::cast<uint32_t>(shapes[i]);
                // make_tensor_external yields a contiguous ChipTensor: row-major strides,
                // start_offset == 0, buffer.size == numel * element_size.
                return make_tensor_external(
                    reinterpret_cast<void *>(static_cast<uintptr_t>(data)), shp, static_cast<uint32_t>(n), dtype,
                    child_memory ? AddressSpace::DEVICE : AddressSpace::HOST
                );
            },
            // The keyword stays `child_memory` while the C++ field is `address_space`: it is the
            // name of a u8 on the remote-L3 tensor wire (see remote_wire.cpp encode_tensor), which
            // renaming here would not change and which this constructor decodes into.
            nb::arg("data"), nb::arg("shapes"), nb::arg("dtype"), nb::arg("child_memory") = false,
            "Create a contiguous ChipTensor over pre-allocated memory. Set child_memory=True when "
            "data is a device pointer allocated by the child process (skips H2D copy in "
            "init_runtime_impl)."
        )

        // `data` is the tensor's memory address — i.e. ChipTensor::buffer.addr.
        .def_prop_rw(
            "data",
            [](const ChipTensor &self) -> uint64_t {
                return self.buffer.addr;
            },
            [](ChipTensor &self, uint64_t v) {
                self.buffer.addr = v;
            }
        )

        .def_prop_rw(
            "shapes",
            [](const ChipTensor &self) -> nb::tuple {
                uint32_t n = self.ndims;
                if (n > MAX_TENSOR_DIMS) n = MAX_TENSOR_DIMS;
                nb::list lst;
                for (uint32_t i = 0; i < n; ++i)
                    lst.append(self.shapes[i]);
                return nb::tuple(lst);
            },
            [](ChipTensor &self, nb::tuple t) {
                size_t n = nb::len(t);
                if (n == 0 || n > MAX_TENSOR_DIMS)
                    throw std::invalid_argument(
                        "shapes tuple length must be in [1, MAX_TENSOR_DIMS] (" + std::to_string(MAX_TENSOR_DIMS) + ")"
                    );
                uint32_t shp[MAX_TENSOR_DIMS];
                for (size_t i = 0; i < n; ++i)
                    shp[i] = nb::cast<uint32_t>(t[i]);
                uint64_t numel = 1;
                for (size_t i = 0; i < n; ++i)
                    numel *= shp[i];
                // Re-establish a contiguous layout over the same buffer base.
                self.init_external(
                    reinterpret_cast<void *>(self.buffer.addr), numel * get_element_size(self.dtype), shp,
                    static_cast<uint32_t>(n), self.dtype, self.address_space
                );
            }
        )

        // Read-only: a raw `ndims` write would desync shapes/strides/buffer.size
        // and could index past the fixed MAX_TENSOR_DIMS arrays. Rank changes go
        // through the `shapes` setter, which rebuilds a valid contiguous layout.
        .def_prop_ro(
            "ndims",
            [](const ChipTensor &self) -> uint32_t {
                return self.ndims;
            }
        )

        .def_prop_rw(
            "dtype",
            [](const ChipTensor &self) -> DataType {
                return self.dtype;
            },
            [](ChipTensor &self, DataType dt) {
                self.dtype = dt;
                self.buffer.size = self.numel() * get_element_size(dt);
            }
        )

        .def_prop_rw(
            "child_memory",
            [](const ChipTensor &self) -> bool {
                return self.is_device_memory();
            },
            [](ChipTensor &self, bool v) {
                self.address_space = v ? AddressSpace::DEVICE : AddressSpace::HOST;
            }
        )

        // Read-only views of the strided metadata (always contiguous for make()).
        .def_prop_ro(
            "strides",
            [](const ChipTensor &self) -> nb::tuple {
                nb::list lst;
                for (uint32_t i = 0; i < self.ndims && i < MAX_TENSOR_DIMS; ++i)
                    lst.append(self.strides[i]);
                return nb::tuple(lst);
            }
        )
        .def_prop_ro(
            "start_offset",
            [](const ChipTensor &self) -> uint64_t {
                return self.start_offset;
            }
        )
        .def_prop_ro(
            "is_contiguous",
            [](const ChipTensor &self) -> bool {
                return self.is_contiguous();
            }
        )

        .def(
            "nbytes",
            [](const ChipTensor &self) -> uint64_t {
                return self.nbytes();
            },
            "Compute total bytes (product of shapes * element_size)."
        )

        .def("__repr__", [](const ChipTensor &self) -> std::string {
            std::ostringstream os;
            os << "ChipTensor(data=0x" << std::hex << self.buffer.addr << std::dec << ", shapes=(";
            for (uint32_t i = 0; i < self.ndims; ++i) {
                if (i) os << ", ";
                os << self.shapes[i];
            }
            os << "), dtype=" << get_dtype_name(self.dtype);
            if (self.is_device_memory()) os << ", child_memory=True";
            os << ")";
            return os.str();
        });

    // --- ChipStorageTaskArgs (fixed-size TaskArgs) ---
    nb::class_<ChipStorageTaskArgs>(m, "ChipStorageTaskArgs")
        .def(nb::init<>())

        .def(
            "add_tensor", &ChipStorageTaskArgs::add_tensor, nb::arg("t"),
            "Add a ChipTensor. Must be called before any add_scalar()."
        )

        .def(
            "add_scalar", &ChipStorageTaskArgs::add_scalar, nb::arg("s"),
            "Add a uint64_t scalar. After this, add_tensor() is no longer allowed."
        )

        .def(
            "tensor",
            [](const ChipStorageTaskArgs &self, int32_t i) -> const ChipTensor & {
                if (i < 0 || i >= self.tensor_count())
                    throw std::out_of_range("ChipStorageTaskArgs tensor index out of range");
                return self.tensor(i);
            },
            nb::arg("i"), nb::rv_policy::reference_internal, "Return the ChipTensor at index i."
        )

        .def(
            "scalar",
            [](const ChipStorageTaskArgs &self, int32_t i) -> uint64_t {
                if (i < 0 || i >= self.scalar_count())
                    throw std::out_of_range("ChipStorageTaskArgs scalar index out of range");
                return self.scalar(i);
            },
            nb::arg("i"), "Return the scalar at index i."
        )

        .def("tensor_count", &ChipStorageTaskArgs::tensor_count)
        .def("scalar_count", &ChipStorageTaskArgs::scalar_count)

        .def("clear", &ChipStorageTaskArgs::clear)

        .def(
            "__len__",
            [](const ChipStorageTaskArgs &self) {
                return self.tensor_count() + self.scalar_count();
            },
            "Return total number of arguments (tensors + scalars)."
        )

        .def(
            "__ptr__",
            [](const ChipStorageTaskArgs &self) -> uint64_t {
                return reinterpret_cast<uint64_t>(&self);
            },
            "Return the memory address of the underlying C++ object."
        )

        .def_static(
            "sizeof",
            []() -> size_t {
                return sizeof(ChipStorageTaskArgs);
            },
            "Return sizeof(ChipStorageTaskArgs) in bytes."
        );

    // --- TensorArgType enum ---
    nb::enum_<TensorArgType>(m, "TensorArgType")
        .value("INPUT", TensorArgType::INPUT)
        .value("OUTPUT", TensorArgType::OUTPUT)
        .value("INOUT", TensorArgType::INOUT)
        .value("OUTPUT_EXISTING", TensorArgType::OUTPUT_EXISTING)
        .value("NO_DEP", TensorArgType::NO_DEP);

    nb::class_<TaskHandle>(m, "TaskHandle");

    // --- TaskArgs (unified vector-backed builder with per-tensor TensorArgType tags) ---
    nb::class_<TaskArgs>(m, "TaskArgs", nb::is_weak_referenceable())
        .def(nb::init<>())

        .def(
            "add_tensor",
            [](TaskArgs &self, const Tensor &t, TensorArgType tag) {
                validate_tensor(t);
                check_access_subset(t.buffer.access, tag);
                self.add_tensor(t, tag);
            },
            nb::arg("t"), nb::arg("tag") = TensorArgType::INPUT,
            "Add a Tensor arg (the self-describing wire view built by Buffer.tensor) with an "
            "optional TensorArgType tag (default INPUT)."
        )

        .def(
            "add_scalar", &TaskArgs::add_scalar, nb::arg("s"),
            "Add a uint64_t scalar. After this, add_tensor() is no longer allowed."
        )

        .def(
            "add_dep",
            [](TaskArgs &self, nb::args deps) {
                if (deps.size() == 0) {
                    throw std::invalid_argument("TaskArgs.add_dep requires at least one TaskHandle");
                }
                for (nb::handle dep : deps) {
                    if (!nb::isinstance<TaskHandle>(dep)) {
                        throw nb::type_error("TaskArgs.add_dep arguments must be TaskHandle objects");
                    }
                }
                for (nb::handle dep : deps) {
                    self.add_dep(nb::cast<const TaskHandle &>(dep));
                }
            },
            "Add dependencies that retain each producer until this task completes."
        )

        .def(
            "add_dep_wait",
            [](TaskArgs &self, nb::args deps) {
                if (deps.size() == 0) {
                    throw std::invalid_argument("TaskArgs.add_dep_wait requires at least one TaskHandle");
                }
                for (nb::handle dep : deps) {
                    if (!nb::isinstance<TaskHandle>(dep)) {
                        throw nb::type_error("TaskArgs.add_dep_wait arguments must be TaskHandle objects");
                    }
                }
                for (nb::handle dep : deps) {
                    self.add_dep_wait(nb::cast<const TaskHandle &>(dep));
                }
            },
            "Add one or more ordering-only dependencies returned by an Orchestrator submit."
        )

        .def(
            "tensor",
            [](const TaskArgs &self, int32_t i) -> Tensor {
                if (i < 0 || i >= self.tensor_count()) throw std::out_of_range("TaskArgs tensor index out of range");
                return self.tensor(i);
            },
            nb::arg("i"), "Return the Tensor at index i."
        )

        .def(
            "scalar",
            [](const TaskArgs &self, int32_t i) -> uint64_t {
                if (i < 0 || i >= self.scalar_count()) throw std::out_of_range("TaskArgs scalar index out of range");
                return self.scalar(i);
            },
            nb::arg("i"), "Return the scalar at index i."
        )

        .def(
            "tag",
            [](const TaskArgs &self, int32_t i) -> TensorArgType {
                if (i < 0 || i >= self.tensor_count()) throw std::out_of_range("TaskArgs tag index out of range");
                return self.tag(i);
            },
            nb::arg("i"), "Return the TensorArgType tag for the tensor at index i."
        )

        .def(
            "set_tag",
            [](TaskArgs &self, int32_t i, TensorArgType tag) {
                if (i < 0 || i >= self.tensor_count()) throw std::out_of_range("TaskArgs set_tag index out of range");
                self.tag(i) = tag;
            },
            nb::arg("i"), nb::arg("tag"), "Set the TensorArgType tag for the tensor at index i."
        )

        .def("tensor_count", &TaskArgs::tensor_count)
        .def("scalar_count", &TaskArgs::scalar_count)

        .def(
            "identities",
            [](const TaskArgs &self) {
                const int32_t n = self.tensor_count();
                nb::list out;
                for (int32_t i = 0; i < n; ++i)
                    out.append(nb::cast(self.tensor(i).buffer.identity));
                return out;
            },
            "Every tensor arg's buffer identity, in argument order."
        )

        .def(
            "has_device_backed_tensor",
            [](const TaskArgs &self) {
                const int32_t n = self.tensor_count();
                for (int32_t i = 0; i < n; ++i) {
                    const auto backend = static_cast<BackendKind>(self.tensor(i).buffer.backend_kind);
                    if (backend == BackendKind::DEVICE_MALLOC || backend == BackendKind::VMM_WINDOW) return true;
                }
                return false;
            },
            "Whether any arg names memory behind a chip boundary, which a dispatch must authorize."
        )

        .def("clear", &TaskArgs::clear)

        .def(
            "__len__",
            [](const TaskArgs &self) {
                return self.tensor_count() + self.scalar_count();
            },
            "Return total number of arguments (tensors + scalars)."
        );

    // --- ProvenanceTable ---
    // The owner's live child device allocations, as the dispatch path consumes them. The Python
    // Worker writes it alongside its own registry and reads it back only through `check_dispatch`.
    m.attr("PROV_NOT_LIVE") = PROV_NOT_LIVE;
    m.attr("PROV_DESCRIPTOR_MISMATCH") = PROV_DESCRIPTOR_MISMATCH;

    nb::class_<ProvenanceTable>(m, "ProvenanceTable")
        .def(nb::init<>())

        .def(
            "insert",
            [](ProvenanceTable &self, const BufferDescriptor &descriptor, int32_t owner_worker_id) {
                self.entries[descriptor.identity] = ProvenanceEntry{descriptor, owner_worker_id};
            },
            nb::arg("descriptor"), nb::arg("owner_worker_id"),
            "Register one allocation, keyed by its descriptor's identity."
        )

        .def(
            "erase",
            [](ProvenanceTable &self, const CanonicalIdentity &identity) {
                self.entries.erase(identity);
            },
            nb::arg("identity"), "Revoke one allocation; absent identities are ignored."
        )

        .def(
            "clear",
            [](ProvenanceTable &self) {
                self.entries.clear();
            }
        )

        .def(
            "__len__",
            [](const ProvenanceTable &self) {
                return self.entries.size();
            }
        )

        .def(
            "__contains__",
            [](const ProvenanceTable &self, const CanonicalIdentity &identity) {
                return self.entries.find(identity) != self.entries.end();
            }
        )

        .def(
            "check_dispatch",
            [](const ProvenanceTable &self, const TaskArgs &args, int32_t target_worker_id) -> nb::object {
                const int32_t n = args.tensor_count();
                for (int32_t i = 0; i < n; ++i) {
                    const BufferDescriptor &d = args.tensor(i).buffer;
                    const auto backend = static_cast<BackendKind>(d.backend_kind);
                    if (backend != BackendKind::DEVICE_MALLOC && backend != BackendKind::VMM_WINDOW) continue;
                    auto it = self.entries.find(d.identity);
                    if (it == self.entries.end() || it->second.owner_worker_id != target_worker_id) {
                        return nb::make_tuple(i, PROV_NOT_LIVE);
                    }
                    // Authorization and execution consume the same Buffer, so a same-identity
                    // descriptor with a changed body/backend/extent/access is a different value.
                    if (!(d == it->second.descriptor)) return nb::make_tuple(i, PROV_DESCRIPTOR_MISMATCH);
                }
                return nb::none();
            },
            nb::arg("args"), nb::arg("target_worker_id"),
            "None when every device arg is live on target_worker_id, else (arg_index, reason)."
        );

    // --- ArgDirection enum ---
    nb::enum_<ArgDirection>(m, "ArgDirection")
        .value("SCALAR", ArgDirection::SCALAR)
        .value("IN", ArgDirection::IN)
        .value("OUT", ArgDirection::OUT)
        .value("INOUT", ArgDirection::INOUT);

    m.def(
        "arg_direction_name",
        [](ArgDirection d) -> std::string {
            return arg_direction_name(d);
        },
        nb::arg("direction"), "Return the string name of an ArgDirection."
    );

    // --- PyCoreCallable wrapper ---
    struct PyCoreCallable {
        std::vector<uint8_t> buffer_;
        const CoreCallable &get() const { return *reinterpret_cast<const CoreCallable *>(buffer_.data()); }
    };

    nb::class_<PyCoreCallable>(m, "CoreCallable")
        .def_static(
            "build",
            [](std::vector<ArgDirection> signature, nb::bytes binary) -> PyCoreCallable {
                auto bin_ptr = reinterpret_cast<const void *>(binary.c_str());
                auto bin_size = static_cast<uint32_t>(binary.size());
                auto buf = make_callable<CORE_MAX_TENSOR_ARGS>(
                    signature.data(), static_cast<int32_t>(signature.size()), bin_ptr, bin_size
                );
                return PyCoreCallable{std::move(buf)};
            },
            nb::arg("signature"), nb::arg("binary"),
            "Build a CoreCallable from a signature list and binary bytes. The dump "
            "maps signature entry i to payload slot i positionally."
        )

        .def(
            "sig",
            [](const PyCoreCallable &self, int32_t i) -> ArgDirection {
                return self.get().sig(i);
            },
            nb::arg("i"), "Return the ArgDirection at signature index i."
        )

        .def_prop_ro(
            "sig_count",
            [](const PyCoreCallable &self) -> int32_t {
                return self.get().sig_count();
            },
            "Number of signature entries."
        )

        .def_prop_ro(
            "binary_size",
            [](const PyCoreCallable &self) -> uint32_t {
                return self.get().binary_size();
            },
            "Size of the binary payload in bytes."
        )

        .def(
            "buffer_ptr",
            [](const PyCoreCallable &self) -> uint64_t {
                return reinterpret_cast<uint64_t>(self.buffer_.data());
            },
            "Return the memory address of the underlying buffer."
        )

        .def(
            "buffer_size",
            [](const PyCoreCallable &self) -> size_t {
                return self.buffer_.size();
            },
            "Return the total size of the underlying buffer in bytes."
        )

        .def("__repr__", [](const PyCoreCallable &self) -> std::string {
            const auto &c = self.get();
            std::ostringstream os;
            os << "CoreCallable(sig_count=" << c.sig_count() << ", binary_size=" << c.binary_size() << ")";
            return os.str();
        });

    // --- PyChipCallable wrapper ---
    struct PyChipCallable {
        std::vector<uint8_t> buffer_;
        const ChipCallable &get() const { return *reinterpret_cast<const ChipCallable *>(buffer_.data()); }
    };

    nb::class_<PyChipCallable>(m, "ChipCallable")
        .def_static(
            "build",
            [](std::vector<ArgDirection> signature, std::string func_name, nb::bytes binary,
               std::vector<std::tuple<int32_t, PyCoreCallable>> children, std::string config_name,
               int32_t scalar_count) -> PyChipCallable {
                auto bin_ptr = reinterpret_cast<const void *>(binary.c_str());
                auto bin_size = static_cast<uint32_t>(binary.size());
                auto child_count = static_cast<int32_t>(children.size());

                std::vector<int32_t> func_ids(children.size());
                std::vector<std::vector<uint8_t>> child_bufs(children.size());
                for (size_t i = 0; i < children.size(); ++i) {
                    func_ids[i] = std::get<0>(children[i]);
                    child_bufs[i] = std::get<1>(children[i]).buffer_;
                }

                auto buf = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
                    signature.data(), static_cast<int32_t>(signature.size()), scalar_count, func_name.c_str(), bin_ptr,
                    bin_size, func_ids.data(), child_bufs.data(), child_count, config_name.c_str()
                );
                return PyChipCallable{std::move(buf)};
            },
            nb::arg("signature"), nb::arg("func_name"), nb::arg("binary"), nb::arg("children"),
            nb::arg("config_name") = "", nb::arg("scalar_count") = 0,
            "Build a ChipCallable from signature, func_name, binary, and list of (func_id, CoreCallable) children. "
            "scalar_count records how many scalar arguments the orchestration expects (0 when it takes none)."
        )

        .def_static(
            "from_bytes",
            [](nb::bytes raw) -> PyChipCallable {
                // Reconstruct a ChipCallable wrapper from the contiguous
                // serialised representation produced by `buffer_ptr()` /
                // `buffer_size()`. Used by the L4 cascade in
                // _child_worker_loop, which receives CTRL_REGISTER bytes
                // through shared memory and needs a typed ChipCallable for
                // digest-owned registration on the child Worker; see
                // docs/callable-identity-registration.md.
                std::vector<uint8_t> buf(
                    reinterpret_cast<const uint8_t *>(raw.c_str()),
                    reinterpret_cast<const uint8_t *>(raw.c_str()) + raw.size()
                );
                return PyChipCallable{std::move(buf)};
            },
            nb::arg("raw"),
            "Reconstruct a ChipCallable from the contiguous bytes that "
            "buffer_ptr() points to (size buffer_size()). Inverse of the "
            "serialisation used to ship a ChipCallable across the L4 "
            "cascade IPC channel."
        )

        .def(
            "sig",
            [](const PyChipCallable &self, int32_t i) -> ArgDirection {
                return self.get().sig(i);
            },
            nb::arg("i"), "Return the ArgDirection at signature index i."
        )

        .def_prop_ro(
            "sig_count",
            [](const PyChipCallable &self) -> int32_t {
                return self.get().sig_count();
            },
            "Number of signature entries."
        )

        .def_prop_ro(
            "binary_size",
            [](const PyChipCallable &self) -> uint32_t {
                return self.get().binary_size();
            },
            "Size of the binary payload in bytes."
        )

        .def_prop_ro(
            "func_name",
            [](const PyChipCallable &self) -> std::string {
                const auto &c = self.get();
                return std::string(c.func_name(), c.func_name_len());
            },
            "The orchestration function name."
        )

        .def_prop_ro(
            "config_name",
            [](const PyChipCallable &self) -> std::string {
                const auto &c = self.get();
                return std::string(c.config_name(), c.config_name_len());
            },
            "The optional orchestration config function name."
        )

        .def_prop_ro(
            "scalar_count",
            [](const PyChipCallable &self) -> int32_t {
                return self.get().scalar_count();
            },
            "Number of scalar arguments the orchestration expects. 0 also for "
            "legacy artifacts built before the count was recorded."
        )

        .def_prop_ro(
            "child_count",
            [](const PyChipCallable &self) -> int32_t {
                return self.get().child_count();
            },
            "Number of child callables."
        )

        .def(
            "child_func_id",
            [](const PyChipCallable &self, int32_t i) -> int32_t {
                return self.get().child_func_id(i);
            },
            nb::arg("i"), "Return the func_id for child at index i."
        )

        .def(
            "child",
            [](const PyChipCallable &self, int32_t i) -> PyCoreCallable {
                const auto &parent = self.get();
                const auto &c = parent.child(i);
                // Reconstruct a PyCoreCallable by copying the child's raw bytes
                auto offset = parent.child_offset(i);
                const uint8_t *child_start = reinterpret_cast<const uint8_t *>(parent.storage_ + offset);
                // Determine child size: from offset to next child or end of buffer
                size_t child_size;
                if (i + 1 < parent.child_count()) {
                    child_size = parent.child_offset(i + 1) - offset;
                } else {
                    size_t header_size = offsetof(ChipCallable, storage_);
                    child_size = self.buffer_.size() - header_size - offset;
                }
                std::vector<uint8_t> child_buf(child_start, child_start + child_size);
                return PyCoreCallable{std::move(child_buf)};
            },
            nb::arg("i"), "Return the CoreCallable child at index i."
        )

        .def(
            "child_offset",
            [](const PyChipCallable &self, int32_t i) -> uint32_t {
                return self.get().child_offset(i);
            },
            nb::arg("i"), "Return the byte offset of child i within storage (must be multiple of 64)."
        )

        .def(
            "buffer_ptr",
            [](const PyChipCallable &self) -> uint64_t {
                return reinterpret_cast<uint64_t>(self.buffer_.data());
            },
            "Return the memory address of the underlying buffer."
        )

        .def(
            "buffer_size",
            [](const PyChipCallable &self) -> size_t {
                return self.buffer_.size();
            },
            "Return the total size of the underlying buffer in bytes."
        )

        .def("__repr__", [](const PyChipCallable &self) -> std::string {
            const auto &c = self.get();
            std::ostringstream os;
            os << "ChipCallable(func_name=\"" << std::string(c.func_name(), c.func_name_len()) << "\", config_name=\""
               << std::string(c.config_name(), c.config_name_len()) << "\", sig_count=" << c.sig_count()
               << ", binary_size=" << c.binary_size() << ", child_count=" << c.child_count() << ")";
            return os.str();
        });

    // --- RuntimeEnv (per-task ring sizing; nested under CallConfig.runtime_env) ---
    // Each ring resource is exposed as ONE property that accepts either an int
    // (broadcast to every ring) or a list of RUNTIME_ENV_RING_COUNT ints
    // (per-ring). The value always reads back as a list — the wire layout is the
    // four-entry array, so a broadcast scalar is stored as [v, v, v, v].
    auto get_ring_values = [](const uint64_t values[RUNTIME_ENV_RING_COUNT]) -> std::vector<uint64_t> {
        std::vector<uint64_t> out;
        out.reserve(RUNTIME_ENV_RING_COUNT);
        for (int i = 0; i < RUNTIME_ENV_RING_COUNT; ++i) {
            out.push_back(values[i]);
        }
        return out;
    };
    auto set_ring_values = [](uint64_t values[RUNTIME_ENV_RING_COUNT], nb::handle obj, const char *name) {
        uint64_t scalar = 0;
        if (nb::try_cast<uint64_t>(obj, scalar)) {  // int -> broadcast to every ring
            for (int i = 0; i < RUNTIME_ENV_RING_COUNT; ++i) {
                values[i] = scalar;
            }
            return;
        }
        std::vector<uint64_t> input;
        if (nb::try_cast<std::vector<uint64_t>>(obj, input)) {  // list -> per-ring
            if (input.size() != RUNTIME_ENV_RING_COUNT) {
                throw std::invalid_argument(
                    std::string("RuntimeEnv.") + name + " list must contain exactly " +
                    std::to_string(RUNTIME_ENV_RING_COUNT) + " entries"
                );
            }
            for (int i = 0; i < RUNTIME_ENV_RING_COUNT; ++i) {
                values[i] = input[static_cast<size_t>(i)];
            }
            return;
        }
        throw std::invalid_argument(
            std::string("RuntimeEnv.") + name + " must be an int (broadcast) or a list of " +
            std::to_string(RUNTIME_ENV_RING_COUNT) + " ints"
        );
    };
    auto append_ring_values = [](std::ostringstream &os, const char *name, bool leading_comma,
                                 const uint64_t values[RUNTIME_ENV_RING_COUNT]) {
        if (leading_comma) {
            os << ", ";
        }
        os << name << "=[";
        for (int i = 0; i < RUNTIME_ENV_RING_COUNT; ++i) {
            if (i != 0) {
                os << ", ";
            }
            os << values[i];
        }
        os << "]";
    };

    nb::class_<RuntimeEnv>(m, "RuntimeEnv")
        .def(nb::init<>())
        .def_prop_rw(
            "ring_task_window",
            [get_ring_values](const RuntimeEnv &self) {
                return get_ring_values(self.ring_task_window);
            },
            [set_ring_values](RuntimeEnv &self, nb::handle value) {
                set_ring_values(self.ring_task_window, value, "ring_task_window");
            }
        )
        .def_prop_rw(
            "ring_heap",
            [get_ring_values](const RuntimeEnv &self) {
                return get_ring_values(self.ring_heap);
            },
            [set_ring_values](RuntimeEnv &self, nb::handle value) {
                set_ring_values(self.ring_heap, value, "ring_heap");
            }
        )
        .def_prop_rw(
            "ring_dep_pool",
            [get_ring_values](const RuntimeEnv &self) {
                return get_ring_values(self.ring_dep_pool);
            },
            [set_ring_values](RuntimeEnv &self, nb::handle value) {
                set_ring_values(self.ring_dep_pool, value, "ring_dep_pool");
            }
        )
        .def("__repr__", [append_ring_values](const RuntimeEnv &self) -> std::string {
            std::ostringstream os;
            os << "RuntimeEnv(";
            append_ring_values(os, "ring_task_window", false, self.ring_task_window);
            append_ring_values(os, "ring_heap", true, self.ring_heap);
            append_ring_values(os, "ring_dep_pool", true, self.ring_dep_pool);
            os << ")";
            return os.str();
        });

    // --- CallConfig ---
    nb::class_<CallConfig>(m, "CallConfig")
        .def(nb::init<>())
        .def_rw("aicpu_thread_num", &CallConfig::aicpu_thread_num)
        // runtime_env returns an internal reference so `cfg.runtime_env.ring_heap = X`
        // writes through to the owning CallConfig (rv_policy::reference_internal).
        .def_prop_rw(
            "runtime_env",
            [](CallConfig &c) -> RuntimeEnv & {
                return c.runtime_env;
            },
            [](CallConfig &c, const RuntimeEnv &re) {
                c.runtime_env = re;
            },
            nb::rv_policy::reference_internal
        )
        .def_prop_rw(
            "enable_chip_swimlane",
            [](const CallConfig &c) {
                return c.enable_chip_swimlane;
            },
            // Accept either an int perf_level (0-4) or a Python bool. `True` maps to
            // level 4 (full collection) to preserve the pre-perf_level semantics for
            // callers that still pass a boolean; `False` maps to 0.
            [](CallConfig &c, nb::object v) {
                if (PyBool_Check(v.ptr())) {
                    c.enable_chip_swimlane = nb::cast<bool>(v) ? 4 : 0;
                } else {
                    int level = nb::cast<int>(v);
                    c.enable_chip_swimlane = (level < 0) ? 0 : (level > 4) ? 4 : level;
                }
            }
        )
        // Accept either an int dump level (0=off, 1=partial, 2=full,
        // 3=hybrid) or a Python bool. `True` maps to level 1
        // (partial) — the default when --dump-args is passed without a
        // value; `False` maps to 0.
        .def_prop_rw(
            "enable_dump_args",
            [](const CallConfig &c) {
                return c.enable_dump_args;
            },
            [](CallConfig &c, nb::object v) {
                if (PyBool_Check(v.ptr())) {
                    c.enable_dump_args = nb::cast<bool>(v) ? 1 : 0;
                } else {
                    int level = nb::cast<int>(v);
                    c.enable_dump_args = (level < 0) ? 0 : (level > 3) ? 3 : level;
                }
            }
        )
        .def_rw("enable_pmu", &CallConfig::enable_pmu)
        .def("validate", &CallConfig::validate)
        .def_prop_rw(
            "enable_dep_gen",
            [](const CallConfig &c) {
                return static_cast<bool>(c.enable_dep_gen);
            },
            [](CallConfig &c, bool v) {
                c.enable_dep_gen = v ? 1 : 0;
            }
        )
        .def_prop_rw(
            "enable_scope_stats",
            [](const CallConfig &c) {
                return static_cast<bool>(c.enable_scope_stats);
            },
            [](CallConfig &c, bool v) {
                c.enable_scope_stats = v ? 1 : 0;
            }
        )
        .def_prop_rw(
            "capture_clock_anchors",
            [](const CallConfig &c) {
                return static_cast<bool>(c.capture_clock_anchors);
            },
            [](CallConfig &c, bool v) {
                c.capture_clock_anchors = v ? 1 : 0;
            }
        )
        .def_prop_rw(
            "output_prefix",
            [](const CallConfig &c) -> std::string {
                return std::string(c.output_prefix, ::strnlen(c.output_prefix, sizeof(c.output_prefix)));
            },
            [](CallConfig &c, const std::string &s) {
                if (s.size() >= sizeof(c.output_prefix)) {
                    throw std::invalid_argument(
                        "CallConfig.output_prefix length " + std::to_string(s.size()) + " exceeds buffer (" +
                        std::to_string(sizeof(c.output_prefix) - 1) + " bytes)"
                    );
                }
                std::memset(c.output_prefix, 0, sizeof(c.output_prefix));
                std::memcpy(c.output_prefix, s.data(), s.size());
            }
        )
        .def("__repr__", [append_ring_values](const CallConfig &self) -> std::string {
            std::ostringstream os;
            os << "CallConfig(aicpu_thread_num=" << self.aicpu_thread_num
               << ", enable_chip_swimlane=" << self.enable_chip_swimlane
               << ", enable_dump_args=" << self.enable_dump_args << ", enable_pmu=" << self.enable_pmu
               << ", enable_dep_gen=" << (self.enable_dep_gen ? "True" : "False")
               << ", enable_scope_stats=" << (self.enable_scope_stats ? "True" : "False")
               << ", capture_clock_anchors=" << (self.capture_clock_anchors ? "True" : "False");
            if (self.runtime_env.any()) {
                append_ring_values(os, "runtime_env.ring_task_window", true, self.runtime_env.ring_task_window);
                append_ring_values(os, "runtime_env.ring_heap", true, self.runtime_env.ring_heap);
                append_ring_values(os, "runtime_env.ring_dep_pool", true, self.runtime_env.ring_dep_pool);
            }
            if (self.output_prefix_set()) {
                os << ", output_prefix='" << self.output_prefix << "'";
            }
            os << ")";
            return os.str();
        });

    // Log default constant — single source. Mirrored in
    // src/common/log/include/common/log_level.h::simpler::log::kDefaultThreshold; if you change
    // one, change the other.
    m.attr("DEFAULT_LOG_THRESHOLD") = 25;  // TIMING

    // Per-stage run timing (host wall, on-NPU device wall + AICPU phase
    // breakdown) is no longer returned from run(); the platform emits it as
    // `[STRACE]` log markers — parse with simpler_setup.tools.strace_timing.

    nb::class_<DeviceMemoryInfo>(m, "DeviceMemoryInfo")
        .def_ro("free_bytes", &DeviceMemoryInfo::free_bytes)
        .def_ro("total_bytes", &DeviceMemoryInfo::total_bytes)
        .def(
            "__iter__",
            [](const DeviceMemoryInfo &self) {
                return nb::make_tuple(self.free_bytes, self.total_bytes).attr("__iter__")();
            }
        )
        .def("__repr__", [](const DeviceMemoryInfo &self) {
            return "DeviceMemoryInfo(free_bytes=" + std::to_string(self.free_bytes) +
                   ", total_bytes=" + std::to_string(self.total_bytes) + ")";
        });

    nb::class_<ChipWorkerNativeRun>(m, "_ChipWorkerNativeRun")
        .def_ro("slot_id", &ChipWorkerNativeRun::slot_id)
        .def_ro("generation", &ChipWorkerNativeRun::generation)
        .def_ro("run_epoch", &ChipWorkerNativeRun::run_epoch)
        .def_ro("run_id", &ChipWorkerNativeRun::run_id)
        .def_ro("dispatch_id", &ChipWorkerNativeRun::dispatch_id);

    nb::enum_<ChipRunPreparationDisposition>(m, "_ChipRunPreparationDisposition")
        .value("VALIDATED_ONLY", ChipRunPreparationDisposition::VALIDATED_ONLY)
        .value("NATIVE_PREPARED", ChipRunPreparationDisposition::NATIVE_PREPARED);

    nb::class_<ChipRun>(m, "_ChipRun")
        .def("done", &ChipRun::done)
        .def("activate", &ChipRun::activate)
        .def("abandon", &ChipRun::abandon, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("launched", &ChipRun::launched)
        .def_prop_ro("lane_poisoned", &ChipRun::lane_poisoned)
        .def_prop_ro("preparation_disposition", &ChipRun::preparation_disposition)
        .def(
            "wait",
            [](ChipRun &self, double timeout) {
                // NaN compares false against everything, so it reaches the cast
                // unless it is rejected by name. Converting a non-finite or
                // out-of-range double to the clock's integral rep is undefined,
                // and this timeout comes straight from Python.
                if (std::isnan(timeout)) throw std::invalid_argument("ChipRun.wait timeout must not be NaN");
                if (timeout < 0) return self.wait_until(ChipRun::Deadline::max());
                const std::chrono::duration<double> requested(timeout);
                const auto limit =
                    std::chrono::duration_cast<std::chrono::duration<double>>(ChipRun::Clock::duration::max());
                // Saturate rather than reject: a caller asking to wait longer
                // than the clock can express means "effectively forever", and
                // the unbounded path is the one that blocks on the device
                // instead of polling.
                if (requested >= limit) return self.wait_until(ChipRun::Deadline::max());
                return self.wait_until(
                    ChipRun::Clock::now() + std::chrono::duration_cast<ChipRun::Clock::duration>(requested)
                );
            },
            nb::arg("timeout") = -1.0, nb::call_guard<nb::gil_scoped_release>(),
            "Wait for this run's completion fence. A negative timeout waits without a deadline and blocks on the "
            "device rather than polling, as does one past the clock's range. Rejects NaN. Returns whether the run "
            "reached terminal; raises the run's error."
        )
        .def(
            "_raise_if_failed",
            [](ChipRun &self) {
                if (!self.done()) throw std::logic_error("ChipRun is not terminal");
                (void)self.wait_until(ChipRun::Deadline::max());
            },
            nb::call_guard<nb::gil_scoped_release>()
        );

    // --- ChipWorker ---
    nb::class_<ChipWorker>(m, "_ChipWorker")
        .def(nb::init<>())
        .def(
            "init",
            [](ChipWorker &self, const std::string &host_lib_path, const std::string &aicpu_path,
               const std::string &aicore_path, const std::string &dispatcher_path, int device_id,
               std::optional<CallConfig> prewarm_config, bool enable_sdma, const std::string &sim_context_path,
               const std::string &sdma_warmup_path) {
                self.init(
                    host_lib_path, aicpu_path, aicore_path, dispatcher_path, device_id,
                    prewarm_config.has_value() ? &(*prewarm_config) : nullptr, enable_sdma, sim_context_path,
                    sdma_warmup_path
                );
            },
            nb::arg("host_lib_path"), nb::arg("aicpu_path"), nb::arg("aicore_path"), nb::arg("dispatcher_path"),
            nb::arg("device_id"), nb::arg("prewarm_config") = nb::none(), nb::arg("enable_sdma") = false,
            nb::arg("sim_context_path") = "", nb::arg("sdma_warmup_path") = "",
            // Release the GIL for the (potentially long) native device attach so
            // another Python thread can run during it — e.g. a concurrent close()
            // observing INITIALIZING and failing fast (a GIL held for the whole
            // attach would block it until init returned). This does not make
            // ChipWorker cross-thread-safe: init/finalize still run on the owning
            // thread (enforced by Worker.close()).
            nb::call_guard<nb::gil_scoped_release>(),
            "Bind the runtime library and attach to device_id. When prewarm_config is "
            "given, its ring sizing is built + cached inside init (fork-constant, no "
            "cross-process control command). A no-op for runtimes without a prebuilt arena. "
            "When enable_sdma is True, provisions the async-DMA (SDMA) workspace at init so "
            "kernels can use get_dma_workspace; init raises if the platform lacks support for the requested workspace."
        )
        .def("finalize", &ChipWorker::finalize)
        .def(
            "register_callable",
            [](ChipWorker &self, int32_t callable_id, const PyChipCallable &callable) {
                self.register_callable(callable_id, callable.buffer_.data());
            },
            nb::arg("callable_id"), nb::arg("callable"),
            "Stage a ChipCallable under callable_id for cheap repeated launches "
            "via run. Variants without per-callable_id support raise."
        )
        .def(
            "register_callable_from_blob",
            [](ChipWorker &self, int32_t callable_id, uint64_t blob_ptr) {
                self.register_callable(callable_id, reinterpret_cast<const void *>(blob_ptr));
            },
            nb::arg("callable_id"), nb::arg("blob_ptr"),
            "Stage a ChipCallable from a raw contiguous-buffer pointer (used by "
            "post-fork dynamic register handlers that receive the ChipCallable "
            "bytes via shared memory; see docs/callable-identity-registration.md). "
            "Equivalent to register_callable(callable_id, ChipCallable) but accepts the "
            "ChipCallable layout pointer directly so chip-child loops can prepare "
            "from shm without rebuilding a PyChipCallable wrapper."
        )
        .def(
            "run",
            [](ChipWorker &self, int32_t callable_id, ChipStorageTaskArgs &args, const CallConfig &config) {
                self.run(callable_id, &args, config);
            },
            nb::arg("callable_id"), nb::arg("args"), nb::arg("config"),
            "Launch a callable_id previously staged via register_callable. Returns "
            "None; per-stage timing is emitted as `[STRACE]` log markers."
        )
        .def(
            "_run_with_pipeline_lease",
            [](ChipWorker &self, int32_t callable_id, ChipStorageTaskArgs &args, const CallConfig &config,
               uint32_t slot_id, uint64_t generation) {
                self.run_with_lease(callable_id, &args, config, PipelineSlotLease{slot_id, 0, generation});
            },
            nb::arg("callable_id"), nb::arg("args"), nb::arg("config"), nb::arg("slot_id"), nb::arg("generation"),
            "Internal generation-safe pipeline-slot launch. Takes the runtime.so-ABI POD, "
            "which is what every lease caller already holds."
        )
        .def(
            "_prepare_native_run_with_pipeline_lease",
            [](ChipWorker &self, int32_t callable_id, ChipStorageTaskArgs &args, const CallConfig &config,
               uint32_t slot_id, uint64_t generation) {
                return self.prepare_native_run(callable_id, &args, config, PipelineSlotLease{slot_id, 0, generation});
            },
            nb::arg("callable_id"), nb::arg("args"), nb::arg("config"), nb::arg("slot_id"), nb::arg("generation"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Prepare a native run from pre-encoded task args after lease admission."
        )
        .def(
            "_submit_chip_run_materialized",
            [](ChipWorker &self, int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config,
               uint32_t slot_id, uint64_t generation, uint64_t run_id, uint64_t dispatch_id,
               uint64_t accepted_state_addr, int32_t accepted_value, bool activated) {
                return self.submit_chip_run(
                    callable_id, args, config, PipelineSlotLease{slot_id, 0, generation}, run_id, dispatch_id,
                    reinterpret_cast<volatile int32_t *>(accepted_state_addr), accepted_value, activated
                );
            },
            nb::arg("callable_id"), nb::arg("args"), nb::arg("config"), nb::arg("slot_id"), nb::arg("generation"),
            nb::arg("run_id"), nb::arg("dispatch_id"), nb::arg("accepted_state_addr"), nb::arg("accepted_value"),
            nb::arg("activated"), nb::call_guard<nb::gil_scoped_release>(),
            "Submit materialized task args to the chip native-run lane."
        )
        .def(
            "_close_chip_run_lane", &ChipWorker::close_chip_run_lane, nb::call_guard<nb::gil_scoped_release>(),
            "Drain active native ownership, abandon unlaunched work, and close the chip run lane."
        )
        .def(
            "_submit_chip_run_direct",
            [](ChipWorker &self, int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config) {
                return self.submit_chip_run(callable_id, args, config);
            },
            nb::arg("callable_id"), nb::arg("args"), nb::arg("config"), nb::call_guard<nb::gil_scoped_release>(),
            "Submit materialized task args to the chip native-run lane without a pipeline lease and return the live "
            "run. The lane follows the runtime PipelineContract: compatible runs admit one active plus one prepared "
            "successor; otherwise this call drains its predecessor before admitting."
        )
        .def(
            "_prepare_native_run_materialized",
            [](ChipWorker &self, int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config,
               uint32_t slot_id, uint64_t generation, uint64_t run_id, uint64_t dispatch_id,
               uint64_t accepted_state_addr, int32_t accepted_value) {
                return self.prepare_native_run(
                    callable_id, &args, config, PipelineSlotLease{slot_id, 0, generation}, run_id, dispatch_id,
                    reinterpret_cast<volatile int32_t *>(accepted_state_addr), accepted_value
                );
            },
            nb::arg("callable_id"), nb::arg("args"), nb::arg("config"), nb::arg("slot_id"), nb::arg("generation"),
            nb::arg("run_id") = 0, nb::arg("dispatch_id") = 0, nb::arg("accepted_state_addr") = 0,
            nb::arg("accepted_value") = 0, nb::call_guard<nb::gil_scoped_release>(),
            "Prepare a native run from materialized task args after lease admission."
        )
        .def(
            "_launch_native_run", &ChipWorker::launch_native_run, nb::arg("run"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Launch a prepared native run and return after its real device launch fence."
        )
        .def(
            "_poll_native_run", &ChipWorker::poll_native_run, nb::arg("run"),
            "Return whether a launched native run has reached its completion fence."
        )
        .def(
            "_wait_native_run", &ChipWorker::wait_native_run, nb::arg("run"), nb::call_guard<nb::gil_scoped_release>(),
            "Wait for a launched native run's completion fence."
        )
        .def(
            "_finalize_native_run", &ChipWorker::finalize_native_run, nb::arg("run"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Validate, copy back, emit diagnostics, and destroy a prepared native run."
        )
        .def(
            "run_materialized",
            [](ChipWorker &self, int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config,
               uint64_t accepted_state_addr, int32_t accepted_value, uint32_t pipeline_slot,
               uint64_t pipeline_generation) {
                if (pipeline_generation == 0) {
                    self.run(
                        callable_id, &args, config, reinterpret_cast<volatile int32_t *>(accepted_state_addr),
                        accepted_value
                    );
                } else {
                    self.run_with_lease(
                        callable_id, &args, config, PipelineSlotLease{pipeline_slot, 0, pipeline_generation},
                        reinterpret_cast<volatile int32_t *>(accepted_state_addr), accepted_value
                    );
                }
            },
            nb::arg("callable_id"), nb::arg("args"), nb::arg("config"), nb::arg("accepted_state_addr") = 0,
            nb::arg("accepted_value") = 0, nb::arg("pipeline_slot") = 0, nb::arg("pipeline_generation") = 0,
            "Launch a callable_id from the runtime.so-ABI POD a chip-child mailbox loop built with "
            "materialize_task_args, so no Python code re-implements the tensor/scalar layout."
        )
        .def(
            "unregister_callable",
            [](ChipWorker &self, int32_t callable_id) {
                self.unregister_callable(callable_id);
            },
            nb::arg("callable_id"),
            "Drop the prepared state for callable_id; releases the per-id share "
            "of the device orch SO buffer (kernel binaries stay resident until "
            "finalize)."
        )
        .def_prop_ro("device_id", &ChipWorker::device_id)
        .def_prop_ro("initialized", &ChipWorker::initialized)
        .def_prop_ro("pipeline_depth", &ChipWorker::pipeline_depth)
        .def_prop_ro("runtime_slot_count", &ChipWorker::runtime_slot_count)
        .def_prop_ro(
            "supports_concurrent_native_prepare", &ChipWorker::supports_concurrent_native_prepare,
            "Whether non-diagnostic native preparation may overlap one active run in another slot."
        )
        .def_prop_ro(
            "runtime_buffer_addrs", &ChipWorker::runtime_buffer_addrs,
            "Host Runtime staging buffer address of every copy the runtime's "
            "PipelineContract asked for, in slot order."
        )
        .def(
            "retained_temp_addr", &ChipWorker::retained_temp_addr, nb::arg("slot_id"),
            "Retained temporary-buffer address held for one pipeline slot, or 0 "
            "while that slot holds none."
        )
        .def(
            "arena_bank_gm_heap_base", &ChipWorker::arena_bank_gm_heap_base, nb::arg("bank_id"),
            "Committed GM heap base of one arena bank on the bound runner, or 0 "
            "when that bank has never been committed."
        )
        .def_prop_ro(
            "aicpu_dlopen_count", &ChipWorker::aicpu_dlopen_count,
            "Number of distinct callable entries the AICPU has dlopened for on the "
            "bound device. Equals 0 when not initialized or the runtime "
            "variant lacks prepared-callable registration. Tests assert this to verify "
            "register_callable + repeated run do not redundantly dlopen."
        )
        .def_prop_ro(
            "host_dlopen_count", &ChipWorker::host_dlopen_count,
            "Number of host-side dlopens triggered by register_callable on "
            "host_build_graph variants. Mirrors aicpu_dlopen_count for the "
            "host-orchestration path; 0 on device-orch variants."
        )
        .def_prop_ro(
            "run_stream_set_create_count", &ChipWorker::run_stream_set_create_count,
            "Number of AICore run streams the bound runner has created. One AICPU "
            "+ AICore pair serves every run for the runner's lifetime. The AICPU "
            "stream persists; the AICore stream is recreated when a new code upload "
            "makes it stale, and destroyed when an unproven completion retires it. "
            "Platforms using the persistent bootstrap pair report 0."
        )
        .def_prop_ro(
            "committed_device_memory", &ChipWorker::committed_device_memory,
            "Total device HBM (bytes) currently committed by this worker's "
            "MemoryAllocator (user tensors + pooled arenas + runtime buffers). "
            "Excludes HCCL/VMM comm windows. 0 when not "
            "initialized. Lets downstream runtimes subtract simpler's own HBM "
            "from their cache budget (it may be invisible to aclrtGetMemInfo)."
        )
        .def(
            "device_memory_info",
            [](const ChipWorker &self) {
                try {
                    return self.device_memory_info();
                } catch (const UnsupportedRuntimeOperation &e) {
                    PyErr_SetString(PyExc_NotImplementedError, e.what());
                    throw nb::python_error();
                }
            },
            "Return the ACL_HBM_MEM free/total byte snapshot for this worker's device."
        )
        .def("malloc", &ChipWorker::malloc, nb::arg("size"))
        .def("free", &ChipWorker::free, nb::arg("ptr"))
        .def("copy_to", &ChipWorker::copy_to, nb::arg("dst"), nb::arg("src"), nb::arg("size"))
        .def("copy_from", &ChipWorker::copy_from, nb::arg("dst"), nb::arg("src"), nb::arg("size"))
        .def(
            "comm_init", &ChipWorker::comm_init, nb::arg("rank"), nb::arg("nranks"), nb::arg("rootinfo_path"),
            "Initialize a communicator for this rank.  ChipWorker owns ACL + stream "
            "lifetime internally (onboard drives ensure_acl_ready + aclrtCreateStream; "
            "sim ignores both).  Pair with comm_destroy for cleanup."
        )
        .def(
            "comm_alloc_windows", &ChipWorker::comm_alloc_windows, nb::arg("comm_handle"), nb::arg("win_size"),
            "Allocate per-rank windows and return the device CommContext pointer."
        )
        .def(
            "comm_get_local_window_base", &ChipWorker::comm_get_local_window_base, nb::arg("comm_handle"),
            "Return this rank's local window base address."
        )
        .def(
            "comm_get_window_size", &ChipWorker::comm_get_window_size, nb::arg("comm_handle"),
            "Return the actual per-rank window size (may differ from the hint)."
        )
        .def(
            "comm_derive_context", &ChipWorker::comm_derive_context, nb::arg("comm_handle"), nb::arg("rank_ids"),
            nb::arg("domain_rank"), nb::arg("window_offset"), nb::arg("window_size"),
            "Derive a domain-local CommContext from an allocated base communicator."
        )
        .def(
            "comm_alloc_domain_windows",
            [](ChipWorker &self, uint64_t comm_handle, uint64_t allocation_id, const std::vector<uint32_t> &rank_ids,
               uint32_t domain_rank, size_t window_size, uint64_t commit_flag_address) {
                if (commit_flag_address == 0 || commit_flag_address % alignof(uint64_t) != 0) {
                    throw std::invalid_argument("comm_alloc_domain_windows: commit flag address is invalid");
                }
                auto [device_ctx, local_window_base] =
                    self.comm_alloc_domain_windows(comm_handle, allocation_id, rank_ids, domain_rank, window_size);
                __atomic_store_n(
                    reinterpret_cast<uint64_t *>(static_cast<uintptr_t>(commit_flag_address)), uint64_t{1},
                    __ATOMIC_RELEASE
                );
                return nb::make_tuple(device_ctx, local_window_base);
            },
            nb::arg("comm_handle"), nb::arg("allocation_id"), nb::arg("rank_ids"), nb::arg("domain_rank"),
            nb::arg("window_size"), nb::arg("commit_flag_address"),
            "Collectively allocate a fresh per-rank pool for a subset; returns "
            "(device_ctx, local_window_base) for this rank and publishes the commit flag before result conversion."
        )
        .def(
            "comm_release_domain_windows", &ChipWorker::comm_release_domain_windows, nb::arg("comm_handle"),
            nb::arg("allocation_id"), nb::arg("rank_count"), nb::arg("domain_rank"),
            "Pair to comm_alloc_domain_windows: collectively release the per-rank pool."
        )
        .def(
            "comm_global_domain_prepare",
            [](ChipWorker &self, uint64_t domain_id, uint32_t domain_rank, uint32_t rank_count, size_t window_size,
               uint32_t profile) {
                auto [descriptor, local_window_base, actual_window_size] =
                    self.comm_global_domain_prepare(domain_id, domain_rank, rank_count, window_size, profile);
                return nb::make_tuple(
                    nb::bytes(reinterpret_cast<const char *>(descriptor.data()), descriptor.size()), local_window_base,
                    actual_window_size
                );
            },
            nb::arg("domain_id"), nb::arg("domain_rank"), nb::arg("rank_count"), nb::arg("window_size"),
            nb::arg("profile"), "Create a Global CommDomain local window and return its transport descriptor."
        )
        .def(
            "comm_global_domain_import",
            [](ChipWorker &self, uint64_t domain_id, nb::bytes descriptors) {
                std::vector<uint8_t> descriptor_bytes(
                    reinterpret_cast<const uint8_t *>(descriptors.c_str()),
                    reinterpret_cast<const uint8_t *>(descriptors.c_str()) + descriptors.size()
                );
                return self.comm_global_domain_import(domain_id, descriptor_bytes);
            },
            nb::arg("domain_id"), nb::arg("descriptors"),
            "Import a rank-ordered Global CommDomain descriptor table and return the device context."
        )
        .def(
            "comm_global_domain_release", &ChipWorker::comm_global_domain_release, nb::arg("domain_id"),
            "Release a prepared or imported Global CommDomain."
        )
        .def("comm_barrier", &ChipWorker::comm_barrier, nb::arg("comm_handle"), "Synchronize all ranks.")
        .def(
            "comm_destroy", &ChipWorker::comm_destroy, nb::arg("comm_handle"),
            "Destroy the communicator and release its resources."
        )
        .def("comm_destroy_all", &ChipWorker::comm_destroy_all, "Destroy all owned communicators in LIFO order.");

    // --- Standalone blob helpers ---

    m.def(
        "materialize_task_args",
        [](const TaskArgs &args, nb::dict resolved) -> ChipStorageTaskArgs {
            ChipStorageTaskArgs out;
            for (int32_t i = 0; i < args.tensor_count(); i++) {
                out.add_tensor(materialize_one(args.tensor(i), resolved));
            }
            for (int32_t i = 0; i < args.scalar_count(); i++) {
                out.add_scalar(args.scalar(i));
            }
            return out;
        },
        nb::arg("args"), nb::arg("resolved"),
        "Materialize a TaskArgs held in this process into the runtime.so-ABI ChipStorageTaskArgs "
        "POD — the sole path to that POD, whether the args are an L2 leaf's own or a chip child's "
        "read back from its mailbox with read_args_from_blob. Each tensor's embedded buffer "
        "identity is resolved via `resolved` {CanonicalIdentity: (local_base, address_space)}; "
        "addr = base + byte_offset. The caller pre-populates `resolved` by materializing each "
        "embedded descriptor on first receipt. Strided views (transpose / permute / step-slice) "
        "materialize to strided ChipTensors. Rejects an unknown identity and a non-dtype-aligned "
        "byte_offset."
    );

    m.def(
        "read_args_from_blob",
        [](uint64_t blob_ptr, size_t capacity) -> TaskArgs {
            TaskArgsView view = read_blob(reinterpret_cast<const uint8_t *>(blob_ptr), capacity);
            TaskArgs args;
            for (int32_t i = 0; i < view.tensor_count; i++) {
                args.add_tensor(view.tensors(i));
            }
            for (int32_t i = 0; i < view.scalar_count; i++) {
                args.add_scalar(view.scalars[i]);
            }
            return args;
        },
        nb::arg("blob_ptr"), nb::arg("capacity"),
        "Reconstruct a TaskArgs from the length-prefixed blob at blob_ptr. `capacity` bounds how far "
        "the reader may walk and belongs to the caller's mapping — the mailbox frame's args region, "
        "or the length of a buffer the caller owns. Every element is gated by validate_tensor on the "
        "way out. Tags are not preserved (the wire format strips them)."
    );

    nb::class_<RegionVmmExport>(m, "_RegionVmmExport")
        .def_ro("device_addr", &RegionVmmExport::device_addr)
        .def_ro("mapping_bytes", &RegionVmmExport::mapping_bytes)
        .def_ro("shareable_handle", &RegionVmmExport::shareable_handle)
        .def_ro("registry_handle", &RegionVmmExport::registry_handle);

    nb::class_<RegionVmmInspect>(m, "_RegionVmmInspect")
        .def_ro("present", &RegionVmmInspect::present)
        .def_ro("device_id", &RegionVmmInspect::device_id)
        .def_ro("device_addr", &RegionVmmInspect::device_addr)
        .def_ro("mapping_bytes", &RegionVmmInspect::mapping_bytes)
        .def_ro("shareable_handle", &RegionVmmInspect::shareable_handle)
        .def_ro("physical_handle", &RegionVmmInspect::physical_handle)
        .def_ro("physical_allocated", &RegionVmmInspect::physical_allocated)
        .def_ro("va_reserved", &RegionVmmInspect::va_reserved)
        .def_ro("mapped", &RegionVmmInspect::mapped)
        .def_ro("access_set", &RegionVmmInspect::access_set)
        .def_ro("exported", &RegionVmmInspect::exported)
        .def_ro("unmap_attempted", &RegionVmmInspect::unmap_attempted)
        .def_ro("unmap_complete", &RegionVmmInspect::unmap_complete)
        .def_ro("va_release_attempted", &RegionVmmInspect::va_release_attempted)
        .def_ro("va_release_complete", &RegionVmmInspect::va_release_complete)
        .def_ro("physical_free_attempted", &RegionVmmInspect::physical_free_attempted)
        .def_ro("physical_free_complete", &RegionVmmInspect::physical_free_complete)
        .def_ro("release_once_done", &RegionVmmInspect::release_once_done)
        .def_ro("first_cleanup_failure", &RegionVmmInspect::first_cleanup_failure)
        .def_ro("local_cleanup_details", &RegionVmmInspect::local_cleanup_details);

    nb::class_<RegionHandle>(m, "_RegionHandle").def("__int__", &RegionHandle::value);

    m.def(
        "_region_import_sim",
        [](const std::string &token, uint64_t mapping_bytes, const std::string &owner_token) -> RegionHandle {
            return import_sim_region(
                token, mapping_bytes, owner_token, "mapped-region sim import requires a positive mapping size",
                "mapped-region import requires a non-empty owner token", "mapped-region sim import shm_open failed",
                "mapped-region sim import mmap failed: "
            );
        },
        nb::arg("token"), nb::arg("mapping_bytes"), nb::arg("owner_token"), nb::call_guard<nb::gil_scoped_release>(),
        "Import a sim POSIX shm mapped region."
    );
    m.def(
        "_region_import_onboard",
        [](int device_id, uint64_t shareable_handle, uint64_t mapping_bytes,
           const std::string &owner_token) -> RegionHandle {
            return import_onboard_region(
                device_id, shareable_handle, mapping_bytes, owner_token,
                "onboard mapped-region import requires a non-negative device id",
                "onboard mapped-region import requires a positive mapping size",
                "mapped-region import requires a non-empty owner token"
            );
        },
        nb::arg("device_id"), nb::arg("shareable_handle"), nb::arg("mapping_bytes"), nb::arg("owner_token"),
        nb::call_guard<nb::gil_scoped_release>(), "Import an onboard VMM mapped region."
    );
    m.def(
        "_region_close",
        [](uint64_t handle) {
            close_region(handle);
        },
        nb::arg("handle"), nb::call_guard<nb::gil_scoped_release>(), "Close a mapped-region handle."
    );
    m.def(
        "_region_active_leases",
        [](uint64_t handle) {
            return region_registry().active_leases(handle);
        },
        nb::arg("handle"), "Return the number of in-flight native operations holding this mapped region."
    );
    m.def(
        "_region_closing_for_test",
        [](uint64_t handle) {
            return region_registry().closing(handle);
        },
        nb::arg("handle"), "Report whether close is waiting for mapped-region leases."
    );
    m.def(
        "_region_take_cleanup_error",
        [](const std::string &owner_token) {
            if (owner_token.empty()) {
                throw std::invalid_argument("cleanup-error lookup requires a non-empty owner token");
            }
            return region_cleanup_errors().take(owner_token);
        },
        nb::arg("owner_token"), "Take a mapped-region cleanup error for one owner."
    );
    m.def(
        "_region_peek_cleanup_error",
        [](const std::string &owner_token) {
            if (owner_token.empty()) {
                throw std::invalid_argument("cleanup-error lookup requires a non-empty owner token");
            }
            return region_cleanup_errors().peek(owner_token);
        },
        nb::arg("owner_token"), "Read one mapped-region cleanup error without consuming it."
    );
    m.def(
        "_region_ack_cleanup_error",
        [](const std::string &owner_token, const std::string &observed) {
            if (owner_token.empty()) {
                throw std::invalid_argument("cleanup-error acknowledgement requires an owner token");
            }
            region_cleanup_errors().acknowledge(owner_token, observed);
        },
        nb::arg("owner_token"), nb::arg("observed"), "Acknowledge a mapped-region cleanup error."
    );
    m.def(
        "_region_record_cleanup_error_for_test",
        [](const std::string &owner_token, const std::string &message) {
            if (owner_token.empty()) {
                throw std::invalid_argument("cleanup-error injection requires a non-empty owner token");
            }
            region_cleanup_errors().record(owner_token, message);
        },
        nb::arg("owner_token"), nb::arg("message"), "Inject one owner-scoped mapped-region cleanup error."
    );
    m.def(
        "_region_fail_next_registry_insert_for_test",
        []() {
            region_registry().fail_next_insert_for_test();
        },
        "Inject one mapped-region registry insertion failure after native acquisition."
    );
    m.def(
        "_host_vmm_copy_to",
        [](uint64_t handle, uint64_t mapping_offset, uint64_t host_ptr, uint64_t nbytes) {
            if (host_ptr == 0) {
                throw std::invalid_argument("host_vmm_copy_to host_ptr must be nonzero");
            }
            RegionLease mapping = region_registry().lease(handle);
            mapping->copy_to(mapping_offset, reinterpret_cast<const void *>(static_cast<uintptr_t>(host_ptr)), nbytes);
        },
        nb::arg("handle"), nb::arg("mapping_offset"), nb::arg("host_ptr"), nb::arg("nbytes"),
        nb::call_guard<nb::gil_scoped_release>(), "Copy host bytes into an imported mapped-region byte range."
    );
    m.def(
        "_host_vmm_copy_to_with_delay_for_test",
        [](uint64_t handle, uint64_t mapping_offset, uint64_t host_ptr, uint64_t nbytes, uint64_t delay_ms) {
            if (host_ptr == 0) {
                throw std::invalid_argument("host_vmm_copy_to host_ptr must be nonzero");
            }
            RegionLease mapping = region_registry().lease(handle);
            mapping->copy_to(mapping_offset, reinterpret_cast<const void *>(static_cast<uintptr_t>(host_ptr)), nbytes);
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        },
        nb::arg("handle"), nb::arg("mapping_offset"), nb::arg("host_ptr"), nb::arg("nbytes"), nb::arg("delay_ms"),
        nb::call_guard<nb::gil_scoped_release>(),
        "Copy host bytes into an imported mapped-region byte range while holding the native lease for tests."
    );
    m.def(
        "_host_vmm_copy_from",
        [](uint64_t handle, uint64_t mapping_offset, uint64_t host_ptr, uint64_t nbytes) {
            if (host_ptr == 0) {
                throw std::invalid_argument("host_vmm_copy_from host_ptr must be nonzero");
            }
            RegionLease mapping = region_registry().lease(handle);
            mapping->copy_from(reinterpret_cast<void *>(static_cast<uintptr_t>(host_ptr)), mapping_offset, nbytes);
        },
        nb::arg("handle"), nb::arg("mapping_offset"), nb::arg("host_ptr"), nb::arg("nbytes"),
        nb::call_guard<nb::gil_scoped_release>(), "Copy imported mapped-region bytes into host memory."
    );
    m.def(
        "_region_counter_notify",
        [](uint64_t handle, uint64_t counter_offset, int32_t value, int op) {
            RegionLease mapping = region_registry().lease(handle);
            mapping->notify_counter(counter_offset, value, checked_region_notify_op(op));
        },
        nb::arg("handle"), nb::arg("counter_offset"), nb::arg("value"), nb::arg("op"),
        nb::call_guard<nb::gil_scoped_release>(), "Store or add one mapped-region signal counter."
    );
    m.def(
        "_region_counter_test",
        [](uint64_t handle, uint64_t counter_offset, int32_t operand, int cmp) -> std::tuple<bool, int32_t> {
            RegionLease mapping = region_registry().lease(handle);
            return mapping->test_counter(counter_offset, operand, checked_region_wait_cmp(cmp));
        },
        nb::arg("handle"), nb::arg("counter_offset"), nb::arg("operand"), nb::arg("cmp"),
        nb::call_guard<nb::gil_scoped_release>(), "Load and compare one mapped-region signal counter."
    );
    m.def(
        "_region_counter_wait",
        [](uint64_t handle, uint64_t counter_offset, int32_t operand, int cmp,
           uint64_t timeout_ns) -> std::tuple<int, int, int32_t, bool, std::string> {
            RegionLease mapping = region_registry().lease(handle);
            return mapping->wait_counter(counter_offset, operand, checked_region_wait_cmp(cmp), timeout_ns);
        },
        nb::arg("handle"), nb::arg("counter_offset"), nb::arg("operand"), nb::arg("cmp"), nb::arg("timeout_ns"),
        nb::call_guard<nb::gil_scoped_release>(), "Poll one mapped-region signal counter until match or timeout."
    );

    m.def(
        "_worker_host_mapped_region_import_sim",
        [](const std::string &token, uint64_t mapping_bytes, const std::string &owner_token) -> RegionHandle {
            return import_sim_region(
                token, mapping_bytes, owner_token,
                "L3-L2 sim L3 Host mapped-region import requires a positive mapping size",
                "L3-L2 mapped-region import requires a non-empty Worker owner token",
                "L3-L2 sim L3 Host mapped-region import shm_open failed",
                "L3-L2 sim L3 Host mapped-region import mmap failed: "
            );
        },
        nb::arg("token"), nb::arg("mapping_bytes"), nb::arg("owner_token"), nb::call_guard<nb::gil_scoped_release>(),
        "Import a sim L3-L2 POSIX shm region for L3 Host mapped-region access."
    );
    m.def(
        "_worker_host_mapped_region_import_onboard",
        [](int device_id, uint64_t shareable_handle, uint64_t mapping_bytes,
           const std::string &owner_token) -> RegionHandle {
            return import_onboard_region(
                device_id, shareable_handle, mapping_bytes, owner_token,
                "L3-L2 onboard mapped-region import requires a non-negative device id",
                "L3-L2 onboard mapped-region import requires a positive mapping size",
                "L3-L2 mapped-region import requires a non-empty Worker owner token"
            );
        },
        nb::arg("device_id"), nb::arg("shareable_handle"), nb::arg("mapping_bytes"), nb::arg("owner_token"),
        nb::call_guard<nb::gil_scoped_release>(), "Import an onboard VMM L3-L2 region for L3 Host mapped-region access."
    );
    m.def(
        "_worker_host_mapped_region_close",
        [](uint64_t handle) {
            close_region(handle);
        },
        nb::arg("handle"), nb::call_guard<nb::gil_scoped_release>(), "Close an L3 Host mapped-region handle."
    );
    m.def(
        "_worker_host_mapped_region_active_leases",
        [](uint64_t handle) {
            return region_registry().active_leases(handle);
        },
        nb::arg("handle"), "Return the number of in-flight native operations holding this mapped region."
    );
    m.def(
        "_worker_host_mapped_region_closing_for_test",
        [](uint64_t handle) {
            return region_registry().closing(handle);
        },
        nb::arg("handle"), "Report whether close is waiting for L3 Host mapped-region leases."
    );
    m.def(
        "_worker_host_mapped_region_take_cleanup_error",
        [](const std::string &owner_token) {
            if (owner_token.empty()) {
                throw std::invalid_argument("L3-L2 cleanup-error lookup requires a non-empty Worker owner token");
            }
            return region_cleanup_errors().take(owner_token);
        },
        nb::arg("owner_token"),
        "Take a cleanup error recorded by an unadopted native mapped-region owner for one Worker."
    );
    m.def(
        "_worker_host_mapped_region_peek_cleanup_error",
        [](const std::string &owner_token) {
            if (owner_token.empty()) {
                throw std::invalid_argument("L3-L2 cleanup-error lookup requires a non-empty Worker owner token");
            }
            return region_cleanup_errors().peek(owner_token);
        },
        nb::arg("owner_token"), "Read one Worker's mapped-region cleanup error without consuming it."
    );
    m.def(
        "_worker_host_mapped_region_ack_cleanup_error",
        [](const std::string &owner_token, const std::string &observed) {
            if (owner_token.empty()) {
                throw std::invalid_argument("L3-L2 cleanup-error acknowledgement requires a Worker owner token");
            }
            region_cleanup_errors().acknowledge(owner_token, observed);
        },
        nb::arg("owner_token"), nb::arg("observed"),
        "Acknowledge the mapped-region cleanup error already published by one Worker."
    );
    m.def(
        "_worker_host_mapped_region_record_cleanup_error_for_test",
        [](const std::string &owner_token, const std::string &message) {
            if (owner_token.empty()) {
                throw std::invalid_argument("L3-L2 cleanup-error injection requires a non-empty Worker owner token");
            }
            region_cleanup_errors().record(owner_token, message);
        },
        nb::arg("owner_token"), nb::arg("message"), "Inject one Worker-owned mapped-region cleanup error."
    );
    m.def(
        "_worker_host_mapped_region_fail_next_registry_insert_for_test",
        []() {
            region_registry().fail_next_insert_for_test();
        },
        "Inject one mapped-region registry insertion failure after native acquisition."
    );
    m.def(
        "_worker_host_mapped_payload_write",
        [](uint64_t handle, uint64_t payload_offset, uint64_t host_ptr, uint64_t nbytes) {
            if (host_ptr == 0) {
                throw std::invalid_argument("L3-L2 payload_write host_ptr must be nonzero");
            }
            RegionLease mapping = region_registry().lease(handle);
            mapping->copy_to(payload_offset, reinterpret_cast<const void *>(static_cast<uintptr_t>(host_ptr)), nbytes);
        },
        nb::arg("handle"), nb::arg("payload_offset"), nb::arg("host_ptr"), nb::arg("nbytes"),
        nb::call_guard<nb::gil_scoped_release>(), "Copy L3 Host bytes into an imported L3-L2 payload range."
    );
    m.def(
        "_worker_host_mapped_payload_read",
        [](uint64_t handle, uint64_t payload_offset, uint64_t host_ptr, uint64_t nbytes) {
            if (host_ptr == 0) {
                throw std::invalid_argument("L3-L2 payload_read host_ptr must be nonzero");
            }
            RegionLease mapping = region_registry().lease(handle);
            mapping->copy_from(reinterpret_cast<void *>(static_cast<uintptr_t>(host_ptr)), payload_offset, nbytes);
        },
        nb::arg("handle"), nb::arg("payload_offset"), nb::arg("host_ptr"), nb::arg("nbytes"),
        nb::call_guard<nb::gil_scoped_release>(), "Copy imported L3-L2 payload bytes into L3 Host memory."
    );
    m.def(
        "_worker_host_mapped_counter_notify",
        [](uint64_t handle, uint64_t counter_offset, int32_t value, int op) {
            RegionLease mapping = region_registry().lease(handle);
            mapping->notify_counter(counter_offset, value, checked_region_notify_op(op));
        },
        nb::arg("handle"), nb::arg("counter_offset"), nb::arg("value"), nb::arg("op"),
        nb::call_guard<nb::gil_scoped_release>(), "Store or add one L3 Host-side L3-L2 signal counter."
    );
    m.def(
        "_worker_host_mapped_counter_test",
        [](uint64_t handle, uint64_t counter_offset, int32_t operand, int cmp) -> std::tuple<bool, int32_t> {
            RegionLease mapping = region_registry().lease(handle);
            return mapping->test_counter(counter_offset, operand, checked_region_wait_cmp(cmp));
        },
        nb::arg("handle"), nb::arg("counter_offset"), nb::arg("operand"), nb::arg("cmp"),
        nb::call_guard<nb::gil_scoped_release>(), "Load and compare one L3 Host-side L3-L2 signal counter."
    );
    m.def(
        "_worker_host_mapped_counter_wait",
        [](uint64_t handle, uint64_t counter_offset, int32_t operand, int cmp,
           uint64_t timeout_ns) -> std::tuple<int, int, int32_t, bool, std::string> {
            RegionLease mapping = region_registry().lease(handle);
            return mapping->wait_counter(counter_offset, operand, checked_region_wait_cmp(cmp), timeout_ns);
        },
        nb::arg("handle"), nb::arg("counter_offset"), nb::arg("operand"), nb::arg("cmp"), nb::arg("timeout_ns"),
        nb::call_guard<nb::gil_scoped_release>(), "Poll one L3 Host-side L3-L2 signal counter until match or timeout."
    );
    m.def(
        "_region_vmm_begin",
        [](int device_id) -> uint64_t {
            return region_vmm_begin(device_id);
        },
        nb::arg("device_id"), nb::call_guard<nb::gil_scoped_release>(),
        "Insert an empty native VMM owner record with no physical side effect."
    );
    m.def(
        "_region_vmm_allocate_export",
        [](uint64_t handle, uint64_t logical_bytes) -> RegionVmmExport {
            return region_vmm_allocate_export(handle, logical_bytes);
        },
        nb::arg("handle"), nb::arg("logical_bytes"), nb::call_guard<nb::gil_scoped_release>(),
        "Physically allocate, map, and export one native VMM record without rollback."
    );
    m.def(
        "_region_vmm_release",
        [](uint64_t handle) {
            region_vmm_release(handle);
        },
        nb::arg("handle"), nb::call_guard<nb::gil_scoped_release>(),
        "Make one dependency-aware cleanup pass and delete the record only after complete success."
    );
    m.def(
        "_region_vmm_zero_bytes",
        [](uint64_t handle, uint64_t offset, uint64_t nbytes) {
            region_vmm_zero_range(handle, offset, nbytes);
        },
        nb::arg("handle"), nb::arg("offset"), nb::arg("nbytes"), nb::call_guard<nb::gil_scoped_release>(),
        "Zero a mapped native VMM byte range."
    );
    m.def(
        "_region_vmm_inspect",
        [](uint64_t handle) -> RegionVmmInspect {
            return region_vmm_inspect(handle);
        },
        nb::arg("handle"), "Inspect one native VMM ledger record."
    );
    m.def(
        "_region_vmm_test_use_fake_driver",
        []() {
            region_vmm_test_use_fake_driver();
        },
        "Route native VMM physical calls through the in-process fake driver."
    );
    m.def(
        "_region_vmm_test_reset_hooks",
        []() {
            region_vmm_test_reset_hooks();
        },
        "Clear native VMM fail-injection and issued-op logs without dropping live records."
    );
    m.def(
        "_region_vmm_test_fail_stage",
        [](const std::string &stage, const std::string &when) {
            region_vmm_test_fail_stage(stage, when);
        },
        nb::arg("stage"), nb::arg("when"), "Fail one named native VMM stage before or after ledger mutation."
    );
    m.def(
        "_region_vmm_test_set_unmap_already_gone",
        []() {
            region_vmm_test_set_unmap_already_gone();
        },
        "Make the next fake unmap report a backend-defined already-gone outcome."
    );
    m.def(
        "_region_vmm_test_issued_ops",
        []() -> std::vector<std::string> {
            return region_vmm_issued_ops();
        },
        "Return native VMM driver operations issued since the last hook reset."
    );
    m.def(
        "_region_vmm_test_live_handles",
        []() -> std::vector<uint64_t> {
            return region_vmm_live_handles();
        },
        "Return currently retained native VMM registry handles."
    );
    m.def(
        "_region_vmm_test_fake_bytes",
        [](uint64_t handle) -> std::vector<uint8_t> {
            return region_vmm_test_fake_bytes(handle);
        },
        nb::arg("handle"), "Return fake-driver VA backing bytes for one native VMM record."
    );
    bind_worker(m);
}

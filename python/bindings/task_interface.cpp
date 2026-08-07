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
 * data_type.h / tensor.h / task_args.h / arg_direction.h / callable.h.
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
#include <array>
#include <condition_variable>
#include <cstring>
#include <cstdint>
#include <exception>
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
#include "buffer.h"
#include "callable.h"
#include "callable_protocol.h"
#include "chip_worker.h"
#include "data_type.h"
#include "dma_workspace.h"
#include "host_trace.h"
#include "worker_chip_orch_comm.h"
#include "worker_chip_orch_region_access.h"
#include "worker_bind.h"
#include "task_args.h"
#include "tensor.h"

namespace nb = nanobind;

namespace {

std::string shm_name_for_open(const std::string &token) {
    if (token.empty()) {
        throw std::invalid_argument("L3-L2 sim backing shm token must be non-empty");
    }
    if (token[0] == '/') {
        return token;
    }
    return "/" + token;
}

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

class WorkerHostMappedRegionCleanupErrors {
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

WorkerHostMappedRegionCleanupErrors &worker_host_mapped_region_cleanup_errors() {
    // Return-boundary owners can be finalized after their importing call has
    // unwound. The process-lifetime registry preserves Worker-keyed diagnostics
    // until that same Worker reaches an admission or close boundary.
    static auto *errors = new WorkerHostMappedRegionCleanupErrors();
    return *errors;
}

class WorkerHostMappedRegion {
public:
    WorkerHostMappedRegion() = default;
    WorkerHostMappedRegion(const WorkerHostMappedRegion &) = delete;
    WorkerHostMappedRegion &operator=(const WorkerHostMappedRegion &) = delete;

    ~WorkerHostMappedRegion() noexcept {
        try {
            std::string cleanup_error;
            close_collecting(cleanup_error);
            if (!cleanup_error.empty()) {
                worker_host_mapped_region_cleanup_errors().record(owner_token, cleanup_error);
            }
        } catch (...) {
            worker_host_mapped_region_cleanup_errors().record(
                owner_token, "L3-L2 mapped-region cleanup failed with an unknown error"
            );
        }
    }

    std::string owner_token;
    WorkerChipRegionAccessProfile profile{WorkerChipRegionAccessProfile::SIM_POSIX_SHM};
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
            throw std::runtime_error("L3-L2 onboard mapped-region handle has no device id");
        }
        acl_api().bind_device_with_check(device_id);
    }

    void validate_mapping_range_or_throw(uint64_t offset, uint64_t nbytes) const {
        if (nbytes == 0 || offset > mapping_bytes || nbytes > mapping_bytes - offset) {
            throw std::out_of_range("L3-L2 L3 Host mapped-region access is out of range");
        }
    }

    void copy_to(uint64_t offset, const void *host_ptr, uint64_t nbytes) const {
        validate_mapping_range_or_throw(offset, nbytes);
        if (profile == WorkerChipRegionAccessProfile::SIM_POSIX_SHM) {
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
        if (profile == WorkerChipRegionAccessProfile::SIM_POSIX_SHM) {
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
            throw std::invalid_argument("L3-L2 counter offset must be 4-byte aligned");
        }
        if (!worker_chip_orch_comm::valid_notify_op(op)) {
            throw std::invalid_argument("L3-L2 counter notify op is invalid");
        }
        if (op == WorkerChipOrchNotifyOp::Add) {
            value = load_counter(offset) + value;
        }
        store_counter(offset, value);
    }

    std::tuple<bool, int32_t> test_counter(uint64_t offset, int32_t operand, WorkerChipOrchWaitCmp cmp) const {
        if (offset % sizeof(int32_t) != 0) {
            throw std::invalid_argument("L3-L2 counter offset must be 4-byte aligned");
        }
        if (!worker_chip_orch_comm::valid_wait_cmp(cmp)) {
            throw std::invalid_argument("L3-L2 counter wait comparison is invalid");
        }
        int32_t observed = load_counter(offset);
        return std::make_tuple(worker_chip_orch_comm::compare_counter(observed, operand, cmp), observed);
    }

    // Returns (status, error_kind, observed, matched, message). The status/error
    // values are the wire contract with the Python facade
    // (_WAIT_STATUS_TIMEOUT / _WAIT_ERROR_SIGNAL_TIMEOUT in worker_chip_orch_comm.py).
    std::tuple<int, int, int32_t, bool, std::string>
    wait_counter(uint64_t offset, int32_t operand, WorkerChipOrchWaitCmp cmp, uint64_t timeout_ns) const {
        if (offset % sizeof(int32_t) != 0) {
            throw std::invalid_argument("L3-L2 counter offset must be 4-byte aligned");
        }
        if (!worker_chip_orch_comm::valid_wait_cmp(cmp)) {
            throw std::invalid_argument("L3-L2 counter wait comparison is invalid");
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

        if (profile == WorkerChipRegionAccessProfile::ONBOARD_VMM) {
            if (mapped_addr == 0 && physical_handle == nullptr) {
                return;
            }
            try {
                if (mapped_device_id < 0) {
                    throw std::runtime_error("L3-L2 onboard mapped-region handle has no device id");
                }
                AclRuntimeApi &api = acl_api();
                api.bind_device_with_check(mapped_device_id);
                api.vmm_release_collecting(
                    reinterpret_cast<void *>(static_cast<uintptr_t>(mapped_addr)), physical_handle, cleanup_error
                );
            } catch (const std::exception &exc) {
                append_cleanup_error(cleanup_error, exc.what());
            } catch (...) {
                append_cleanup_error(cleanup_error, "L3-L2 onboard mapped-region cleanup failed");
            }
            return;
        }

        if (mapped_addr != 0 &&
            munmap(reinterpret_cast<void *>(static_cast<uintptr_t>(mapped_addr)), mapped_bytes) != 0) {
            int err = errno;
            append_cleanup_error(
                cleanup_error, std::string("L3-L2 sim L3 Host mapped-region munmap failed: ") + std::strerror(err)
            );
        }
        if (mapped_fd >= 0 && ::close(mapped_fd) != 0) {
            int err = errno;
            append_cleanup_error(
                cleanup_error, std::string("L3-L2 sim L3 Host mapped-region close failed: ") + std::strerror(err)
            );
        }
    }

    static constexpr int kWaitStatusOk = 0;
    static constexpr int kWaitStatusTimeout = -1;
    static constexpr int kWaitErrorNone = 0;
    static constexpr int kWaitErrorSignalTimeout = 7;
    static constexpr int64_t kWaitPollIntervalNs = 50000;
};

class ChipChildOnboardRegion {
public:
    int device_id{-1};
    uint64_t device_addr{0};
    uint64_t mapping_bytes{0};
    uint64_t shareable_handle{0};
    void *vmm_handle{nullptr};

    void bind_acl_device() const {
        if (device_id < 0) {
            throw std::runtime_error("L3-L2 onboard child region has no device id");
        }
        acl_api().bind_device_with_check(device_id);
    }
};

struct ChipChildOnboardRegionExport {
    uint64_t device_addr{0};
    uint64_t mapping_bytes{0};
    uint64_t shareable_handle{0};
    uint64_t registry_handle{0};
};

class WorkerHostMappedRegionEntry {
public:
    explicit WorkerHostMappedRegionEntry(std::unique_ptr<WorkerHostMappedRegion> mapping) :
        mapping_(std::move(mapping)) {}

    void acquire() {
        std::lock_guard<std::mutex> lk(mu_);
        if (state_ != State::OPEN) {
            throw std::runtime_error("L3-L2 L3 Host mapped-region handle is closed or unknown");
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

    WorkerHostMappedRegion &mapping() { return *mapping_; }

    size_t active_leases() const {
        std::lock_guard<std::mutex> lk(mu_);
        return active_leases_;
    }

    void close() {
        std::unique_ptr<WorkerHostMappedRegion> mapping;
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
            // A counter_wait lease may remain held for the rest of its timeout;
            // the mapping stays valid until every such waiter has returned.
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

    std::unique_ptr<WorkerHostMappedRegion> mapping_;
    mutable std::mutex mu_;
    std::condition_variable idle_;
    size_t active_leases_{0};
    State state_{State::OPEN};
    std::exception_ptr close_error_;
};

class WorkerHostMappedRegionLease {
public:
    explicit WorkerHostMappedRegionLease(std::shared_ptr<WorkerHostMappedRegionEntry> entry) :
        entry_(std::move(entry)) {
        entry_->acquire();
    }
    WorkerHostMappedRegionLease(const WorkerHostMappedRegionLease &) = delete;
    WorkerHostMappedRegionLease &operator=(const WorkerHostMappedRegionLease &) = delete;
    WorkerHostMappedRegionLease(WorkerHostMappedRegionLease &&) noexcept = default;
    WorkerHostMappedRegionLease &operator=(WorkerHostMappedRegionLease &&) = delete;
    ~WorkerHostMappedRegionLease() {
        if (entry_ != nullptr) {
            entry_->release();
        }
    }

    WorkerHostMappedRegion *operator->() { return &entry_->mapping(); }

private:
    std::shared_ptr<WorkerHostMappedRegionEntry> entry_;
};

class WorkerHostMappedRegionRegistry {
public:
    uint64_t emplace(std::unique_ptr<WorkerHostMappedRegion> mapping) {
        auto entry = std::make_shared<WorkerHostMappedRegionEntry>(std::move(mapping));
        std::lock_guard<std::mutex> lk(mu_);
        if (std::exchange(fail_next_insert_for_test_, false)) {
            throw std::runtime_error("injected mapped-region registry insertion failure");
        }
        uint64_t handle = next_handle_;
        auto result = regions_.emplace(handle, std::move(entry));
        if (!result.second) {
            throw std::overflow_error("L3-L2 L3 Host mapped-region handle space is exhausted");
        }
        next_handle_ += 1;
        if (next_handle_ == 0) {
            next_handle_ = 1;
        }
        return handle;
    }

    WorkerHostMappedRegionLease lease(uint64_t handle) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = regions_.find(handle);
        if (it == regions_.end()) {
            throw std::runtime_error("L3-L2 L3 Host mapped-region handle is closed or unknown");
        }
        return WorkerHostMappedRegionLease(it->second);
    }

    size_t active_leases(uint64_t handle) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = regions_.find(handle);
        if (it == regions_.end()) {
            throw std::runtime_error("L3-L2 L3 Host mapped-region handle is closed or unknown");
        }
        return it->second->active_leases();
    }

    void close(uint64_t handle) {
        // Retain the entry while it closes: its state rejects new leases, and
        // duplicate close callers join the same physical-cleanup completion.
        std::shared_ptr<WorkerHostMappedRegionEntry> entry;
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
    std::unordered_map<uint64_t, std::shared_ptr<WorkerHostMappedRegionEntry>> regions_;
    uint64_t next_handle_{1};
    bool fail_next_insert_for_test_{false};
};

WorkerHostMappedRegionRegistry &worker_host_mapped_region_registry() {
    // Python owners may be finalized after ordinary C++ static destruction has
    // begun. The registry and ACL dispatch table therefore have process
    // lifetime; the OS reclaims any entries still open at process exit.
    static auto *registry = new WorkerHostMappedRegionRegistry();
    return *registry;
}

void close_worker_host_mapped_region(uint64_t handle) { worker_host_mapped_region_registry().close(handle); }

class WorkerHostMappedRegionHandle {
public:
    explicit WorkerHostMappedRegionHandle(uint64_t handle, std::string owner_token) :
        handle_(handle),
        owner_token_(std::move(owner_token)) {}
    WorkerHostMappedRegionHandle(const WorkerHostMappedRegionHandle &) = delete;
    WorkerHostMappedRegionHandle &operator=(const WorkerHostMappedRegionHandle &) = delete;
    WorkerHostMappedRegionHandle(WorkerHostMappedRegionHandle &&other) noexcept :
        handle_(std::exchange(other.handle_, 0)),
        owner_token_(std::move(other.owner_token_)) {}
    WorkerHostMappedRegionHandle &operator=(WorkerHostMappedRegionHandle &&) = delete;

    ~WorkerHostMappedRegionHandle() noexcept {
        if (handle_ == 0) {
            return;
        }
        try {
            close_worker_host_mapped_region(handle_);
        } catch (const std::exception &exc) {
            worker_host_mapped_region_cleanup_errors().record(owner_token_, exc.what());
        } catch (...) {
            worker_host_mapped_region_cleanup_errors().record(
                owner_token_, "L3-L2 mapped-region owner cleanup failed with an unknown error"
            );
        }
    }

    uint64_t value() const { return handle_; }

private:
    uint64_t handle_{0};
    std::string owner_token_;
};

class ChipChildOnboardRegionRegistry {
public:
    uint64_t emplace(ChipChildOnboardRegion region) {
        std::lock_guard<std::mutex> lk(mu_);
        uint64_t handle = next_handle_++;
        regions_.emplace(handle, std::move(region));
        return handle;
    }

    std::optional<ChipChildOnboardRegion> remove(uint64_t handle) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = regions_.find(handle);
        if (it == regions_.end()) {
            return std::nullopt;
        }
        ChipChildOnboardRegion region = std::move(it->second);
        regions_.erase(it);
        return region;
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<uint64_t, ChipChildOnboardRegion> regions_;
    uint64_t next_handle_{1};
};

ChipChildOnboardRegionRegistry g_chip_child_onboard_regions;

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

WorkerChipOrchNotifyOp checked_notify_op(int op) {
    auto typed = static_cast<WorkerChipOrchNotifyOp>(op);
    if (!worker_chip_orch_comm::valid_notify_op(typed)) {
        throw std::invalid_argument("L3-L2 counter notify op is invalid");
    }
    return typed;
}

WorkerChipOrchWaitCmp checked_wait_cmp(int cmp) {
    auto typed = static_cast<WorkerChipOrchWaitCmp>(cmp);
    if (!worker_chip_orch_comm::valid_wait_cmp(typed)) {
        throw std::invalid_argument("L3-L2 counter wait comparison is invalid");
    }
    return typed;
}

void append_cleanup_error(std::string &cleanup_error, const std::string &message) {
    if (!cleanup_error.empty()) {
        cleanup_error += "; ";
    }
    cleanup_error += message;
}

// The int wire value of a dtype given either a DataType enumerator or its int value. The nanobind
// DataType enum is not arithmetic, so a caller holding one has only `.value`; accept both forms.
uint8_t datatype_wire_value(nb::object dtype) {
    if (nb::hasattr(dtype, "value")) dtype = dtype.attr("value");
    return nb::cast<uint8_t>(dtype);
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
        "_bind_host_span_sink", &simpler::host_trace::bind_process_sink,
        "Resolve the process-global host-span sink after libsimpler_log.so is loaded."
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
        "Emit one explicitly timed host span through the process-global logger."
    );
    // Byte size of a ChipTensor and the offset of its child_memory flag within it.
    // A task-args blob stores ChipTensors as a raw memcpy array, so a Python-side
    // blob walker locates tensor i's fields at i * CHIP_TENSOR_STRIDE_BYTES without
    // reimplementing the struct layout.
    m.attr("CHIP_TENSOR_STRIDE_BYTES") = static_cast<int>(sizeof(ChipTensor));
    m.attr("CHIP_TENSOR_CHILD_MEMORY_OFFSET") = static_cast<int>(offsetof(ChipTensor, child_memory));

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
    // receives one already decoded; turning mailbox bytes into a Tensor is task_args.h's job, and
    // keeping that the only decode path is what makes validate_tensor a gate rather than a habit.
    nb::class_<Tensor>(m, "Tensor")
        .def(
            "__init__",
            [](Tensor *self, const BufferDescriptor &buffer, uint64_t byte_offset, nb::sequence shapes,
               nb::sequence strides, nb::object dtype) {
                const size_t ndims = nb::len(shapes);
                if (ndims != nb::len(strides)) {
                    throw std::invalid_argument("Tensor shapes and strides must have equal length");
                }
                if (ndims == 0 || ndims > static_cast<size_t>(MAX_TENSOR_DIMS)) {
                    throw std::invalid_argument(
                        "Tensor ndims must be in [1, " + std::to_string(MAX_TENSOR_DIMS) + "], got " +
                        std::to_string(ndims)
                    );
                }
                new (self) Tensor{};
                self->buffer = buffer;
                self->byte_offset = byte_offset;
                self->ndims = static_cast<uint32_t>(ndims);
                for (size_t i = 0; i < ndims; ++i) {
                    self->shapes[i] = nb::cast<uint32_t>(shapes[i]);
                    self->strides[i] = nb::cast<uint32_t>(strides[i]);
                }
                self->dtype = static_cast<DataType>(datatype_wire_value(dtype));
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
                    /*manual_dep=*/false, /*version=*/0, child_memory ? 1 : 0
                );
            },
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
                    static_cast<uint32_t>(n), self.dtype, self.version, self.manual_dep, self.child_memory
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
                return self.is_child_memory();
            },
            [](ChipTensor &self, bool v) {
                self.child_memory = v ? 1 : 0;
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
                return self.is_contiguous;
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
            if (self.is_child_memory()) os << ", child_memory=True";
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

    // --- TaskArgs (unified vector-backed builder with per-tensor TensorArgType tags) ---
    nb::class_<TaskArgs>(m, "TaskArgs", nb::is_weak_referenceable())
        .def(nb::init<>())

        .def(
            "add_tensor",
            [](TaskArgs &self, const ChipTensor &t, TensorArgType tag) {
                self.add_tensor(t, tag);
            },
            nb::arg("t"), nb::arg("tag") = TensorArgType::INPUT,
            "Add a ChipTensor with an optional TensorArgType tag (default INPUT)."
        )

        .def(
            "add_scalar", &TaskArgs::add_scalar, nb::arg("s"),
            "Add a uint64_t scalar. After this, add_tensor() is no longer allowed."
        )

        .def(
            "tensor",
            [](const TaskArgs &self, int32_t i) -> const ChipTensor & {
                if (i < 0 || i >= self.tensor_count()) throw std::out_of_range("TaskArgs tensor index out of range");
                return self.tensor(i);
            },
            nb::arg("i"), nb::rv_policy::reference_internal, "Return the ChipTensor at index i."
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

        .def("clear", &TaskArgs::clear)

        .def(
            "__len__",
            [](const TaskArgs &self) {
                return self.tensor_count() + self.scalar_count();
            },
            "Return total number of arguments (tensors + scalars)."
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
               std::vector<std::tuple<int32_t, PyCoreCallable>> children, std::string config_name) -> PyChipCallable {
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
                    signature.data(), static_cast<int32_t>(signature.size()), func_name.c_str(), bin_ptr, bin_size,
                    func_ids.data(), child_bufs.data(), child_count, config_name.c_str()
                );
                return PyChipCallable{std::move(buf)};
            },
            nb::arg("signature"), nb::arg("func_name"), nb::arg("binary"), nb::arg("children"),
            nb::arg("config_name") = "",
            "Build a ChipCallable from signature, func_name, binary, and list of (func_id, CoreCallable) children."
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

    // --- RuntimeEnv (per-task PTO2_RING_* overrides; nested under CallConfig.runtime_env) ---
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
               << ", enable_scope_stats=" << (self.enable_scope_stats ? "True" : "False");
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

    nb::class_<ChipWorkerNativeRun>(m, "_ChipWorkerNativeRun")
        .def_ro("slot_id", &ChipWorkerNativeRun::slot_id)
        .def_ro("generation", &ChipWorkerNativeRun::generation)
        .def_ro("run_epoch", &ChipWorkerNativeRun::run_epoch)
        .def_ro("run_id", &ChipWorkerNativeRun::run_id)
        .def_ro("dispatch_id", &ChipWorkerNativeRun::dispatch_id);

    // --- ChipWorker ---
    nb::class_<ChipWorker>(m, "_ChipWorker")
        .def(nb::init<>())
        .def(
            "init",
            [](ChipWorker &self, const std::string &host_lib_path, const std::string &aicpu_path,
               const std::string &aicore_path, const std::string &dispatcher_path, int device_id,
               std::optional<CallConfig> prewarm_config, bool enable_sdma) {
                // Translate the Python bool into a DmaWorkspaceKind bitmask so the
                // platform-agnostic ChipWorker stays free of the enum. Empty mask
                // when disabled leaves the Worker with no async-DMA provisioning.
                uint32_t dma_workspace_mask = enable_sdma ? (uint32_t{1} << DMA_WORKSPACE_SDMA) : 0;
                self.init(
                    host_lib_path, aicpu_path, aicore_path, dispatcher_path, device_id,
                    prewarm_config.has_value() ? &(*prewarm_config) : nullptr, dma_workspace_mask
                );
            },
            nb::arg("host_lib_path"), nb::arg("aicpu_path"), nb::arg("aicore_path"), nb::arg("dispatcher_path"),
            nb::arg("device_id"), nb::arg("prewarm_config") = nb::none(), nb::arg("enable_sdma") = false,
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
            "kernels can use get_dma_workspace; init raises if the platform lacks SDMA."
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
            "run",
            [](ChipWorker &self, int32_t callable_id, TaskArgs &args, const CallConfig &config) {
                TaskArgsView view = make_view(args);
                self.run(callable_id, view, config);
            },
            nb::arg("callable_id"), nb::arg("args"), nb::arg("config"),
            "Launch a callable_id from a TaskArgs (used for in-process callers). "
            "Returns None; timing is emitted as `[STRACE]` log markers."
        )
        .def(
            "_run_with_pipeline_lease",
            [](ChipWorker &self, int32_t callable_id, TaskArgs &args, const CallConfig &config, uint32_t slot_id,
               uint64_t generation) {
                self.run_with_lease(callable_id, make_view(args), config, PipelineSlotLease{slot_id, 0, generation});
            },
            nb::arg("callable_id"), nb::arg("args"), nb::arg("config"), nb::arg("slot_id"), nb::arg("generation"),
            "Internal generation-safe pipeline-slot launch used by hierarchical admission."
        )
        .def(
            "_run_with_pipeline_lease",
            [](ChipWorker &self, int32_t callable_id, ChipStorageTaskArgs &args, const CallConfig &config,
               uint32_t slot_id, uint64_t generation) {
                self.run_with_lease(callable_id, &args, config, PipelineSlotLease{slot_id, 0, generation});
            },
            nb::arg("callable_id"), nb::arg("args"), nb::arg("config"), nb::arg("slot_id"), nb::arg("generation"),
            "Internal generation-safe pipeline-slot launch for pre-encoded task args."
        )
        .def(
            "_prepare_native_run_with_pipeline_lease",
            [](ChipWorker &self, int32_t callable_id, TaskArgs &args, const CallConfig &config, uint32_t slot_id,
               uint64_t generation) {
                return self.prepare_native_run(
                    callable_id, make_view(args), config, PipelineSlotLease{slot_id, 0, generation}
                );
            },
            nb::arg("callable_id"), nb::arg("args"), nb::arg("config"), nb::arg("slot_id"), nb::arg("generation"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Prepare a native run after lease admission without crossing its device launch fence."
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
            "_prepare_native_run_from_blob",
            [](ChipWorker &self, int32_t callable_id, uint64_t args_blob_ptr, size_t blob_capacity,
               const CallConfig &config, uint32_t slot_id, uint64_t generation, uint64_t run_id, uint64_t dispatch_id,
               uint64_t accepted_state_addr, int32_t accepted_value) {
                TaskArgsView view = read_blob(reinterpret_cast<const uint8_t *>(args_blob_ptr), blob_capacity);
                return self.prepare_native_run(
                    callable_id, view, config, PipelineSlotLease{slot_id, 0, generation}, run_id, dispatch_id,
                    reinterpret_cast<volatile int32_t *>(accepted_state_addr), accepted_value
                );
            },
            nb::arg("callable_id"), nb::arg("args_blob_ptr"), nb::arg("blob_capacity"), nb::arg("config"),
            nb::arg("slot_id"), nb::arg("generation"), nb::arg("run_id") = 0, nb::arg("dispatch_id") = 0,
            nb::arg("accepted_state_addr") = 0, nb::arg("accepted_value") = 0, nb::call_guard<nb::gil_scoped_release>(),
            "Prepare a native run from a raw mailbox TaskArgs blob after lease admission."
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
            "run_from_blob",
            [](ChipWorker &self, int32_t callable_id, uint64_t args_blob_ptr, size_t blob_capacity,
               const CallConfig &config, uint64_t accepted_state_addr, int32_t accepted_value, uint32_t pipeline_slot,
               uint64_t pipeline_generation) {
                // The mailbox region is the on-wire format `write_blob` produced;
                // `read_blob` is the matching reader that returns a zero-copy
                // TaskArgsView into the caller-owned bytes. Forwards to the
                // existing `run(cid, view, config)` path so chip-child
                // loops never re-implement the tensor/scalar layout in Python
                // (where it has historically dropped fields like child_memory).
                TaskArgsView view = read_blob(reinterpret_cast<const uint8_t *>(args_blob_ptr), blob_capacity);
                if (pipeline_generation == 0) {
                    self.run(
                        callable_id, view, config, reinterpret_cast<volatile int32_t *>(accepted_state_addr),
                        accepted_value
                    );
                } else {
                    self.run_with_lease(
                        callable_id, view, config, PipelineSlotLease{pipeline_slot, 0, pipeline_generation},
                        reinterpret_cast<volatile int32_t *>(accepted_state_addr), accepted_value
                    );
                }
            },
            nb::arg("callable_id"), nb::arg("args_blob_ptr"), nb::arg("blob_capacity"), nb::arg("config"),
            nb::arg("accepted_state_addr") = 0, nb::arg("accepted_value") = 0, nb::arg("pipeline_slot") = 0,
            nb::arg("pipeline_generation") = 0,
            "Launch a callable_id from a raw mailbox-blob pointer + capacity "
            "(used by chip-child mailbox loops to avoid Python-side re-deserialisation "
            "of the per-task tensor/scalar layout). The blob must be in the format "
            "produced by `write_blob`; read_blob enforces capacity bounds against shm corruption."
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
            "Number of AICore run streams the bound runner has created. The AICPU "
            "stream belongs to a pipeline slot for the worker's lifetime, while each "
            "run creates and retires its own AICore stream, so this advances once per "
            "run; platforms whose runs use the persistent bootstrap pair report 0."
        )
        .def_prop_ro(
            "committed_device_memory", &ChipWorker::committed_device_memory,
            "Total device HBM (bytes) currently committed by this worker's "
            "MemoryAllocator (user tensors + pooled arenas + runtime buffers). "
            "Excludes HCCL/VMM comm windows. 0 when not "
            "initialized. Lets downstream runtimes subtract simpler's own HBM "
            "from their cache budget (it may be invisible to aclrtGetMemInfo)."
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
        .def("comm_barrier", &ChipWorker::comm_barrier, nb::arg("comm_handle"), "Synchronize all ranks.")
        .def(
            "comm_destroy", &ChipWorker::comm_destroy, nb::arg("comm_handle"),
            "Destroy the communicator and release its resources."
        )
        .def("comm_destroy_all", &ChipWorker::comm_destroy_all, "Destroy all owned communicators in LIFO order.");

    // --- Standalone blob helpers ---
    m.def(
        "read_args_from_blob",
        [](uint64_t blob_ptr) {
            TaskArgsView view = read_blob(reinterpret_cast<const uint8_t *>(blob_ptr), MAILBOX_ARGS_CAPACITY);
            TaskArgs args;
            for (int32_t i = 0; i < view.tensor_count; i++) {
                args.add_tensor(view.tensors(i));
            }
            for (int32_t i = 0; i < view.scalar_count; i++) {
                args.add_scalar(view.scalars[i]);
            }
            return args;
        },
        nb::arg("blob_ptr"),
        "Reconstruct a TaskArgs from a length-prefixed blob at blob_ptr. "
        "Tags are not preserved (blob wire format strips them)."
    );

    nb::class_<ChipChildOnboardRegionExport>(m, "_ChipChildOnboardRegionExport")
        .def_ro("device_addr", &ChipChildOnboardRegionExport::device_addr)
        .def_ro("mapping_bytes", &ChipChildOnboardRegionExport::mapping_bytes)
        .def_ro("shareable_handle", &ChipChildOnboardRegionExport::shareable_handle)
        .def_ro("registry_handle", &ChipChildOnboardRegionExport::registry_handle);

    nb::class_<WorkerHostMappedRegionHandle>(m, "_WorkerHostMappedRegionHandle")
        .def("__int__", &WorkerHostMappedRegionHandle::value);

    m.def(
        "_worker_host_mapped_region_import_sim",
        [](const std::string &token, uint64_t mapping_bytes,
           const std::string &owner_token) -> WorkerHostMappedRegionHandle {
            if (mapping_bytes == 0 || mapping_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::invalid_argument("L3-L2 sim L3 Host mapped-region import requires a positive mapping size");
            }
            if (owner_token.empty()) {
                throw std::invalid_argument("L3-L2 mapped-region import requires a non-empty Worker owner token");
            }
            std::string handle_owner_token = owner_token;
            std::string name = shm_name_for_open(token);
            auto mapping = std::make_unique<WorkerHostMappedRegion>();
            mapping->owner_token = owner_token;
            mapping->fd = shm_open(name.c_str(), O_RDWR, 0);
            if (mapping->fd < 0) {
                throw std::runtime_error("L3-L2 sim L3 Host mapped-region import shm_open failed");
            }
            void *base =
                mmap(nullptr, static_cast<size_t>(mapping_bytes), PROT_READ | PROT_WRITE, MAP_SHARED, mapping->fd, 0);
            if (base == MAP_FAILED) {
                int err = errno;
                throw std::runtime_error(
                    std::string("L3-L2 sim L3 Host mapped-region import mmap failed: ") + std::strerror(err)
                );
            }
            mapping->profile = WorkerChipRegionAccessProfile::SIM_POSIX_SHM;
            mapping->device_addr = reinterpret_cast<uint64_t>(base);
            mapping->mapping_bytes = mapping_bytes;
            uint64_t handle = worker_host_mapped_region_registry().emplace(std::move(mapping));
            return WorkerHostMappedRegionHandle(handle, std::move(handle_owner_token));
        },
        nb::arg("token"), nb::arg("mapping_bytes"), nb::arg("owner_token"), nb::call_guard<nb::gil_scoped_release>(),
        "Import a sim L3-L2 POSIX shm region for L3 Host mapped-region access."
    );
    m.def(
        "_worker_host_mapped_region_import_onboard",
        [](int device_id, uint64_t shareable_handle, uint64_t mapping_bytes,
           const std::string &owner_token) -> WorkerHostMappedRegionHandle {
            if (device_id < 0) {
                throw std::invalid_argument("L3-L2 onboard mapped-region import requires a non-negative device id");
            }
            if (mapping_bytes == 0 || mapping_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::invalid_argument("L3-L2 onboard mapped-region import requires a positive mapping size");
            }
            if (owner_token.empty()) {
                throw std::invalid_argument("L3-L2 mapped-region import requires a non-empty Worker owner token");
            }
            std::string handle_owner_token = owner_token;
            auto mapping = std::make_unique<WorkerHostMappedRegion>();
            mapping->owner_token = owner_token;
            mapping->profile = WorkerChipRegionAccessProfile::ONBOARD_VMM;
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
            uint64_t handle = worker_host_mapped_region_registry().emplace(std::move(mapping));
            return WorkerHostMappedRegionHandle(handle, std::move(handle_owner_token));
        },
        nb::arg("device_id"), nb::arg("shareable_handle"), nb::arg("mapping_bytes"), nb::arg("owner_token"),
        nb::call_guard<nb::gil_scoped_release>(), "Import an onboard VMM L3-L2 region for L3 Host mapped-region access."
    );
    m.def(
        "_worker_host_mapped_region_close",
        [](uint64_t handle) {
            close_worker_host_mapped_region(handle);
        },
        nb::arg("handle"), nb::call_guard<nb::gil_scoped_release>(), "Close an L3 Host mapped-region handle."
    );
    m.def(
        "_worker_host_mapped_region_active_leases",
        [](uint64_t handle) {
            return worker_host_mapped_region_registry().active_leases(handle);
        },
        nb::arg("handle"), "Return the number of in-flight native operations holding this mapped region."
    );
    m.def(
        "_worker_host_mapped_region_take_cleanup_error",
        [](const std::string &owner_token) {
            if (owner_token.empty()) {
                throw std::invalid_argument("L3-L2 cleanup-error lookup requires a non-empty Worker owner token");
            }
            return worker_host_mapped_region_cleanup_errors().take(owner_token);
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
            return worker_host_mapped_region_cleanup_errors().peek(owner_token);
        },
        nb::arg("owner_token"), "Read one Worker's mapped-region cleanup error without consuming it."
    );
    m.def(
        "_worker_host_mapped_region_ack_cleanup_error",
        [](const std::string &owner_token, const std::string &observed) {
            if (owner_token.empty()) {
                throw std::invalid_argument("L3-L2 cleanup-error acknowledgement requires a Worker owner token");
            }
            worker_host_mapped_region_cleanup_errors().acknowledge(owner_token, observed);
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
            worker_host_mapped_region_cleanup_errors().record(owner_token, message);
        },
        nb::arg("owner_token"), nb::arg("message"), "Inject one Worker-owned mapped-region cleanup error."
    );
    m.def(
        "_worker_host_mapped_region_fail_next_registry_insert_for_test",
        []() {
            worker_host_mapped_region_registry().fail_next_insert_for_test();
        },
        "Inject one mapped-region registry insertion failure after native acquisition."
    );
    m.def(
        "_worker_host_mapped_payload_write",
        [](uint64_t handle, uint64_t payload_offset, uint64_t host_ptr, uint64_t nbytes) {
            if (host_ptr == 0) {
                throw std::invalid_argument("L3-L2 payload_write host_ptr must be nonzero");
            }
            WorkerHostMappedRegionLease mapping = worker_host_mapped_region_registry().lease(handle);
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
            WorkerHostMappedRegionLease mapping = worker_host_mapped_region_registry().lease(handle);
            mapping->copy_from(reinterpret_cast<void *>(static_cast<uintptr_t>(host_ptr)), payload_offset, nbytes);
        },
        nb::arg("handle"), nb::arg("payload_offset"), nb::arg("host_ptr"), nb::arg("nbytes"),
        nb::call_guard<nb::gil_scoped_release>(), "Copy imported L3-L2 payload bytes into L3 Host memory."
    );
    m.def(
        "_worker_host_mapped_counter_notify",
        [](uint64_t handle, uint64_t counter_offset, int32_t value, int op) {
            WorkerHostMappedRegionLease mapping = worker_host_mapped_region_registry().lease(handle);
            mapping->notify_counter(counter_offset, value, checked_notify_op(op));
        },
        nb::arg("handle"), nb::arg("counter_offset"), nb::arg("value"), nb::arg("op"),
        nb::call_guard<nb::gil_scoped_release>(), "Store or add one L3 Host-side L3-L2 signal counter."
    );
    m.def(
        "_worker_host_mapped_counter_test",
        [](uint64_t handle, uint64_t counter_offset, int32_t operand, int cmp) -> std::tuple<bool, int32_t> {
            WorkerHostMappedRegionLease mapping = worker_host_mapped_region_registry().lease(handle);
            return mapping->test_counter(counter_offset, operand, checked_wait_cmp(cmp));
        },
        nb::arg("handle"), nb::arg("counter_offset"), nb::arg("operand"), nb::arg("cmp"),
        nb::call_guard<nb::gil_scoped_release>(), "Load and compare one L3 Host-side L3-L2 signal counter."
    );
    m.def(
        "_worker_host_mapped_counter_wait",
        [](uint64_t handle, uint64_t counter_offset, int32_t operand, int cmp,
           uint64_t timeout_ns) -> std::tuple<int, int, int32_t, bool, std::string> {
            WorkerHostMappedRegionLease mapping = worker_host_mapped_region_registry().lease(handle);
            return mapping->wait_counter(counter_offset, operand, checked_wait_cmp(cmp), timeout_ns);
        },
        nb::arg("handle"), nb::arg("counter_offset"), nb::arg("operand"), nb::arg("cmp"), nb::arg("timeout_ns"),
        nb::call_guard<nb::gil_scoped_release>(), "Poll one L3 Host-side L3-L2 signal counter until match or timeout."
    );
    m.def(
        "_l3_child_onboard_region_create",
        [](uint64_t nbytes) -> ChipChildOnboardRegionExport {
            if (nbytes == 0 || nbytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::invalid_argument("L3-L2 onboard child region requires a positive mapping size");
            }
            AclRuntimeApi &api = acl_api();
            int device_id = api.current_device_with_check();
            uint64_t mapping_bytes = align_vmm_bytes(nbytes, api.vmm_granularity_with_check(device_id));
            ChipChildOnboardRegion region{};
            region.device_id = device_id;
            region.mapping_bytes = mapping_bytes;
            region.vmm_handle = api.vmm_malloc_physical_with_check(mapping_bytes, device_id);
            void *mapped_addr = nullptr;
            try {
                mapped_addr = api.vmm_reserve_with_check(mapping_bytes);
                api.vmm_map_with_check(mapped_addr, mapping_bytes, region.vmm_handle);
                api.vmm_set_access_with_check(mapped_addr, mapping_bytes, device_id);
                region.shareable_handle = api.vmm_export_shareable_with_check(region.vmm_handle);
            } catch (...) {
                std::string cleanup_error;
                api.vmm_release_collecting(mapped_addr, region.vmm_handle, cleanup_error);
                throw;
            }
            region.device_addr = reinterpret_cast<uint64_t>(mapped_addr);
            uint64_t registry_handle = g_chip_child_onboard_regions.emplace(region);
            return ChipChildOnboardRegionExport{
                region.device_addr,
                region.mapping_bytes,
                region.shareable_handle,
                registry_handle,
            };
        },
        nb::arg("nbytes"), nb::call_guard<nb::gil_scoped_release>(),
        "Create and export a child-owned onboard VMM region."
    );
    m.def(
        "_l3_child_onboard_region_close",
        [](uint64_t registry_handle) {
            std::optional<ChipChildOnboardRegion> removed = g_chip_child_onboard_regions.remove(registry_handle);
            if (!removed.has_value()) {
                return;
            }
            ChipChildOnboardRegion region = *removed;
            region.bind_acl_device();
            std::string cleanup_error;
            acl_api().vmm_release_collecting(
                reinterpret_cast<void *>(static_cast<uintptr_t>(region.device_addr)), region.vmm_handle, cleanup_error
            );
            if (!cleanup_error.empty()) {
                throw std::runtime_error(cleanup_error);
            }
        },
        nb::arg("registry_handle"), nb::call_guard<nb::gil_scoped_release>(), "Close a child-owned onboard VMM region."
    );

    bind_worker(m);
}

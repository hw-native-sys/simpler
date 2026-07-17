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
 * Wraps DataType, Tensor, ChipStorageTaskArgs, TaskArgs (unified
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
#include <cstring>
#include <cstdint>
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
#include "chip_worker.h"
#include "data_type.h"
#include "dma_workspace.h"
#include "l3_l2_orch_comm.h"
#include "l3_l2_orch_region_access.h"
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
    static std::unique_ptr<AclRuntimeApi> api;
    std::call_once(once, []() {
        auto candidate = std::make_unique<AclRuntimeApi>();
        candidate->load();
        candidate->init();
        api = std::move(candidate);
    });
    return *api;
}

class L3HostMappedRegion {
public:
    L3L2RegionAccessProfile profile{L3L2RegionAccessProfile::SIM_POSIX_SHM};
    int fd{-1};
    uint64_t device_addr{0};
    int device_id{-1};
    uint64_t shareable_handle{0};
    void *vmm_handle{nullptr};
    uint64_t mapping_bytes{0};

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
        if (profile == L3L2RegionAccessProfile::SIM_POSIX_SHM) {
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
        if (profile == L3L2RegionAccessProfile::SIM_POSIX_SHM) {
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

    void notify_counter(uint64_t offset, int32_t value, L3L2OrchNotifyOp op) const {
        if (offset % sizeof(int32_t) != 0) {
            throw std::invalid_argument("L3-L2 counter offset must be 4-byte aligned");
        }
        if (!l3_l2_orch_comm::valid_notify_op(op)) {
            throw std::invalid_argument("L3-L2 counter notify op is invalid");
        }
        if (op == L3L2OrchNotifyOp::Add) {
            value = load_counter(offset) + value;
        }
        store_counter(offset, value);
    }

    std::tuple<bool, int32_t> test_counter(uint64_t offset, int32_t operand, L3L2OrchWaitCmp cmp) const {
        if (offset % sizeof(int32_t) != 0) {
            throw std::invalid_argument("L3-L2 counter offset must be 4-byte aligned");
        }
        if (!l3_l2_orch_comm::valid_wait_cmp(cmp)) {
            throw std::invalid_argument("L3-L2 counter wait comparison is invalid");
        }
        int32_t observed = load_counter(offset);
        return std::make_tuple(l3_l2_orch_comm::compare_counter(observed, operand, cmp), observed);
    }

    // Returns (status, error_kind, observed, matched, message). The status/error
    // values are the wire contract with the Python facade
    // (_WAIT_STATUS_TIMEOUT / _WAIT_ERROR_SIGNAL_TIMEOUT in l3_l2_orch_comm.py).
    std::tuple<int, int, int32_t, bool, std::string>
    wait_counter(uint64_t offset, int32_t operand, L3L2OrchWaitCmp cmp, uint64_t timeout_ns) const {
        if (offset % sizeof(int32_t) != 0) {
            throw std::invalid_argument("L3-L2 counter offset must be 4-byte aligned");
        }
        if (!l3_l2_orch_comm::valid_wait_cmp(cmp)) {
            throw std::invalid_argument("L3-L2 counter wait comparison is invalid");
        }
        auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(timeout_ns);
        while (true) {
            int32_t observed = load_counter(offset);
            bool matched = l3_l2_orch_comm::compare_counter(observed, operand, cmp);
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
    static constexpr int kWaitStatusOk = 0;
    static constexpr int kWaitStatusTimeout = -1;
    static constexpr int kWaitErrorNone = 0;
    static constexpr int kWaitErrorSignalTimeout = 7;
    static constexpr int64_t kWaitPollIntervalNs = 50000;
};

class L2ChildOnboardRegion {
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

struct L2ChildOnboardRegionExport {
    uint64_t device_addr{0};
    uint64_t mapping_bytes{0};
    uint64_t shareable_handle{0};
    uint64_t registry_handle{0};
};

class L3HostMappedRegionRegistry {
public:
    uint64_t emplace(L3HostMappedRegion mapping) {
        std::lock_guard<std::mutex> lk(mu_);
        uint64_t handle = next_handle_++;
        regions_.emplace(handle, std::move(mapping));
        return handle;
    }

    L3HostMappedRegion find(uint64_t handle) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = regions_.find(handle);
        if (it == regions_.end()) {
            throw std::runtime_error("L3-L2 L3 Host mapped-region handle is closed or unknown");
        }
        return it->second;
    }

    std::optional<L3HostMappedRegion> remove(uint64_t handle) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = regions_.find(handle);
        if (it == regions_.end()) {
            return std::nullopt;
        }
        L3HostMappedRegion mapping = std::move(it->second);
        regions_.erase(it);
        return mapping;
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<uint64_t, L3HostMappedRegion> regions_;
    uint64_t next_handle_{1};
};

L3HostMappedRegionRegistry g_l3_host_mapped_regions;

class L2ChildOnboardRegionRegistry {
public:
    uint64_t emplace(L2ChildOnboardRegion region) {
        std::lock_guard<std::mutex> lk(mu_);
        uint64_t handle = next_handle_++;
        regions_.emplace(handle, std::move(region));
        return handle;
    }

    std::optional<L2ChildOnboardRegion> remove(uint64_t handle) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = regions_.find(handle);
        if (it == regions_.end()) {
            return std::nullopt;
        }
        L2ChildOnboardRegion region = std::move(it->second);
        regions_.erase(it);
        return region;
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<uint64_t, L2ChildOnboardRegion> regions_;
    uint64_t next_handle_{1};
};

L2ChildOnboardRegionRegistry g_l2_child_onboard_regions;

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

L3L2OrchNotifyOp checked_notify_op(int op) {
    auto typed = static_cast<L3L2OrchNotifyOp>(op);
    if (!l3_l2_orch_comm::valid_notify_op(typed)) {
        throw std::invalid_argument("L3-L2 counter notify op is invalid");
    }
    return typed;
}

L3L2OrchWaitCmp checked_wait_cmp(int cmp) {
    auto typed = static_cast<L3L2OrchWaitCmp>(cmp);
    if (!l3_l2_orch_comm::valid_wait_cmp(typed)) {
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

}  // namespace

// ============================================================================
// Module definition
// ============================================================================

NB_MODULE(_task_interface, m) {
    m.doc() = "Nanobind bindings for task_interface (DataType, Tensor, TaskArgs variants)";

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
        .value("UINT32", DataType::UINT32);

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
    // Byte size of a Tensor and the offset of its child_memory flag within it.
    // A task-args blob stores Tensors as a raw memcpy array, so a Python-side
    // blob walker locates tensor i's fields at i * TENSOR_STRIDE_BYTES without
    // reimplementing the struct layout.
    m.attr("TENSOR_STRIDE_BYTES") = static_cast<int>(sizeof(Tensor));
    m.attr("TENSOR_CHILD_MEMORY_OFFSET") = static_cast<int>(offsetof(Tensor, child_memory));

    // --- Tensor ---
    // The unified strided tensor descriptor. Constructed contiguous via make()
    // (row-major strides, start_offset == 0); see src/common/task_interface/tensor.h.
    nb::class_<Tensor>(m, "Tensor")
        .def(nb::init<>())

        .def_static(
            "make",
            [](uint64_t data, nb::tuple shapes, DataType dtype, bool child_memory) -> Tensor {
                size_t n = nb::len(shapes);
                if (n == 0 || n > MAX_TENSOR_DIMS)
                    throw std::invalid_argument("Tensor.make: shapes length must be in [1, MAX_TENSOR_DIMS]");
                uint32_t shp[MAX_TENSOR_DIMS];
                for (size_t i = 0; i < n; ++i)
                    shp[i] = nb::cast<uint32_t>(shapes[i]);
                // make_tensor_external yields a contiguous Tensor: row-major strides,
                // start_offset == 0, buffer.size == numel * element_size.
                return make_tensor_external(
                    reinterpret_cast<void *>(static_cast<uintptr_t>(data)), shp, static_cast<uint32_t>(n), dtype,
                    /*manual_dep=*/false, /*version=*/0, child_memory ? 1 : 0
                );
            },
            nb::arg("data"), nb::arg("shapes"), nb::arg("dtype"), nb::arg("child_memory") = false,
            "Create a contiguous Tensor over pre-allocated memory. Set child_memory=True when "
            "data is a device pointer allocated by the child process (skips H2D copy in "
            "init_runtime_impl)."
        )

        // `data` is the tensor's memory address — i.e. Tensor::buffer.addr.
        .def_prop_rw(
            "data",
            [](const Tensor &self) -> uint64_t {
                return self.buffer.addr;
            },
            [](Tensor &self, uint64_t v) {
                self.buffer.addr = v;
            }
        )

        .def_prop_rw(
            "shapes",
            [](const Tensor &self) -> nb::tuple {
                uint32_t n = self.ndims;
                if (n > MAX_TENSOR_DIMS) n = MAX_TENSOR_DIMS;
                nb::list lst;
                for (uint32_t i = 0; i < n; ++i)
                    lst.append(self.shapes[i]);
                return nb::tuple(lst);
            },
            [](Tensor &self, nb::tuple t) {
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
            [](const Tensor &self) -> uint32_t {
                return self.ndims;
            }
        )

        .def_prop_rw(
            "dtype",
            [](const Tensor &self) -> DataType {
                return self.dtype;
            },
            [](Tensor &self, DataType dt) {
                self.dtype = dt;
                self.buffer.size = self.numel() * get_element_size(dt);
            }
        )

        .def_prop_rw(
            "child_memory",
            [](const Tensor &self) -> bool {
                return self.is_child_memory();
            },
            [](Tensor &self, bool v) {
                self.child_memory = v ? 1 : 0;
            }
        )

        // Read-only views of the strided metadata (always contiguous for make()).
        .def_prop_ro(
            "strides",
            [](const Tensor &self) -> nb::tuple {
                nb::list lst;
                for (uint32_t i = 0; i < self.ndims && i < MAX_TENSOR_DIMS; ++i)
                    lst.append(self.strides[i]);
                return nb::tuple(lst);
            }
        )
        .def_prop_ro(
            "start_offset",
            [](const Tensor &self) -> uint64_t {
                return self.start_offset;
            }
        )
        .def_prop_ro(
            "is_contiguous",
            [](const Tensor &self) -> bool {
                return self.is_contiguous;
            }
        )

        .def(
            "nbytes",
            [](const Tensor &self) -> uint64_t {
                return self.nbytes();
            },
            "Compute total bytes (product of shapes * element_size)."
        )

        .def("__repr__", [](const Tensor &self) -> std::string {
            std::ostringstream os;
            os << "Tensor(data=0x" << std::hex << self.buffer.addr << std::dec << ", shapes=(";
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
            "Add a Tensor. Must be called before any add_scalar()."
        )

        .def(
            "add_scalar", &ChipStorageTaskArgs::add_scalar, nb::arg("s"),
            "Add a uint64_t scalar. After this, add_tensor() is no longer allowed."
        )

        .def(
            "tensor",
            [](const ChipStorageTaskArgs &self, int32_t i) -> const Tensor & {
                if (i < 0 || i >= self.tensor_count())
                    throw std::out_of_range("ChipStorageTaskArgs tensor index out of range");
                return self.tensor(i);
            },
            nb::arg("i"), nb::rv_policy::reference_internal, "Return the Tensor at index i."
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
            [](TaskArgs &self, const Tensor &t, TensorArgType tag) {
                self.add_tensor(t, tag);
            },
            nb::arg("t"), nb::arg("tag") = TensorArgType::INPUT,
            "Add a Tensor with an optional TensorArgType tag (default INPUT)."
        )

        .def(
            "add_scalar", &TaskArgs::add_scalar, nb::arg("s"),
            "Add a uint64_t scalar. After this, add_tensor() is no longer allowed."
        )

        .def(
            "tensor",
            [](const TaskArgs &self, int32_t i) -> const Tensor & {
                if (i < 0 || i >= self.tensor_count()) throw std::out_of_range("TaskArgs tensor index out of range");
                return self.tensor(i);
            },
            nb::arg("i"), nb::rv_policy::reference_internal, "Return the Tensor at index i."
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
            "_write_blob",
            [](const TaskArgs &self, uint64_t dst_ptr, size_t capacity) {
                const size_t needed = task_args_blob_size(self);
                if (dst_ptr == 0 || needed > capacity) {
                    throw std::runtime_error(
                        "TaskArgs._write_blob: need " + std::to_string(needed) + ", capacity " +
                        std::to_string(capacity)
                    );
                }
                write_blob(reinterpret_cast<uint8_t *>(dst_ptr), self);
                return needed;
            },
            nb::arg("dst_ptr"), nb::arg("capacity"), "Serialize this TaskArgs into caller-owned shared memory."
        )

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
        .def_prop_ro_static(
            "_blob_size",
            [](nb::handle) {
                return sizeof(CallConfig);
            }
        )
        .def(
            "_write_blob",
            [](const CallConfig &self, uint64_t dst_ptr, size_t capacity) {
                if (dst_ptr == 0 || capacity < sizeof(CallConfig)) {
                    throw std::runtime_error("CallConfig._write_blob: destination is null or too small");
                }
                std::memcpy(reinterpret_cast<void *>(dst_ptr), &self, sizeof(CallConfig));
                return sizeof(CallConfig);
            },
            nb::arg("dst_ptr"), nb::arg("capacity"), "Serialize the CallConfig POD into shared memory."
        )
        .def_rw("block_dim", &CallConfig::block_dim)
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
            "enable_l2_swimlane",
            [](const CallConfig &c) {
                return c.enable_l2_swimlane;
            },
            // Accept either an int perf_level (0-4) or a Python bool. `True` maps to
            // level 4 (full collection) to preserve the pre-perf_level semantics for
            // callers that still pass a boolean; `False` maps to 0.
            [](CallConfig &c, nb::object v) {
                if (PyBool_Check(v.ptr())) {
                    c.enable_l2_swimlane = nb::cast<bool>(v) ? 4 : 0;
                } else {
                    int level = nb::cast<int>(v);
                    c.enable_l2_swimlane = (level < 0) ? 0 : (level > 4) ? 4 : level;
                }
            }
        )
        // Accept either an int dump level (0=off, 1=partial, 2=full,
        // 3=full_json_only) or a Python bool. `True` maps to level 1
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
            os << "CallConfig(block_dim=" << self.block_dim << ", aicpu_thread_num=" << self.aicpu_thread_num
               << ", enable_l2_swimlane=" << self.enable_l2_swimlane << ", enable_dump_args=" << self.enable_dump_args
               << ", enable_pmu=" << self.enable_pmu << ", enable_dep_gen=" << (self.enable_dep_gen ? "True" : "False")
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
    // src/common/log/host_log.h::simpler::log::kDefaultThreshold; if you change
    // one, change the other.
    m.attr("DEFAULT_LOG_THRESHOLD") = 20;  // V5 = Python INFO

    // Per-stage run timing (host wall, on-NPU device wall + AICPU phase
    // breakdown) is no longer returned from run(); the platform emits it as
    // `[STRACE]` log markers — parse with simpler_setup.tools.strace_timing.

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
            "run_from_blob",
            [](ChipWorker &self, int32_t callable_id, uint64_t args_blob_ptr, size_t blob_capacity,
               const CallConfig &config) {
                // The mailbox region is the on-wire format `write_blob` produced;
                // `read_blob` is the matching reader that returns a zero-copy
                // TaskArgsView into the caller-owned bytes. Forwards to the
                // existing `run(cid, view, config)` path so chip-child
                // loops never re-implement the tensor/scalar layout in Python
                // (where it has historically dropped fields like child_memory).
                TaskArgsView view = read_blob(reinterpret_cast<const uint8_t *>(args_blob_ptr), blob_capacity);
                self.run(callable_id, view, config);
            },
            nb::arg("callable_id"), nb::arg("args_blob_ptr"), nb::arg("blob_capacity"), nb::arg("config"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Launch a callable_id from a raw mailbox-blob pointer + capacity "
            "(used by chip-child mailbox loops to avoid Python-side re-deserialisation "
            "of the per-task tensor/scalar layout). The blob must be in the format "
            "produced by `write_blob`; read_blob enforces capacity bounds against shm corruption."
        )
        .def(
            "prepare_from_blob",
            [](ChipWorker &self, uint64_t request_id, int32_t callable_id, uint64_t args_blob_ptr, size_t blob_capacity,
               const CallConfig &config, unsigned arena_bank) {
                TaskArgsView view = read_blob(reinterpret_cast<const uint8_t *>(args_blob_ptr), blob_capacity);
                self.prepare(request_id, callable_id, view, config, arena_bank);
            },
            nb::arg("request_id"), nb::arg("callable_id"), nb::arg("args_blob_ptr"), nb::arg("blob_capacity"),
            nb::arg("config"), nb::arg("arena_bank"), nb::call_guard<nb::gil_scoped_release>(),
            "Bind a request and start Host work without launching Device execution."
        )
        .def(
            "prepare_from_request_blob",
            [](ChipWorker &self, uint64_t request_id, int32_t callable_id, uint64_t args_blob_ptr,
               size_t args_blob_capacity, uint64_t config_blob_ptr, size_t config_blob_size, unsigned arena_bank) {
                if (config_blob_ptr == 0 || config_blob_size != sizeof(CallConfig)) {
                    throw std::runtime_error("prepare_from_request_blob: invalid CallConfig blob size");
                }
                CallConfig config;
                std::memcpy(&config, reinterpret_cast<const void *>(config_blob_ptr), sizeof(config));
                TaskArgsView view = read_blob(reinterpret_cast<const uint8_t *>(args_blob_ptr), args_blob_capacity);
                self.prepare(request_id, callable_id, view, config, arena_bank);
            },
            nb::arg("request_id"), nb::arg("callable_id"), nb::arg("args_blob_ptr"), nb::arg("args_blob_capacity"),
            nb::arg("config_blob_ptr"), nb::arg("config_blob_size"), nb::arg("arena_bank"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Prepare a request from TaskArgs and CallConfig PODs stored in shared memory."
        )
        .def(
            "execute_prepared", &ChipWorker::execute_prepared, nb::arg("request_id"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Launch Device execution for a request previously prepared from the admission lane."
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
               uint32_t domain_rank, size_t window_size) {
                auto [device_ctx, local_window_base] =
                    self.comm_alloc_domain_windows(comm_handle, allocation_id, rank_ids, domain_rank, window_size);
                return nb::make_tuple(device_ctx, local_window_base);
            },
            nb::arg("comm_handle"), nb::arg("allocation_id"), nb::arg("rank_ids"), nb::arg("domain_rank"),
            nb::arg("window_size"),
            "Collectively allocate a fresh per-rank pool for a subset; returns "
            "(device_ctx, local_window_base) for this rank."
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

    nb::class_<L2ChildOnboardRegionExport>(m, "_L2ChildOnboardRegionExport")
        .def_ro("device_addr", &L2ChildOnboardRegionExport::device_addr)
        .def_ro("mapping_bytes", &L2ChildOnboardRegionExport::mapping_bytes)
        .def_ro("shareable_handle", &L2ChildOnboardRegionExport::shareable_handle)
        .def_ro("registry_handle", &L2ChildOnboardRegionExport::registry_handle);

    m.def(
        "_l3_host_mapped_region_import_sim",
        [](const std::string &token, uint64_t mapping_bytes) -> uint64_t {
            if (mapping_bytes == 0 || mapping_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::invalid_argument("L3-L2 sim L3 Host mapped-region import requires a positive mapping size");
            }
            std::string name = shm_name_for_open(token);
            int fd = shm_open(name.c_str(), O_RDWR, 0);
            if (fd < 0) {
                throw std::runtime_error("L3-L2 sim L3 Host mapped-region import shm_open failed");
            }
            void *base = mmap(nullptr, static_cast<size_t>(mapping_bytes), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (base == MAP_FAILED) {
                int err = errno;
                ::close(fd);
                throw std::runtime_error(
                    std::string("L3-L2 sim L3 Host mapped-region import mmap failed: ") + std::strerror(err)
                );
            }
            L3HostMappedRegion mapping{};
            mapping.profile = L3L2RegionAccessProfile::SIM_POSIX_SHM;
            mapping.fd = fd;
            mapping.device_addr = reinterpret_cast<uint64_t>(base);
            mapping.mapping_bytes = mapping_bytes;
            return g_l3_host_mapped_regions.emplace(mapping);
        },
        nb::arg("token"), nb::arg("mapping_bytes"), nb::call_guard<nb::gil_scoped_release>(),
        "Import a sim L3-L2 POSIX shm region for L3 Host mapped-region access."
    );
    m.def(
        "_l3_host_mapped_region_import_onboard",
        [](int device_id, uint64_t shareable_handle, uint64_t mapping_bytes) -> uint64_t {
            if (device_id < 0) {
                throw std::invalid_argument("L3-L2 onboard mapped-region import requires a non-negative device id");
            }
            if (mapping_bytes == 0 || mapping_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::invalid_argument("L3-L2 onboard mapped-region import requires a positive mapping size");
            }
            L3HostMappedRegion mapping{};
            mapping.profile = L3L2RegionAccessProfile::ONBOARD_VMM;
            mapping.device_id = device_id;
            mapping.mapping_bytes = mapping_bytes;
            mapping.bind_acl_device();
            AclRuntimeApi &api = acl_api();
            mapping.shareable_handle = shareable_handle;
            mapping.vmm_handle = api.vmm_import_shareable_with_check(shareable_handle, device_id);
            void *mapped_addr = nullptr;
            try {
                mapped_addr = api.vmm_reserve_with_check(mapping_bytes);
                api.vmm_map_with_check(mapped_addr, mapping_bytes, mapping.vmm_handle);
                api.vmm_set_access_with_check(mapped_addr, mapping_bytes, device_id);
            } catch (...) {
                std::string cleanup_error;
                api.vmm_release_collecting(mapped_addr, mapping.vmm_handle, cleanup_error);
                throw;
            }
            mapping.device_addr = reinterpret_cast<uint64_t>(mapped_addr);
            return g_l3_host_mapped_regions.emplace(mapping);
        },
        nb::arg("device_id"), nb::arg("shareable_handle"), nb::arg("mapping_bytes"),
        nb::call_guard<nb::gil_scoped_release>(), "Import an onboard VMM L3-L2 region for L3 Host mapped-region access."
    );
    m.def(
        "_l3_host_mapped_region_close",
        [](uint64_t handle) {
            std::optional<L3HostMappedRegion> removed = g_l3_host_mapped_regions.remove(handle);
            if (!removed.has_value()) {
                return;
            }
            L3HostMappedRegion mapping = *removed;
            if (mapping.profile == L3L2RegionAccessProfile::ONBOARD_VMM) {
                mapping.bind_acl_device();
                std::string cleanup_error;
                acl_api().vmm_release_collecting(
                    reinterpret_cast<void *>(static_cast<uintptr_t>(mapping.device_addr)), mapping.vmm_handle,
                    cleanup_error
                );
                if (!cleanup_error.empty()) {
                    throw std::runtime_error(cleanup_error);
                }
                return;
            }

            std::string cleanup_error;
            if (mapping.device_addr != 0 &&
                munmap(reinterpret_cast<void *>(static_cast<uintptr_t>(mapping.device_addr)), mapping.mapping_bytes) !=
                    0) {
                int err = errno;
                append_cleanup_error(
                    cleanup_error, std::string("L3-L2 sim L3 Host mapped-region munmap failed: ") + std::strerror(err)
                );
            }
            if (mapping.fd >= 0 && ::close(mapping.fd) != 0) {
                int err = errno;
                append_cleanup_error(
                    cleanup_error, std::string("L3-L2 sim L3 Host mapped-region close failed: ") + std::strerror(err)
                );
            }
            if (!cleanup_error.empty()) {
                throw std::runtime_error(cleanup_error);
            }
        },
        nb::arg("handle"), nb::call_guard<nb::gil_scoped_release>(), "Close an L3 Host mapped-region handle."
    );
    m.def(
        "_l3_host_mapped_payload_write",
        [](uint64_t handle, uint64_t payload_offset, uint64_t host_ptr, uint64_t nbytes) {
            if (host_ptr == 0) {
                throw std::invalid_argument("L3-L2 payload_write host_ptr must be nonzero");
            }
            L3HostMappedRegion mapping = g_l3_host_mapped_regions.find(handle);
            mapping.copy_to(payload_offset, reinterpret_cast<const void *>(static_cast<uintptr_t>(host_ptr)), nbytes);
        },
        nb::arg("handle"), nb::arg("payload_offset"), nb::arg("host_ptr"), nb::arg("nbytes"),
        nb::call_guard<nb::gil_scoped_release>(), "Copy L3 Host bytes into an imported L3-L2 payload range."
    );
    m.def(
        "_l3_host_mapped_payload_read",
        [](uint64_t handle, uint64_t payload_offset, uint64_t host_ptr, uint64_t nbytes) {
            if (host_ptr == 0) {
                throw std::invalid_argument("L3-L2 payload_read host_ptr must be nonzero");
            }
            L3HostMappedRegion mapping = g_l3_host_mapped_regions.find(handle);
            mapping.copy_from(reinterpret_cast<void *>(static_cast<uintptr_t>(host_ptr)), payload_offset, nbytes);
        },
        nb::arg("handle"), nb::arg("payload_offset"), nb::arg("host_ptr"), nb::arg("nbytes"),
        nb::call_guard<nb::gil_scoped_release>(), "Copy imported L3-L2 payload bytes into L3 Host memory."
    );
    m.def(
        "_l3_host_mapped_counter_notify",
        [](uint64_t handle, uint64_t counter_offset, int32_t value, int op) {
            L3HostMappedRegion mapping = g_l3_host_mapped_regions.find(handle);
            mapping.notify_counter(counter_offset, value, checked_notify_op(op));
        },
        nb::arg("handle"), nb::arg("counter_offset"), nb::arg("value"), nb::arg("op"),
        nb::call_guard<nb::gil_scoped_release>(), "Store or add one L3 Host-side L3-L2 signal counter."
    );
    m.def(
        "_l3_host_mapped_counter_test",
        [](uint64_t handle, uint64_t counter_offset, int32_t operand, int cmp) -> std::tuple<bool, int32_t> {
            L3HostMappedRegion mapping = g_l3_host_mapped_regions.find(handle);
            return mapping.test_counter(counter_offset, operand, checked_wait_cmp(cmp));
        },
        nb::arg("handle"), nb::arg("counter_offset"), nb::arg("operand"), nb::arg("cmp"),
        nb::call_guard<nb::gil_scoped_release>(), "Load and compare one L3 Host-side L3-L2 signal counter."
    );
    m.def(
        "_l3_host_mapped_counter_wait",
        [](uint64_t handle, uint64_t counter_offset, int32_t operand, int cmp,
           uint64_t timeout_ns) -> std::tuple<int, int, int32_t, bool, std::string> {
            L3HostMappedRegion mapping = g_l3_host_mapped_regions.find(handle);
            return mapping.wait_counter(counter_offset, operand, checked_wait_cmp(cmp), timeout_ns);
        },
        nb::arg("handle"), nb::arg("counter_offset"), nb::arg("operand"), nb::arg("cmp"), nb::arg("timeout_ns"),
        nb::call_guard<nb::gil_scoped_release>(), "Poll one L3 Host-side L3-L2 signal counter until match or timeout."
    );
    m.def(
        "_l3_child_onboard_region_create",
        [](uint64_t nbytes) -> L2ChildOnboardRegionExport {
            if (nbytes == 0 || nbytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::invalid_argument("L3-L2 onboard child region requires a positive mapping size");
            }
            AclRuntimeApi &api = acl_api();
            int device_id = api.current_device_with_check();
            uint64_t mapping_bytes = align_vmm_bytes(nbytes, api.vmm_granularity_with_check(device_id));
            L2ChildOnboardRegion region{};
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
            uint64_t registry_handle = g_l2_child_onboard_regions.emplace(region);
            return L2ChildOnboardRegionExport{
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
            std::optional<L2ChildOnboardRegion> removed = g_l2_child_onboard_regions.remove(registry_handle);
            if (!removed.has_value()) {
                return;
            }
            L2ChildOnboardRegion region = *removed;
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

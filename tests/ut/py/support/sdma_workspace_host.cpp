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
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "acl/acl.h"

namespace sdma_fixture {
enum class Failure { NONE, BEFORE_BIND, THROW_BEFORE_BIND, AFTER_BIND, THROW_AFTER_BIND };
Failure failure = Failure::NONE;
bool zero_size = false;
bool fail_malloc = false;
bool fail_free = false;
int free_attempts = 0;
int allocations = 0;
int frees = 0;
int initializations = 0;
int destructions = 0;
int live_streams = 0;
std::unordered_set<void *> buffers;
}  // namespace sdma_fixture

namespace pto::comm::sdma {
inline constexpr uint32_t kSdmaMaxChannelGroups = 48;
class SdmaWorkspaceManager {
public:
    static inline SdmaWorkspaceManager *last = nullptr;
    SdmaWorkspaceManager() { last = this; }
    ~SdmaWorkspaceManager() {
        if (address_ != nullptr) assert(sdma_fixture::buffers.count(address_) == 1);
        if (stream_created_) --sdma_fixture::live_streams;
        ++sdma_fixture::destructions;
        last = nullptr;
    }
    static aclError GetWorkspaceSize(uint64_t *size) {
        *size = sdma_fixture::zero_size ? 0 : 16384;
        return sdma_fixture::zero_size ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS;
    }
    bool Init(void *address, uint64_t size) {
        using namespace sdma_fixture;
        ++initializations;
        assert(address != nullptr && size == 16384 && buffers.count(address) == 1);
        // ISA creates STARS streams before binding the external workspace.
        stream_created_ = true;
        ++live_streams;
        if (failure == Failure::BEFORE_BIND) return false;
        if (failure == Failure::THROW_BEFORE_BIND) throw std::runtime_error("before binding");
        address_ = address;
        if (failure == Failure::THROW_AFTER_BIND) throw std::runtime_error("after binding");
        return failure != Failure::AFTER_BIND;
    }
    void *GetWorkspaceAddr() const { return address_; }

private:
    void *address_ = nullptr;
    bool stream_created_ = false;
};
}  // namespace pto::comm::sdma

#include "comm_hccl.cpp"

extern "C" aclError aclrtUnmapMem(void *) { std::abort(); }
extern "C" aclError aclrtReleaseMemAddress(void *) { std::abort(); }
extern "C" aclError aclrtFreePhysical(aclrtDrvMemHandle) { std::abort(); }

extern "C" aclError aclrtMalloc(void **out, size_t bytes, aclrtMemMallocPolicy) {
    if (sdma_fixture::fail_malloc) return ACL_ERROR_BAD_ALLOC;
    *out = std::malloc(bytes);
    assert(*out != nullptr);
    assert(sdma_fixture::buffers.insert(*out).second);
    ++sdma_fixture::allocations;
    return ACL_SUCCESS;
}

extern "C" aclError aclrtFree(void *address) {
    ++sdma_fixture::free_attempts;
    if (sdma_fixture::fail_free) return ACL_ERROR_FAILURE;
    assert(sdma_fixture::buffers.erase(address) == 1);
    std::free(address);
    ++sdma_fixture::frees;
    return ACL_SUCCESS;
}

int main(int argc, char **argv) {
    using namespace sdma_fixture;
    using Manager = pto::comm::sdma::SdmaWorkspaceManager;
    assert(argc == 2);
    const std::string scenario = argv[1];
    if (scenario == "before_bind_free_failure") {
        failure = Failure::BEFORE_BIND;
        fail_free = true;
        uint64_t addresses[DMA_WORKSPACE_KIND_COUNT]{};
        void *handle = nullptr;
        assert(dma_workspace_provision(1U << DMA_WORKSPACE_SDMA, addresses, DMA_WORKSPACE_KIND_COUNT, &handle) != 0);
        assert(handle != nullptr);
        assert(live_streams == 1 && destructions == 0);
        assert(buffers.size() == 1 && addresses[DMA_WORKSPACE_SDMA] == 0);
        const int attempts = free_attempts;
        assert(dma_workspace_release(handle) == COMM_RECLAMATION_UNCONFIRMED);
        assert(free_attempts == attempts && buffers.size() == 1);
        auto *owner = static_cast<SdmaWorkspaceOwner *>(handle);
        assert(owner->allocation != nullptr && buffers.count(owner->allocation) == 1);
        fail_free = false;
        void *allocation = owner->allocation;
        delete owner;
        assert(aclrtFree(allocation) == ACL_SUCCESS);
        return 0;
    }
    if (scenario == "release_free_failure") {
        uint64_t addresses[DMA_WORKSPACE_KIND_COUNT]{};
        void *handle = nullptr;
        assert(dma_workspace_provision(1U << DMA_WORKSPACE_SDMA, addresses, DMA_WORKSPACE_KIND_COUNT, &handle) == 0);
        fail_free = true;
        assert(dma_workspace_release(handle) == COMM_RECLAMATION_UNCONFIRMED);
        auto *owner = static_cast<SdmaWorkspaceOwner *>(handle);
        assert(!owner->manager && buffers.count(owner->allocation) == 1);
        const int attempts = free_attempts;
        assert(dma_workspace_release(handle) == COMM_RECLAMATION_UNCONFIRMED);
        assert(free_attempts == attempts);
        fail_free = false;
        void *allocation = owner->allocation;
        delete owner;
        assert(aclrtFree(allocation) == ACL_SUCCESS);
        return 0;
    }
    if (scenario == "before_bind") failure = Failure::BEFORE_BIND;
    else if (scenario == "throw_before_bind") failure = Failure::THROW_BEFORE_BIND;
    else if (scenario == "after_bind") failure = Failure::AFTER_BIND;
    else if (scenario == "throw_after_bind") failure = Failure::THROW_AFTER_BIND;
    else if (scenario == "zero_size") zero_size = true;
    else if (scenario == "allocation_failure") fail_malloc = true;
    else assert(scenario == "success");

    const bool initialized = !zero_size && !fail_malloc;
    const bool bound_failure = failure == Failure::AFTER_BIND || failure == Failure::THROW_AFTER_BIND;
    const bool success = initialized && failure == Failure::NONE;
    for (int attempt = 0; attempt < 3; ++attempt) {
        const int destroyed_before = destructions;
        const int freed_before = frees;
        uint64_t addresses[DMA_WORKSPACE_KIND_COUNT];
        for (auto &address : addresses)
            address = UINT64_MAX;
        void *handle = reinterpret_cast<void *>(1);
        const int rc = dma_workspace_provision(1U << DMA_WORKSPACE_SDMA, addresses, DMA_WORKSPACE_KIND_COUNT, &handle);
        assert(allocations == (initialized ? attempt + 1 : 0));
        assert(initializations == (initialized ? attempt + 1 : 0));
        if (success) {
            assert(rc == 0 && handle != nullptr);
            assert(static_cast<SdmaWorkspaceOwner *>(handle)->manager.get() == Manager::last);
            assert(addresses[DMA_WORKSPACE_SDMA] == reinterpret_cast<uint64_t>(Manager::last->GetWorkspaceAddr()));
            assert(buffers.size() == 1 && destructions == destroyed_before && frees == freed_before);
            assert(dma_workspace_release(handle) == 0);
            assert(destructions == destroyed_before + 1 && frees == freed_before + 1);
        } else {
            assert(rc != 0 && (handle != nullptr) == initialized);
            for (auto address : addresses)
                assert(address == 0);
            assert(destructions == destroyed_before + (initialized ? 0 : 1));
            assert(live_streams == (initialized ? 1 : 0));
            assert(frees == freed_before + (initialized && !bound_failure ? 1 : 0));
            assert(buffers.size() == (bound_failure ? 1U : 0U));
            // The fixture reclaims abandoned host objects after checking the provider's failure policy.
            auto *owner = static_cast<SdmaWorkspaceOwner *>(handle);
            if (owner) assert(dma_workspace_release(handle) == COMM_RECLAMATION_UNCONFIRMED);
            void *bound = owner ? owner->allocation : nullptr;
            delete owner;
            if (bound != nullptr) assert(aclrtFree(bound) == ACL_SUCCESS);
        }
        assert(buffers.empty());
        assert(live_streams == 0);
    }
    assert(dma_workspace_release(nullptr) == 0);
    assert(allocations == frees);
}

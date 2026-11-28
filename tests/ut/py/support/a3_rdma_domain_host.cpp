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
#define main unused_a2a3_lifecycle_main
#include "../a2a3/toolkit/a2a3_domain_lifecycle_host.cpp"
#undef main

#include <filesystem>

extern "C" aclError aclrtGetPhyDevIdByLogicDevId(int32_t, int32_t *physical) {
    *physical = static_cast<int32_t>(getpid());
    return ACL_SUCCESS;
}

namespace {

struct TestDomain {
    std::unique_ptr<CommHandle_> handle = std::make_unique<CommHandle_>();
    std::string folder;
    uint64_t context = 0;
    uint64_t window = 0;
    size_t size = 0;

    TestDomain() {
        char pattern[] = "/tmp/simpler-rdma-ut-XXXXXX";
        assert(mkdtemp(pattern));
        folder = pattern;
        handle->rootinfo_path = folder + "/root";
        handle->run_token = 7;
        handle->rank = 0;
        handle->nranks = 1;
        handle->hccl_comm = reinterpret_cast<HcclComm>(uintptr_t{1});
    }
    ~TestDomain() { std::filesystem::remove_all(folder); }

    int allocate(uint32_t port = 0) {
        const uint32_t ranks[]{0};
        const CommWorkspaceConfig options{1, 0, COMM_CCU_MODE_SCHED, 2, port, {}};
        return comm_alloc_domain_windows(
            handle.get(), 1, ranks, 1, 0, 4096, comm_workspace_bit(CommWorkspaceKind::RDMA), &context, &window, &size,
            &options
        );
    }
    int release(bool abort = false) {
        return abort ? comm_abort_domain_windows(handle.get(), 1, 1, 0) :
                       comm_release_domain_windows(handle.get(), 1, 1, 0);
    }
};

void test_exchange() {
    TestDomain domain;
    DomainAllocation left, right;
    for (auto *owner : {&left, &right}) {
        owner->nranks = 2;
        owner->rdma_rootinfo = domain.handle->rootinfo_path;
        owner->rdma_run_token = 7;
        owner->rdma_allocation_id = 1;
        owner->rdma_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    }
    left.rank = 0;
    right.rank = 1;
    const uint32_t sent[2]{0x12003400, 0xab00cd00};
    uint32_t received[2][2]{};
    auto a = std::async(std::launch::async, [&] {
        return domain_host_all_gather(&left, &sent[0], 4, received[0]);
    });
    auto b = std::async(std::launch::async, [&] {
        return domain_host_all_gather(&right, &sent[1], 4, received[1]);
    });
    assert(a.get() && b.get());
    for (const auto &values : received)
        assert(values[0] == sent[0] && values[1] == sent[1]);
    left.rdma_deadline = std::chrono::steady_clock::now();
    assert(!domain_host_all_gather(&left, &sent[0], 4, received[0]));
}

}  // namespace

int main(int argc, char **argv) {
    assert(argc == 2);
    const std::string scenario = argv[1];
    if (scenario == "exchange") {
        test_exchange();
        return 0;
    }
    TestDomain first;
    if (scenario == "initialization") {
        rdma_fixture::fail_initialization = true;
        rdma_fixture::fail_cleanup = true;
        assert(first.allocate() == COMM_INITIALIZATION_ABORT_REQUIRED);
        assert(first.context == 0 && first.window == 0 && !mappings.empty());
        const auto live = mappings.size();
        assert(first.release(true) == COMM_RECLAMATION_UNCONFIRMED);
        assert(mappings.size() == live && rdma_fixture::providers == 1);
        rdma_fixture::fail_cleanup = false;
        assert(first.release(true) == 0);
    } else {
        assert(first.allocate() == 0);
        assert(first.size == 4096 && first.context != 0 && first.window != 0);
        const auto *context = reinterpret_cast<const CommContext *>(first.context);
        assert(context->windowsIn[0] == first.window && context->windowsOut[0] == first.window);
        assert(comm_workspace<CommWorkspaceKind::RDMA>(*context).addr != 0);
        assert(rdma_fixture::providers == 1);
        if (scenario == "gm_cleanup" || scenario == "ctx_cleanup") {
            fail_device_free_call = device_free_calls + (scenario == "gm_cleanup" ? 1 : 4);
            assert(first.release() == COMM_RECLAMATION_UNCONFIRMED);
            const auto freed = device_free_calls;
            assert(first.handle->domain_allocations.count(1) == 1);
            assert(first.handle->domain_allocations.at(1)->release_failed);
            assert(first.handle->domain_allocations.at(1)->rdma_port_fd >= 0);
            assert(first.release() == COMM_RECLAMATION_UNCONFIRMED);
            assert(device_free_calls == freed);
            assert(comm_destroy(first.handle.get()) == COMM_RECLAMATION_UNCONFIRMED);
            return 0;
        }
        if (scenario == "port") {
            TestDomain second;
            const auto port = first.handle->domain_allocations.at(1)->rdma_port;
            assert(second.allocate(port) == COMM_INITIALIZATION_ABORT_REQUIRED);
            assert(rdma_fixture::providers == 1);
            assert(second.release(true) == 0);
        } else if (scenario == "cleanup") {
            rdma_fixture::fail_cleanup = true;
            const auto live = mappings.size();
            assert(first.release() == COMM_RECLAMATION_UNCONFIRMED);
            assert(mappings.size() == live && first.handle->domain_allocations.count(1) == 1);
            assert(comm_destroy(first.handle.get()) == COMM_RECLAMATION_UNCONFIRMED);
            rdma_fixture::fail_cleanup = false;
        }
        assert(first.release() == 0);
    }
    assert(mappings.empty() && rdma_fixture::providers == 0 && first.handle->domain_allocations.empty());
}

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
#include <string>
#include <vector>
#include "comm_hccl.cpp"

int live_comms = 1;
int finalized = 0;
int sync_result = 0;
int destroy_result = 0;
int destroy_calls = 0;
int fail_destroy_at = 0;
std::vector<int> order;
extern "C" aclError aclrtSynchronizeDeviceWithTimeout(int32_t) {
    order.push_back(1);
    return sync_result;
}
extern "C" HcclResult HcclCommDestroy(HcclComm) {
    order.push_back(2);
    ++destroy_calls;
    const int result = destroy_result || (fail_destroy_at && destroy_calls == fail_destroy_at);
    if (!result) --live_comms;
    return static_cast<HcclResult>(result);
}
extern "C" aclError aclrtFree(void *) {
    assert(live_comms == 0 && finalized > 0);
    order.push_back(4);
    return ACL_SUCCESS;
}
extern "C" aclError aclrtUnmapMem(void *) {
    assert(live_comms == 0 && finalized > 0);
    order.push_back(5);
    return ACL_SUCCESS;
}
extern "C" aclError aclrtReleaseMemAddress(void *) { return ACL_SUCCESS; }
extern "C" aclError aclrtFreePhysical(aclrtDrvMemHandle) { return ACL_SUCCESS; }
extern "C" aclError aclrtMemcpy(void *, size_t, const void *, size_t, aclrtMemcpyKind) { return 1; }

int main(int argc, char **argv) {
    assert(argc == 2);
    char dir[] = "/tmp/simpler-urma-lifecycle-XXXXXX";
    assert(mkdtemp(dir));
    auto *h = new CommHandle_{};
    h->rank = 0;
    h->nranks = 1;
    h->rootinfo_path = std::string(dir) + "/root";
    h->hccl_comm = reinterpret_cast<HcclComm>(1);
    std::string scenario = argv[1];
    if (scenario == "init_failure") {
        h->host_ctx.winSize = 64;
        h->host_ctx.windowsIn[0] = 64;
        assert(!ensure_base_urma_workspace(h));
        assert(finalized == 0);
        assert(h->urma_workspace);
    } else if (scenario == "domain_init_failure") {
        live_comms = 2;
        auto alloc = std::make_unique<DomainAllocation>();
        alloc->rank = 0;
        alloc->nranks = 1;
        alloc->urma_comm = reinterpret_cast<HcclComm>(2);
        alloc->local_buf = reinterpret_cast<void *>(64);
        alloc->own_handle = reinterpret_cast<aclrtDrvMemHandle>(64);
        assert(!init_urma_workspace(h, alloc->urma_comm, 0, 1, alloc->local_buf, 64, alloc->urma_workspace));
        assert(finalized == 0);
        h->domain_allocations.emplace(1, std::move(alloc));
        assert(comm_release_domain_windows(h, 1, 1, 0) == 0);
        assert(finalized == 0);
    } else if (scenario == "domains" || scenario == "retired_domains" || scenario == "child_destroy_failure") {
        live_comms = 3;
        for (uint64_t id : {1, 2}) {
            auto alloc = std::make_unique<DomainAllocation>();
            alloc->urma_comm = reinterpret_cast<HcclComm>(id + 1);
            alloc->rank = 0;
            alloc->nranks = 1;
            alloc->urma_workspace = std::make_unique<pto::comm::urma::UrmaWorkspaceManager>();
            h->domain_allocations.emplace(id, std::move(alloc));
        }
        assert(comm_release_domain_windows(h, 1, 1, 0) == 0);
        assert(finalized == 0);
        assert(h->domain_allocations.count(2) == 1);
        assert(comm_release_domain_windows(h, 1, 1, 0) != 0);
        if (scenario == "retired_domains") {
            assert(comm_release_domain_windows(h, 2, 1, 0) == 0);
            assert(finalized == 0);
        }
        if (scenario == "child_destroy_failure") fail_destroy_at = 2;
    } else {
        h->urma_workspace = std::make_unique<pto::comm::urma::UrmaWorkspaceManager>();
    }
    if (scenario == "stop_failure") sync_result = 1;
    if (scenario == "drain_failure") h->urma_ready = true;
    if (scenario == "handshake_failure") h->rootinfo_path = "/dev/null/rootinfo";
    if (scenario == "destroy_failure") destroy_result = 1;
    const int rc = comm_destroy(h);
    if (sync_result || destroy_result || fail_destroy_at || scenario == "drain_failure" ||
        scenario == "handshake_failure") {
        assert(rc != 0);
        assert(live_comms == (fail_destroy_at ? 2 : 1));
        assert(finalized == 0);
        assert(comm_destroy(h) != 0);
        assert(finalized == 0);
    } else {
        assert(rc == 0);
        assert(live_comms == 0);
        assert(finalized == (scenario == "domains" || scenario == "retired_domains" ? 2 : 1));
        assert(order.front() == 1);
        assert(order[destroy_calls] == 2);
        assert(order[destroy_calls + 1] == 3);
    }
    std::filesystem::remove_all(dir);
}

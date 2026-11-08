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
#define SIMPLER_COMM_VMM_ACTOR_STUBS_ONLY
#include "a2a3_domain_lifecycle_host.cpp"

#include <sys/wait.h>
#include <signal.h>

static void actor(int rank, int ready, int release, const std::string &folder, const std::string &scenario, int late) {
    alarm(6);
    auto *h = new CommHandle_{};
    h->rank = rank;
    h->nranks = 2;
    h->run_token = 9;
    h->rootinfo_path = folder + "/root";
    h->hccl_comm = reinterpret_cast<HcclComm>(1);
    if (scenario == "legacy_conflict" && rank == 1) assert(h->sdma_owner.bind(CommSdmaAccess::LEGACY, 0));
    if (scenario == "q_conflict" && rank == 1) assert(h->sdma_owner.bind(CommSdmaAccess::SDK_ONLY, 2));
    const bool upload_failure = scenario == "upload_failure";
    fail_context_upload = upload_failure;
    const uint32_t ranks[] = {0, 1};
    if (rank == late) std::this_thread::sleep_for(std::chrono::milliseconds(120));
    uint64_t context = 0, window = 0;
    size_t bytes = 0;
    const bool legacy = scenario == "legacy" || (scenario == "mixed_entry" && rank == 1);
    const uint32_t q = scenario == "request_mismatch" && rank == 1 ? 1 : 3;
    const int rc = legacy ?
                       comm_alloc_domain_windows(h, 1, ranks, 2, rank, 4096, 1, &context, &window, &bytes) :
                       comm_alloc_domain_windows_sdma_sdk_v1(h, 1, ranks, 2, rank, 4096, q, &context, &window, &bytes);
    const bool success = scenario == "managed" || scenario == "legacy";
    assert(rc == (success ? 0 : COMM_INITIALIZATION_ABORT_REQUIRED));
    const auto marker = barrier_marker_path(h->rootinfo_path, h->run_token, domain_barrier_tag(1, "p2p_ready"), rank);
    assert(std::filesystem::exists(marker) == (success || upload_failure));
    assert((peer_import_calls > 0) == (success || upload_failure));
    if (success) {
        assert(h->domain_allocations.count(1) == 1);
        assert(h->sdma_owner.access == (legacy ? CommSdmaAccess::LEGACY : CommSdmaAccess::SDK_ONLY));
        assert(h->sdma_owner.queue_num == (legacy ? 0u : 3u));
        assert(
            reinterpret_cast<uint64_t>(h->sdma_workspace_addr) ==
            comm_workspace<CommWorkspaceKind::SDMA>(h->host_ctx).addr
        );
    } else {
        assert(context == 0 && window == 0);
        assert(h->domain_allocations.count(1) == 1);  // still retained until both Init calls return
    }
    assert(write(ready, &rc, sizeof(rc)) == sizeof(rc));
    char allow = 0;
    assert(read(release, &allow, 1) == 1);  // parent joins both Init outcomes before any abort
    assert((success ? comm_release_domain_windows(h, 1, 2, rank) : comm_abort_domain_windows(h, 1, 2, rank)) == 0);
    if (upload_failure) {
        assert(h->sdma_owner.access == CommSdmaAccess::UNKNOWN && h->sdma_owner.queue_num == 0);
        fail_context_upload = false;
        assert(comm_alloc_domain_windows_sdma_sdk_v1(h, 2, ranks, 2, rank, 4096, 2, &context, &window, &bytes) == 0);
        assert(h->sdma_owner.access == CommSdmaAccess::SDK_ONLY && h->sdma_owner.queue_num == 2);
        const int returned = 1;
        assert(write(ready, &returned, sizeof(returned)) == sizeof(returned));
        assert(read(release, &allow, 1) == 1);
        assert(comm_release_domain_windows(h, 2, 2, rank) == 0);
    }
    if (success) {
        // Domain release must not erase the shared owner's partition/history.
        assert(h->sdma_owner.access == (legacy ? CommSdmaAccess::LEGACY : CommSdmaAccess::SDK_ONLY));
        assert(h->sdma_owner.queue_num == (legacy ? 0u : 3u));
        assert(h->domain_allocations.count(1) == 0);
        assert(
            comm_alloc_domain_windows_sdma_sdk_v1(h, 2, ranks, 2, rank, 4096, 2, &context, &window, &bytes) ==
            COMM_INITIALIZATION_ABORT_REQUIRED
        );
        const auto second =
            barrier_marker_path(h->rootinfo_path, h->run_token, domain_barrier_tag(2, "p2p_ready"), rank);
        assert(!std::filesystem::exists(second));
        const int returned = 1;
        assert(write(ready, &returned, sizeof(returned)) == sizeof(returned));
        assert(read(release, &allow, 1) == 1);
        assert(comm_abort_domain_windows(h, 2, 2, rank) == 0);
    }
    assert(comm_destroy(h) == 0);
    assert(mappings.empty() && physical_handles == 0);
}

int main(int argc, char **argv) {
    assert(argc == 3);
    alarm(7);
    char folder[] = "/tmp/simpler-sdma-owner-XXXXXX";
    assert(mkdtemp(folder));
    const std::string scenario = argv[1];
    const int late = std::stoi(argv[2]);
    int ready[2][2], release[2][2];
    pid_t children[2];
    for (int rank = 0; rank < 2; ++rank) {
        assert(pipe(ready[rank]) == 0 && pipe(release[rank]) == 0);
        children[rank] = fork();
        assert(children[rank] >= 0);
        if (children[rank] == 0) {
            actor(rank, ready[rank][1], release[rank][0], folder, scenario, late);
            _exit(0);
        }
    }
    for (int round = 0;
         round < ((scenario == "managed" || scenario == "legacy" || scenario == "upload_failure") ? 2 : 1); ++round) {
        for (auto &channel : ready) {
            int result = 0;
            assert(read(channel[0], &result, sizeof(result)) == sizeof(result));
        }
        for (auto &channel : release)
            assert(write(channel[1], "1", 1) == 1);
    }
    for (auto child : children) {
        int status = 0;
        assert(waitpid(child, &status, 0) == child);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    std::filesystem::remove_all(folder);
}

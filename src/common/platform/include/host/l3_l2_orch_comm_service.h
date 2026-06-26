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

#ifndef SRC_COMMON_PLATFORM_INCLUDE_HOST_L3_L2_ORCH_COMM_SERVICE_H_
#define SRC_COMMON_PLATFORM_INCLUDE_HOST_L3_L2_ORCH_COMM_SERVICE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "common/l3_l2_orch_comm.h"

enum class L3L2OrchCommControlState : uint32_t {
    IDLE = 0,
    READY = 1,
    RUNNING = 2,
    DONE = 3,
};

struct L3L2OrchCommControlBlock {
    std::atomic<uint32_t> state{static_cast<uint32_t>(L3L2OrchCommControlState::IDLE)};
    uint32_t reserved0{0};
    L3L2OrchCommRequest request{};
    L3L2OrchCommResponse response{};
};

class L3L2OrchCommBackend {
public:
    virtual ~L3L2OrchCommBackend() = default;

    virtual void *l3_l2_allocate_region_bytes(uint64_t bytes) = 0;
    virtual void l3_l2_free_region_bytes(void *ptr) = 0;
    virtual int l3_l2_copy_to_device(void *dev_ptr, const void *host_ptr, uint64_t bytes) = 0;
    virtual int l3_l2_copy_from_device(void *host_ptr, const void *dev_ptr, uint64_t bytes) = 0;
    virtual std::thread l3_l2_create_service_thread(std::function<void()> fn) = 0;
};

class L3L2OrchCommClient {
public:
    int attach(void *control_block, size_t control_block_size);
    int submit(const L3L2OrchCommRequest &request, L3L2OrchCommResponse *response, uint64_t timeout_ns);

private:
    L3L2OrchCommControlBlock *control_{nullptr};
    std::mutex mu_;
};

class L3L2OrchCommService {
public:
    L3L2OrchCommService() = default;
    ~L3L2OrchCommService();

    L3L2OrchCommService(const L3L2OrchCommService &) = delete;
    L3L2OrchCommService &operator=(const L3L2OrchCommService &) = delete;

    int start(L3L2OrchCommBackend *backend, void *control_block, size_t control_block_size);
    int stop();
    bool started() const { return started_.load(std::memory_order_acquire); }

private:
    struct Region {
        uint64_t region_id{0};
        void *payload_dev{nullptr};
        uint64_t payload_bytes{0};
        void *counter_dev{nullptr};
        uint64_t counter_bytes{0};
        bool poisoned{false};
    };

    void loop();
    void execute_request(const L3L2OrchCommRequest &request, L3L2OrchCommResponse *response);
    void alloc_region(const L3L2OrchCommRequest &request, L3L2OrchCommResponse *response);
    void free_region(uint64_t region_id, L3L2OrchCommResponse *response);
    void payload_write(const L3L2OrchCommRequest &request, L3L2OrchCommResponse *response);
    void payload_read(const L3L2OrchCommRequest &request, L3L2OrchCommResponse *response);
    void signal_notify(const L3L2OrchCommRequest &request, L3L2OrchCommResponse *response);
    void signal_test(const L3L2OrchCommRequest &request, L3L2OrchCommResponse *response);
    void signal_wait(const L3L2OrchCommRequest &request, L3L2OrchCommResponse *response);

    Region *find_live_region(uint64_t region_id, L3L2OrchCommResponse *response);
    L3L2OrchRegionDesc desc_for_region(const Region &region) const;
    void *counter_ptr(Region &region, uint64_t counter_addr, L3L2OrchCommResponse *response);
    void release_region(Region &region);
    void release_all_regions();

    L3L2OrchCommBackend *backend_{nullptr};
    L3L2OrchCommControlBlock *control_{nullptr};
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> started_{false};
    std::mutex regions_mu_;
    std::unordered_map<uint64_t, Region> regions_;
    uint64_t next_region_id_{1};
};

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_L3_L2_ORCH_COMM_SERVICE_H_

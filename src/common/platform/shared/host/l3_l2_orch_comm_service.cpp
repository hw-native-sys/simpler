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

#include "host/l3_l2_orch_comm_service.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

enum class ServiceError : uint32_t {
    OK = 0,
    BAD_REQUEST = 1,
    UNKNOWN_COMMAND = 2,
    UNKNOWN_REGION = 3,
    POISONED_REGION = 4,
    ALLOC_FAILED = 5,
    COPY_FAILED = 6,
    SIGNAL_TIMEOUT = 7,
};

constexpr uint64_t kPollSleepNs = 50000;

std::chrono::nanoseconds clamp_timeout_ns(uint64_t timeout_ns) {
    const auto max_timeout = std::chrono::nanoseconds::max();
    if (timeout_ns > static_cast<uint64_t>(max_timeout.count())) {
        return max_timeout;
    }
    return std::chrono::nanoseconds(static_cast<std::chrono::nanoseconds::rep>(timeout_ns));
}

std::chrono::steady_clock::time_point make_deadline(uint64_t timeout_ns) {
    const auto timeout = clamp_timeout_ns(timeout_ns);
    if (timeout == std::chrono::nanoseconds::max()) {
        return std::chrono::steady_clock::time_point::max();
    }
    return std::chrono::steady_clock::now() + timeout;
}

L3L2OrchCommControlState load_state(L3L2OrchCommControlBlock *control) {
    return static_cast<L3L2OrchCommControlState>(control->state.load(std::memory_order_acquire));
}

void store_state(L3L2OrchCommControlBlock *control, L3L2OrchCommControlState state) {
    control->state.store(static_cast<uint32_t>(state), std::memory_order_release);
}

void set_response(
    L3L2OrchCommResponse *response, int32_t status, ServiceError error, uint64_t region_id, const char *message
) {
    if (response == nullptr) {
        return;
    }
    response->status = status;
    response->error_kind = static_cast<uint32_t>(error);
    response->region_id = region_id;
    if (message != nullptr) {
        std::snprintf(response->message, sizeof(response->message), "%s", message);
    }
}

}  // namespace

int L3L2OrchCommClient::attach(void *control_block, size_t control_block_size) {
    if (control_block == nullptr || control_block_size < sizeof(L3L2OrchCommControlBlock)) {
        return -1;
    }
    control_ = static_cast<L3L2OrchCommControlBlock *>(control_block);
    return 0;
}

int L3L2OrchCommClient::submit(
    const L3L2OrchCommRequest &request, L3L2OrchCommResponse *response, uint64_t timeout_ns
) {
    if (control_ == nullptr || response == nullptr) {
        return -1;
    }

    std::lock_guard<std::mutex> lk(mu_);
    const auto deadline = make_deadline(timeout_ns);

    while (load_state(control_) != L3L2OrchCommControlState::IDLE) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::nanoseconds(kPollSleepNs));
    }

    control_->request = request;
    control_->response = L3L2OrchCommResponse{};
    store_state(control_, L3L2OrchCommControlState::READY);

    while (load_state(control_) != L3L2OrchCommControlState::DONE) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::nanoseconds(kPollSleepNs));
    }

    *response = control_->response;
    store_state(control_, L3L2OrchCommControlState::IDLE);
    return 0;
}

L3L2OrchCommService::~L3L2OrchCommService() { stop(); }

int L3L2OrchCommService::start(L3L2OrchCommBackend *backend, void *control_block, size_t control_block_size) {
    if (started()) {
        return 0;
    }
    if (backend == nullptr || control_block == nullptr || control_block_size < sizeof(L3L2OrchCommControlBlock)) {
        return -1;
    }
    backend_ = backend;
    control_ = static_cast<L3L2OrchCommControlBlock *>(control_block);
    store_state(control_, L3L2OrchCommControlState::IDLE);
    stop_.store(false, std::memory_order_release);
    thread_ = backend_->l3_l2_create_service_thread([this]() {
        loop();
    });
    started_.store(true, std::memory_order_release);
    return 0;
}

int L3L2OrchCommService::stop() {
    if (!started_.exchange(false, std::memory_order_acq_rel)) {
        return 0;
    }
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
    release_all_regions();
    return 0;
}

void L3L2OrchCommService::loop() {
    while (!stop_.load(std::memory_order_acquire)) {
        if (load_state(control_) != L3L2OrchCommControlState::READY) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(kPollSleepNs));
            continue;
        }
        store_state(control_, L3L2OrchCommControlState::RUNNING);
        L3L2OrchCommRequest request = control_->request;
        L3L2OrchCommResponse response{};
        execute_request(request, &response);
        control_->response = response;
        store_state(control_, L3L2OrchCommControlState::DONE);
    }
}

void L3L2OrchCommService::execute_request(const L3L2OrchCommRequest &request, L3L2OrchCommResponse *response) {
    if (request.reserved0 != 0) {
        set_response(response, -1, ServiceError::BAD_REQUEST, request.region_id, "reserved0 must be zero");
        return;
    }
    switch (static_cast<L3L2OrchCommCmd>(request.cmd)) {
    case L3L2OrchCommCmd::ALLOC_REGION:
        alloc_region(request, response);
        return;
    case L3L2OrchCommCmd::FREE_REGION:
        free_region(request.region_id, response);
        return;
    case L3L2OrchCommCmd::PAYLOAD_WRITE:
        payload_write(request, response);
        return;
    case L3L2OrchCommCmd::PAYLOAD_READ:
        payload_read(request, response);
        return;
    case L3L2OrchCommCmd::SIGNAL_NOTIFY:
        signal_notify(request, response);
        return;
    case L3L2OrchCommCmd::SIGNAL_WAIT:
        signal_wait(request, response);
        return;
    case L3L2OrchCommCmd::SIGNAL_TEST:
        signal_test(request, response);
        return;
    }
    set_response(response, -1, ServiceError::UNKNOWN_COMMAND, request.region_id, "unknown L3-L2 orch comm command");
}

void L3L2OrchCommService::alloc_region(const L3L2OrchCommRequest &request, L3L2OrchCommResponse *response) {
    if (request.payload_bytes == 0 || request.counter_bytes == 0 ||
        (request.counter_bytes % L3L2_ORCH_COMM_COUNTER_BYTES) != 0) {
        set_response(
            response, -1, ServiceError::BAD_REQUEST, 0, "ALLOC_REGION payload and counter sizes must be valid"
        );
        return;
    }

    Region region{};
    region.region_id = next_region_id_++;
    region.payload_bytes = request.payload_bytes;
    region.counter_bytes = request.counter_bytes;
    region.payload_dev = backend_->l3_l2_allocate_region_bytes(request.payload_bytes);
    region.counter_dev = backend_->l3_l2_allocate_region_bytes(request.counter_bytes);

    std::vector<uint8_t> zeros(static_cast<size_t>(request.counter_bytes), 0);
    bool ok = region.payload_dev != nullptr && region.counter_dev != nullptr &&
              backend_->l3_l2_copy_to_device(region.counter_dev, zeros.data(), request.counter_bytes) == 0;
    if (!ok) {
        release_region(region);
        set_response(response, -1, ServiceError::ALLOC_FAILED, 0, "ALLOC_REGION failed");
        return;
    }

    response->desc = desc_for_region(region);
    if (l3_l2_orch_comm_validate_desc(response->desc) != L3L2OrchCommValidationError::OK) {
        release_region(region);
        set_response(response, -1, ServiceError::ALLOC_FAILED, 0, "ALLOC_REGION produced invalid descriptor");
        return;
    }

    {
        std::lock_guard<std::mutex> lk(regions_mu_);
        regions_.emplace(region.region_id, region);
    }
    set_response(response, 0, ServiceError::OK, region.region_id, "");
}

void L3L2OrchCommService::free_region(uint64_t region_id, L3L2OrchCommResponse *response) {
    Region region{};
    bool found = false;
    {
        std::lock_guard<std::mutex> lk(regions_mu_);
        auto it = regions_.find(region_id);
        if (it != regions_.end()) {
            region = it->second;
            regions_.erase(it);
            found = true;
        }
    }
    if (found) {
        release_region(region);
    }
    set_response(response, 0, ServiceError::OK, region_id, "");
}

void L3L2OrchCommService::payload_write(const L3L2OrchCommRequest &request, L3L2OrchCommResponse *response) {
    Region *region = find_live_region(request.region_id, response);
    if (region == nullptr) {
        return;
    }
    L3L2OrchCommValidationError bounds =
        l3_l2_orch_comm_validate_payload_bounds(request.payload_offset, request.payload_bytes, region->payload_bytes);
    if (request.host_ptr == 0 || bounds != L3L2OrchCommValidationError::OK) {
        set_response(response, -1, ServiceError::BAD_REQUEST, request.region_id, "invalid PAYLOAD_WRITE request");
        return;
    }
    char *dst = static_cast<char *>(region->payload_dev) + request.payload_offset;
    if (backend_->l3_l2_copy_to_device(dst, reinterpret_cast<const void *>(request.host_ptr), request.payload_bytes) !=
        0) {
        region->poisoned = true;
        set_response(response, -1, ServiceError::COPY_FAILED, request.region_id, "PAYLOAD_WRITE copy failed");
        return;
    }
    set_response(response, 0, ServiceError::OK, request.region_id, "");
}

void L3L2OrchCommService::payload_read(const L3L2OrchCommRequest &request, L3L2OrchCommResponse *response) {
    Region *region = find_live_region(request.region_id, response);
    if (region == nullptr) {
        return;
    }
    L3L2OrchCommValidationError bounds =
        l3_l2_orch_comm_validate_payload_bounds(request.payload_offset, request.payload_bytes, region->payload_bytes);
    if (request.host_ptr == 0 || bounds != L3L2OrchCommValidationError::OK) {
        set_response(response, -1, ServiceError::BAD_REQUEST, request.region_id, "invalid PAYLOAD_READ request");
        return;
    }
    const char *src = static_cast<const char *>(region->payload_dev) + request.payload_offset;
    if (backend_->l3_l2_copy_from_device(reinterpret_cast<void *>(request.host_ptr), src, request.payload_bytes) != 0) {
        region->poisoned = true;
        set_response(response, -1, ServiceError::COPY_FAILED, request.region_id, "PAYLOAD_READ copy failed");
        return;
    }
    set_response(response, 0, ServiceError::OK, request.region_id, "");
}

void L3L2OrchCommService::signal_notify(const L3L2OrchCommRequest &request, L3L2OrchCommResponse *response) {
    Region *region = find_live_region(request.region_id, response);
    if (region == nullptr) {
        return;
    }
    L3L2OrchNotifyOp op = static_cast<L3L2OrchNotifyOp>(request.op);
    if (!l3_l2_orch_comm_valid_notify_op(op)) {
        set_response(response, -1, ServiceError::BAD_REQUEST, request.region_id, "invalid SIGNAL_NOTIFY request");
        return;
    }
    void *counter = counter_ptr(*region, request.counter_addr, response);
    if (counter == nullptr) {
        return;
    }

    int32_t next_value = request.counter_operand;
    if (op == L3L2OrchNotifyOp::Add) {
        int32_t observed = 0;
        if (backend_->l3_l2_copy_from_device(&observed, counter, sizeof(observed)) != 0) {
            region->poisoned = true;
            set_response(response, -1, ServiceError::COPY_FAILED, request.region_id, "SIGNAL_NOTIFY load failed");
            return;
        }
        next_value = observed + request.counter_operand;
    }

    std::atomic_thread_fence(std::memory_order_release);
    if (backend_->l3_l2_copy_to_device(counter, &next_value, sizeof(next_value)) != 0) {
        region->poisoned = true;
        set_response(response, -1, ServiceError::COPY_FAILED, request.region_id, "SIGNAL_NOTIFY store failed");
        return;
    }
    set_response(response, 0, ServiceError::OK, request.region_id, "");
}

void L3L2OrchCommService::signal_test(const L3L2OrchCommRequest &request, L3L2OrchCommResponse *response) {
    Region *region = find_live_region(request.region_id, response);
    if (region == nullptr) {
        return;
    }
    L3L2OrchWaitCmp cmp = static_cast<L3L2OrchWaitCmp>(request.op);
    if (!l3_l2_orch_comm_valid_wait_cmp(cmp)) {
        set_response(response, -1, ServiceError::BAD_REQUEST, request.region_id, "invalid SIGNAL_TEST request");
        return;
    }
    void *counter = counter_ptr(*region, request.counter_addr, response);
    if (counter == nullptr) {
        return;
    }

    int32_t observed = 0;
    if (backend_->l3_l2_copy_from_device(&observed, counter, sizeof(observed)) != 0) {
        region->poisoned = true;
        response->observed_counter = observed;
        set_response(response, -1, ServiceError::COPY_FAILED, request.region_id, "SIGNAL_TEST load failed");
        return;
    }

    response->observed_counter = observed;
    response->matched = l3_l2_orch_comm_compare_counter(observed, request.counter_operand, cmp) ? 1u : 0u;
    if (response->matched != 0) {
        std::atomic_thread_fence(std::memory_order_acquire);
    }
    set_response(response, 0, ServiceError::OK, request.region_id, "");
}

void L3L2OrchCommService::signal_wait(const L3L2OrchCommRequest &request, L3L2OrchCommResponse *response) {
    Region *region = find_live_region(request.region_id, response);
    if (region == nullptr) {
        return;
    }
    L3L2OrchWaitCmp cmp = static_cast<L3L2OrchWaitCmp>(request.op);
    if (!l3_l2_orch_comm_valid_wait_cmp(cmp)) {
        set_response(response, -1, ServiceError::BAD_REQUEST, request.region_id, "invalid SIGNAL_WAIT request");
        return;
    }
    void *counter = counter_ptr(*region, request.counter_addr, response);
    if (counter == nullptr) {
        return;
    }

    const auto deadline = make_deadline(request.timeout_ns);
    int32_t observed = 0;
    while (true) {
        if (backend_->l3_l2_copy_from_device(&observed, counter, sizeof(observed)) != 0) {
            region->poisoned = true;
            response->observed_counter = observed;
            set_response(response, -1, ServiceError::COPY_FAILED, request.region_id, "SIGNAL_WAIT load failed");
            return;
        }
        response->observed_counter = observed;
        response->matched = l3_l2_orch_comm_compare_counter(observed, request.counter_operand, cmp) ? 1u : 0u;
        if (response->matched != 0) {
            std::atomic_thread_fence(std::memory_order_acquire);
            set_response(response, 0, ServiceError::OK, request.region_id, "");
            return;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            set_response(response, -1, ServiceError::SIGNAL_TIMEOUT, request.region_id, "SIGNAL_WAIT timed out");
            return;
        }
        std::this_thread::sleep_for(std::chrono::nanoseconds(kPollSleepNs));
    }
}

L3L2OrchCommService::Region *L3L2OrchCommService::find_live_region(uint64_t region_id, L3L2OrchCommResponse *response) {
    std::lock_guard<std::mutex> lk(regions_mu_);
    auto it = regions_.find(region_id);
    if (it == regions_.end()) {
        set_response(response, -1, ServiceError::UNKNOWN_REGION, region_id, "unknown region_id");
        return nullptr;
    }
    if (it->second.poisoned) {
        set_response(response, -1, ServiceError::POISONED_REGION, region_id, "region is poisoned");
        return nullptr;
    }
    // The returned raw pointer is used after releasing regions_mu_; callers
    // rely on the single-threaded service loop.
    return &it->second;
}

L3L2OrchRegionDesc L3L2OrchCommService::desc_for_region(const Region &region) const {
    return L3L2OrchRegionDesc{
        l3_l2_orch_comm_magic_version(),
        region.region_id,
        reinterpret_cast<uint64_t>(region.payload_dev),
        region.payload_bytes,
        reinterpret_cast<uint64_t>(region.counter_dev),
        region.counter_bytes,
    };
}

void *L3L2OrchCommService::counter_ptr(
    L3L2OrchCommService::Region &region, uint64_t counter_addr, L3L2OrchCommResponse *response
) {
    L3L2OrchRegionDesc desc = desc_for_region(region);
    L3L2OrchCommValidationError error = l3_l2_orch_comm_validate_counter_addr(desc, counter_addr);
    if (error != L3L2OrchCommValidationError::OK) {
        set_response(response, -1, ServiceError::BAD_REQUEST, region.region_id, "invalid counter address");
        return nullptr;
    }
    uint64_t offset = counter_addr - desc.counter_base;
    return static_cast<char *>(region.counter_dev) + offset;
}

void L3L2OrchCommService::release_region(Region &region) {
    backend_->l3_l2_free_region_bytes(region.payload_dev);
    backend_->l3_l2_free_region_bytes(region.counter_dev);
    region.payload_dev = nullptr;
    region.counter_dev = nullptr;
}

void L3L2OrchCommService::release_all_regions() {
    std::lock_guard<std::mutex> lk(regions_mu_);
    for (auto &kv : regions_) {
        release_region(kv.second);
    }
    regions_.clear();
}

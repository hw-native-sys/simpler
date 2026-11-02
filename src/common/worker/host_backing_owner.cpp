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

#include "host_backing_owner.h"

#include <array>
#include <stdexcept>

#include "../platform_comm/comm.h"

namespace simpler::host {

int query_host_backing_identity(const RuntimeModule &module) {
    auto query = module.required<decltype(&host_backing_process_identity)>("host_backing_process_identity");
    std::array<char, 1024> error{};
    int32_t identity = 0;
    check_host_backing_result(query(&identity, error.data(), error.size()), error.data());
    return identity;
}

HostBackingOwner::HostBackingOwner(const std::string &runtime_path) :
    module_(std::make_shared<RuntimeModule>(runtime_path.c_str())),
    allocate_(module_->required<decltype(allocate_)>("host_backing_allocate")),
    authorize_(module_->required<decltype(authorize_)>("host_backing_authorize")),
    base_(module_->required<decltype(base_)>("host_backing_get_base")),
    release_(module_->required<decltype(release_)>("host_backing_release")) {
    (void)module_->required<decltype(&host_backing_process_identity)>("host_backing_process_identity");
}

void HostBackingOwner::require_mapping_support() const {
    (void)module_->required<decltype(&comm_attach_domain_host)>("comm_attach_domain_host");
    (void)module_->required<decltype(&comm_detach_domain_host)>("comm_detach_domain_host");
}

HostBackingOwner::~HostBackingOwner() {
    // Unknown imported-view lifetime forbids backing cleanup and code unload.
    if (owner_) module_->retain_until_process_exit();
}

std::pair<uint64_t, std::vector<uint8_t>> HostBackingOwner::allocate(int device, uint32_t numa, uint64_t bytes) {
    std::array<char, 1024> error{};
    std::vector<uint8_t> token(COMM_GLOBAL_DOMAIN_HANDLE_BYTES);
    uint64_t size = 0;
    size_t count = 0;
    check_host_backing_result(
        allocate_(&owner_, device, numa, bytes, &size, token.data(), token.size(), &count, error.data(), error.size()),
        error.data()
    );
    if (count > token.size() || !count || size < bytes) throw std::runtime_error("Invalid Host backing export");
    token.resize(count);
    return {size, std::move(token)};
}

void HostBackingOwner::authorize(const std::vector<int32_t> &identities) {
    std::array<char, 1024> error{};
    check_host_backing_result(
        authorize_(owner_, identities.data(), identities.size(), error.data(), error.size()), error.data()
    );
}

uint64_t HostBackingOwner::base() const {
    std::array<char, 1024> error{};
    uint64_t address = 0;
    check_host_backing_result(base_(owner_, &address, error.data(), error.size()), error.data());
    return address;
}

void HostBackingOwner::release() {
    std::array<char, 1024> error{};
    check_host_backing_result(release_(&owner_, error.data(), error.size()), error.data());
}

int HostBackingOwner::process_identity() const { return query_host_backing_identity(*module_); }

}  // namespace simpler::host

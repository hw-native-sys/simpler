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

namespace simpler::host {
HostRegistration::HostRegistration(const std::string &runtime_path) :
    module_(std::make_shared<RuntimeModule>(runtime_path.c_str())),
    register_(module_->required<decltype(register_)>("host_register_memory")),
    base_(module_->required<decltype(base_)>("host_registration_base")),
    release_(module_->required<decltype(release_)>("host_registration_release")) {}

HostRegistration::~HostRegistration() {
    // Only explicit release proves that device users have stopped.
    if (owner_) module_->retain_until_process_exit();
}
void HostRegistration::register_memory(int device, uint64_t host_base, uint64_t bytes) {
    std::array<char, 1024> error{};
    check_host_backing_result(register_(&owner_, device, host_base, bytes, error.data(), error.size()), error.data());
}
uint64_t HostRegistration::base() const {
    std::array<char, 1024> error{};
    uint64_t address = 0;
    check_host_backing_result(base_(owner_, &address, error.data(), error.size()), error.data());
    return address;
}
void HostRegistration::release() {
    std::array<char, 1024> error{};
    check_host_backing_result(release_(&owner_, error.data(), error.size()), error.data());
}
}  // namespace simpler::host

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

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "host/host_backing.h"
#include "runtime_module.h"

namespace simpler::host {
class HostRegistration {
public:
    explicit HostRegistration(const std::string &runtime_path);
    ~HostRegistration();
    HostRegistration(const HostRegistration &) = delete;
    HostRegistration &operator=(const HostRegistration &) = delete;
    void register_memory(int device, uint64_t host_base, uint64_t bytes);
    uint64_t base() const;
    void release();

private:
    std::shared_ptr<RuntimeModule> module_;
    void *owner_ = nullptr;
    decltype(&host_register_memory) register_;
    decltype(&host_registration_base) base_;
    decltype(&host_registration_release) release_;
};
}  // namespace simpler::host

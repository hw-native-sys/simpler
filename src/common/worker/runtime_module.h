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

/** Host runtime code ownership; context and native-run storage share this owner. */
class RuntimeModule {
public:
    explicit RuntimeModule(const char *path);
    ~RuntimeModule();
    RuntimeModule(const RuntimeModule &) = delete;
    RuntimeModule &operator=(const RuntimeModule &) = delete;

    void *handle() const noexcept { return handle_; }

    template <typename Function>
    Function required(const char *name) const {
        return reinterpret_cast<Function>(required_symbol(name));
    }

    template <typename Function>
    Function optional(const char *name) const noexcept {
        return reinterpret_cast<Function>(optional_symbol(name));
    }

    /** Terminal abandonment retains code reachable from unproven device/host state. */
    void retain_until_process_exit() noexcept { unload_allowed_ = false; }

private:
    void *required_symbol(const char *name) const;
    void *optional_symbol(const char *name) const noexcept;
    void *handle_{nullptr};
    bool unload_allowed_{true};
};

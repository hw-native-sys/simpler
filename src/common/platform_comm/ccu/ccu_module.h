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

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "ccu_abi.h"
#include "ccu_registry.h"
#include "task_interface/ccu_function.h"

namespace simpler::ccu {

// This registry accepts the internal resource-builder ABI. Public callable
// modules use CCU_BUILDER_ABI_VERSION and are loaded by the platform session.

class Module {
public:
    explicit Module(const std::vector<uint8_t> &binary) {
        CcuFunction{binary, "validate"}.validate();
        char path[] = "/tmp/simpler-ccu-XXXXXX";
        const int fd = mkstemp(path);
        if (fd < 0) throw std::runtime_error("CCU module temporary file creation failed");
        size_t written = 0;
        while (written < binary.size()) {
            const ssize_t count = write(fd, binary.data() + written, binary.size() - written);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) {
                ::close(fd);
                unlink(path);
                throw std::runtime_error("CCU module temporary file write failed");
            }
            written += static_cast<size_t>(count);
        }
        if (::close(fd) != 0) {
            unlink(path);
            throw std::runtime_error("CCU module temporary file close failed");
        }
        handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        const std::string error = handle_ ? "" : dlerror();
        unlink(path);
        if (!handle_) throw std::runtime_error("CCU module load failed: " + error);
        try {
            auto version = reinterpret_cast<ModuleVersion>(symbol("simpler_ccu_module_abi_version"));
            if (version() != ABI_VERSION) throw std::runtime_error("unsupported CCU module ABI version");
        } catch (...) {
            dlclose(handle_);
            handle_ = nullptr;
            throw;
        }
    }

    ~Module() {
        if (handle_) dlclose(handle_);
    }
    Module(const Module &) = delete;
    Module &operator=(const Module &) = delete;

    Builder builder(const std::string &entry) const {
        if (entry.empty() || entry.find('\0') != std::string::npos) {
            throw std::invalid_argument("invalid CCU builder entry");
        }
        return reinterpret_cast<Builder>(symbol(entry.c_str()));
    }

private:
    void *symbol(const char *name) const {
        dlerror();
        void *result = dlsym(handle_, name);
        const char *error = dlerror();
        if (error || !result) {
            throw std::runtime_error("CCU module missing symbol " + std::string(name));
        }
        return result;
    }

    void *handle_ = nullptr;
};

class ModuleCache {
public:
    std::vector<FunctionDefinition> load_functions(const std::vector<CcuFunction> &functions) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<FunctionDefinition> result;
        result.reserve(functions.size());
        try {
            for (const auto &function : functions) {
                function.validate();
                auto &cached = modules_[function.binary];
                auto module = cached.lock();
                if (!module) {
                    module = std::make_shared<Module>(function.binary);
                    cached = module;
                }
                const Builder builder = module->builder(function.entry);
                result.push_back({std::move(module), builder});
            }
        } catch (...) {
            result.clear();
            prune_expired();
            throw;
        }
        prune_expired();
        return result;
    }

    size_t cached_module_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return modules_.size();
    }

private:
    void prune_expired() {
        for (auto it = modules_.begin(); it != modules_.end();) {
            if (it->second.expired()) it = modules_.erase(it);
            else ++it;
        }
    }
    mutable std::mutex mutex_;
    std::map<std::vector<uint8_t>, std::weak_ptr<Module>> modules_;
};

}  // namespace simpler::ccu

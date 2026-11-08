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

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "ccu_abi.h"

namespace simpler::ccu {

using CallableIdentity = std::array<uint8_t, 32>;

struct FunctionDefinition {
    std::shared_ptr<void> module;
    Builder build;
};

struct Domain {
    DomainIdentity identity;
    uint32_t rank;
    std::vector<Window> windows;
    std::shared_ptr<void> owner;
    void *platform_context = nullptr;
};

class Instance {
public:
    Instance(std::vector<FunctionDefinition> functions, Domain domain) :
        resources_(std::make_shared<Resources>(Resources{std::move(functions), std::move(domain), {}})) {
        const auto &d = resources_->domain;
        if (!d.identity.owner || !d.identity.generation || !d.owner || d.windows.empty() || d.windows.size() > 64 ||
            d.rank >= d.windows.size()) {
            throw std::invalid_argument("invalid CCU domain identity or window table");
        }
        for (const auto &window : d.windows) {
            if (!window.base || !window.size || window.size > UINT64_MAX - window.base) {
                throw std::invalid_argument("invalid CCU domain window credentials or range");
            }
        }
        if (resources_->functions.empty()) throw std::invalid_argument("CCU callable has no functions");
        for (const auto &function : resources_->functions) {
            if (!function.module || !function.build) throw std::invalid_argument("invalid CCU function definition");
        }
    }

    ~Instance() {
        if (!close()) {
            // Unconfirmed hardware ownership outlives the registry and its caller.
            auto &debt = quarantine();
            std::lock_guard<std::mutex> lock(debt.mutex);
            debt.resources.push_back(std::move(resources_));
        }
    }
    Instance(const Instance &) = delete;
    Instance &operator=(const Instance &) = delete;

    void prepare() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (poisoned_ || closed_) throw std::runtime_error("CCU instance cannot be prepared after failure or close");
        if (ready_) return;
        const auto &d = resources_->domain;
        BuildContext context{
            ABI_VERSION,      sizeof(BuildContext), d.identity, d.rank, static_cast<uint32_t>(d.windows.size()),
            d.windows.data(), d.platform_context
        };
        resources_->programs.reserve(resources_->functions.size());
        try {
            for (const auto &function : resources_->functions) {
                resources_->programs.push_back({ABI_VERSION, sizeof(Program), nullptr, 0, nullptr, nullptr, nullptr});
                auto &program = resources_->programs.back();
                const auto result = function.build(&context, &program);
                if (result != 0 || program.abi_version != ABI_VERSION || program.struct_size != sizeof(Program) ||
                    !program.handle || !program.device_program || !program.drain || !program.close || !program.poll) {
                    throw std::runtime_error("CCU builder failed or returned an incompatible program");
                }
            }
            ready_ = true;
        } catch (...) {
            poisoned_ = true;
            close_locked();
            throw;
        }
    }

    bool matches(const Domain &domain) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto &prior = resources_->domain;
        if (!(prior.identity == domain.identity) || prior.rank != domain.rank || prior.owner != domain.owner ||
            prior.platform_context != domain.platform_context || prior.windows.size() != domain.windows.size())
            return false;
        for (size_t i = 0; i < prior.windows.size(); ++i) {
            const auto &a = prior.windows[i];
            const auto &b = domain.windows[i];
            if (a.base != b.base || a.size != b.size || a.credential != b.credential) return false;
        }
        return true;
    }

    uint64_t device_program(uint32_t ccu_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        require_ready();
        if (ccu_id >= resources_->programs.size()) throw std::out_of_range("CCU function ID out of range");
        return resources_->programs[ccu_id].device_program;
    }

    void retain_run() {
        std::lock_guard<std::mutex> lock(mutex_);
        require_ready();
        if (active_runs_) throw std::runtime_error("CCU execution context already belongs to an active run");
        active_runs_ = 1;
        run_completion_proven_ = false;
    }

    // The caller releases only after every Host/device task reader exits.
    void release_run(bool success) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!active_runs_) throw std::runtime_error("CCU run reference underflow");
        if (!run_completion_proven_) throw std::runtime_error("CCU run completion is not proven");
        --active_runs_;
        if (!success) poisoned_ = true;
    }

    PollResult poll_run_completion(bool ordinary_stream_complete) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!active_runs_) throw std::runtime_error("CCU run is not retained");
        if (!ordinary_stream_complete) return PollResult::PENDING;
        if (run_completion_proven_) return PollResult::COMPLETE;
        bool complete = true;
        for (const auto &program : resources_->programs) {
            PollResult result = PollResult::FAILED;
            try {
                result = program.poll(program.handle);
            } catch (...) {}
            if (result == PollResult::PENDING) complete = false;
            else if (result != PollResult::COMPLETE) {
                poisoned_ = true;
                return PollResult::FAILED;
            }
        }
        run_completion_proven_ = complete;
        return complete ? PollResult::COMPLETE : PollResult::PENDING;
    }

    bool drain_run_completion(bool ordinary_stream_complete) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!active_runs_) throw std::runtime_error("CCU run is not retained");
        if (!ordinary_stream_complete) return false;
        if (run_completion_proven_) return true;
        bool confirmed = true;
        for (const auto &program : resources_->programs) {
            try {
                if (program.drain(program.handle) != StopResult::CONFIRMED) confirmed = false;
            } catch (...) {
                confirmed = false;
            }
        }
        run_completion_proven_ = confirmed;
        if (!confirmed) poisoned_ = true;
        return confirmed;
    }
    bool close() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return close_locked();
    }

private:
    friend class InstanceRunLease;
    // Only a lease constructor that has not returned can roll back an admission without a device proof.
    void rollback_run_reference() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        active_runs_ = 0;
        run_completion_proven_ = false;
    }
    struct Resources {
        std::vector<FunctionDefinition> functions;
        Domain domain;
        std::vector<Program> programs;
    };
    struct Quarantine {
        std::mutex mutex;
        std::vector<std::shared_ptr<Resources>> resources;
    };
    static Quarantine &quarantine() {
        static auto *debt = new Quarantine;
        return *debt;
    }
    void require_ready() const {
        if (!ready_ || poisoned_ || closed_) throw std::runtime_error("CCU instance is not executable");
    }
    bool close_locked() noexcept {
        if (closed_) return true;
        poisoned_ = true;
        ready_ = false;
        if (active_runs_) return false;
        bool confirmed = true;
        for (auto it = resources_->programs.rbegin(); it != resources_->programs.rend(); ++it) {
            if (!it->handle) continue;
            try {
                if (it->abi_version != ABI_VERSION || it->struct_size != sizeof(Program) || !it->drain || !it->close ||
                    it->drain(it->handle) != StopResult::CONFIRMED || it->close(it->handle) != StopResult::CONFIRMED) {
                    confirmed = false;
                    continue;
                }
                it->handle = nullptr;
            } catch (...) {
                confirmed = false;
            }
        }
        if (confirmed) {
            closed_ = true;
            resources_->programs.clear();
            resources_->functions.clear();
            resources_->domain.owner.reset();
        }
        return confirmed;
    }

    std::shared_ptr<Resources> resources_;
    mutable std::mutex mutex_;
    bool ready_ = false;
    bool poisoned_ = false;
    bool closed_ = false;
    uint64_t active_runs_ = 0;
    bool run_completion_proven_ = false;
};

class Registry {
public:
    void register_callable(const CallableIdentity &identity, std::vector<FunctionDefinition> functions) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closing_) throw std::runtime_error("CCU registry is closing");
        if (definitions_.count(identity)) throw std::invalid_argument("CCU callable is already registered");
        if (functions.empty()) throw std::invalid_argument("CCU callable has no functions");
        for (const auto &function : functions) {
            if (!function.module || !function.build) throw std::invalid_argument("invalid CCU function definition");
        }
        disabled_callables_.erase(identity);
        definitions_.emplace(identity, std::move(functions));
    }

    // The Host domain owner validates the current generation and retires old generations before publication.
    std::shared_ptr<Instance> prepare(const CallableIdentity &identity, const Domain &domain) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closing_ || disabled_callables_.count(identity) ||
            disabled_domains_.count({domain.identity.owner, domain.identity.id, domain.identity.generation})) {
            throw std::runtime_error("CCU callable or domain is closing");
        }
        const auto definition = definitions_.find(identity);
        if (definition == definitions_.end()) throw std::out_of_range("CCU callable is not registered");
        const Key key{identity, domain.identity.owner, domain.identity.id, domain.identity.generation, domain.rank};
        const auto existing = instances_.find(key);
        if (existing != instances_.end()) {
            if (!existing->second->matches(domain)) throw std::invalid_argument("CCU domain resources changed");
            existing->second->prepare();
            return existing->second;
        }
        auto instance = std::make_shared<Instance>(definition->second, domain);
        instances_.emplace(key, instance);
        try {
            instance->prepare();
        } catch (...) {
            if (instance->close()) instances_.erase(key);
            throw;
        }
        return instance;
    }

    bool close_domain(const DomainIdentity &domain) {
        std::lock_guard<std::mutex> lock(mutex_);
        disabled_domains_.emplace(domain.owner, domain.id, domain.generation);
        return close_matching([&](const Key &key) {
            return std::get<1>(key) == domain.owner && std::get<2>(key) == domain.id &&
                   std::get<3>(key) == domain.generation;
        });
    }

    bool unregister_callable(const CallableIdentity &identity) {
        std::lock_guard<std::mutex> lock(mutex_);
        disabled_callables_.insert(identity);
        const bool confirmed = close_matching([&](const Key &key) {
            return std::get<0>(key) == identity;
        });
        if (confirmed) definitions_.erase(identity);
        return confirmed;
    }

    bool close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closing_ = true;
        const bool confirmed = close_matching([](const Key &) {
            return true;
        });
        if (confirmed) definitions_.clear();
        return confirmed;
    }

private:
    using Key = std::tuple<CallableIdentity, uint64_t, uint64_t, uint64_t, uint32_t>;
    template <typename Predicate>
    bool close_matching(Predicate predicate) {
        bool confirmed = true;
        for (auto it = instances_.begin(); it != instances_.end();) {
            if (!predicate(it->first)) {
                ++it;
                continue;
            }
            if (it->second->close()) it = instances_.erase(it);
            else {
                confirmed = false;
                ++it;
            }
        }
        return confirmed;
    }
    std::mutex mutex_;
    std::map<CallableIdentity, std::vector<FunctionDefinition>> definitions_;
    std::map<Key, std::shared_ptr<Instance>> instances_;
    std::set<CallableIdentity> disabled_callables_;
    std::set<std::tuple<uint64_t, uint64_t, uint64_t>> disabled_domains_;
    bool closing_ = false;
};

}  // namespace simpler::ccu

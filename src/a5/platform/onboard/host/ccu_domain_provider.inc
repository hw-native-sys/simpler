/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <atomic>
#include <mutex>
#include <set>
#include "platform_comm/ccu/ccu_host.h"
#include "platform_comm/ccu/ccu_provider.h"
#include "platform_comm/ccu/ccu_publication.h"
#include "platform_comm/ccu/ccu_shutdown.h"

struct CcuPreparedInstance {
    void *runner{};
    int32_t callable_id{};
    std::shared_ptr<CcuSession> session;
    std::shared_ptr<simpler::ccu::PublicationImage> image;
    uint64_t image_addr{};
    std::atomic<uint32_t> active_runs{0};
    bool closed{false};
};

struct CcuDomainState {
    simpler::ccu::PublicationDomain identity{};
    std::vector<std::shared_ptr<CcuPreparedInstance>> instances;
    std::vector<void *> allocations;
    std::vector<std::shared_ptr<simpler::ccu::PublicationImage>> directories;
    bool failed{false};
};

namespace {
using CcuCallableKey = std::pair<void *, int32_t>;
std::mutex ccu_provider_mutex;
std::map<CcuCallableKey, std::vector<CcuEmbeddedFunction>> ccu_definitions;
std::map<uint64_t, std::weak_ptr<CcuDomainState>> ccu_domains;
// Non-owning reset blockers. ChipWorker retains each handle until comm_destroy
// completes; failed cleanup keeps that existing owner and this blocker intact.
std::map<CommHandle, std::set<void *>> ccu_parent_comms;

void ccu_forget_parent_comm(CommHandle handle) {
    std::lock_guard<std::mutex> lock(ccu_provider_mutex);
    ccu_parent_comms.erase(handle);
}

void ccu_check(CcuStatus status, const char *operation) {
    if (status != CcuStatus::OK)
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(static_cast<uint32_t>(status)));
}
void ccu_acl(aclError status, const char *operation) {
    if (status != ACL_SUCCESS) throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
}

int ccu_file_exchange(const std::string &prefix, uint32_t rank, uint64_t step, uint32_t phase,
                      const void *input, uint32_t bytes, void *output, uint32_t expected, uint32_t timeout_ms) {
    if (!input || !output || bytes != expected || !bytes || rank > 1) return -1;
    const auto stem = prefix + "-" + std::to_string(step) + "-" + std::to_string(phase);
    const auto local = stem + "-" + std::to_string(rank);
    const auto temporary = local + ".tmp-" + std::to_string(getpid());
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file.write(static_cast<const char *>(input), bytes);
        file.close();
        if (!file || std::rename(temporary.c_str(), local.c_str()) != 0) return -1;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (uint32_t peer = 0; peer < 2; ++peer) {
        const auto path = stem + "-" + std::to_string(peer);
        while (true) {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (file && file.tellg() == static_cast<std::streamoff>(bytes)) {
                file.seekg(0);
                if (file.read(static_cast<char *>(output) + peer * bytes, bytes)) break;
            }
            if (std::chrono::steady_clock::now() >= deadline) return -1;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    return 0;
}

uint64_t ccu_upload(CcuDomainState &state, const void *data, size_t bytes) {
    state.allocations.push_back(nullptr);
    ccu_acl(aclrtMalloc(&state.allocations.back(), bytes, ACL_MEM_MALLOC_HUGE_FIRST), "CCU directory allocation");
    ccu_acl(aclrtMemcpy(state.allocations.back(), bytes, data, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "CCU image upload");
    return reinterpret_cast<uint64_t>(state.allocations.back());
}

void ccu_publish_directory(DomainAllocation &allocation) {
    auto &state = *allocation.ccu;
    CommContext context{};
    ccu_acl(aclrtMemcpy(&context, sizeof(context), allocation.device_ctx, sizeof(context), ACL_MEMCPY_DEVICE_TO_HOST),
            "CCU read domain context");
    if (state.instances.empty()) {
        set_comm_workspace(context, CommWorkspaceKind::CCU, 0, 0);
    } else {
        std::vector<simpler::ccu::PublicationImage::UploadedInstance> entries;
        for (const auto &instance : state.instances) {
            if (instance->active_runs.load() != 0) throw std::runtime_error("CCU domain still has run readers");
            entries.push_back({instance->callable_id, instance->image_addr, instance->image, instance});
        }
        auto directory = simpler::ccu::PublicationImage::directory(state.identity, entries);
        directory->open_for_publication();
        const auto address = ccu_upload(state, directory->data(), directory->size());
        set_comm_workspace(context, CommWorkspaceKind::CCU, address, directory->size());
        state.directories.push_back(std::move(directory));
    }
    ccu_acl(aclrtMemcpy(allocation.device_ctx, sizeof(context), &context, sizeof(context), ACL_MEMCPY_HOST_TO_DEVICE),
            "CCU publish domain directory");
}

class CcuSessionRunLease final : public simpler::ccu::RunLease {
public:
    explicit CcuSessionRunLease(std::vector<std::shared_ptr<CcuPreparedInstance>> instances) :
        instances_(std::move(instances)) {
        size_t retained = 0;
        try {
            for (const auto &instance : instances_) {
                ccu_check(instance->session->retain_run(), "CCU retain run");
                ++instance->active_runs;
                ++retained;
            }
        } catch (...) {
            for (size_t i = 0; i < retained; ++i) {
                (void)instances_[i]->session->release_run();
                --instances_[i]->active_runs;
            }
            throw;
        }
    }
    ~CcuSessionRunLease() override {
        if (stopped_) {
            for (const auto &instance : instances_) {
                if (instance->session->release_run() == CcuStatus::OK) --instance->active_runs;
            }
        } else {
            // An unobserved run can still own a request even if its ordinary task has failed.
            static auto *retained = new std::vector<std::shared_ptr<CcuPreparedInstance>>;
            static auto *mutex = new std::mutex;
            std::lock_guard<std::mutex> lock(*mutex);
            retained->insert(retained->end(), instances_.begin(), instances_.end());
        }
    }
    simpler::ccu::ReaderResult poll() noexcept override {
        using simpler::ccu::ReaderState;
        if (stopped_) return {ReaderState::STOPPED, 0};
        try {
            for (const auto &instance : instances_) {
                const auto result = instance->session->poll_idle();
                if (result.state == CcuRequestState::PENDING) return {ReaderState::PENDING, 0};
                if (result.state != CcuRequestState::SUCCEEDED || result.status != CcuStatus::OK)
                    return {ReaderState::UNCONFIRMED, -1};
            }
            stopped_ = true;
            return {ReaderState::STOPPED, 0};
        } catch (...) { return {ReaderState::UNCONFIRMED, -1}; }
    }
    simpler::ccu::ReaderResult drain() noexcept override {
        using simpler::ccu::ReaderState;
        if (stopped_) return {ReaderState::STOPPED, 0};
        try {
            for (const auto &instance : instances_)
                if (instance->session->drain(5000) != CcuStatus::OK) return {ReaderState::UNCONFIRMED, -1};
            stopped_ = true;
            return {ReaderState::STOPPED, 0};
        } catch (...) { return {ReaderState::UNCONFIRMED, -1}; }
    }
    simpler::ccu::ReaderResult cancel_unlaunched() noexcept override {
        stopped_ = true;
        return {simpler::ccu::ReaderState::STOPPED, 0};
    }
private:
    std::vector<std::shared_ptr<CcuPreparedInstance>> instances_;
    bool stopped_{false};
};
}  // namespace

int platform_ccu_register(void *runner, int32_t callable_id, const ChipCallable &callable) {
    if (!callable.ccu_size_) return 0;
    try {
        std::vector<CcuEmbeddedFunction> definitions;
        for (const auto &function : callable.ccu_functions()) definitions.push_back({function.binary, function.entry});
        std::lock_guard<std::mutex> lock(ccu_provider_mutex);
        const auto key = CcuCallableKey{runner, callable_id};
        auto found = ccu_definitions.find(key);
        if (found != ccu_definitions.end()) {
            if (found->second.size() != definitions.size()) return -1;
            for (size_t i = 0; i < definitions.size(); ++i)
                if (found->second[i].binary != definitions[i].binary || found->second[i].entry != definitions[i].entry)
                    return -1;
            return 0;
        }
        ccu_definitions.emplace(key, std::move(definitions));
        return 0;
    } catch (...) { return -1; }
}

int platform_ccu_unregister(void *runner, int32_t callable_id) {
    std::lock_guard<std::mutex> lock(ccu_provider_mutex);
    // Instances keep their module bytes and session independently of registry definitions.
    // Their callable slots must not be recycled while a domain still publishes them.
    for (const auto &[address, weak] : ccu_domains) {
        (void)address;
        if (auto domain = weak.lock()) {
            for (const auto &instance : domain->instances)
                if (instance->runner == runner && instance->callable_id == callable_id) return -1;
        }
    }
    ccu_definitions.erase({runner, callable_id});
    return 0;
}

void platform_ccu_forget_runner(void *runner) {
    std::lock_guard<std::mutex> lock(ccu_provider_mutex);
    // Finalized runner slots are gone. Published instances independently retain
    // their sessions and bytes if an unresolved domain still needs quarantine.
    for (auto it = ccu_definitions.begin(); it != ccu_definitions.end();) {
        if (it->first.first == runner) it = ccu_definitions.erase(it);
        else ++it;
    }
}

bool platform_ccu_has_device_owners(void *runner) {
    std::lock_guard<std::mutex> lock(ccu_provider_mutex);
    for (const auto &[handle, runners] : ccu_parent_comms) {
        (void)handle;
        if (runners.count(runner)) return true;
    }
    for (const auto &[address, weak] : ccu_domains) {
        (void)address;
        if (auto domain = weak.lock()) {
            for (const auto &instance : domain->instances) {
                if (instance->runner == runner && !instance->closed) return true;
            }
        }
    }
    return false;
}

bool platform_ccu_close_runner(void *runner) try {
    std::lock_guard<std::mutex> lock(ccu_provider_mutex);
    std::vector<std::shared_ptr<CcuPreparedInstance>> instances;
    for (const auto &[address, weak] : ccu_domains) {
        (void)address;
        if (auto domain = weak.lock()) {
            for (const auto &instance : domain->instances) {
                if (instance->runner != runner) continue;
                // Finalization permanently revokes submission through this domain,
                // including when closing only a prefix succeeds.
                domain->failed = true;
                instances.push_back(instance);
            }
        }
    }
    bool parent_comms_released = true;
    for (const auto &[handle, runners] : ccu_parent_comms) {
        (void)handle;
        if (runners.count(runner)) parent_comms_released = false;
    }
    return simpler::ccu::close_services_before_device_teardown(
        instances, parent_comms_released, [](CcuPreparedInstance &instance) {
            return !instance.session || instance.session->close(CcuCloseMode::NORMAL) == CcuStatus::OK;
        });
} catch (...) { return false; }

std::shared_ptr<simpler::ccu::RunLease>
platform_ccu_acquire_run(void *runner, int32_t callable_id, const ChipStorageTaskArgs &args) {
    std::lock_guard<std::mutex> lock(ccu_provider_mutex);
    if (!ccu_definitions.count({runner, callable_id})) return {};
    std::vector<std::shared_ptr<CcuPreparedInstance>> instances;
    std::set<uint64_t> seen;
    for (int32_t i = 0; i < args.scalar_count(); ++i) {
        const auto address = args.scalar(i);
        if (!seen.insert(address).second) continue;
        const auto found = ccu_domains.find(address);
        if (found == ccu_domains.end()) continue;
        auto domain = found->second.lock();
        if (!domain || domain->failed) throw std::runtime_error("CCU selected domain is unavailable");
        bool matched = false;
        for (const auto &instance : domain->instances) {
            if (instance->runner == runner && instance->callable_id == callable_id) {
                instances.push_back(instance);
                matched = true;
            }
        }
        if (!matched) throw std::runtime_error("CCU callable/domain was not prepared before submission");
    }
    if (instances.empty()) throw std::runtime_error("CCU run has no prepared domain context");
    return std::make_shared<CcuSessionRunLease>(std::move(instances));
}

extern "C" int comm_prepare_ccu(void *runner, CommHandle handle, int32_t callable_id, uint64_t allocation_id,
                                 uint64_t owner, uint64_t generation, const uint8_t *digest) try {
    if (!runner || !handle || !digest || !owner || !generation) return -1;
    std::lock_guard<std::mutex> lock(ccu_provider_mutex);
    const auto found = handle->domain_allocations.find(allocation_id);
    const auto definitions = ccu_definitions.find({runner, callable_id});
    if (found == handle->domain_allocations.end() || definitions == ccu_definitions.end()) return -1;
    auto allocation = found->second;
    if (!allocation->initialized || !allocation->ccu_requested || allocation->nranks != 2 || allocation->cleanup_failed)
        return -1;
    if (!allocation->ccu) {
        allocation->ccu = std::make_shared<CcuDomainState>();
        allocation->ccu->identity = {owner, allocation_id, generation, static_cast<uint32_t>(allocation->rank), 2};
    }
    auto &state = *allocation->ccu;
    if (state.failed || state.identity.owner != owner || state.identity.generation != generation) return -1;
    // Failed initialization may already own SDK resources. Make those sessions
    // visible to the device shutdown gate before their first SDK operation.
    ccu_domains[reinterpret_cast<uint64_t>(allocation->device_ctx)] = allocation->ccu;
    ccu_parent_comms[handle].insert(runner);
    for (const auto &instance : state.instances) {
        if (instance->active_runs.load()) return -1;
        if (instance->runner == runner && instance->callable_id == callable_id) return 0;
    }
    std::string key;
    static constexpr char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < 32; ++i) { key += hex[digest[i] >> 4]; key += hex[digest[i] & 15]; }
    const auto prefix = handle->rootinfo_path + ".ccu-" + std::to_string(handle->run_token) + "-" +
                        std::to_string(allocation_id) + "-" + std::to_string(generation) + "-" + key;
    const auto rank = static_cast<uint32_t>(allocation->rank);
    auto exchange = [prefix, rank](uint64_t step, uint32_t phase, const void *in, uint32_t bytes, void *out,
                                  uint32_t expected, uint32_t timeout) {
        return ccu_file_exchange(prefix, rank, step, phase, in, bytes, out, expected, timeout);
    };
    auto instance = std::make_shared<CcuPreparedInstance>();
    instance->runner = runner;
    instance->callable_id = callable_id;
    // Every partially created session remains owned by the allocation until confirmed close.
    state.instances.push_back(instance);
    try {
        // The mailbox digest routes a common definition. Verify the hidden slot too:
        // different local registration orders must never publish mismatched identities.
        const std::array<uint64_t, 4> identity{{owner, allocation_id, generation,
                                             static_cast<uint64_t>(callable_id)}};
        std::array<uint64_t, 8> identities{};
        if (exchange(UINT64_MAX, 0, identity.data(), sizeof(identity), identities.data(), sizeof(identity), 5000) != 0 ||
            !std::equal(identities.begin(), identities.begin() + 4, identities.begin() + 4))
            throw std::runtime_error("CCU peer callable/domain identity mismatch");
        HcclRootInfo root{};
        if (rank == 0 && HcclGetRootInfo(&root) != HCCL_SUCCESS) throw std::runtime_error("CCU root-info failed");
        std::array<HcclRootInfo, 2> roots{};
        if (exchange(0, 0, &root, sizeof(root), roots.data(), sizeof(root), 5000) != 0)
            throw std::runtime_error("CCU root-info exchange failed");
        CcuSessionConfig config;
        config.allocation_id = allocation_id;
        config.epoch = generation;
        config.domain_owner = owner;
        config.callable_id = callable_id;
        ccu_acl(aclrtGetDevice(&config.device_id), "CCU current device");
        config.rank = rank;
        config.rank_count = 2;
        config.parent_owner = allocation;
        config.exchange = exchange;
        const auto *root_bytes = reinterpret_cast<const uint8_t *>(&roots[0]);
        config.root_info.assign(root_bytes, root_bytes + sizeof(HcclRootInfo));
        instance->session = std::make_shared<CcuSession>(std::move(config));
        ccu_check(instance->session->initialize(), "CCU initialize");
        ccu_check(instance->session->register_functions(definitions->second), "CCU register embedded functions");
        ccu_check(instance->session->prepare_memory(allocation_id, reinterpret_cast<uint64_t>(allocation->local_buf),
                                                   allocation->requested_bytes), "CCU prepare domain window");
        CcuDeviceContextView view{};
        ccu_check(instance->session->get_device_context(view), "CCU publish session");
        CcuDeviceContext header{};
        ccu_acl(aclrtMemcpy(&header, sizeof(header), reinterpret_cast<void *>(view.addr), sizeof(header),
                           ACL_MEMCPY_DEVICE_TO_HOST), "CCU read session layout");
        std::vector<CcuProgramBinding> bindings(header.binding_count);
        ccu_acl(aclrtMemcpy(bindings.data(), bindings.size() * sizeof(CcuProgramBinding),
                           reinterpret_cast<void *>(view.addr + header.bindings_offset),
                           bindings.size() * sizeof(CcuProgramBinding), ACL_MEMCPY_DEVICE_TO_HOST), "CCU read bindings");
        std::vector<simpler::ccu::PreparedProgramBinding> programs;
        for (const auto &binding : bindings) programs.push_back({view.addr, binding.gate_addr, (uint64_t{1} << 63) | 1});
        instance->image = simpler::ccu::PublicationImage::instance(
            state.identity, callable_id, {{reinterpret_cast<uint64_t>(allocation->local_buf), allocation->requested_bytes, 0}},
            programs, uint64_t{5000000000}, instance->session, view.addr, view.size);
        instance->image->open_for_publication();
        instance->image_addr = ccu_upload(state, instance->image->data(), instance->image->size());
        ccu_publish_directory(*allocation);
        return 0;
    } catch (...) {
        state.failed = true;
        throw;
    }
} catch (const std::exception &error) {
    LOG_ERROR("CCU domain preparation: %s", error.what());
    return -1;
} catch (...) { return -1; }

static bool ccu_domain_drained(const DomainAllocation &allocation) {
    std::lock_guard<std::mutex> lock(ccu_provider_mutex);
    if (!allocation.ccu) return true;
    for (const auto &instance : allocation.ccu->instances) {
        if (instance->active_runs.load()) return false;
        // Unconfirmed runs keep their active reference. A partially initialized
        // session has no run reader; let close() attempt its bounded SDK cleanup.
    }
    return true;
}

static bool close_ccu_domain(DomainAllocation &allocation) {
    std::lock_guard<std::mutex> lock(ccu_provider_mutex);
    if (!allocation.ccu) return true;
    auto &state = *allocation.ccu;
    if (!simpler::ccu::close_idle_services(state.instances, [](CcuPreparedInstance &instance) {
            return !instance.session || instance.session->close(CcuCloseMode::NORMAL) == CcuStatus::OK;
        })) return false;
    for (void *&address : state.allocations) {
        if (address && aclrtFree(address) != ACL_SUCCESS) return false;
        address = nullptr;
    }
    state.directories.clear();
    state.instances.clear();
    ccu_domains.erase(reinterpret_cast<uint64_t>(allocation.device_ctx));
    allocation.ccu.reset();
    return true;
}

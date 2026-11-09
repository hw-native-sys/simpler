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
#include "ccu_service_program.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <sstream>
#include <thread>

#include "acl/acl.h"
#include "hccl/hccl_ccu_res.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"
#include "hcomm/ccu/ccu_launch.h"
#include "hcomm/ccu/ccu_res.h"
#include "hcomm/ccu/ccu_primitives.hpp"
#include "host/ccu_device.h"
#include "ccu_build_config.h"

#pragma weak HcommCcuInsQueryResDesc
#pragma weak HcommCcuInsResDescQueryNum

namespace {
void Check(int code, const char *operation) {
    if (code != 0) throw CcuSdkError(code, operation);
}
CcuResult Probe(const ChannelHandle *channel) { return AscendC::ccu::NotifyWait(*channel, 0); }
using Clock = std::chrono::steady_clock;

class SessionSdk final : public CcuSessionBackend {
public:
    void Initialize(const CcuSessionConfig &config) override {
        config_ = config;
        if (!ccu_device_available())
            throw CcuBackendError(CcuStatus::UNSUPPORTED, CCU_E_UNAVAIL, "Missing service SDK symbols");
        descriptors_.reserve(2);
        SetDevice();
        if (config.root_info.size() != sizeof(HcclRootInfo)) throw CcuSdkError(CCU_E_PARA, "HCCL root-info size");
        HcclRootInfo root{};
        std::memcpy(&root, config.root_info.data(), sizeof(root));
        HcclCommConfig comm_config{};
        HcclCommConfigInit(&comm_config);
        // The config wire value is 6; HcclOpExpansionMode reports CCU_SCHED as 5.
        comm_config.hcclOpExpansionMode = 6;
        if (!HcclCommInitRootInfoConfig)
            throw CcuBackendError(CcuStatus::UNSUPPORTED, CCU_E_UNAVAIL, "Missing configured communicator API");
        Check(
            HcclCommInitRootInfoConfig(2, &root, config.rank, &comm_config, &comm_),
            "HcclCommInitRootInfoConfig(service CCU_SCHED)"
        );
        Check(aclrtCreateStream(&stream_), "CreateStream(service)");
        Check(aclrtCreateStream(&trigger_stream_), "CreateStream(service stop)");
        Check(HcclThreadAcquireWithStream(comm_, COMM_ENGINE_CCU, stream_, 1, &thread_), "ThreadAcquire(service)");
        CommLink *links{};
        uint32_t count{};
        Check(
            HcclRankGraphGetLinks(comm_, 0, config.rank, 1 - config.rank, &links, &count), "RankGraphGetLinks(service)"
        );
        bool found = false;
        for (uint32_t i = 0; i < count; ++i) {
            if (links[i].linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) continue;
            HcclChannelDesc desc{};
            Check(HcclChannelDescInit(&desc, 1), "ChannelDescInit(service)");
            desc.remoteRank = 1 - config.rank;
            desc.notifyNum = 4;
            desc.channelProtocol = COMM_PROTOCOL_UBC_CTP;
            desc.localEndpoint = links[i].srcEndpointDesc;
            desc.remoteEndpoint = links[i].dstEndpointDesc;
            Check(HcclChannelAcquire(comm_, COMM_ENGINE_CCU, &desc, 1, &channel_), "ChannelAcquire(service)");
            found = true;
            break;
        }
        if (!found) throw CcuBackendError(CcuStatus::UNSUPPORTED, CCU_E_UNAVAIL, "No service UBC_CTP channel");
        uint32_t instances{};
        Check(HcclCommQueryCcuIns(comm_, &instance_, &instances), "CommQueryCcuIns(service)");
        if (instances != 1) throw CcuBackendError(CcuStatus::UNSUPPORTED, CCU_E_UNAVAIL, "Service CCU instance count");
        int selected = -1;
        const void *probe_args[] = {&channel_};
        for (uint32_t die = 0; die < 2; ++die) {
            HcommCcuResDescHandle descriptor{};
            Check(HcommCcuInsResDescCreate(die, &descriptor), "InsResDescCreate");
            descriptors_.push_back(descriptor);
            const auto rc = HcommCcuKernelQueryResReq(reinterpret_cast<const void *>(Probe), probe_args, 1, descriptor);
            Check(HcommCcuInsResDescDestroy(descriptor), "InsResDescDestroy");
            descriptors_.pop_back();
            if (rc == CCU_SUCCESS) {
                if (selected >= 0) throw CcuSdkError(CCU_E_PARA, "Ambiguous service IO die");
                selected = die;
            } else if (rc != CCU_E_PARA && rc != CCU_E_UNAVAIL) Check(rc, "KernelQueryResReq(service die)");
        }
        if (selected < 0) throw CcuBackendError(CcuStatus::UNSUPPORTED, CCU_E_UNAVAIL, "No service IO die");
        die_ = selected;
        Check(HcommCcuEventAlloc(instance_, die_, 1, &gate_), "EventAlloc(service)");
        Check(HcommCcuEventGetAddr(gate_, 0, &gate_addr_), "EventGetAddr(service)");
        Dl_info info{};
        if (dladdr(reinterpret_cast<const void *>(ccu_make_session_backend), &info) == 0 || !info.dli_fname)
            throw CcuSdkError(-1, "Cannot locate service stop trigger");
        const auto path = (std::filesystem::path(info.dli_fname).parent_path() / "ccu_trigger.o").string();
        Check(aclrtBinaryLoadFromFile(path.c_str(), nullptr, &binary_), "BinaryLoad(service stop)");
        Check(aclrtBinaryGetFunction(binary_, "simpler_ccu_trigger", &trigger_), "BinaryGetFunction(service stop)");
    }
    uint64_t PrepareMemory(uint64_t addr, uint64_t size) override {
        SetDevice();
        aclrtPtrAttributes attributes{};
        Check(
            aclrtPointerGetAttributes(reinterpret_cast<const void *>(addr), &attributes),
            "PointerGetAttributes(service)"
        );
        if (attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE ||
            attributes.location.id != static_cast<uint32_t>(config_.device_id))
            throw CcuBackendError(CcuStatus::UNSUPPORTED, CCU_E_PARA, "Service memory type/device mismatch");
        uint64_t token{};
        Check(HcommCcuGetMemToken(addr, size, &token), "GetMemToken(service)");
        return token;
    }
    void ReleaseMemory(uint64_t) override {
        // GetMemToken borrows an existing allocation registration, owned by PR-3.
        // No deregistration handle is acquired by this SDK query.
        if (!Idle()) throw CcuSdkError(CCU_E_UNAVAIL, "Memory still referenced by service request");
    }
    bool Idle() override {
        if (!image_) return true;
        const auto record = ReadRecord();
        return record.slot_state == CcuSlotState::FREE;
    }
    bool Drain(uint64_t timeout_ms) override {
        if (!image_) return true;
        SetDevice();
        const auto start = Clock::now();
        do {
            const auto record = ReadRecord();
            if (record.slot_state == CcuSlotState::FREE) return true;
            const bool terminal =
                (record.completed_state == CcuRequestState::SUCCEEDED && record.status == CcuStatus::OK) ||
                (record.completed_state == CcuRequestState::FAILED && record.status != CcuStatus::OK);
            if (record.slot_state == CcuSlotState::RESERVED ||
                (record.slot_state == CcuSlotState::SUBMITTED &&
                 (record.local_state == CcuRequestState::FAILED ||
                  (terminal && record.sequence && record.completed_sequence == record.sequence &&
                   record.completed_epoch == record.session_epoch &&
                   record.completed_program == record.program_index)))) {
                const auto free = CcuSlotState::FREE;
                Check(
                    aclrtMemcpy(
                        static_cast<uint8_t *>(image_) + header_.record_offset + offsetof(CcuRequestRecord, slot_state),
                        sizeof(free), &free, sizeof(free), ACL_MEMCPY_HOST_TO_DEVICE
                    ),
                    "Drain service request"
                );
                return true;
            }
            if (std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count() >=
                static_cast<int64_t>(std::min<uint64_t>(timeout_ms, INT64_MAX)))
                return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (true);
    }
    void PreventRequests() override {
        if (!image_) return;
        SetDevice();
        const uint64_t closed = 1;
        Check(
            aclrtMemcpy(
                static_cast<uint8_t *>(image_) + header_.admission_offset, sizeof(closed), &closed, sizeof(closed),
                ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "Close service admission"
        );
    }
    void Publish(
        const std::vector<CcuRegisteredFunction> &functions, std::vector<CcuImageBlock> image,
        CcuDeviceContextView &view
    ) override {
        SetDevice();
        if (!Idle() || !Stop(config_.timeout_ms))
            throw CcuSdkError(CCU_E_UNAVAIL, "Service still active during publication");
        auto *header = reinterpret_cast<CcuDeviceContext *>(image.data());
        if (header->arguments_size > UINT16_MAX - 128)
            throw CcuSdkError(CCU_E_PARA, "Service argument layout too large");
        if (image_) {
            CcuDeviceContext previous{};
            Check(
                aclrtMemcpy(&previous, sizeof(previous), image_, sizeof(previous), ACL_MEMCPY_DEVICE_TO_HOST),
                "Read previous sequence"
            );
            header->next_sequence = previous.next_sequence;
        }
        const auto bytes = image.size() * sizeof(CcuImageBlock);
        // Every allocation is recorded before later calls can fail.
        const auto image_index = Allocate(bytes);
        const auto mailbox_index = Allocate(128 + header->arguments_size + 128);
        const auto scratch_index = Allocate(128);
        auto *new_image = allocations_.at(image_index);
        auto *new_mailbox = allocations_.at(mailbox_index);
        auto *new_scratch = allocations_.at(scratch_index);
        auto *bindings =
            reinterpret_cast<CcuProgramBinding *>(reinterpret_cast<uint8_t *>(image.data()) + header->bindings_offset);
        for (uint32_t i = 0; i < header->binding_count; ++i)
            bindings[i].gate_addr = gate_addr_;
        Check(aclrtMemcpy(new_image, bytes, image.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "Publish device context");
        const uint64_t context_token = PrepareMemory(reinterpret_cast<uint64_t>(new_image), bytes);
        const uint64_t mailbox_token =
            PrepareMemory(reinterpret_cast<uint64_t>(new_mailbox), 128 + header->arguments_size + 128);
        const uint64_t scratch_token = PrepareMemory(reinterpret_cast<uint64_t>(new_scratch), 128);
        const uint64_t generation = ++publication_;
        const std::vector<uint64_t> manifest_header{
            CCU_DEVICE_ABI_VERSION, config_.epoch, functions.size(), header->arguments_size, generation
        };
        const auto manifest_headers = Exchange(manifest_header, generation, 0);
        if (!std::equal(
                manifest_headers.begin(), manifest_headers.begin() + manifest_header.size(),
                manifest_headers.begin() + manifest_header.size()
            ))
            throw CcuSdkError(CCU_E_PARA, "Different service manifest layouts across ranks");
        std::vector<uint64_t> manifest(4 + functions.size() * 4);
        manifest[0] = config_.epoch;
        manifest[1] = functions.size();
        manifest[2] = header->arguments_size;
        manifest[3] = generation;
        for (size_t i = 0; i < functions.size(); ++i)
            std::memcpy(manifest.data() + 4 + 4 * i, functions[i].spec.program_key.bytes, 32);
        auto peers = Exchange(manifest, generation, 1);
        if (!std::equal(peers.begin(), peers.begin() + manifest.size(), peers.begin() + manifest.size()))
            throw CcuSdkError(CCU_E_PARA, "Different CCU definitions across ranks");
        const std::vector<uint64_t> local{config_.rank, reinterpret_cast<uint64_t>(new_mailbox), mailbox_token};
        const auto remote = Exchange(local, generation, 2);
        if (remote[0] != 0 || remote[3] != 1 || remote[1] == 0 || remote[4] == 0)
            throw CcuSdkError(CCU_E_PARA, "Invalid peer service mailbox");
        const uint32_t peer = 1 - config_.rank;
        std::vector<CcuProgramKey> keys;
        for (const auto &function : functions)
            keys.push_back(function.spec.program_key);
        const bool same_program =
            keys.size() == program_keys_.size() &&
            std::equal(keys.begin(), keys.end(), program_keys_.begin(), [](const auto &a, const auto &b) {
                return std::memcmp(a.bytes, b.bytes, sizeof(a.bytes)) == 0;
            });
        if (!program_ || !same_program) {
            const CcuServiceBuildArgs build{channel_,      gate_,     config_.rank, header->arguments_size,
                                            config_.epoch, &functions};
            const void *arguments[] = {&build};
            CcuKernelHandle program{};
            std::ostringstream name;
            name << "ccu_svc_" << std::hex << reinterpret_cast<uint64_t>(comm_) << '_' << generation;
            const auto program_name = name.str();
            Check(HcommCcuKernelRegisterStart(instance_), "RegisterStart(service)");
            try {
                const auto registration = HcommCcuKernelRegister(
                    instance_, die_, program_name.c_str(), reinterpret_cast<const void *>(CcuServiceProgram), arguments,
                    1, &program
                );
                if (registration == CCU_E_UNAVAIL) DiagnoseResources(arguments);
                Check(registration, "Register(service)");
            } catch (...) {
                // A successful Start owns one End, including an aborted registration round.
                (void)HcommCcuKernelRegisterEnd(instance_);
                throw;
            }
            Check(HcommCcuKernelRegisterEnd(instance_), "RegisterEnd(service)");
            uint32_t task_count{};
            Check(HcommCcuGetTaskArgsNum(program, &task_count), "TaskArgsNum(service)");
            if (task_count != 9) throw CcuSdkError(CCU_E_PARA, "Service task argument ABI");
            program_ = program;
            program_keys_.swap(keys);
        }
        image_ = new_image;
        header_ = *header;
        const std::array<uint64_t, 9> task{
            reinterpret_cast<uint64_t>(new_image) + header->record_offset,
            reinterpret_cast<uint64_t>(new_image) + header->arguments_offset,
            context_token,
            reinterpret_cast<uint64_t>(new_mailbox),
            mailbox_token,
            remote[peer * 3 + 1],
            remote[peer * 3 + 2],
            reinterpret_cast<uint64_t>(new_scratch),
            scratch_token
        };
        launched_ = true;
        stopped_ = false;
        stop_submitted_ = false;
        Check(HcommCcuKernelLaunch(thread_, program_, task.data(), task.size()), "Launch(persistent service)");
        const std::vector<uint64_t> ready{config_.rank, generation, 1};
        const auto armed = Exchange(ready, generation, 3);
        if (armed[0] != 0 || armed[3] != 1 || armed[1] != generation || armed[4] != generation || armed[2] != 1 ||
            armed[5] != 1)
            throw CcuSdkError(CCU_E_PARA, "Service startup not confirmed by both ranks");
        for (size_t i = 0; i < image_index; ++i)
            if (allocations_[i]) {
                Check(aclrtFree(allocations_[i]), "Free retired service allocation");
                allocations_[i] = nullptr;
            }
        view = {reinterpret_cast<uint64_t>(new_image), bytes, CCU_DEVICE_ABI_VERSION, 0};
    }
    bool Stop(uint64_t timeout) override {
        if (!launched_ || stopped_) return true;
        SetDevice();
        const auto now = Clock::now();
        const auto maximum =
            std::chrono::duration_cast<std::chrono::milliseconds>(Clock::time_point::max() - now).count();
        const auto deadline = now + std::chrono::milliseconds(std::min<uint64_t>(timeout, maximum));
        if (!stop_submitted_) {
            for (;;) {
                const auto record = ReadRecord();
                const bool terminal =
                    record.slot_state == CcuSlotState::SUBMITTED &&
                    ((record.local_state == CcuRequestState::FAILED) ||
                     (record.completed_sequence == record.sequence && record.completed_epoch == record.session_epoch &&
                      record.completed_program == record.program_index &&
                      ((record.completed_state == CcuRequestState::SUCCEEDED && record.status == CcuStatus::OK) ||
                       (record.completed_state == CcuRequestState::FAILED && record.status != CcuStatus::OK))));
                if (record.slot_state != CcuSlotState::SUBMITTED || terminal) break;
                if (Clock::now() >= deadline) return false;
                std::this_thread::sleep_until(std::min(deadline, Clock::now() + std::chrono::milliseconds(1)));
            }
            const auto state = CcuSessionState::QUIESCING;
            Check(
                aclrtMemcpy(
                    static_cast<uint8_t *>(image_) + offsetof(CcuDeviceContext, state), sizeof(state), &state,
                    sizeof(state), ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "Quiesce device context"
            );
            const uint64_t stop = UINT64_MAX;
            auto *command =
                static_cast<uint8_t *>(image_) + header_.record_offset + offsetof(CcuRequestRecord, program_index);
            Check(
                aclrtMemcpy(command, sizeof(stop), &stop, sizeof(stop), ACL_MEMCPY_HOST_TO_DEVICE),
                "Write service stop command"
            );
            std::array<uint64_t, 2> task{gate_addr_, 1};
            aclrtLaunchKernelAttr engine{};
            engine.id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
            engine.value.engineType = ACL_RT_ENGINE_TYPE_AIV;
            aclrtLaunchKernelCfg launch_config{&engine, 1};
            stop_submitted_ = true;
            Check(
                aclrtLaunchKernelWithHostArgs(
                    trigger_, 1, trigger_stream_, &launch_config, task.data(), sizeof(task), nullptr, 0
                ),
                "Submit service stop trigger"
            );
        }
        do {
            aclrtStreamStatus service{}, trigger{};
            Check(aclrtStreamQuery(stream_, &service), "Query service stop");
            Check(aclrtStreamQuery(trigger_stream_, &trigger), "Query stop trigger");
            if (service == ACL_STREAM_STATUS_COMPLETE && trigger == ACL_STREAM_STATUS_COMPLETE) {
                stopped_ = true;
                stop_submitted_ = false;
                return true;
            }
            std::this_thread::yield();
        } while (Clock::now() < deadline);
        return false;
    }
    void Close() override {
        if (launched_ && !stopped_) throw CcuSdkError(CCU_E_UNAVAIL, "Service has no drain proof");
        if (!comm_ && !stream_ && !trigger_stream_ && !binary_ && allocations_.empty() && descriptors_.empty()) {
            config_.exchange = {};
            config_.parent_owner.reset();
            return;
        }
        SetDevice();
        while (!descriptors_.empty()) {
            Check(HcommCcuInsResDescDestroy(descriptors_.back()), "Destroy retained descriptor");
            descriptors_.pop_back();
        }
        if (comm_) {
            Check(HcclCommDestroy(comm_), "Destroy service communicator");
            comm_ = nullptr;
        }
        if (stream_) {
            Check(aclrtDestroyStream(stream_), "Destroy service stream");
            stream_ = nullptr;
        }
        if (trigger_stream_) {
            Check(aclrtDestroyStream(trigger_stream_), "Destroy stop stream");
            trigger_stream_ = nullptr;
        }
        for (auto &allocation : allocations_)
            if (allocation) {
                Check(aclrtFree(allocation), "Free service allocation");
                if (allocation == image_) image_ = nullptr;
                allocation = nullptr;
            }
        image_ = nullptr;
        if (binary_) {
            Check(aclrtBinaryUnLoad(binary_), "Unload service stop binary");
            binary_ = nullptr;
        }
        config_.exchange = {};
        config_.parent_owner.reset();
    }

private:
    void DiagnoseResources(const void **arguments) noexcept {
        if (!HcommCcuInsQueryResDesc || !HcommCcuInsResDescQueryNum) return;
        const auto retained = descriptors_.size();
        try {
            descriptors_.reserve(retained + 2);
            HcommCcuResDescHandle required{}, allocated{};
            Check(HcommCcuInsResDescCreate(die_, &required), "Create required resource descriptor");
            descriptors_.push_back(required);
            Check(HcommCcuInsResDescCreate(die_, &allocated), "Create allocated resource descriptor");
            descriptors_.push_back(allocated);
            Check(
                HcommCcuKernelQueryResReq(reinterpret_cast<const void *>(CcuServiceProgram), arguments, 1, required),
                "Query service resource requirements"
            );
            Check(HcommCcuInsQueryResDesc(instance_, allocated), "Query allocated service resources");
            static constexpr const char *names[]{"LOOP",  "CCU_BUF",    "VARIABLE",    "ADDRESS",
                                                 "EVENT", "CCU_THREAD", "INSTRUCTION", "CASC_CNT"};
            for (uint32_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
                uint32_t need{}, total{};
                const auto type = static_cast<HcommCcuResType>(i);
                const auto need_rc = HcommCcuInsResDescQueryNum(required, type, &need);
                const auto total_rc = HcommCcuInsResDescQueryNum(allocated, type, &total);
                if (need_rc == CCU_SUCCESS && total_rc == CCU_SUCCESS)
                    std::fprintf(
                        stderr, "CCU resources die=%u type=%s required=%u allocated=%u\n", static_cast<unsigned>(die_),
                        names[i], need, total
                    );
                else
                    std::fprintf(
                        stderr, "CCU resources die=%u type=%s query_errors=%d,%d\n", static_cast<unsigned>(die_),
                        names[i], static_cast<int>(need_rc), static_cast<int>(total_rc)
                    );
            }
        } catch (const CcuSdkError &error) {
            std::fprintf(stderr, "CCU resource diagnostic failed: %d\n", error.code);
        } catch (...) {
            std::fprintf(stderr, "CCU resource diagnostic failed\n");
        }
        // Failed descriptor destruction stays owned for Close() to retry.
        while (descriptors_.size() > retained) {
            try {
                const auto result = HcommCcuInsResDescDestroy(descriptors_.back());
                if (result != CCU_SUCCESS) {
                    std::fprintf(stderr, "CCU resource diagnostic cleanup failed: %d\n", static_cast<int>(result));
                    break;
                }
                descriptors_.pop_back();
            } catch (...) {
                std::fprintf(stderr, "CCU resource diagnostic cleanup failed\n");
                break;
            }
        }
    }
    void SetDevice() { Check(aclrtSetDevice(config_.device_id), "SetDevice(service)"); }
    size_t Allocate(uint64_t size) {
        allocations_.push_back(nullptr);
        aclrtMallocAttrValue value{};
        value.moduleId = 3;
        aclrtMallocAttribute attr{ACL_RT_MEM_ATTR_MODULE_ID, value};
        aclrtMallocConfig cfg{&attr, 1};
        Check(aclrtMallocWithCfg(&allocations_.back(), size, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg), "Malloc(service)");
        return allocations_.size() - 1;
    }
    CcuRequestRecord ReadRecord() {
        SetDevice();
        CcuRequestRecord record{};
        Check(
            aclrtMemcpy(
                &record, sizeof(record), static_cast<uint8_t *>(image_) + header_.record_offset, sizeof(record),
                ACL_MEMCPY_DEVICE_TO_HOST
            ),
            "Read service request"
        );
        return record;
    }
    std::vector<uint64_t> Exchange(const std::vector<uint64_t> &local, uint64_t id, uint32_t phase) {
        if (local.size() > UINT32_MAX / sizeof(uint64_t)) throw CcuSdkError(CCU_E_PARA, "Manifest too large");
        std::vector<uint64_t> ranks(local.size() * 2);
        Check(
            config_.exchange(
                id, phase, local.data(), local.size() * 8, ranks.data(), local.size() * 8, config_.timeout_ms
            ),
            "Bootstrap service exchange"
        );
        if (!std::equal(local.begin(), local.end(), ranks.begin() + config_.rank * local.size()))
            throw CcuSdkError(CCU_E_PARA, "Service exchange local echo mismatch");
        return ranks;
    }
    CcuSessionConfig config_;
    HcclComm comm_{};
    aclrtStream stream_{}, trigger_stream_{};
    ThreadHandle thread_{};
    ChannelHandle channel_{};
    CcuInsHandle instance_{};
    CcuKernelHandle program_{};
    std::vector<CcuProgramKey> program_keys_;
    CcuEventHandle gate_{};
    uint64_t gate_addr_{}, publication_{};
    uint8_t die_{};
    std::vector<HcommCcuResDescHandle> descriptors_;
    std::vector<void *> allocations_;
    void *image_{};
    CcuDeviceContext header_{};
    aclrtBinHandle binary_{};
    aclrtFuncHandle trigger_{};
    bool launched_{}, stopped_{}, stop_submitted_{};
};
}  // namespace

std::unique_ptr<CcuSessionBackend> ccu_make_session_backend() { return std::make_unique<SessionSdk>(); }
const char *ccu_session_sdk_build_id() { return CCU_BUILD_SDK_ID; }

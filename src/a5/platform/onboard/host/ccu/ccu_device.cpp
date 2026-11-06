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
#include "host/ccu_device.h"

#include <cstring>
#include <filesystem>
#include <dlfcn.h>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_ccu_res.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"
#include "hcomm/ccu/ccu_launch.h"
#include "hcomm/ccu/ccu_primitives.hpp"
#include "hcomm/ccu/ccu_res.h"

namespace {
void Check(int code, const char *operation) {
    if (code != 0) throw CcuSdkError(code, operation);
}

struct BroadcastBuildArgs {
    ChannelHandle channel;
    CcuEventHandle gate;
    uint64_t completed_seq;
    bool is_root;
};

// Registration executes on Host and records the device program.
// 注册回调在 Host 上记录指令；真正搬运和完成写回发生在 CCU。
CcuResult BroadcastProgram(const BroadcastBuildArgs *build) noexcept {
    namespace ccu = AscendC::ccu;
    try {
        ccu::Variable input, output, input_token, output_token, peer, peer_token, bytes, sequence;
        const std::array<ccu::Variable, 8> args{input, output,     input_token, output_token,
                                                peer,  peer_token, bytes,       sequence};
        for (uint32_t i = 0; i < args.size(); ++i)
            Check(ccu::LoadArg(args[i], i), "LoadArg");
        ccu::Event gate(build->gate);
        Check(ccu::EventWait(gate, 1), "EventWait(gate)");
        Check(ccu::NotifyRecord(build->channel, 0, 1), "NotifyRecord(ready)");
        Check(ccu::NotifyWait(build->channel, 0, 1), "NotifyWait(ready)");
        if (build->is_root) {
            ccu::LocalAddr src, dst;
            ccu::RemoteAddr remote;
            src.addr = input;
            src.token = input_token;
            dst.addr = output;
            dst.token = output_token;
            remote.addr = peer;
            remote.token = peer_token;
            ccu::CcuBuffer buffer;
            ccu::Event copied;
            Check(ccu::LocalCopy(buffer, src, bytes, copied), "LocalCopy(read)");
            Check(ccu::EventWait(copied), "EventWait(read)");
            Check(ccu::LocalCopy(dst, buffer, bytes, copied), "LocalCopy(write)");
            Check(ccu::EventWait(copied), "EventWait(write)");
            Check(ccu::Write(build->channel, remote, src, bytes, copied), "Write(peer)");
            Check(ccu::EventWait(copied), "EventWait(peer)");
        }
        Check(ccu::NotifyRecord(build->channel, 0, 2), "NotifyRecord(done)");
        Check(ccu::NotifyWait(build->channel, 0, 2), "NotifyWait(done)");
        // Store emits its own completion wait. Host never writes a success sequence.
        // Store 包含写回完成等待；Host 不写入成功序号。
        Check(ccu::Store(build->completed_seq, sequence), "Store(completed_seq)");
        return CCU_SUCCESS;
    } catch (...) {
        return CCU_E_INTERNAL;
    }
}

CcuResult ProbeDie(const ChannelHandle *channel) { return AscendC::ccu::NotifyWait(*channel, 0); }

uint8_t FindDie(ChannelHandle channel, std::vector<HcommCcuResDescHandle> &descriptors) {
    int found = -1;
    const void *args[] = {&channel};
    for (uint32_t die = 0; die < 2; ++die) {
        HcommCcuResDescHandle desc{};
        Check(HcommCcuInsResDescCreate(die, &desc), "InsResDescCreate");
        descriptors.push_back(desc);
        const auto rc = HcommCcuKernelQueryResReq(reinterpret_cast<const void *>(ProbeDie), args, 1, desc);
        Check(HcommCcuInsResDescDestroy(desc), "InsResDescDestroy");
        descriptors.pop_back();
        if (rc == CCU_SUCCESS) {
            if (found >= 0) throw CcuSdkError(CCU_E_PARA, "Channel matches multiple IO dies");
            found = die;
        } else if (rc != CCU_E_PARA && rc != CCU_E_UNAVAIL) Check(rc, "KernelQueryResReq");
    }
    if (found < 0) throw CcuSdkError(CCU_E_UNAVAIL, "No IO die for channel");
    return static_cast<uint8_t>(found);
}

std::string DefaultTriggerPath() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void *>(ccu_make_device), &info) == 0 || info.dli_fname == nullptr)
        throw CcuSdkError(-1, "Cannot locate CCU trigger binary");
    return (std::filesystem::path(info.dli_fname).parent_path() / "ccu_trigger.o").string();
}

class A5Device final : public CcuDevice {
public:
    void Open(const CcuCreateOptionsV1 &options, uint64_t context, uint64_t generation) override {
        probe_descriptors_.reserve(2);
        device_ = options.device_id;
        SetDevice();
        HcclRootInfo root{};
        if (options.root_info_size != sizeof(root)) throw CcuSdkError(-1, "Invalid HCCL root-info size");
        std::memcpy(&root, options.root_info, sizeof(root));
        const auto path =
            options.trigger_binary_path != nullptr ? std::string(options.trigger_binary_path) : DefaultTriggerPath();
        Check(aclrtBinaryLoadFromFile(path.c_str(), nullptr, &binary_), "BinaryLoadFromFile(ccu_trigger.o)");
        Check(aclrtBinaryGetFunction(binary_, "simpler_ccu_trigger", &trigger_), "BinaryGetFunction");
        Check(HcclCommInitRootInfo(2, &root, options.rank, &comm_), "HcclCommInitRootInfo");
        Check(aclrtCreateStream(&ccu_stream_), "CreateStream(CCU)");
        Check(aclrtCreateStream(&trigger_stream_), "CreateStream(AIV)");
        Check(HcclThreadAcquireWithStream(comm_, COMM_ENGINE_CCU, ccu_stream_, 1, &thread_), "ThreadAcquire");
        CommLink *links = nullptr;
        uint32_t count{};
        Check(HcclRankGraphGetLinks(comm_, 0, options.rank, 1 - options.rank, &links, &count), "RankGraphGetLinks");
        bool acquired = false;
        for (uint32_t i = 0; i < count; ++i) {
            if (links[i].linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) continue;
            HcclChannelDesc desc{};
            Check(HcclChannelDescInit(&desc, 1), "ChannelDescInit");
            desc.remoteRank = 1 - options.rank;
            desc.notifyNum = 4;
            desc.channelProtocol = COMM_PROTOCOL_UBC_CTP;
            desc.localEndpoint = links[i].srcEndpointDesc;
            desc.remoteEndpoint = links[i].dstEndpointDesc;
            Check(HcclChannelAcquire(comm_, COMM_ENGINE_CCU, &desc, 1, &channel_), "ChannelAcquire");
            acquired = true;
            break;
        }
        if (!acquired) throw CcuSdkError(CCU_E_UNAVAIL, "No UBC_CTP link");
        CcuInsHandle instance{};
        uint32_t instances{};
        Check(HcclCommQueryCcuIns(comm_, &instance, &instances), "CommQueryCcuIns");
        if (instances != 1) throw CcuSdkError(CCU_E_UNAVAIL, "Expected one CCU instance");
        const auto die = FindDie(channel_, probe_descriptors_);
        CcuEventHandle gate{};
        Check(HcommCcuEventAlloc(instance, die, 1, &gate), "EventAlloc");
        Check(HcommCcuEventGetAddr(gate, 0, &gate_), "EventGetAddr");
        if (gate_ == 0) throw CcuSdkError(-1, "Empty gate mapping");
        aclrtMallocAttrValue value{};
        value.moduleId = 3;
        aclrtMallocAttribute attr{ACL_RT_MEM_ATTR_MODULE_ID, value};
        aclrtMallocConfig config{&attr, 1};
        Check(
            aclrtMallocWithCfg(&completion_, sizeof(CcuCompletionRecordV1), ACL_MEM_TYPE_HIGH_BAND_WIDTH, &config),
            "MallocWithCfg(completion)"
        );
        if (Completion() % alignof(CcuCompletionRecordV1) != 0) throw CcuSdkError(-1, "Completion alignment");
        CcuCompletionRecordV1 initial{};
        initial.abi_version = CCU_EXECUTION_ABI_V1;
        initial.struct_size = sizeof(initial);
        initial.context_id = context;
        initial.context_generation = generation;
        Check(
            aclrtMemcpy(completion_, sizeof(initial), &initial, sizeof(initial), ACL_MEMCPY_HOST_TO_DEVICE),
            "Memcpy(completion initialization)"
        );
        // Token lookup validates that the completion allocation is CCU-accessible.
        // 完成区使用 CCU 模块分配，并通过 SDK 查询访问 token。
        (void)Token(Completion(), sizeof(initial));
        Check(HcommCcuKernelRegisterStart(instance), "KernelRegisterStart");
        for (uint32_t root_rank = 0; root_rank < 2; ++root_rank) {
            const BroadcastBuildArgs build{
                channel_, gate, Completion() + offsetof(CcuCompletionRecordV1, completed_seq), options.rank == root_rank
            };
            const void *args[] = {&build};
            Check(
                HcommCcuKernelRegister(
                    instance, die, root_rank == 0 ? "SimplerBroadcastRoot0" : "SimplerBroadcastRoot1",
                    reinterpret_cast<const void *>(BroadcastProgram), args, 1, &programs_[root_rank]
                ),
                "KernelRegister(Broadcast)"
            );
        }
        Check(HcommCcuKernelRegisterEnd(instance), "KernelRegisterEnd");
        for (const auto program : programs_) {
            uint32_t args{};
            Check(HcommCcuGetTaskArgsNum(program, &args), "GetTaskArgsNum");
            if (args != 8) throw CcuSdkError(-1, "Broadcast requires eight task arguments");
        }
    }

    uint64_t Token(uint64_t base, uint64_t size) override {
        SetDevice();
        aclrtPtrAttributes attributes{};
        Check(aclrtPointerGetAttributes(reinterpret_cast<const void *>(base), &attributes), "PointerGetAttributes");
        if (attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE ||
            attributes.location.id != static_cast<uint32_t>(device_))
            throw CcuSdkError(CCU_E_PARA, "Window belongs to a different device or memory type");
        uint64_t token{};
        Check(HcommCcuGetMemToken(base, size, &token), "GetMemToken");
        return token;
    }
    uint64_t Program(uint32_t root) const override { return programs_.at(root); }
    void Launch(uint32_t root, const std::array<uint64_t, 8> &args) override {
        SetDevice();
        Check(HcommCcuKernelLaunch(thread_, programs_.at(root), args.data(), args.size()), "KernelLaunch(CCU)");
    }
    void Trigger() override {
        SetDevice();
        const std::array<uint64_t, 2> args{gate_, 1};
        Check(aclrtLaunchKernel(trigger_, 1, args.data(), sizeof(args), trigger_stream_), "LaunchKernel(AIV trigger)");
    }
    CcuDeviceObservation Observe() override {
        SetDevice();
        CcuDeviceObservation observation;
        aclrtStreamStatus trigger_status{}, ccu_status{};
        Check(aclrtStreamQuery(trigger_stream_, &trigger_status), "StreamQuery(trigger)");
        Check(aclrtStreamQuery(ccu_stream_, &ccu_status), "StreamQuery(CCU)");
        observation.trigger_drained = trigger_status == ACL_STREAM_STATUS_COMPLETE;
        observation.ccu_drained = ccu_status == ACL_STREAM_STATUS_COMPLETE;
        Check(
            aclrtMemcpy(
                &observation.record, sizeof(observation.record), completion_, sizeof(observation.record),
                ACL_MEMCPY_DEVICE_TO_HOST
            ),
            "Memcpy(completion observation)"
        );
        return observation;
    }
    uint64_t Gate() const override { return gate_; }
    uint64_t Completion() const override { return reinterpret_cast<uint64_t>(completion_); }
    void Close() override {
        SetDevice();
        while (!probe_descriptors_.empty()) {
            Check(HcommCcuInsResDescDestroy(probe_descriptors_.back()), "InsResDescDestroy(rollback)");
            probe_descriptors_.pop_back();
        }
        // The communicator owns channels, thread, instance, gates and programs.
        // 先销毁引用完成区的 communicator 资源，再释放完成区和二进制。
        if (comm_) {
            Check(HcclCommDestroy(comm_), "CommDestroy");
            comm_ = nullptr;
        }
        if (ccu_stream_) {
            Check(aclrtDestroyStream(ccu_stream_), "DestroyStream(CCU)");
            ccu_stream_ = nullptr;
        }
        if (trigger_stream_) {
            Check(aclrtDestroyStream(trigger_stream_), "DestroyStream(trigger)");
            trigger_stream_ = nullptr;
        }
        if (completion_) {
            Check(aclrtFree(completion_), "Free(completion)");
            completion_ = nullptr;
        }
        if (binary_) {
            Check(aclrtBinaryUnLoad(binary_), "BinaryUnLoad");
            binary_ = nullptr;
        }
    }

private:
    void SetDevice() { Check(aclrtSetDevice(device_), "SetDevice"); }
    int device_{-1};
    HcclComm comm_{};
    aclrtStream ccu_stream_{}, trigger_stream_{};
    ThreadHandle thread_{};
    ChannelHandle channel_{};
    std::array<CcuKernelHandle, 2> programs_{};
    std::vector<HcommCcuResDescHandle> probe_descriptors_;
    uint64_t gate_{};
    void *completion_{};
    aclrtBinHandle binary_{};
    aclrtFuncHandle trigger_{};
};
}  // namespace

std::shared_ptr<CcuDevice> ccu_make_device() { return std::make_shared<A5Device>(); }
bool ccu_device_available() {
    return HcclGetRootInfo && HcclCommInitRootInfo && HcommCcuKernelRegisterStart && HcommCcuKernelRegister &&
           HcommCcuKernelRegisterEnd && HcommCcuKernelLaunch;
}
uint32_t ccu_device_root_info_size() { return sizeof(HcclRootInfo); }
void ccu_device_root_info(int device, void *bytes) {
    Check(aclrtSetDevice(device), "SetDevice(root-info)");
    Check(HcclGetRootInfo(static_cast<HcclRootInfo *>(bytes)), "GetRootInfo");
}

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
#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
#include <cstring>
#include <map>
#include "ccu_service_program.h"
#include "acl/acl.h"
#include "hccl/hccl_ccu_res.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"
#include "hcomm/ccu/ccu_launch.h"
#include "hcomm/ccu/ccu_res.h"

namespace {
// Link-time SDK fault injection exercises the production adapter without a device.
struct State {
    std::map<void *, size_t> allocations;
    std::map<HcommCcuResDescHandle, uint32_t> descriptors;
    HcommCcuResDescHandle allocated_descriptor{};
    uint64_t next_handle{100};
    int registrations{}, launches{}, triggers{}, frees{}, fail_free_at{}, streams{}, communicators{};
    int resource_queries{};
    int record_reads{}, complete_after_reads{};
    int registration_starts{}, registration_ends{};
    bool registration_active{}, fail_registration_start{}, fail_registration_end{}, fail_communicator_destroy{};
    bool complete{}, fail_thread{};
    bool fail_registration{}, fail_resource_query{}, fail_descriptor_destroy{};
} state;
void *Handle() { return reinterpret_cast<void *>(++state.next_handle); }
bool Covered(const void *address, size_t size) {
    const auto addr = reinterpret_cast<uint64_t>(address);
    for (const auto &entry : state.allocations) {
        const auto base = reinterpret_cast<uint64_t>(entry.first);
        if (addr >= base && addr - base <= entry.second && size <= entry.second - (addr - base)) return true;
    }
    return false;
}
CcuSessionConfig Config() {
    CcuSessionConfig config;
    config.allocation_id = 11;
    config.epoch = 29;
    config.device_id = 0;
    config.parent_owner = std::make_shared<int>(9);
    config.root_info.resize(sizeof(HcclRootInfo));
    config.exchange = [](uint64_t, uint32_t phase, const void *local, uint32_t size, void *ranks, uint32_t stride,
                         uint32_t) {
        if (size != stride) return 1;
        std::memcpy(ranks, local, size);
        std::memcpy(static_cast<uint8_t *>(ranks) + stride, local, size);
        if (phase == 2 || phase == 3) reinterpret_cast<uint64_t *>(static_cast<uint8_t *>(ranks) + stride)[0] = 1;
        return 0;
    };
    return config;
}
std::vector<CcuImageBlock> Image() {
    std::vector<CcuImageBlock> image(10);
    auto *header = reinterpret_cast<CcuDeviceContext *>(image.data());
    header->abi_version = CCU_DEVICE_ABI_VERSION;
    header->struct_size = sizeof(*header);
    header->image_size = image.size() * sizeof(CcuImageBlock);
    header->allocation_id = 11;
    header->session_epoch = 29;
    header->next_sequence = 1;
    header->admission_offset = 128;
    header->record_offset = 256;
    header->arguments_offset = 512;
    header->arguments_size = sizeof(CcuPositionalArguments);
    return image;
}
std::vector<CcuRegisteredFunction> Functions() {
    CcuRegisteredFunction f{};
    f.digest.back() = 7;
    return {f};
}
class CcuSessionSdkTest : public testing::Test {
protected:
    void SetUp() override { state = {}; }
    void TearDown() override {
        EXPECT_TRUE(state.allocations.empty());
        EXPECT_EQ(state.streams, 0);
        EXPECT_EQ(state.communicators, 0);
        EXPECT_TRUE(state.descriptors.empty());
        for (const auto &entry : state.allocations)
            std::free(entry.first);
    }
};
}  // namespace

CcuResult CcuServiceProgram(const CcuServiceBuildArgs *) noexcept { return CCU_SUCCESS; }
extern "C" {
int memset_s(void *dst, size_t maximum, int value, size_t count) {
    if (!dst || count > maximum) return 22;
    std::memset(dst, value, count);
    return 0;
}
aclError aclrtSetDevice(int32_t) { return ACL_SUCCESS; }
aclError aclrtCreateStream(aclrtStream *out) {
    *out = Handle();
    ++state.streams;
    return ACL_SUCCESS;
}
aclError aclrtDestroyStream(aclrtStream) {
    --state.streams;
    return ACL_SUCCESS;
}
aclError aclrtStreamQuery(aclrtStream, aclrtStreamStatus *out) {
    *out = state.complete ? ACL_STREAM_STATUS_COMPLETE : ACL_STREAM_STATUS_NOT_READY;
    return ACL_SUCCESS;
}
aclError aclrtMallocWithCfg(void **out, size_t size, aclrtMemMallocPolicy, aclrtMallocConfig *) {
    *out = std::aligned_alloc(128, (size + 127) / 128 * 128);
    if (!*out) return ACL_ERROR_BAD_ALLOC;
    std::memset(*out, 0, size);
    state.allocations.emplace(*out, size);
    return ACL_SUCCESS;
}
aclError aclrtFree(void *ptr) {
    if (++state.frees == state.fail_free_at) return ACL_ERROR_INTERNAL_ERROR;
    if (!state.allocations.erase(ptr)) return ACL_ERROR_INVALID_PARAM;
    std::free(ptr);
    return ACL_SUCCESS;
}
aclError aclrtMemcpy(void *dst, size_t maximum, const void *src, size_t bytes, aclrtMemcpyKind kind) {
    if (bytes > maximum || (kind == ACL_MEMCPY_HOST_TO_DEVICE && !Covered(dst, bytes)) ||
        (kind == ACL_MEMCPY_DEVICE_TO_HOST && !Covered(src, bytes)))
        return ACL_ERROR_INVALID_PARAM;
    if (kind == ACL_MEMCPY_DEVICE_TO_HOST && bytes == sizeof(CcuRequestRecord)) {
        ++state.record_reads;
        if (state.complete_after_reads && state.record_reads >= state.complete_after_reads) {
            auto *record = const_cast<CcuRequestRecord *>(static_cast<const CcuRequestRecord *>(src));
            record->completed_sequence = record->sequence;
            record->completed_epoch = record->session_epoch;
            record->completed_program = record->program_index;
            record->completed_state = CcuRequestState::SUCCEEDED;
            record->status = CcuStatus::OK;
        }
    }
    std::memcpy(dst, src, bytes);
    return ACL_SUCCESS;
}
aclError aclrtPointerGetAttributes(const void *, aclrtPtrAttributes *out) {
    out->location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    out->location.id = 0;
    return ACL_SUCCESS;
}
aclError aclrtBinaryLoadFromFile(const char *, aclrtBinaryLoadOptions *options, aclrtBinHandle *out) {
    EXPECT_EQ(options, nullptr);
    if (options) return ACL_ERROR_INVALID_PARAM;
    *out = Handle();
    return ACL_SUCCESS;
}
aclError aclrtBinaryGetFunction(aclrtBinHandle, const char *, aclrtFuncHandle *out) {
    *out = Handle();
    return ACL_SUCCESS;
}
aclError aclrtBinaryUnLoad(aclrtBinHandle) { return ACL_SUCCESS; }
aclError aclrtLaunchKernel(aclrtFuncHandle, uint32_t, const void *, size_t, aclrtStream) {
    ADD_FAILURE() << "Host trigger arguments require aclrtLaunchKernelWithHostArgs";
    return ACL_ERROR_INVALID_PARAM;
}
aclError aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle, uint32_t blocks, aclrtStream, aclrtLaunchKernelCfg *config, void *host_args, size_t bytes,
    aclrtPlaceHolderInfo *placeholders, size_t placeholder_count
) {
    EXPECT_EQ(blocks, 1U);
    EXPECT_NE(config, nullptr);
    EXPECT_NE(host_args, nullptr);
    EXPECT_EQ(bytes, sizeof(uint64_t) * 2);
    EXPECT_EQ(placeholders, nullptr);
    EXPECT_EQ(placeholder_count, 0U);
    if (!config || !host_args || bytes != sizeof(uint64_t) * 2) return ACL_ERROR_INVALID_PARAM;
    EXPECT_EQ(config->numAttrs, 1U);
    EXPECT_NE(config->attrs, nullptr);
    if (config->numAttrs != 1 || !config->attrs) return ACL_ERROR_INVALID_PARAM;
    EXPECT_EQ(config->attrs[0].id, ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE);
    EXPECT_EQ(config->attrs[0].value.engineType, ACL_RT_ENGINE_TYPE_AIV);
    EXPECT_FALSE(Covered(host_args, bytes));
    std::array<uint64_t, 2> arguments{};
    std::memcpy(arguments.data(), host_args, bytes);
    EXPECT_NE(arguments[0], 0U);
    EXPECT_EQ(arguments[1], 1U);
    ++state.triggers;
    return ACL_SUCCESS;
}
HcclResult
HcclCommInitRootInfoConfig(uint32_t, const HcclRootInfo *, uint32_t, const HcclCommConfig *config, HcclComm *out) {
    EXPECT_NE(config, nullptr);
    if (!config || config->hcclOpExpansionMode != 6) return HCCL_E_PARA;
    *out = Handle();
    ++state.communicators;
    return HCCL_SUCCESS;
}
HcclResult HcclCommDestroy(HcclComm) {
    if (state.fail_communicator_destroy) return HCCL_E_INTERNAL;
    state.registration_active = false;
    --state.communicators;
    return HCCL_SUCCESS;
}
HcclResult HcclThreadAcquireWithStream(HcclComm, CommEngine, aclrtStream, uint32_t, ThreadHandle *out) {
    if (state.fail_thread) return HCCL_E_INTERNAL;
    *out = ++state.next_handle;
    return HCCL_SUCCESS;
}
HcclResult HcclRankGraphGetLinks(HcclComm, uint32_t, uint32_t, uint32_t, CommLink **out, uint32_t *count) {
    static CommLink link{};
    link.linkAttr.linkProtocol = COMM_PROTOCOL_UBC_CTP;
    *out = &link;
    *count = 1;
    return HCCL_SUCCESS;
}
HcclResult HcclChannelAcquire(HcclComm, CommEngine, const HcclChannelDesc *, uint32_t, ChannelHandle *out) {
    *out = ++state.next_handle;
    return HCCL_SUCCESS;
}
HcclResult HcclCommQueryCcuIns(HcclComm, CcuInsHandle *out, uint32_t *count) {
    *out = ++state.next_handle;
    *count = 1;
    return HCCL_SUCCESS;
}
CcuResult HcommCcuInsResDescCreate(uint32_t die, HcommCcuResDescHandle *out) {
    *out = ++state.next_handle;
    state.descriptors[*out] = die;
    return CCU_SUCCESS;
}
CcuResult HcommCcuInsResDescDestroy(HcommCcuResDescHandle descriptor) {
    if (state.fail_descriptor_destroy) return CCU_E_INTERNAL;
    return state.descriptors.erase(descriptor) == 1 ? CCU_SUCCESS : CCU_E_NOT_FOUND;
}
CcuResult
HcommCcuKernelQueryResReq(const void *function, const void **args, uint32_t count, HcommCcuResDescHandle descriptor) {
    if (function == reinterpret_cast<const void *>(CcuServiceProgram)) {
        ++state.resource_queries;
        EXPECT_EQ(count, 1U);
        EXPECT_NE(args, nullptr);
        if (args && count == 1) {
            EXPECT_NE(args[0], nullptr);
        }
        return state.fail_resource_query ? CCU_E_INTERNAL : CCU_SUCCESS;
    }
    return state.descriptors.at(descriptor) == 0 ? CCU_SUCCESS : CCU_E_UNAVAIL;
}
CcuResult HcommCcuInsQueryResDesc(CcuInsHandle, HcommCcuResDescHandle descriptor) {
    state.allocated_descriptor = descriptor;
    return CCU_SUCCESS;
}
CcuResult HcommCcuInsResDescQueryNum(HcommCcuResDescHandle descriptor, HcommCcuResType type, uint32_t *out) {
    EXPECT_EQ(state.descriptors.count(descriptor), 1U);
    *out = type == HCOMM_CCU_RES_TYPE_VARIABLE ? (descriptor == state.allocated_descriptor ? 400 : 440) : 1;
    return CCU_SUCCESS;
}
CcuResult HcommCcuEventAlloc(CcuInsHandle, uint8_t, uint32_t, CcuEventHandle *out) {
    *out = ++state.next_handle;
    return CCU_SUCCESS;
}
CcuResult HcommCcuEventGetAddr(CcuEventHandle, uint32_t, uint64_t *out) {
    *out = 0x80000;
    return CCU_SUCCESS;
}
CcuResult HcommCcuGetMemToken(uint64_t, uint64_t, uint64_t *out) {
    *out = 17;
    return CCU_SUCCESS;
}
CcuResult HcommCcuKernelRegisterStart(CcuInsHandle) {
    ++state.registration_starts;
    if (state.fail_registration_start || state.registration_active) return CCU_E_INTERNAL;
    state.registration_active = true;
    return CCU_SUCCESS;
}
CcuResult HcommCcuKernelRegister(
    CcuInsHandle, uint32_t, const char *, const void *, const void **, uint32_t, CcuKernelHandle *out
) {
    ++state.registrations;
    EXPECT_TRUE(state.registration_active);
    if (state.fail_registration) return CCU_E_UNAVAIL;
    *out = ++state.next_handle;
    return CCU_SUCCESS;
}
CcuResult HcommCcuKernelRegisterEnd(CcuInsHandle) {
    ++state.registration_ends;
    EXPECT_TRUE(state.registration_active);
    state.registration_active = false;
    return state.fail_registration_end ? CCU_E_RUNTIME : CCU_SUCCESS;
}
CcuResult HcommCcuGetTaskArgsNum(CcuKernelHandle, uint32_t *out) {
    *out = 9;
    return CCU_SUCCESS;
}
CcuResult HcommCcuKernelLaunch(ThreadHandle, CcuKernelHandle, const void *, uint32_t) {
    ++state.launches;
    return CCU_SUCCESS;
}
CcuResult CcuNotifyWait(ChannelHandle, uint32_t, uint16_t) { return CCU_SUCCESS; }
}

TEST_F(CcuSessionSdkTest, FailedRegistrationEndsItsRoundOnceAndPreservesOriginalError) {
    for (bool end_fails : {false, true}) {
        auto backend = ccu_make_session_backend();
        backend->Initialize(Config());
        state.fail_registration = true;
        state.fail_registration_end = end_fails;
        const auto starts = state.registration_starts;
        const auto ends = state.registration_ends;
        CcuDeviceContextView view{};
        try {
            backend->Publish(Functions(), Image(), view);
            ADD_FAILURE() << "Register failure must be returned";
        } catch (const CcuSdkError &error) {
            EXPECT_EQ(error.code, CCU_E_UNAVAIL);
            EXPECT_STREQ(error.what(), "Register(service)");
        }
        EXPECT_EQ(state.registration_starts, starts + 1);
        EXPECT_EQ(state.registration_ends, ends + 1);
        EXPECT_FALSE(state.registration_active);
        EXPECT_EQ(state.launches, 0);
        EXPECT_EQ(view.addr, 0U);
        state.fail_communicator_destroy = true;
        EXPECT_THROW(backend->Close(), CcuSdkError);
        EXPECT_EQ(state.communicators, 1);
        EXPECT_FALSE(state.allocations.empty());
        state.fail_communicator_destroy = false;
        EXPECT_NO_THROW(backend->Close());
        EXPECT_EQ(state.registration_ends, ends + 1);
        state.fail_registration = false;
    }
}

TEST_F(CcuSessionSdkTest, StopRetryOnlyQueriesDrainAfterCommandOverwritesRequestProgram) {
    auto backend = ccu_make_session_backend();
    backend->Initialize(Config());
    CcuDeviceContextView view{};
    backend->Publish(Functions(), Image(), view);
    auto *record = reinterpret_cast<CcuRequestRecord *>(view.addr + 256);
    record->slot_state = CcuSlotState::SUBMITTED;
    record->sequence = 7;
    record->session_epoch = 29;
    record->completed_sequence = 7;
    record->completed_epoch = 29;
    record->completed_state = CcuRequestState::SUCCEEDED;
    EXPECT_FALSE(backend->Stop(0));
    EXPECT_EQ(state.triggers, 1);
    EXPECT_EQ(record->program_index, UINT64_MAX);
    EXPECT_THROW(backend->Close(), CcuSdkError);
    EXPECT_EQ(state.communicators, 1);
    EXPECT_FALSE(state.allocations.empty());
    state.complete = true;
    EXPECT_TRUE(backend->Stop(0));
    EXPECT_EQ(state.triggers, 1);
    backend->Close();
}

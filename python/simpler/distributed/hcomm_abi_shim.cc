/*
 * Copyright (c) 2026.
 *
 * ABI shim for loading stock local HCOMM builds from Simpler tests.
 *
 * Some local libhcomm.so builds reference a C++ template instantiation that is
 * not exported by the linked HCOMM sidecar libraries.  Keep the fix outside
 * HCOMM by providing the missing instantiation from Simpler's adapter layer.
 */

#include "hccl_communicator.h"

namespace hccl {

template <>
HcclResult HcclCommunicator::GenIbvAiRMAInfo<HcclAiRMAInfo>(
    u32 rankid,
    const std::shared_ptr<Transport> &transport,
    const std::string &tag,
    HcclAiRMAInfo *aiRMAInfoPtr)
{
    (void)tag;
    std::vector<HcclAiRMAQueueInfo> aiQpVec;
    HcclResult ret = transport->GetAiRMAQueueInfo(aiQpVec);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    if (aiRMAInfoPtr == nullptr) {
        return HCCL_E_PTR;
    }
    if (aiQpVec.size() != aiRMAInfoPtr->qpNum) {
        return HCCL_E_INTERNAL;
    }
    if (aiSqMem_ == nullptr || aiScqMem_ == nullptr || aiRqMem_ == nullptr || aiRcqMem_ == nullptr) {
        return HCCL_E_PTR;
    }

    HcclAiRMAWQ *aiSqHost = reinterpret_cast<HcclAiRMAWQ *>(aiSqMem_->ptr());
    HcclAiRMACQ *aiScqHost = reinterpret_cast<HcclAiRMACQ *>(aiScqMem_->ptr());
    HcclAiRMAWQ *aiRqHost = reinterpret_cast<HcclAiRMAWQ *>(aiRqMem_->ptr());
    HcclAiRMACQ *aiRcqHost = reinterpret_cast<HcclAiRMACQ *>(aiRcqMem_->ptr());
    if (aiSqHost == nullptr || aiScqHost == nullptr || aiRqHost == nullptr || aiRcqHost == nullptr) {
        return HCCL_E_PTR;
    }

    for (u32 j = 0; j < aiRMAInfoPtr->qpNum; ++j) {
        const auto &aiQpInfo = aiQpVec[j];
        aiSqHost[rankid * aiRMAInfoPtr->qpNum + j] = aiQpInfo.sq;
        aiScqHost[rankid * aiRMAInfoPtr->qpNum + j] = aiQpInfo.scq;
        aiRqHost[rankid * aiRMAInfoPtr->qpNum + j] = aiQpInfo.rq;
        aiRcqHost[rankid * aiRMAInfoPtr->qpNum + j] = aiQpInfo.rcq;
    }
    return HCCL_SUCCESS;
}

} // namespace hccl

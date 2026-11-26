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
#include <acl/acl.h>
#include <driver/ascend_hal.h>
#include <rt_external_device.h>
#include <fstream>
#include <iterator>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "common/acl_hal_device.h"
#include "common/platform_config.h"
#include "host/host_regs.h"
#include "host/scheduler_notification_address.h"
#include "probe.h"

using namespace peer_dmb;

static void check(int rc, const char *name) {
    if (rc != 0) throw std::runtime_error(std::string(name) + " rc=" + std::to_string(rc));
}
#define CHECK(call) check((call), #call)

class CoreWindows {
public:
    explicit CoreWindows(int device) :
        device_(pto::acl_to_hal_device_id(device)) {}
    void bind(Config &config) {
        for (unsigned i = 0; i < DAV_3510::PLATFORM_MAX_PHYSICAL_CORES; ++i) {
            res_map_info info{};
            info.target_proc_type = PROCESS_CP1;
            info.res_type = RES_AICORE;
            info.res_id = i;
            unsigned long va = 0;
            unsigned bytes = REG_AICORE_MAP_SIZE;
            CHECK(halResMap(device_, &info, &va, &bytes));
            ++count_;
            if (!va || bytes < REG_SPR_DATA_MAIN_BASE_OFFSET + sizeof(uint32_t))
                throw std::runtime_error("HAL mapping does not cover the AIC DMB");
            const unsigned slot =
                (i / PLATFORM_AICORE_PER_DIE) * PLATFORM_AICORE_PER_DIE * PLATFORM_CORES_PER_BLOCKDIM +
                i % PLATFORM_AICORE_PER_DIE;
            config.dmb[slot] = va + REG_SPR_DATA_MAIN_BASE_OFFSET;
        }
    }
    // A failed synchronization can leave a kernel using the table until runtime teardown.
    void stream_pending(bool pending) { busy_ = pending; }
    ~CoreWindows() {
        if (busy_) return;
        for (unsigned i = 0; i < count_; ++i) {
            res_map_info info{};
            info.target_proc_type = PROCESS_CP1;
            info.res_type = RES_AICORE;
            info.res_id = i;
            const int rc = halResUnmap(device_, &info);
            if (rc) std::fprintf(stderr, "halResUnmap core=%u rc=%d\n", i, rc);
        }
    }

private:
    unsigned device_, count_ = 0;
    bool busy_ = false;
};

static int run(int device, const char *binary, bool attempt, Fence fence) {
    aclrtStream stream{};
    CHECK(aclrtCreateStream(&stream));
    aclrtCntNotify notify{};
    CHECK(aclrtCntNotifyCreate(&notify, 0));
    uint32_t id = 0, bytes = 0;
    CHECK(aclrtCntNotifyGetId(notify, &id));
    rtDevResInfo mapping{0, RT_PROCESS_CP1, RT_RES_TYPE_STARS_CNT_NOTIFY_RECORD, id, 0};
    Config cfg{};
    rtDevResAddrInfo out{&cfg.set, &bytes};
    CHECK(rtGetDevResAddress(&mapping, &out));
    cfg.inc = a5_cnt_notify_increment_address(cfg.set);
    if (!cfg.inc || bytes < 4) throw std::runtime_error("invalid CntNotify mapping");
    CoreWindows windows(device);
    windows.bind(cfg);
    void *config_dev{}, *state_dev{};
    CHECK(aclrtMalloc(&config_dev, sizeof(cfg), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&state_dev, sizeof(State), ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtBinaryLoadOption magic{};
    magic.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
    magic.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
    aclrtBinaryLoadOptions options{&magic, 1};
    aclrtBinHandle bin{};
    aclrtFuncHandle fn{};
    std::ifstream input(binary, std::ios::binary);
    std::vector<char> elf{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (elf.empty()) throw std::runtime_error("empty kernel binary");
    CHECK(aclrtBinaryLoadFromData(elf.data(), elf.size(), &options, &bin));
    CHECK(aclrtBinaryGetFunctionByEntry(bin, 0, &fn));
    int result = 0;
    for (Mode mode : {GM_EARLY, GM_ORDERED, PEER_DMB}) {
        if (mode == PEER_DMB && !attempt) break;
        cfg.mode = mode;
        cfg.fence = mode == PEER_DMB ? fence : DDR;
        State data{};
        data.receiver.first = data.receiver.eventual = kUnset;
        data.sender.core = data.receiver.core = kUnset;
        CHECK(aclrtMemcpy(config_dev, sizeof(cfg), &cfg, sizeof(cfg), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK(aclrtMemcpy(state_dev, sizeof(data), &data, sizeof(data), ACL_MEMCPY_HOST_TO_DEVICE));
        struct {
            void *cfg, *state;
        } args{config_dev, state_dev};
        const char *arm = mode == GM_EARLY ? "gm-early" : mode == GM_ORDERED ? "gm-ordered" : "peer-dmb";
        std::printf("launch arm=%s fence=%u blocks=2\n", arm, cfg.fence);
        CHECK(aclrtLaunchKernelWithHostArgs(fn, 2, stream, nullptr, &args, sizeof(args), nullptr, 0));
        windows.stream_pending(true);
        const int sync_rc = aclrtSynchronizeStreamWithTimeout(stream, 5000);
        windows.stream_pending(sync_rc != 0);
        std::printf("arm=%s stream_rc=%d\n", arm, sync_rc);
        const int copy_rc = aclrtMemcpy(&data, sizeof(data), state_dev, sizeof(data), ACL_MEMCPY_DEVICE_TO_HOST);
        std::printf(
            "arm=%s copy_rc=%d tx_core=%u rx_core=%u tx_stage=%u rx_status=%u"
            " initial=0x%x token=0x%x observed=0x%x first=%u eventual=%u\n",
            arm, copy_rc, data.sender.core, data.receiver.core, data.sender.stage, data.receiver.status,
            data.receiver.initial, data.receiver.token, data.receiver.observed, data.receiver.first,
            data.receiver.eventual
        );
        CHECK(sync_rc);
        CHECK(copy_rc);
        if (data.sender.stage != 2 || data.receiver.status != 1 || data.sender.core == data.receiver.core ||
            data.receiver.observed != data.receiver.token || data.receiver.eventual != 2 ||
            data.receiver.first != (mode == GM_EARLY ? 1U : 2U)) {
            result = 1;
            break;
        }
    }
    CHECK(aclrtBinaryUnLoad(bin));
    CHECK(aclrtFree(state_dev));
    CHECK(aclrtFree(config_dev));
    CHECK(rtReleaseDevResAddress(&mapping));
    CHECK(aclrtCntNotifyDestroy(notify));
    CHECK(aclrtDestroyStream(stream));
    return result;
}

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        if (argc != 3) throw std::runtime_error("usage: launch probe.bin control|peer-none|peer-ddr|peer-all");
        const std::string arm(argv[2]);
        if (arm != "control" && arm != "peer-none" && arm != "peer-ddr" && arm != "peer-all")
            throw std::runtime_error("unknown arm");
        const char *allocation = std::getenv("TASK_DEVICE");
        if (!allocation || !*allocation || std::string(allocation).find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("requires one queue-allocated TASK_DEVICE");
        const int device = std::stoi(allocation);
        CHECK(aclInit(nullptr));
        CHECK(aclrtSetDevice(device));
        const char *soc = aclrtGetSocName();
        std::printf("TASK_DEVICE=%d soc=%s\n", device, soc ? soc : "unknown");
        if (!soc || std::string(soc).find("Ascend950") != 0) throw std::runtime_error("A5 required");
        const Fence fence = arm == "peer-none" ? NONE : arm == "peer-all" ? ALL : DDR;
        const int rc = run(device, argv[1], arm != "control", fence);
        CHECK(aclrtResetDevice(device));  // Release this process's ACL context.
        CHECK(aclFinalize());
        return rc;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
}

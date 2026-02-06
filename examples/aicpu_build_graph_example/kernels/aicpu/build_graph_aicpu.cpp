#include <cstdint>

#include "runtime.h"

namespace {
union ScalarConverter {
    float f32;
    uint64_t u64;
};

constexpr int ORCH_ARGC_MIN = 7;

enum OrchArgIdx : int {
    DEV_A = 0,
    DEV_B = 1,
    DEV_C = 2,
    DEV_D = 3,
    DEV_E = 4,
    DEV_F = 5,
    SIZE = 6,
};
}  // namespace

extern "C" int build_graph_aicpu(Runtime* runtime) {
    if (runtime == nullptr) {
        return -1;
    }

    if (runtime->orch_argc < ORCH_ARGC_MIN) {
        return -1;
    }

    const uint64_t dev_a = runtime->orch_args[DEV_A];
    const uint64_t dev_b = runtime->orch_args[DEV_B];
    const uint64_t dev_c = runtime->orch_args[DEV_C];
    const uint64_t dev_d = runtime->orch_args[DEV_D];
    const uint64_t dev_e = runtime->orch_args[DEV_E];
    const uint64_t dev_f = runtime->orch_args[DEV_F];
    const int size = static_cast<int>(runtime->orch_args[SIZE]);

    if (dev_a == 0 || dev_b == 0 || dev_c == 0 || dev_d == 0 || dev_e == 0 || dev_f == 0 || size <= 0) {
        return -1;
    }

    const AicpuBuildApi& api = runtime->aicpu_build_api;
    if (api.add_task == nullptr || api.add_successor_conditional == nullptr || api.publish_task == nullptr) {
        return -1;
    }

    // Task 0: c = a + b (func_id=0, AIV)
    uint64_t args_t0[4];
    args_t0[0] = dev_a;
    args_t0[1] = dev_b;
    args_t0[2] = dev_c;
    args_t0[3] = static_cast<uint64_t>(size);
    // Pass function_bin_addr=0 to use the runtime's func_id -> kernel_addrs[] binding.
    int t0 = api.add_task(runtime, args_t0, 4, 0, CoreType::AIV, 0);
    if (t0 < 0) return -1;
    api.publish_task(runtime, t0);

    // Task 1: d = c + 1 (func_id=1, AIV)
    ScalarConverter s1{};
    s1.f32 = 1.0f;
    uint64_t args_t1[4];
    args_t1[0] = dev_c;
    args_t1[1] = s1.u64;
    args_t1[2] = dev_d;
    args_t1[3] = static_cast<uint64_t>(size);
    int t1 = api.add_task(runtime, args_t1, 4, 1, CoreType::AIV, 0);
    if (t1 < 0) return -1;
    api.add_successor_conditional(runtime, t0, t1);
    api.publish_task(runtime, t1);

    // Task 2: e = c + 2 (func_id=1, AIV)
    ScalarConverter s2{};
    s2.f32 = 2.0f;
    uint64_t args_t2[4];
    args_t2[0] = dev_c;
    args_t2[1] = s2.u64;
    args_t2[2] = dev_e;
    args_t2[3] = static_cast<uint64_t>(size);
    int t2 = api.add_task(runtime, args_t2, 4, 1, CoreType::AIV, 0);
    if (t2 < 0) return -1;
    api.add_successor_conditional(runtime, t0, t2);
    api.publish_task(runtime, t2);

    // Task 3: f = d * e (func_id=2, AIV)
    uint64_t args_t3[4];
    args_t3[0] = dev_d;
    args_t3[1] = dev_e;
    args_t3[2] = dev_f;
    args_t3[3] = static_cast<uint64_t>(size);
    int t3 = api.add_task(runtime, args_t3, 4, 2, CoreType::AIV, 0);
    if (t3 < 0) return -1;
    api.add_successor_conditional(runtime, t1, t3);
    api.add_successor_conditional(runtime, t2, t3);
    api.publish_task(runtime, t3);

    // Minimal sanity: kernel addresses must exist (0 indicates not registered).
    if (runtime->kernel_addrs[0] == 0 || runtime->kernel_addrs[1] == 0 || runtime->kernel_addrs[2] == 0) {
        return -1;
    }

    return 0;
}

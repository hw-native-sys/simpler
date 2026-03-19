/**
 * AllReduce Orchestration — aicpu_build_graph runtime.
 *
 * This orchestration plugin runs on AICPU. It reads args from
 * runtime->orch_args[] (populated by init_runtime from func_args[])
 * and builds a single AIV task via the aicpu_build_api.
 *
 * orch_args layout (same as host_build_graph variant):
 *   [0] = input device pointer  (in RDMA window)
 *   [1] = output device pointer (regular device memory)
 *   [2] = nranks
 *   [3] = root rank
 *   [4] = CommDeviceContext device pointer
 */

#include "runtime.h"
#include <cstdint>

extern "C" int build_allreduce_graph(Runtime* runtime) {
    if (runtime == nullptr || runtime->orch_argc < 5) {
        return -1;
    }

    uint64_t task_args[5];
    task_args[0] = runtime->orch_args[0];
    task_args[1] = runtime->orch_args[1];
    task_args[2] = runtime->orch_args[2];
    task_args[3] = runtime->orch_args[3];
    task_args[4] = runtime->orch_args[4];

    const AicpuBuildApi& api = runtime->aicpu_build_api;
    int t0 = api.add_task(runtime, task_args, 5, 0, CoreType::AIV, 0);
    if (t0 < 0) return -1;
    api.publish_task(runtime, t0);

    return 0;
}

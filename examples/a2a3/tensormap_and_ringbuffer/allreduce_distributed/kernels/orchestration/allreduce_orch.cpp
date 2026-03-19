/**
 * AllReduce Orchestration — tensormap_and_ringbuffer runtime (PTO2 API).
 *
 * All five arguments are passed as SCALAR params so the kernel receives
 * raw uint64_t values (device pointers + integers) in the same flat
 * args[] layout as host_build_graph / aicpu_build_graph.
 *
 * The Tensor/PTOParam system maps tensor params to Tensor-struct pointers
 * (not device addresses) — that would break the allreduce kernel which reads
 * args[] as raw pointers.  Using all-scalar avoids this incompatibility.
 *
 * args layout:
 *   [0] = input device pointer  (in RDMA window)
 *   [1] = output device pointer (regular device memory)
 *   [2] = nranks
 *   [3] = root rank
 *   [4] = CommDeviceContext device pointer
 */

#include <stdint.h>
#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 5,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count,
                                int orch_thread_num, int orch_thread_index) {
    (void)arg_count;
    (void)orch_thread_num;

    if (orch_thread_index != 0) return;

    PTOParam params;
    params.add_scalar(args[0]);
    params.add_scalar(args[1]);
    params.add_scalar(args[2]);
    params.add_scalar(args[3]);
    params.add_scalar(args[4]);
    pto2_rt_submit_aiv_task(rt, 0, params);
}

}  // extern "C"

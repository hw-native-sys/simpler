/**
 * TREDUCE Orchestration Function
 *
 * All arguments are device pointers / scalars already set up by the
 * distributed_worker (comm window addresses, device context pointer).
 * Creates a single AIV task with func_id=0 (treduce kernel).
 *
 * args layout:
 *   args[0] = input device pointer  (in RDMA window)
 *   args[1] = output device pointer (regular device memory)
 *   args[2] = nranks
 *   args[3] = root rank
 *   args[4] = CommDeviceContext device pointer
 */

#include "runtime.h"
#include <cstdint>
#include <iostream>

extern "C" {

int build_treduce_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    if (arg_count < 5) {
        std::cerr << "build_treduce_graph: need 5 args, got " << arg_count << '\n';
        return -1;
    }

    uint64_t input_dev    = args[0];
    uint64_t output_dev   = args[1];
    uint64_t nranks       = args[2];
    uint64_t root         = args[3];
    uint64_t comm_ctx_dev = args[4];

    std::cout << "\n=== build_treduce_graph ===" << '\n';
    std::cout << "  input_dev  = 0x" << std::hex << input_dev << '\n';
    std::cout << "  output_dev = 0x" << output_dev << '\n';
    std::cout << "  comm_ctx   = 0x" << comm_ctx_dev << std::dec << '\n';
    std::cout << "  nranks     = " << nranks << '\n';
    std::cout << "  root       = " << root << '\n';

    uint64_t task_args[5];
    task_args[0] = input_dev;
    task_args[1] = output_dev;
    task_args[2] = nranks;
    task_args[3] = root;
    task_args[4] = comm_ctx_dev;

    int t0 = runtime->add_task(task_args, 5, 0, CoreType::AIV);
    std::cout << "  Created task " << t0 << " (treduce, func_id=0, AIV)" << '\n';

    return 0;
}

}  // extern "C"

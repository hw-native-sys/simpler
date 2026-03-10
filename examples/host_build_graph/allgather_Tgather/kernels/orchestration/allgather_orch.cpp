/**
 * AllGather (TGATHER): N sequential Gathers for performance comparison.
 *
 * Flow: for r in [0, n_ranks): Barrier -> Gather(root=r) -> [rank r: WindowMemCopyOut]
 * Only root calls TGATHER per round. Avoids deadlock when all ranks call TGATHER.
 *
 * Args (10): [0] host_src, [1] host_out, [2] size_src, [3] size_out,
 *   [4] device_ctx_ptr, [5] win_in_base, [6] win_out_base,
 *   [7] n_ranks, [8] root (unused), [9] rank_id
 */

#include "runtime.h"
#include <iostream>
#include <cstdint>
#include <cstring>

extern "C" {

constexpr int GATHER_COUNT = 64;
constexpr size_t HCCL_WIN_SYNC_PREFIX = 64 * sizeof(int32_t);

#define FUNC_WIN_MEMCOPY_IN  0
#define FUNC_GATHER          1
#define FUNC_WIN_MEMCOPY_OUT 2
#define FUNC_COMM_BARRIER    3

int build_allgather_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    if (arg_count < 10) {
        std::cerr << "build_allgather_graph: Expected at least 10 args, got " << arg_count << '\n';
        return -1;
    }

    void* host_src  = reinterpret_cast<void*>(args[0]);
    void* host_out  = reinterpret_cast<void*>(args[1]);
    size_t size_src = static_cast<size_t>(args[2]);
    size_t size_out = static_cast<size_t>(args[3]);
    uint64_t device_ctx_ptr = args[4];
    uint64_t win_in_base    = args[5];
    uint64_t win_out_base   = args[6];
    int n_ranks = static_cast<int>(args[7]);
    int rank_id = static_cast<int>(args[9]);

    std::cout << "\n=== build_allgather_graph (TGATHER, N sequential) ===" << '\n';
    std::cout << "  n_ranks=" << n_ranks << " rank_id=" << rank_id << '\n';

    size_t barrier_size = static_cast<size_t>(n_ranks) * sizeof(int32_t);
    size_t total_barrier_bytes = barrier_size * (static_cast<size_t>(n_ranks) + 1);
    uint64_t barrier_base_0 = win_in_base + HCCL_WIN_SYNC_PREFIX;
    uint64_t win_src = barrier_base_0 + total_barrier_bytes;
    uint64_t win_dst = win_src + GATHER_COUNT * sizeof(float);

    int32_t zeros[64] = {};
    std::memset(zeros, 0, sizeof(zeros));
    runtime->host_api.copy_to_device(reinterpret_cast<void*>(barrier_base_0),
                                     zeros, total_barrier_bytes);

    void* dev_src = runtime->host_api.device_malloc(size_src);
    if (!dev_src) return -1;
    runtime->host_api.copy_to_device(dev_src, host_src, size_src);

    void* dev_out = runtime->host_api.device_malloc(size_out);
    if (!dev_out) {
        runtime->host_api.device_free(dev_src);
        return -1;
    }
    runtime->record_tensor_pair(host_out, dev_out, size_out);

    uint64_t args_wmin[3] = {
        win_src,
        reinterpret_cast<uint64_t>(dev_src),
        static_cast<uint64_t>(GATHER_COUNT)
    };
    int t0 = runtime->add_task(args_wmin, 3, FUNC_WIN_MEMCOPY_IN, CoreType::AIV);

    int t_prev = t0;
    for (int r = 0; r < n_ranks; r++) {
        uint64_t barrier_base_r = barrier_base_0 + static_cast<uint64_t>(r) * barrier_size;
        uint64_t args_barrier[4] = {
            barrier_base_r, device_ctx_ptr,
            static_cast<uint64_t>(n_ranks), static_cast<uint64_t>(0)
        };
        int t_barrier = runtime->add_task(args_barrier, 4, FUNC_COMM_BARRIER, CoreType::AIV);
        runtime->add_successor(t_prev, t_barrier);

        uint64_t args_gather[5] = {
            win_dst, win_src, device_ctx_ptr,
            static_cast<uint64_t>(n_ranks), static_cast<uint64_t>(r)
        };
        int t_gather = runtime->add_task(args_gather, 5, FUNC_GATHER, CoreType::AIV);
        runtime->add_successor(t_barrier, t_gather);

        if (rank_id == r) {
            uint64_t args_wmout[3] = {
                reinterpret_cast<uint64_t>(dev_out),
                win_dst,
                static_cast<uint64_t>(n_ranks * GATHER_COUNT)
            };
            int t_wmout = runtime->add_task(args_wmout, 3, FUNC_WIN_MEMCOPY_OUT, CoreType::AIV);
            runtime->add_successor(t_gather, t_wmout);
            t_prev = t_wmout;
        } else {
            t_prev = t_gather;
        }
    }

    uint64_t barrier_base_post = barrier_base_0 + static_cast<uint64_t>(n_ranks) * barrier_size;
    uint64_t args_barrier_post[4] = {
        barrier_base_post, device_ctx_ptr,
        static_cast<uint64_t>(n_ranks), static_cast<uint64_t>(0)
    };
    int t_post = runtime->add_task(args_barrier_post, 4, FUNC_COMM_BARRIER, CoreType::AIV);
    runtime->add_successor(t_prev, t_post);

    std::cout << "  task" << t0 << ": WindowMemCopyIn [AIV]\n";
    std::cout << "  tasks: " << n_ranks << "x (Barrier -> Gather(root=r) -> [rank r: WinCopyOut])\n";
    std::cout << "  task" << t_post << ": CommBarrierAll (post) [AIV]\n";

    return 0;
}

}  // extern "C"

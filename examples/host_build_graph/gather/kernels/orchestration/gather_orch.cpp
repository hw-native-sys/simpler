/**
 * Gather-only orchestration: WindowMemCopyIn -> CommBarrier -> TGATHER -> WindowMemCopyOut (root only).
 *
 * No computation. Each rank has local src data; first GATHER_COUNT elements are gathered to root.
 * CommBarrier uses TNOTIFY/TWAIT for device-side cross-rank synchronization.
 *
 * Args (10):
 *   [0] host_src
 *   [1] host_out (root only, output buffer)
 *   [2] size_src
 *   [3] size_out
 *   [4] device_ctx_ptr
 *   [5] win_in_base
 *   [6] win_out_base
 *   [7] n_ranks
 *   [8] root
 *   [9] rank_id
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

int build_gather_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    if (arg_count < 10) {
        std::cerr << "build_gather_graph: Expected at least 10 args, got " << arg_count << '\n';
        return -1;
    }

    void* host_src  = reinterpret_cast<void*>(args[0]);
    void* host_out = reinterpret_cast<void*>(args[1]);
    size_t size_src  = static_cast<size_t>(args[2]);
    size_t size_out  = static_cast<size_t>(args[3]);
    uint64_t device_ctx_ptr = args[4];
    uint64_t win_in_base    = args[5];
    uint64_t win_out_base   = args[6];
    int n_ranks  = static_cast<int>(args[7]);
    int root     = static_cast<int>(args[8]);
    int rank_id  = static_cast<int>(args[9]);

    std::cout << "\n=== build_gather_graph ===" << '\n';
    std::cout << "  n_ranks=" << n_ranks << " root=" << root
              << " rank_id=" << rank_id << '\n';

    /* ── Window layout ──────────────────────────────────────────────── */
    size_t barrier_size = static_cast<size_t>(n_ranks) * sizeof(int32_t);
    uint64_t barrier_base = win_in_base + HCCL_WIN_SYNC_PREFIX;
    uint64_t win_src = barrier_base + barrier_size;
    uint64_t win_dst = win_src + GATHER_COUNT * sizeof(float);

    int32_t zeros[64] = {};
    std::memset(zeros, 0, sizeof(zeros));
    runtime->host_api.copy_to_device(reinterpret_cast<void*>(barrier_base),
                                     zeros, barrier_size);

    /* ── Allocate device memory for src ─────────────────────────────── */
    void* dev_src = runtime->host_api.device_malloc(size_src);
    if (!dev_src) return -1;
    runtime->host_api.copy_to_device(dev_src, host_src, size_src);

    void* dev_out = nullptr;
    if (rank_id == root) {
        dev_out = runtime->host_api.device_malloc(size_out);
        if (!dev_out) {
            runtime->host_api.device_free(dev_src);
            return -1;
        }
        runtime->record_tensor_pair(host_out, dev_out, size_out);
    }

    /* ── Task 0: WindowMemCopyIn ─────────────────────────────────────── */
    uint64_t args_wmin[3] = {
        win_src,
        reinterpret_cast<uint64_t>(dev_src),
        static_cast<uint64_t>(GATHER_COUNT)
    };
    int t0 = runtime->add_task(args_wmin, 3, FUNC_WIN_MEMCOPY_IN, CoreType::AIV);

    /* ── Task 1: CommBarrier (TNOTIFY/TWAIT) ─────────────────────────── */
    uint64_t args_barrier[4] = {
        barrier_base, device_ctx_ptr,
        static_cast<uint64_t>(n_ranks), static_cast<uint64_t>(root)
    };
    int t1 = runtime->add_task(args_barrier, 4, FUNC_COMM_BARRIER, CoreType::AIV);
    runtime->add_successor(t0, t1);

    /* ── Task 2: TGATHER ─────────────────────────────────────────────── */
    uint64_t args_gather[5] = {
        win_dst, win_src, device_ctx_ptr,
        static_cast<uint64_t>(n_ranks), static_cast<uint64_t>(root)
    };
    int t2 = runtime->add_task(args_gather, 5, FUNC_GATHER, CoreType::AIV);
    runtime->add_successor(t1, t2);

    int t3 = -1;
    if (dev_out != nullptr) {
        /* ── Task 3: WindowMemCopyOut (root only) ─────────────────────── */
        uint64_t args_wmout[3] = {
            reinterpret_cast<uint64_t>(dev_out),
            win_dst,
            static_cast<uint64_t>(n_ranks * GATHER_COUNT)
        };
        t3 = runtime->add_task(args_wmout, 3, FUNC_WIN_MEMCOPY_OUT, CoreType::AIV);
        runtime->add_successor(t2, t3);
    }

    std::cout << "  task" << t0 << ": WindowMemCopyIn [AIV]\n";
    std::cout << "  task" << t1 << ": CommBarrier [AIV]\n";
    std::cout << "  task" << t2 << ": Gather [AIV]\n";
    if (t3 >= 0) std::cout << "  task" << t3 << ": WindowMemCopyOut [AIV]\n";

    return 0;
}

}  // extern "C"

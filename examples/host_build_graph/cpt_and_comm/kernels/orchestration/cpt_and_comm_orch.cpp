/**
 * cpt_and_comm orchestration — split into compute and comm phases so the
 * runner can insert an HcclBarrier between them.
 *
 * Phase 1 (compute): GEMM -> WindowMemCopyIn      (all ranks)
 * Phase 2 (comm):    TGATHER -> WindowMemCopyOut   (root collects)
 *
 * Both functions accept the same arg layout:
 *   host_A, host_B, host_C, host_out, size_A, size_B, size_C, size_out,
 *   device_ctx_ptr, win_in_base, win_out_base, n_ranks, root, rank_id
 */

#include "runtime.h"
#include <iostream>
#include <cstdint>

extern "C" {

constexpr int TILE = 64;
constexpr int GATHER_COUNT = 64;
constexpr size_t HCCL_WIN_SYNC_PREFIX = 64 * sizeof(int32_t);

// ── helpers ──────────────────────────────────────────────────────────────

struct CptCommArgs {
    void* host_A;
    void* host_B;
    void* host_C;
    void* host_out;
    size_t size_A;
    size_t size_B;
    size_t size_C;
    size_t size_out;
    uint64_t device_ctx_ptr;
    uint64_t win_in_base;
    uint64_t win_out_base;
    int n_ranks;
    int root;
    int rank_id;
};

static int parse_args(uint64_t* args, int arg_count, CptCommArgs& out) {
    if (arg_count < 14) {
        std::cerr << "cpt_and_comm_orch: Expected at least 14 args, got "
                  << arg_count << '\n';
        return -1;
    }
    out.host_A   = reinterpret_cast<void*>(args[0]);
    out.host_B   = reinterpret_cast<void*>(args[1]);
    out.host_C   = reinterpret_cast<void*>(args[2]);
    out.host_out = reinterpret_cast<void*>(args[3]);
    out.size_A   = static_cast<size_t>(args[4]);
    out.size_B   = static_cast<size_t>(args[5]);
    out.size_C   = static_cast<size_t>(args[6]);
    out.size_out = static_cast<size_t>(args[7]);
    out.device_ctx_ptr = args[8];
    out.win_in_base    = args[9];
    out.win_out_base   = args[10];
    out.n_ranks  = static_cast<int>(args[11]);
    out.root     = static_cast<int>(args[12]);
    out.rank_id  = static_cast<int>(args[13]);
    return 0;
}

// ── Phase 1: compute ─────────────────────────────────────────────────────

int build_cpt_compute_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    CptCommArgs a{};
    if (parse_args(args, arg_count, a) != 0) return -1;

    std::cout << "\n=== build_cpt_compute_graph ===" << '\n';
    std::cout << "  n_ranks=" << a.n_ranks << " root=" << a.root
              << " rank_id=" << a.rank_id << '\n';

    // Allocate device memory for GEMM operands
    void* dev_A = runtime->host_api.device_malloc(a.size_A);
    if (!dev_A) return -1;
    runtime->host_api.copy_to_device(dev_A, a.host_A, a.size_A);

    void* dev_B = runtime->host_api.device_malloc(a.size_B);
    if (!dev_B) { runtime->host_api.device_free(dev_A); return -1; }
    runtime->host_api.copy_to_device(dev_B, a.host_B, a.size_B);

    void* dev_C = runtime->host_api.device_malloc(a.size_C);
    if (!dev_C) {
        runtime->host_api.device_free(dev_A);
        runtime->host_api.device_free(dev_B);
        return -1;
    }
    runtime->host_api.copy_to_device(dev_C, a.host_C, a.size_C);

    // Window src address (same layout as comm phase)
    uint64_t win_src = a.win_in_base + HCCL_WIN_SYNC_PREFIX;

    // Task 0: GEMM  C = A @ B
    uint64_t args_gemm[3];
    args_gemm[0] = reinterpret_cast<uint64_t>(dev_A);
    args_gemm[1] = reinterpret_cast<uint64_t>(dev_B);
    args_gemm[2] = reinterpret_cast<uint64_t>(dev_C);
    int t0 = runtime->add_task(args_gemm, 3, 0, CoreType::AIC);

    // Task 1: WindowMemCopyIn — copy first GATHER_COUNT of dev_C to window
    uint64_t args_wmin[3];
    args_wmin[0] = win_src;
    args_wmin[1] = reinterpret_cast<uint64_t>(dev_C);
    args_wmin[2] = static_cast<uint64_t>(GATHER_COUNT);
    int t1 = runtime->add_task(args_wmin, 3, 1, CoreType::AIV);

    runtime->add_successor(t0, t1);

    std::cout << "  task" << t0 << ": GEMM [AIC]\n";
    std::cout << "  task" << t1 << ": WindowMemCopyIn [AIV]\n";
    return 0;
}

// ── Phase 2: comm ────────────────────────────────────────────────────────

int build_cpt_comm_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    CptCommArgs a{};
    if (parse_args(args, arg_count, a) != 0) return -1;

    std::cout << "\n=== build_cpt_comm_graph ===" << '\n';
    std::cout << "  n_ranks=" << a.n_ranks << " root=" << a.root
              << " rank_id=" << a.rank_id << '\n';

    // Window layout (matches pto-comm-isa TGATHER test pattern):
    //   [0, SYNC_PREFIX)                              : sync prefix
    //   [SYNC_PREFIX, SYNC_PREFIX + GATHER_COUNT*4)   : src  (per-rank slice)
    //   [SYNC_PREFIX + GATHER_COUNT*4, ...)           : dst  (gathered, root)
    uint64_t win_src = a.win_in_base + HCCL_WIN_SYNC_PREFIX;
    uint64_t win_dst = a.win_in_base + HCCL_WIN_SYNC_PREFIX
                       + GATHER_COUNT * sizeof(float);

    // Allocate dev_out for root (to receive gathered result)
    void* dev_out = nullptr;
    if (a.rank_id == a.root) {
        dev_out = runtime->host_api.device_malloc(a.size_out);
        if (!dev_out) return -1;
        runtime->record_tensor_pair(a.host_out, dev_out, a.size_out);
    }

    // Task 0: Gather — root collects from all ranks
    uint64_t args_gather[5];
    args_gather[0] = win_dst;
    args_gather[1] = win_src;
    args_gather[2] = a.device_ctx_ptr;
    args_gather[3] = static_cast<uint64_t>(a.n_ranks);
    args_gather[4] = static_cast<uint64_t>(a.root);
    int t0 = runtime->add_task(args_gather, 5, 2, CoreType::AIV);

    int t1 = -1;
    if (dev_out != nullptr) {
        // Task 1: WindowMemCopyOut — root copies gathered result to device
        uint64_t args_wmout[3];
        args_wmout[0] = reinterpret_cast<uint64_t>(dev_out);
        args_wmout[1] = win_dst;
        args_wmout[2] = static_cast<uint64_t>(a.n_ranks * GATHER_COUNT);
        t1 = runtime->add_task(args_wmout, 3, 3, CoreType::AIV);
        runtime->add_successor(t0, t1);
    }

    std::cout << "  task" << t0 << ": Gather [AIV]\n";
    if (t1 >= 0) std::cout << "  task" << t1 << ": WindowMemCopyOut [AIV]\n";
    return 0;
}

}  // extern "C"

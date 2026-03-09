/**
 * Mega Kernel + Communication orchestration.
 *
 * Phase 1 (compute): Full paged attention graph (QK → SF → PV → UP chains).
 * Phase 2 (comm):    WindowMemCopyIn → CommBarrier → TGATHER → WindowMemCopyOut.
 *
 * All tasks are in a single graph launched once; cross-rank synchronization
 * uses TNOTIFY/TWAIT (CommBarrier) on the device side.
 *
 * Args (22):
 *   [0..7]   host pointers: query, key_cache, value_cache, block_table,
 *            context_lens, attn_out, gather_out, config
 *   [8..15]  sizes (bytes): same order as pointers
 *   [16..21] HCCL: device_ctx_ptr, win_in_base, win_out_base, n_ranks, root, rank_id
 */

#include "runtime.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdint>

#define FUNC_QK_MATMUL       0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL       2
#define FUNC_ONLINE_UPDATE   3
#define FUNC_WIN_MEMCOPY_IN  4
#define FUNC_GATHER          5
#define FUNC_WIN_MEMCOPY_OUT 6
#define FUNC_COMM_BARRIER    7

constexpr int GATHER_COUNT = 64;
constexpr size_t HCCL_WIN_SYNC_PREFIX = 64 * sizeof(int32_t);

extern "C" {

int build_mega_kernel_comm_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    if (arg_count < 22) {
        std::cerr << "build_mega_kernel_comm_graph: expected >= 22 args, got "
                  << arg_count << '\n';
        return -1;
    }

    /* ── Parse arguments ──────────────────────────────────────────── */

    void*    host_query       = reinterpret_cast<void*>(args[0]);
    void*    host_key_cache   = reinterpret_cast<void*>(args[1]);
    void*    host_value_cache = reinterpret_cast<void*>(args[2]);
    int*     host_block_table = reinterpret_cast<int*>(args[3]);
    int*     host_context_lens = reinterpret_cast<int*>(args[4]);
    void*    host_attn_out    = reinterpret_cast<void*>(args[5]);
    void*    host_gather_out  = reinterpret_cast<void*>(args[6]);
    int64_t* host_config      = reinterpret_cast<int64_t*>(args[7]);

    size_t query_size        = static_cast<size_t>(args[8]);
    size_t key_cache_size    = static_cast<size_t>(args[9]);
    size_t value_cache_size  = static_cast<size_t>(args[10]);
    // args[11] block_table_size  – used only on host
    // args[12] context_lens_size – used only on host
    size_t attn_out_size     = static_cast<size_t>(args[13]);
    size_t gather_out_size   = static_cast<size_t>(args[14]);
    // args[15] config_size       – used only on host

    uint64_t device_ctx_ptr = args[16];
    uint64_t win_in_base    = args[17];
    uint64_t win_out_base   = args[18];
    int n_ranks  = static_cast<int>(args[19]);
    int root     = static_cast<int>(args[20]);
    int rank_id  = static_cast<int>(args[21]);

    int batch           = static_cast<int>(host_config[0]);
    int num_heads       = static_cast<int>(host_config[1]);
    int kv_head_num     = static_cast<int>(host_config[2]);
    int head_dim        = static_cast<int>(host_config[3]);
    int block_size      = static_cast<int>(host_config[4]);
    int max_num_blocks  = static_cast<int>(host_config[5]);
    uint64_t scale_bits = static_cast<uint64_t>(host_config[6]);

    int q_tile_size    = std::min(num_heads, 128);
    int num_head_tiles = (num_heads + q_tile_size - 1) / q_tile_size;

    std::cout << "\n=== build_mega_kernel_comm_graph ===" << '\n';
    std::cout << "  batch=" << batch << " num_heads=" << num_heads
              << " kv_head_num=" << kv_head_num << " head_dim=" << head_dim << '\n';
    std::cout << "  q_tile_size=" << q_tile_size
              << " num_head_tiles=" << num_head_tiles << '\n';
    std::cout << "  n_ranks=" << n_ranks << " root=" << root
              << " rank_id=" << rank_id << '\n';

    /* ── Allocate device memory for paged-attention inputs ────────── */

    void* dev_query       = runtime->host_api.device_malloc(query_size);
    void* dev_key_cache   = runtime->host_api.device_malloc(key_cache_size);
    void* dev_value_cache = runtime->host_api.device_malloc(value_cache_size);
    void* dev_attn_out    = runtime->host_api.device_malloc(attn_out_size);

    if (!dev_query || !dev_key_cache || !dev_value_cache || !dev_attn_out) {
        std::cerr << "Failed to allocate device memory for attention\n";
        return -1;
    }

    runtime->host_api.copy_to_device(dev_query, host_query, query_size);
    runtime->host_api.copy_to_device(dev_key_cache, host_key_cache, key_cache_size);
    runtime->host_api.copy_to_device(dev_value_cache, host_value_cache, value_cache_size);
    runtime->record_tensor_pair(host_attn_out, dev_attn_out, attn_out_size);

    /* ── Intermediate buffers (same as paged_attention_orch) ──────── */

    size_t sij_size    = static_cast<size_t>(q_tile_size) * block_size * sizeof(float);
    size_t pij_size    = static_cast<size_t>(q_tile_size) * block_size * sizeof(uint16_t);
    size_t mij_size    = static_cast<size_t>(q_tile_size) * sizeof(float);
    size_t lij_size    = mij_size;
    size_t oi_new_size = static_cast<size_t>(q_tile_size) * head_dim * sizeof(float);

    int total_buffers = batch * max_num_blocks;
    void** dev_sij_arr    = new void*[total_buffers];
    void** dev_pij_arr    = new void*[total_buffers];
    void** dev_mij_arr    = new void*[total_buffers];
    void** dev_lij_arr    = new void*[total_buffers];
    void** dev_oi_new_arr = new void*[total_buffers];

    for (int i = 0; i < total_buffers; i++) {
        dev_sij_arr[i]    = runtime->host_api.device_malloc(sij_size);
        dev_pij_arr[i]    = runtime->host_api.device_malloc(pij_size);
        dev_mij_arr[i]    = runtime->host_api.device_malloc(mij_size);
        dev_lij_arr[i]    = runtime->host_api.device_malloc(lij_size);
        dev_oi_new_arr[i] = runtime->host_api.device_malloc(oi_new_size);
    }

    int total_accums = batch * num_head_tiles;
    size_t mi_size = static_cast<size_t>(q_tile_size) * sizeof(float);
    size_t li_size = mi_size;
    size_t oi_size = static_cast<size_t>(q_tile_size) * head_dim * sizeof(float);

    void** dev_mi_arr = new void*[total_accums];
    void** dev_li_arr = new void*[total_accums];
    void** dev_oi_arr = new void*[total_accums];

    for (int i = 0; i < total_accums; i++) {
        dev_mi_arr[i] = runtime->host_api.device_malloc(mi_size);
        dev_li_arr[i] = runtime->host_api.device_malloc(li_size);
        dev_oi_arr[i] = runtime->host_api.device_malloc(oi_size);
    }

    /* ── Build paged-attention task graph ─────────────────────────── */

    int* last_up_tasks = new int[total_accums];
    for (int i = 0; i < total_accums; i++) last_up_tasks[i] = -1;

    int total_tasks = 0;

    for (int b_idx = 0; b_idx < batch; b_idx++) {
        int cur_seq = host_context_lens[b_idx];
        int bn_this_batch = (cur_seq + block_size - 1) / block_size;

        for (int ht = 0; ht < num_head_tiles; ht++) {
            int cur_offset = ht * q_tile_size;
            int accum_idx  = b_idx * num_head_tiles + ht;

            uint8_t* qi_ptr = reinterpret_cast<uint8_t*>(dev_query)
                + static_cast<int64_t>(b_idx * num_heads + cur_offset)
                  * head_dim * sizeof(uint16_t);

            uint8_t* out_ptr = reinterpret_cast<uint8_t*>(dev_attn_out)
                + static_cast<int64_t>(b_idx * num_heads + cur_offset)
                  * head_dim * sizeof(float);

            int kv_head_idx = cur_offset / (num_heads / kv_head_num);

            void* dev_mi = dev_mi_arr[accum_idx];
            void* dev_li = dev_li_arr[accum_idx];
            void* dev_oi = dev_oi_arr[accum_idx];

            int t_up_prev = -1;

            for (int bn = 0; bn < bn_this_batch; bn++) {
                int cur_block_idx = host_block_table[b_idx * max_num_blocks + bn];

                uint8_t* kj_ptr = reinterpret_cast<uint8_t*>(dev_key_cache)
                    + (static_cast<int64_t>(cur_block_idx) * block_size * kv_head_num
                       + kv_head_idx) * head_dim * sizeof(uint16_t);

                uint8_t* vj_ptr = reinterpret_cast<uint8_t*>(dev_value_cache)
                    + (static_cast<int64_t>(cur_block_idx) * block_size * kv_head_num
                       + kv_head_idx) * head_dim * sizeof(uint16_t);

                int buf_idx = b_idx * max_num_blocks + bn;
                void* dev_sij    = dev_sij_arr[buf_idx];
                void* dev_pij    = dev_pij_arr[buf_idx];
                void* dev_mij    = dev_mij_arr[buf_idx];
                void* dev_lij    = dev_lij_arr[buf_idx];
                void* dev_oi_new = dev_oi_new_arr[buf_idx];

                /* QK: qi @ kj.T → sij */
                uint64_t qk_args[6] = {
                    reinterpret_cast<uint64_t>(qi_ptr),
                    reinterpret_cast<uint64_t>(kj_ptr),
                    reinterpret_cast<uint64_t>(dev_sij),
                    static_cast<uint64_t>(q_tile_size),
                    static_cast<uint64_t>(head_dim),
                    static_cast<uint64_t>(block_size)
                };
                int t_qk = runtime->add_task(qk_args, 6, FUNC_QK_MATMUL, CoreType::AIC);
                total_tasks++;

                /* SF: scale, rowmax, exp, rowsum → pij, mij, lij */
                uint64_t sf_args[7] = {
                    reinterpret_cast<uint64_t>(dev_sij),
                    scale_bits,
                    reinterpret_cast<uint64_t>(dev_pij),
                    reinterpret_cast<uint64_t>(dev_mij),
                    reinterpret_cast<uint64_t>(dev_lij),
                    static_cast<uint64_t>(q_tile_size),
                    static_cast<uint64_t>(block_size)
                };
                int t_sf = runtime->add_task(sf_args, 7, FUNC_SOFTMAX_PREPARE, CoreType::AIV);
                total_tasks++;

                /* PV: pij @ vj → oi_new */
                uint64_t pv_args[6] = {
                    reinterpret_cast<uint64_t>(dev_pij),
                    reinterpret_cast<uint64_t>(vj_ptr),
                    reinterpret_cast<uint64_t>(dev_oi_new),
                    static_cast<uint64_t>(q_tile_size),
                    static_cast<uint64_t>(block_size),
                    static_cast<uint64_t>(head_dim)
                };
                int t_pv = runtime->add_task(pv_args, 6, FUNC_PV_MATMUL, CoreType::AIC);
                total_tasks++;

                runtime->add_successor(t_qk, t_sf);
                runtime->add_successor(t_sf, t_pv);

                /* UP: online softmax update + normalise */
                int is_first = (bn == 0) ? 1 : 0;
                int is_last  = (bn == bn_this_batch - 1) ? 1 : 0;

                uint64_t up_args[11] = {
                    reinterpret_cast<uint64_t>(dev_mij),
                    reinterpret_cast<uint64_t>(dev_lij),
                    reinterpret_cast<uint64_t>(dev_oi_new),
                    reinterpret_cast<uint64_t>(dev_mi),
                    reinterpret_cast<uint64_t>(dev_li),
                    reinterpret_cast<uint64_t>(dev_oi),
                    static_cast<uint64_t>(is_first),
                    static_cast<uint64_t>(is_last),
                    reinterpret_cast<uint64_t>(out_ptr),
                    static_cast<uint64_t>(q_tile_size),
                    static_cast<uint64_t>(head_dim)
                };
                int t_up = runtime->add_task(up_args, 11, FUNC_ONLINE_UPDATE, CoreType::AIV);
                total_tasks++;

                runtime->add_successor(t_pv, t_up);
                if (t_up_prev >= 0) {
                    runtime->add_successor(t_up_prev, t_up);
                }
                t_up_prev = t_up;
            }

            last_up_tasks[accum_idx] = t_up_prev;
        }
    }

    std::cout << "  Paged-attention tasks: " << total_tasks << '\n';

    /* ── Communication tasks ──────────────────────────────────────── */

    size_t barrier_size = static_cast<size_t>(n_ranks) * sizeof(int32_t);
    uint64_t barrier_base = win_in_base + HCCL_WIN_SYNC_PREFIX;
    uint64_t win_src = barrier_base + barrier_size;
    uint64_t win_dst = win_src + GATHER_COUNT * sizeof(float);

    int32_t zeros[64] = {};
    std::memset(zeros, 0, sizeof(zeros));
    runtime->host_api.copy_to_device(reinterpret_cast<void*>(barrier_base),
                                     zeros, barrier_size);

    /* WindowMemCopyIn: first GATHER_COUNT of attn_out → window */
    uint64_t args_wmin[3] = {
        win_src,
        reinterpret_cast<uint64_t>(dev_attn_out),
        static_cast<uint64_t>(GATHER_COUNT)
    };
    int t_wmin = runtime->add_task(args_wmin, 3, FUNC_WIN_MEMCOPY_IN, CoreType::AIV);
    total_tasks++;

    for (int i = 0; i < total_accums; i++) {
        if (last_up_tasks[i] >= 0) {
            runtime->add_successor(last_up_tasks[i], t_wmin);
        }
    }

    /* CommBarrier: TNOTIFY + TWAIT */
    uint64_t args_barrier[4] = {
        barrier_base, device_ctx_ptr,
        static_cast<uint64_t>(n_ranks), static_cast<uint64_t>(root)
    };
    int t_barrier = runtime->add_task(args_barrier, 4, FUNC_COMM_BARRIER, CoreType::AIV);
    runtime->add_successor(t_wmin, t_barrier);
    total_tasks++;

    /* TGATHER: root collects from all ranks */
    uint64_t args_gather[5] = {
        win_dst, win_src, device_ctx_ptr,
        static_cast<uint64_t>(n_ranks), static_cast<uint64_t>(root)
    };
    int t_gather = runtime->add_task(args_gather, 5, FUNC_GATHER, CoreType::AIV);
    runtime->add_successor(t_barrier, t_gather);
    total_tasks++;

    /* WindowMemCopyOut: root copies gathered result to device */
    if (rank_id == root) {
        void* dev_gather_out = runtime->host_api.device_malloc(gather_out_size);
        if (!dev_gather_out) {
            delete[] dev_sij_arr; delete[] dev_pij_arr;
            delete[] dev_mij_arr; delete[] dev_lij_arr;
            delete[] dev_oi_new_arr;
            delete[] dev_mi_arr; delete[] dev_li_arr; delete[] dev_oi_arr;
            delete[] last_up_tasks;
            return -1;
        }
        runtime->record_tensor_pair(host_gather_out, dev_gather_out, gather_out_size);

        uint64_t args_wmout[3] = {
            reinterpret_cast<uint64_t>(dev_gather_out),
            win_dst,
            static_cast<uint64_t>(n_ranks * GATHER_COUNT)
        };
        int t_wmout = runtime->add_task(args_wmout, 3, FUNC_WIN_MEMCOPY_OUT, CoreType::AIV);
        runtime->add_successor(t_gather, t_wmout);
        total_tasks++;
    }

    std::cout << "  Total tasks (attention + comm): " << total_tasks << '\n';

    /* ── Cleanup host arrays ──────────────────────────────────────── */

    delete[] dev_sij_arr;
    delete[] dev_pij_arr;
    delete[] dev_mij_arr;
    delete[] dev_lij_arr;
    delete[] dev_oi_new_arr;
    delete[] dev_mi_arr;
    delete[] dev_li_arr;
    delete[] dev_oi_arr;
    delete[] last_up_tasks;

    return 0;
}

}  // extern "C"

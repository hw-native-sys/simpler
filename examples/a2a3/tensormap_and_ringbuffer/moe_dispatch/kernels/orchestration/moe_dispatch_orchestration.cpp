/**
 * MOE Dispatch V2 Orchestration — 8-rank, 4-phase task DAG
 *
 * Task DAG per rank:
 *
 *   Phase 0: Prepare (func_id=0, RTC)
 *     IN:  tokens, expert_ids
 *     OUT: send_staging, local_counts
 *     Side: writes shmem_data[local slots], send_counts
 *       |
 *       +-- send_staging --> Phase 1: Send (func_id=1, deferred CQ)
 *       |                      14 × TPUT_ASYNC data → peer shmem_data
 *       |                       7 × TPUT_ASYNC counts → peer recv_counts
 *       |                       7 × TNOTIFY → peer notify_counter
 *       |
 *       +-- local_counts --+
 *                          |
 *       Phase 1.5: NotifyWait (func_id=3, deferred CQ)
 *         OUT: dummy_notify (dependency token)
 *         Waits for notify_counter >= NUM_RANKS-1 via CQ poll
 *                          |
 *       Phase 2: RecvAssemble (func_id=2, RTC)
 *         IN:  local_counts, dummy_notify
 *         OUT: expand_x, expert_token_nums
 *         Reads shmem_data + recv_counts after NotifyWait completes
 *
 * args layout (from DISTRIBUTED_CONFIG):
 *   [0]  = tokens            (window, float*)
 *   [1]  = expert_ids        (window, int32*)
 *   [2]  = shmem_data        (window, float*)
 *   [3]  = send_staging      (window, float*)
 *   [4]  = local_counts      (window, int32*)
 *   [5]  = send_counts       (window, int32*)
 *   [6]  = recv_counts       (window, int32*)
 *   [7]  = notify_counter    (window, int32*)
 *   [8]  = expand_x          (device, float*)
 *   [9]  = expert_token_nums (device, int32*)
 *   [10] = CommDeviceContext*
 */

#include <stdint.h>

#include "common/comm_context.h"
#include "pto_orchestration_api.h"

static constexpr int NUM_TOKENS = 16;
static constexpr int HIDDEN_DIM = 128;
static constexpr int NUM_RANKS = 8;
static constexpr int EXPERTS_PER_RANK = 2;
static constexpr int EXPAND_X_ROWS = NUM_TOKENS * NUM_RANKS;
static constexpr int COUNT_PAD = 32;
static constexpr int SLOT_ELEMS = NUM_TOKENS * HIDDEN_DIM;
static constexpr int STAGING_ELEMS = NUM_RANKS * EXPERTS_PER_RANK * SLOT_ELEMS;

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 11,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(uint64_t* args, int arg_count,
                               int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;
    if (orch_thread_index != 0) return;

    if (arg_count != 11) {
        LOG_ERROR("moe_dispatch_v2: expected 11 args, got %d", arg_count);
        return;
    }

    void*    tokens_ptr          = (void*)(uintptr_t)args[0];
    void*    expert_ids_ptr      = (void*)(uintptr_t)args[1];
    uint64_t shmem_data_addr     = args[2];
    void*    send_staging_ptr    = (void*)(uintptr_t)args[3];
    void*    local_counts_ptr    = (void*)(uintptr_t)args[4];
    uint64_t send_counts_addr    = args[5];
    uint64_t recv_counts_addr    = args[6];
    uint64_t notify_counter_addr = args[7];
    void*    expand_x_ptr        = (void*)(uintptr_t)args[8];
    void*    etn_ptr             = (void*)(uintptr_t)args[9];
    auto*    comm_ctx = reinterpret_cast<CommDeviceContext*>((uintptr_t)args[10]);

    int my_rank = (int)comm_ctx->rankId;

    uint32_t tokens_shape[1]     = { (uint32_t)(NUM_TOKENS * HIDDEN_DIM) };
    uint32_t expert_ids_shape[1] = { (uint32_t)NUM_TOKENS };
    uint32_t send_stg_shape[1]   = { (uint32_t)STAGING_ELEMS };
    uint32_t count_shape[1]      = { (uint32_t)COUNT_PAD };
    uint32_t expand_x_shape[1]   = { (uint32_t)(EXPAND_X_ROWS * HIDDEN_DIM) };
    uint32_t etn_shape[1]        = { (uint32_t)EXPERTS_PER_RANK };

    Tensor ext_tokens       = make_tensor_external(tokens_ptr, tokens_shape, 1, DataType::FLOAT32);
    Tensor ext_expert_ids   = make_tensor_external(expert_ids_ptr, expert_ids_shape, 1, DataType::INT32);
    Tensor ext_send_stg     = make_tensor_external(send_staging_ptr, send_stg_shape, 1, DataType::FLOAT32);
    Tensor ext_local_counts = make_tensor_external(local_counts_ptr, count_shape, 1, DataType::INT32);
    Tensor ext_expand_x     = make_tensor_external(expand_x_ptr, expand_x_shape, 1, DataType::FLOAT32);
    Tensor ext_etn          = make_tensor_external(etn_ptr, etn_shape, 1, DataType::INT32);

    uint64_t sdma_context = pto2_rt_get_sdma_context();
    uint64_t cq_send = pto2_rt_alloc_cq();
    uint64_t cq_notify = pto2_rt_alloc_cq();
    if (sdma_context == 0 || cq_send == 0 || cq_notify == 0) {
        LOG_ERROR("moe_dispatch_v2: rank %d failed SDMA context or CQ alloc", my_rank);
        return;
    }

    // Phase 0: Prepare
    PTOParam params_prepare;
    params_prepare.add_input(ext_tokens);
    params_prepare.add_input(ext_expert_ids);
    params_prepare.add_output(ext_send_stg);
    params_prepare.add_output(ext_local_counts);
    params_prepare.add_scalar(shmem_data_addr);
    params_prepare.add_scalar(send_counts_addr);
    params_prepare.add_scalar((uint64_t)(uintptr_t)comm_ctx);
    pto2_rt_submit_aiv_task(0, params_prepare);

    // Phase 1: Send — data + counts + notify (single deferred CQ)
    PTOParam params_send;
    params_send.add_input(ext_send_stg);
    params_send.add_scalar(shmem_data_addr);
    params_send.add_scalar(send_counts_addr);
    params_send.add_scalar(recv_counts_addr);
    params_send.add_scalar(notify_counter_addr);
    params_send.add_scalar((uint64_t)(uintptr_t)comm_ctx);
    params_send.add_scalar(sdma_context);
    pto2_rt_submit_aiv_task_deferred(1, params_send, cq_send);

    // Phase 1.5: NotifyWait — deferred task that waits for notification counter.
    // Produces a dummy_notify tensor so RecvAssemble can depend on it via TensorMap.
    uint32_t dummy_shape[1] = { 1 };
    Tensor dummy_notify = make_tensor(dummy_shape, 1, DataType::INT32);

    PTOParam params_wait;
    params_wait.add_output(dummy_notify);
    params_wait.add_scalar(notify_counter_addr);
    params_wait.add_scalar((uint64_t)(NUM_RANKS - 1));
    pto2_rt_submit_aiv_task_deferred(3, params_wait, cq_notify);

    // Phase 2: RecvAssemble (depends on NotifyWait via dummy_notify)
    PTOParam params_recv;
    params_recv.add_input(dummy_notify);
    params_recv.add_input(ext_local_counts);
    params_recv.add_output(ext_expand_x);
    params_recv.add_output(ext_etn);
    params_recv.add_scalar(shmem_data_addr);
    params_recv.add_scalar(recv_counts_addr);
    params_recv.add_scalar((uint64_t)(uintptr_t)comm_ctx);
    pto2_rt_submit_aiv_task(2, params_recv);

    LOG_INFO("moe_dispatch_v2: rank %d submitted 4-phase DAG (8-rank, expect %d notifs)",
             my_rank, NUM_RANKS - 1);
}

}  // extern "C"

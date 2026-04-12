/**
 * PTO Async Kernel API — unified device-facing async helper entry.
 *
 * This header intentionally aggregates only AICore-side async helpers:
 *   - completion queue data layout and write helpers
 *   - send queue/session helpers
 *   - notify helpers
 *
 * It does not include scheduler/AICPU-side polling logic such as
 * pto_async_wait.h. That boundary is kept explicit so device inline APIs
 * and runtime completion management do not get mixed into one layer.
 */

#ifndef PTO_ASYNC_KERNEL_API_H
#define PTO_ASYNC_KERNEL_API_H

#include "pto_cq_types.h"
#include "pto_cq_kernel_api.h"
#include "pto_sq_kernel_api.h"
#include "pto_notify_kernel_api.h"

#endif  // PTO_ASYNC_KERNEL_API_H

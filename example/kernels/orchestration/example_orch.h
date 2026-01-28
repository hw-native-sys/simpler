/**
 * Example Orchestration Function Header
 *
 * Defines the orchestration function signature and exports the example
 * orchestration function for building the (a+b+1)*(a+b+2) task graph.
 */

#ifndef EXAMPLE_ORCH_H
#define EXAMPLE_ORCH_H

#include <stdint.h>

// Forward declaration
class Runtime;

/**
 * Orchestration function signature.
 *
 * @param runtime   Pointer to Runtime to populate with tasks
 * @param args      Arguments array (host pointers, sizes, etc.)
 * @param arg_count Total number of arguments
 * @return 0 on success, negative on error
 */
typedef int (*OrchestrationFunc)(Runtime* runtime, uint64_t* args, int arg_count);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Example orchestration: builds task graph for (a+b+1)*(a+b+2)
 *
 * This function handles device memory management:
 * 1. Allocates device memory for inputs and output
 * 2. Copies input data from host to device
 * 3. Records output tensor for copy-back during finalize
 * 4. Builds the compute task graph
 *
 * Expected args (7 total):
 *   args[0] = host_a pointer (host memory)
 *   args[1] = host_b pointer (host memory)
 *   args[2] = host_f pointer (host memory, output)
 *   args[3] = size_a (bytes)
 *   args[4] = size_b (bytes)
 *   args[5] = size_f (bytes)
 *   args[6] = SIZE (number of elements)
 *
 * Tasks created:
 *   t0: c = a + b     (func_id=0: kernel_add)
 *   t1: d = c + 1     (func_id=1: kernel_add_scalar)
 *   t2: e = c + 2     (func_id=1: kernel_add_scalar)
 *   t3: f = d * e     (func_id=2: kernel_mul)
 */
int BuildExampleGraph(Runtime* runtime, uint64_t* args, int arg_count);

#ifdef __cplusplus
}
#endif

#endif // EXAMPLE_ORCH_H

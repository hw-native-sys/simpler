/**
 * @file cpu_sim_state.h
 * @brief Internal header for CPU simulation state lifecycle management
 *
 * Declares clear_cpu_sim_shared_storage() for DeviceRunner to call at
 * run() entry and finalize() to reset simulation state between runs.
 */

#ifndef PLATFORM_SIM_HOST_CPU_SIM_STATE_H_
#define PLATFORM_SIM_HOST_CPU_SIM_STATE_H_

void clear_cpu_sim_shared_storage();

#endif  // PLATFORM_SIM_HOST_CPU_SIM_STATE_H_

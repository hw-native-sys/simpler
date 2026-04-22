#ifndef INCLUDE_COMMON_H
#define INCLUDE_COMMON_H

#define CONST_2 2

#define SET_FLAG(trigger, waiter, e) set_flag(PIPE_##trigger, PIPE_##waiter, (e))
#define WAIT_FLAG(trigger, waiter, e) wait_flag(PIPE_##trigger, PIPE_##waiter, (e))
#define PIPE_BARRIER(pipe) pipe_barrier(PIPE_##pipe)

#ifndef __force_inline__
#define __force_inline__ inline __attribute__((always_inline))
#endif

#endif

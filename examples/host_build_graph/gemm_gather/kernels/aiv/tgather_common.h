/**
 * Shape constants for TGATHER (a2a3, comm-isa aligned).
 * Used by kernel_gather.cpp. Same values as pto-comm-isa a2a3 tgather.
 */
#ifndef GEMM_GATHER_TGATHER_COMMON_H
#define GEMM_GATHER_TGATHER_COMMON_H

// runTGather1D_float: src0 (32, 1024), src1 (16, 64), out (16, 64)
#define GATHER_SRC0_ROWS 32
#define GATHER_SRC0_COLS 1024
#define GATHER_SRC1_ROWS 16
#define GATHER_SRC1_COLS 64

#endif

# TFILLPAD_INPLACE Bug at Small Tile Width (N ≤ 16)

## Summary

`TFILLPAD_INPLACE` produces incorrect padding results on Ascend A2/A3 hardware when
the tile column count `N` is small (e.g. N=16 for float32). The bug manifests as
corrupted data in the padded region for certain `valid_len` values, causing downstream
softmax and attention computations to produce wrong results.

## Affected Configuration

- **Platform**: Ascend A2/A3 (tested on hardware, also reproduces on simulator)
- **Data type**: float32 (sizeof=4)
- **Tile shape**: (M, N) = (16, 16) — i.e. 2 × 32-byte blocks per row
- **PTO source**: `include/pto/npu/a2a3/TFillPad.hpp`

The bug does NOT reproduce at larger N values (N=32, 64, 128) where the same
`valid_len` values work correctly.

## Reproduction

In the paged attention example (`examples/tensormap_and_ringbuffer/paged_attention/`),
the softmax preparation kernel uses `TFILLPAD_INPLACE` to mask invalid key positions
with `-inf` before computing softmax:

```cpp
// Tile types
using TileSijDyn = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, -1>;
using TileSijPad = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16,
                        SLayout::NoneBox, 512, PadValue::Min>;

TileSijDyn sijDynTile(valid_len);  // valid_len = number of valid columns
TileSijPad sijPadTile;
// Both assigned to same UB address (in-place)
TASSIGN(sijDynTile, 0x0);
TASSIGN(sijPadTile, 0x0);

// After loading sij from GM:
TFILLPAD_INPLACE(sijPadTile, sijDynTile);
// Expected: columns [valid_len, 16) filled with -inf (0xff800000)
// Actual:   corrupted for certain valid_len values
```

### Test Matrix (N=16, float32, on hardware)

| valid_len | context_len | blocks | TFILLPAD_INPLACE only | SetValue only | TFILLPAD + SetValue |
|-----------|-------------|--------|-----------------------|---------------|---------------------|
| 1         | 17          | 2      | FAIL (27/256)         | PASS          | PASS                |
| 7         | 23          | 2      | FAIL (29/256)         | PASS          | PASS                |
| 8         | 24          | 2      | FAIL (28/256)         | FAIL (182/256)| PASS                |
| 9         | 25          | 2      | PASS                  | PASS          | PASS                |
| 12        | 28          | 2      | PASS                  | PASS          | PASS                |
| 15        | 31          | 2      | PASS                  | PASS          | PASS                |
| 16 (full) | 32          | 2      | PASS                  | PASS          | PASS                |
| 1         | 33          | 3      | FAIL (25/256)         | FAIL (88/256) | PASS                |

### Cross-dimension validation (confirming N=16 is the trigger)

| num_heads | head_dim | block_size (=N) | context_len | valid_len | Result |
|-----------|----------|-----------------|-------------|-----------|--------|
| 16        | 16       | **16**          | 33          | 1         | FAIL   |
| 16        | 16       | **32**          | 33          | 1         | PASS   |
| 16        | **32**   | **16**          | 33          | 1         | FAIL   |

block_size determines N in the softmax tile (M, N). When block_size=32 (N=32),
the same valid_len=1 passes. When block_size=16 (N=16), it fails regardless of
head_dim.

## Root Cause Analysis

The bug is in the `TFillPad` function in `include/pto/npu/a2a3/TFillPad.hpp`.
The function has two internal code paths for filling padding:

### Path A: `Handle32BAlignedPad_Other` (lines 103-134)

Fills the **partial 32-byte block** at the boundary using `vector_dup` with a
norm-mode bitmask. This path is reliable.

### Path B: `PadRightSingleRow` + `PadRightRemainingRows` (lines 136-167)

Fills **complete 32-byte blocks** to the right of the boundary. Uses `vector_dup`
for row 0, then `vcopy` with `srcRepeatStride=0` (broadcast) to replicate to
remaining rows. **This path has the bug.**

### Which path runs depends on `valid_len`

The key variable is `srcValidCol32B` — the valid_len rounded up to the next
32-byte-aligned element count:

```
elements_per_block = 32 / sizeof(float) = 8
srcValidCol32B = ceil(valid_len / 8) * 8
padOffset = srcValidCol32B
padCols = N - srcValidCol32B        // columns for Path B
pad_32B = srcValidCol32B - valid_len // columns for Path A
```

For N=16 (2 blocks of 8 elements each):

```
valid_len ∈ [1, 8]:
    srcValidCol32B = 8
    padOffset = 8,  padCols = 8   → Path B runs (fills block 1)
    pad_32B = 8 - valid_len       → Path A runs if valid_len < 8

valid_len ∈ [9, 15]:
    srcValidCol32B = 16
    padOffset = 16, padCols = 0   → Path B is a NO-OP
    pad_32B = 16 - valid_len      → Path A runs (fills within block 1)

valid_len = 16:
    No padding needed (full block)
```

**Pattern: valid_len ≤ 8 → Path B runs → BUG. valid_len ≥ 9 → only Path A → OK.**

### Path B code trace (the buggy path)

```cpp
// PadRightSingleRow: fill row 0's right padding
set_mask_count();
set_vector_mask(0, padCols);  // padCols = 8
vector_dup(dstPtr + padOffset, dupPadValue, 1, 1, 1, 8, 0);
//         ^-- dstPtr + 8 (element 8 of row 0)
pipe_barrier(PIPE_V);

// PadRightRemainingRows: broadcast row 0's pattern to rows 1..M-1
dstRepeatStride = N * sizeof(float) / 32;  // = 16 * 4 / 32 = 2
_dstPtr = dstPtr + padOffset + copyDstCols; // = dstPtr + 8 + 16 = dstPtr + 24
fillRow = M - 1;  // = 15

vcopy(_dstPtr, dstPtr + padOffset, 15, 1, 0, 2, 0);
//    dst       src                rep  dB sB dR sR
//    row1:8    row0:8             15   1  0  2  0
//
// dstRepeatStride=2 (64 bytes = 1 row), srcRepeatStride=0 (broadcast)
// mask: counter mode, 8 elements (inherited from PadRightSingleRow)
```

The `vcopy` with `srcRepeatStride=0` and `dstRepeatStride=2` at N=16 appears to
produce incorrect results on hardware. The exact hardware failure mode is unclear,
but it consistently corrupts the padding data.

### Why valid_len=8 is special

When `valid_len=8`:
- `pad_32B = 8 - 8 = 0` → Path A computes `mask = 0xff >> 8 << 8 = 0`
- `set_vector_mask(0, 0)` is called, then `vector_dup` with zero mask
- This is effectively a no-op, but may have undefined behavior on hardware
- Path B still runs and produces incorrect results
- Additionally, `SetValue`-only workaround also fails for valid_len=8,
  suggesting the zero-mask `vector_dup` in Path A corrupts pipeline state

## Workaround

The working fix uses **both** `TFILLPAD_INPLACE` and scalar `SetValue` writes:

```cpp
// Step 1: TFILLPAD_INPLACE sets up vector pipeline state correctly
//         (mask modes, barriers, etc.) even though its data output is buggy
TFILLPAD_INPLACE(sijPadTile, sijDynTile);

// Step 2: SetValue patches the actual data with correct -inf values
if (valid_len < static_cast<uint64_t>(N)) {
    constexpr float NEG_INF = -__builtin_huge_valf();
    for (int r = 0; r < M; r++) {
        for (uint64_t c = valid_len; c < N; c++) {
            sijTile.SetValue(static_cast<uint32_t>(r * N + c), NEG_INF);
        }
    }
}
```

**Why both are needed:**

| Approach               | valid_len=1 | valid_len=7 | valid_len=8 |
|------------------------|-------------|-------------|-------------|
| TFILLPAD_INPLACE only  | FAIL        | FAIL        | FAIL        |
| SetValue only          | PASS        | PASS        | FAIL        |
| TFILLPAD + SetValue    | PASS        | PASS        | PASS        |

- `TFILLPAD_INPLACE` alone: Path B produces wrong data
- `SetValue` alone: works for most cases, but valid_len=8 fails because
  Path A's zero-mask `vector_dup` (which runs before SetValue in the
  TFILLPAD-only case) apparently sets up necessary pipeline state that
  subsequent vector operations depend on
- Both together: TFILLPAD handles pipeline state, SetValue fixes the data

## Scope

- **Affected**: Any `TFILLPAD_INPLACE` call with float32 tiles where
  `N ≤ 16` and `valid_len ≤ N/2` (i.e. valid data fits within the first
  32-byte block of each row)
- **Not affected**: N ≥ 32 (tested with N=32, 64, 128 — all pass)
- **Not affected**: Full tiles (valid_len == N)
- **Likely affected**: float16/bfloat16 tiles with N ≤ 32 (untested, but
  the same code path would be triggered since elements_per_block=16 for
  16-bit types, and the same vcopy broadcast pattern is used)

## Files

- Bug location: `include/pto/npu/a2a3/TFillPad.hpp`, functions
  `PadRightSingleRow` (line 136) and `PadRightRemainingRows` (line 146)
- Workaround applied in: `examples/tensormap_and_ringbuffer/paged_attention/kernels/aiv/aiv_softmax_prepare.cpp`
- Test configuration: `examples/tensormap_and_ringbuffer/paged_attention/golden.py`

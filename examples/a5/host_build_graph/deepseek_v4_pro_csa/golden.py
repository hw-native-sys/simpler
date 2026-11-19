# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Original CSA numerical contract, with no PyPTO imports or model compilation."""

if __package__:
    from ._compressor_reference import reference_compressor_cache
    from ._validation import mapped_pool_ratio_allclose, ratio_reldiff
else:
    from _compressor_reference import reference_compressor_cache
    from _validation import mapped_pool_ratio_allclose, ratio_reldiff

B = 4
S = 2
T = B * S
COMPRESS_RATIO = 4
HALF_ROPE = 32
MAIN_STATE_BLOCK_SIZE = 4
INNER_STATE_BLOCK_SIZE = 4
BLOCK_SIZE = 128


def fp8_cache_compare(compare_fn):
    """Widen FP8 exactly for CPU comparison, retaining the mapping/tail checks."""
    import functools

    @functools.wraps(compare_fn)
    def compare(actual, expected, **kwargs):
        return compare_fn(actual.float(), expected.float(), **kwargs)

    return compare


def csa_cache_from_runtime_state_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, **kwargs):
    """Validate state independently, then validate the quantized cache epilogue."""
    required = (
        "compress_state_block_table",
        "cmp_norm_w",
        "position_ids",
        "cmp_slot_mapping",
        "state_slot_mapping",
        "freqs_cos",
        "freqs_sin",
    )
    missing = [name for name in required if name not in inputs]
    if missing or "compress_state" not in actual_outputs or "compress_state" not in expected_outputs:
        return False, f"missing CSA compressor state or inputs: {missing}"
    state_compare = mapped_pool_ratio_allclose(
        "state_slot_mapping",
        mapping_shape=(T,),
        block_size=MAIN_STATE_BLOCK_SIZE,
        pool_name="main compressor state",
        atol=1e-3,
        rtol=1e-3,
        max_error_ratio=0.0,
    )
    ok, detail = state_compare(
        actual_outputs["compress_state"],
        expected_outputs["compress_state"],
        actual_outputs=actual_outputs,
        expected_outputs=expected_outputs,
        inputs=inputs,
        **kwargs,
    )
    if not ok:
        return False, detail

    positions = inputs["position_ids"].reshape(B, S).long()
    first_pos = positions[:, 0]
    cmp_pos = first_pos - first_pos % COMPRESS_RATIO
    _, cache = reference_compressor_cache(
        actual_outputs["compress_state"],
        inputs["compress_state_block_table"],
        inputs["cmp_norm_w"],
        inputs["freqs_cos"][cmp_pos, :HALF_ROPE].float(),
        inputs["freqs_sin"][cmp_pos, :HALF_ROPE].float(),
        positions,
        inputs["cmp_slot_mapping"].reshape(B, S),
        expected,
    )
    cache_compare = mapped_pool_ratio_allclose(
        "cmp_slot_mapping",
        mapping_shape=(T,),
        block_size=BLOCK_SIZE,
        pool_name="compressed KV cache from validated runtime state",
        atol=1e-4,
        rtol=1.0 / 128,
        max_error_ratio=0.0,
    )
    return cache_compare(
        actual,
        cache,
        actual_outputs=actual_outputs,
        expected_outputs=expected_outputs,
        inputs=inputs,
        **kwargs,
    )


def comparators():
    return {
        "x_out": ratio_reldiff(diff_thd=0.004, pct_thd=0.008, max_diff_hd=1),
        "kv_cache": mapped_pool_ratio_allclose(
            "ori_slot_mapping",
            mapping_shape=(T,),
            block_size=BLOCK_SIZE,
            pool_name="KV cache",
            atol=0.0001,
            rtol=1.0 / 128,
            max_error_ratio=0.005,
        ),
        "cmp_kv": csa_cache_from_runtime_state_compare,
        "compress_state": mapped_pool_ratio_allclose(
            "state_slot_mapping",
            mapping_shape=(T,),
            block_size=MAIN_STATE_BLOCK_SIZE,
            pool_name="main compressor state",
            atol=0.001,
            rtol=0.001,
            max_error_ratio=0.0,
        ),
        "inner_compress_state": mapped_pool_ratio_allclose(
            "inner_state_slot_mapping",
            mapping_shape=(T,),
            block_size=INNER_STATE_BLOCK_SIZE,
            pool_name="inner compressor state",
            atol=0.001,
            rtol=0.001,
            max_error_ratio=0.0,
        ),
        "idx_kv_cache": fp8_cache_compare(
            mapped_pool_ratio_allclose(
                "idx_slot_mapping",
                mapping_shape=(T,),
                block_size=BLOCK_SIZE,
                pool_name="index cache",
                atol=0.001953125,
                rtol=0.125,
                max_error_ratio=0.01,
            )
        ),
        "idx_kv_scale": mapped_pool_ratio_allclose(
            "idx_slot_mapping",
            mapping_shape=(T,),
            block_size=BLOCK_SIZE,
            pool_name="index scale cache",
            atol=0.0001,
            rtol=1.0 / 128,
            max_error_ratio=0.01,
        ),
    }


def validate_outputs(actual, expected, inputs):
    comparisons = comparators()
    if set(actual) != set(comparisons) or set(expected) != set(comparisons):
        raise AssertionError("All seven CSA outputs must be provided")
    for name, compare in comparisons.items():
        value, reference = actual[name], expected[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise AssertionError(f"{name}: shape/dtype mismatch")
        ok, detail = compare(
            value, reference, actual_outputs=actual, expected_outputs=expected, inputs=inputs, rtol=1e-2, atol=1e-2
        )
        if not ok:
            raise AssertionError(f"{name}: {detail}")
        print(f"[CSA] {name} PASS", flush=True)

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Small CPU tests of the physical-address/request renaming oracle."""

import pytest
import torch

from examples.a5.host_build_graph.deepseek_v4_pro_csa import business_cases as variants


def small_inputs():
    values = {}
    for pools, table, slots, block_size in variants.POOL_LAYOUTS.values():
        for name in pools:
            value = (torch.arange(5 * block_size * 2).reshape(5, block_size, 1, 2) % 32).float()
            values[name] = value.to(torch.float8_e4m3fn if name == "idx_kv_cache" else torch.float32)
        if table:
            values[table] = torch.tensor([[0, 2, -1], [1, 3, 4], [4, 0, 2], [3, 2, 1]], dtype=torch.int32)
        values[slots] = torch.tensor(
            [0, block_size - 1, block_size, 2 * block_size - 1, 2 * block_size, 3 * block_size, 5 * block_size - 1, -1]
        )
    values.update(
        x_hc=torch.arange(8 * 4 * 2).reshape(8, 4, 2).float(),
        x_out=torch.arange(8 * 4 * 2).reshape(8, 4, 2).float(),
        position_ids=torch.tensor([8192, 8193, 0, 1, 2, 3, 3, 4], dtype=torch.int32),
        window_swa_indices=torch.tensor([[0, 127, 128, 639, -1]] * 8, dtype=torch.int32),
        window_swa_lens=torch.arange(8, dtype=torch.int32),
        kv_seq_lens=torch.tensor([8194, 2, 4, 5], dtype=torch.int32),
        weights=torch.tensor([3.5]),
    )
    return values


def same_bytes(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8))


@pytest.mark.parametrize("group", variants.POOL_LAYOUTS)
@pytest.mark.parametrize("style", ["reverse", "rotate", "scramble"])
def test_page_renaming_preserves_every_physical_lookup_and_input(group, style):
    values = small_inputs()
    saved = {key: value.clone() for key, value in values.items()}
    params = {"pages": {group: style}}
    changed = variants.transform_inputs(values, params)
    pools, table, slots, size = variants.POOL_LAYOUTS[group]
    assert changed["weights"] is values["weights"]
    for name in values:
        same_bytes(values[name], saved[name])
        assert changed[name].dtype == values[name].dtype
        assert changed[name].is_contiguous()
    for pool in pools:
        before = values[pool].reshape(5 * size, -1)
        after = changed[pool].reshape(5 * size, -1)
        for i, old_row in enumerate(values[slots].tolist()):
            new_row = int(changed[slots][i])
            if old_row == -1:
                assert new_row == -1
            else:
                same_bytes(before[old_row], after[new_row])
                assert old_row % size == new_row % size
        if table:
            for old_page, new_page in zip(values[table].flatten(), changed[table].flatten(), strict=True):
                if old_page >= 0:
                    same_bytes(values[pool][old_page], changed[pool][new_page])
                else:
                    assert new_page == -1
    if group == "kv":
        for old, new in zip(
            values["window_swa_indices"].flatten(), changed["window_swa_indices"].flatten(), strict=True
        ):
            if old >= 0:
                same_bytes(before[old], after[new])
            else:
                assert new == -1


@pytest.mark.parametrize("name,params", variants.VARIANTS, ids=[name for name, _ in variants.VARIANTS])
def test_output_renaming_matches_inputs_without_touching_weights_or_fp8_bytes(name, params):
    values = small_inputs()
    inputs = variants.transform_inputs(values, params)
    outputs = variants.transform_outputs(values, params)
    for pools, *_ in variants.POOL_LAYOUTS.values():
        for pool in pools:
            if pool == "idx_kv_scale" and params.get("poison_scale", False):
                rows = inputs[pool].reshape(5 * 128, -1)
                mapping = inputs["idx_slot_mapping"]
                assert torch.all(rows[mapping[mapping >= 0]] == 2.0)
                mask = torch.ones(5 * 128, dtype=torch.bool)
                mask[mapping[mapping >= 0]] = False
                same_bytes(rows[mask], outputs[pool].reshape(5 * 128, -1)[mask])
            else:
                same_bytes(inputs[pool], outputs[pool])
    same_bytes(inputs["x_hc"], outputs["x_out"])
    assert inputs["weights"] is values["weights"]


@pytest.mark.parametrize("order", [[3, 2, 1, 0], [1, 2, 3, 0], [2, 0, 3, 1], [0, 1, 2, 2], [0, 1, 2]])
def test_batch_permutations_are_not_claimed_as_equivalent_for_aliased_state(order):
    with pytest.raises(ValueError, match="batch permutation"):
        variants.transform_inputs(small_inputs(), {"batch": order})


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32, torch.bfloat16, torch.float8_e4m3fn])
def test_row_gather_preserves_one_dimensional_element_boundaries(dtype):
    value = torch.arange(8).to(dtype)
    order = torch.tensor([7, 3, 0, 6, 4, 2, 1, 5])
    actual = variants.take_rows(value, order)
    assert actual.shape == value.shape
    for index, source in enumerate(order):
        same_bytes(actual[index], value[source])


@pytest.mark.parametrize("group", variants.POOL_LAYOUTS)
@pytest.mark.parametrize("bad", ["negative", "one_past_end", "float"])
def test_invalid_slot_values_fail_before_device_execution(group, bad):
    values = small_inputs()
    pools, _, slots, block_size = variants.POOL_LAYOUTS[group]
    if bad == "float":
        values[slots] = values[slots].float()
    else:
        values[slots][0] = -2 if bad == "negative" else values[pools[0]].shape[0] * block_size
    with pytest.raises(ValueError, match="index|indices"):
        variants.transform_inputs(values, {"pages": {group: "reverse"}})


def test_idx_payload_and_scale_cannot_use_different_page_counts():
    values = small_inputs()
    values["idx_kv_scale"] = values["idx_kv_scale"][:4]
    with pytest.raises(ValueError, match="paired cache"):
        variants.transform_inputs(values, {"pages": {"idx": "reverse"}})


@pytest.mark.parametrize("count", [1, 2, 32, 64, 65, 128])
@pytest.mark.parametrize("style", ["reverse", "rotate", "scramble"])
def test_page_maps_are_deterministic_bijections_without_global_rng_side_effect(count, style):
    state = torch.random.get_rng_state()
    order = variants.page_order(count, style)
    assert sorted(order.tolist()) == list(range(count))
    assert torch.equal(order, variants.page_order(count, style))
    assert torch.equal(state, torch.random.get_rng_state())


def test_unknown_page_style_is_not_silently_identity():
    with pytest.raises(ValueError, match="unknown"):
        variants.page_order(65, "typo")

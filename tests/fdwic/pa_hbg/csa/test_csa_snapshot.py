# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Explicit frozen-data tests; missing data fails here rather than silently skipping."""

import pytest
import torch

from examples.a5.host_build_graph.deepseek_v4_pro_csa import business_cases as variants
from examples.a5.host_build_graph.deepseek_v4_pro_csa import fixture, golden

MAPPINGS = {
    "compress_state": "state_slot_mapping",
    "inner_compress_state": "inner_state_slot_mapping",
    "kv_cache": "ori_slot_mapping",
    "cmp_kv": "cmp_slot_mapping",
    "idx_kv_cache": "idx_slot_mapping",
    "idx_kv_scale": "idx_slot_mapping",
}


@pytest.fixture(scope="module")
def snapshot():
    fixture.verify_snapshot()
    inputs = {
        arg["name"]: fixture.load_tensor(arg["name"], "in")
        for arg in fixture.manifest()["arguments"]
        if arg["direction"] != "OUT"
    }
    expected = {name: fixture.load_tensor(name, "out") for name in golden.comparators()}
    return inputs, expected


def compare(name, actual, expected, inputs):
    return golden.comparators()[name](
        actual[name],
        expected[name],
        actual_outputs=actual,
        expected_outputs=expected,
        inputs=inputs,
        rtol=1e-2,
        atol=1e-2,
    )


def test_frozen_business_has_all_four_position_regimes_and_two_compression_writes(snapshot):
    inputs, expected = snapshot
    assert inputs["position_ids"].tolist() == [8192, 8193, 0, 1, 2, 3, 3, 4]
    assert inputs["kv_seq_lens"].tolist() == [8194, 2, 4, 5]
    assert inputs["cmp_slot_mapping"].tolist() == [-1, -1, -1, -1, -1, 256, 384, -1]
    assert inputs["idx_slot_mapping"].tolist() == inputs["cmp_slot_mapping"].tolist()
    for name, mapping in MAPPINGS.items():
        size = inputs[name].shape[0] * inputs[name].shape[1]
        rows = inputs[mapping]
        untouched = torch.ones(size, dtype=torch.bool)
        untouched[rows[rows >= 0]] = False
        before, after = inputs[name].reshape(size, -1), expected[name].reshape(size, -1)
        # FP8 indexing is byte-exact; compressor untouched rows legitimately contain -Inf.
        assert torch.equal(before.view(torch.uint8)[untouched], after.view(torch.uint8)[untouched])


@pytest.mark.parametrize("name,params", variants.VARIANTS, ids=[name for name, _ in variants.VARIANTS])
def test_real_seven_goldens_survive_consistent_physical_renaming(snapshot, name, params):
    inputs, expected = snapshot
    renamed_inputs = variants.transform_inputs(inputs, params)
    renamed_outputs = variants.transform_outputs(expected, params)
    golden.validate_outputs(renamed_outputs, renamed_outputs, renamed_inputs)


@pytest.mark.parametrize(
    "table,mapping,pool",
    [
        ("compress_state_block_table", "state_slot_mapping", "compress_state"),
        ("inner_compress_state_block_table", "inner_state_slot_mapping", "inner_compress_state"),
    ],
)
def test_cross_request_state_alias_makes_arbitrary_batch_permutation_invalid(snapshot, table, mapping, pool):
    inputs, expected = snapshot
    assert inputs[table][0, 8192 // 4] == inputs[table][2, 0] == 2
    assert inputs[mapping][:2].tolist() == [8, 9]
    before = inputs[pool].reshape(260, -1)[8:10]
    after = expected[pool].reshape(260, -1)[8:10]
    assert not torch.equal(before, after)


@pytest.mark.parametrize("name", MAPPINGS)
@pytest.mark.parametrize("where", ["first_written", "last_written", "first_unmapped", "last_unmapped"])
def test_real_oracle_rejects_a_lost_store_or_stray_cache_write(snapshot, name, where):
    inputs, expected = snapshot
    actual = dict(expected, **{name: expected[name].clone()})
    tensor = actual[name]
    rows = tensor.reshape(tensor.shape[0] * tensor.shape[1], -1)
    selected = sorted(set(int(row) for row in inputs[MAPPINGS[name]] if row >= 0))
    untouched = sorted(set(range(len(rows))) - set(selected))
    row = {
        "first_written": selected[0],
        "last_written": selected[-1],
        "first_unmapped": untouched[0],
        "last_unmapped": untouched[-1],
    }[where]
    rows[row].fill_(96)
    ok, detail = compare(name, actual, expected, inputs)
    assert not ok, (name, row)
    if "unmapped" in where:
        assert "unmapped" in detail


@pytest.mark.parametrize("name", list(golden.comparators()))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_real_oracle_rejects_nonfinite_mapped_values(snapshot, name, value):
    inputs, expected = snapshot
    actual = dict(expected, **{name: expected[name].clone()})
    tensor = actual[name]
    if name == "x_out":
        tensor.flatten()[0] = value
    else:
        row = next(int(row) for row in inputs[MAPPINGS[name]] if row >= 0)
        tensor.reshape(tensor.shape[0] * tensor.shape[1], -1)[row, 0] = value
    ok, detail = compare(name, actual, expected, inputs)
    assert not ok, detail


@pytest.mark.parametrize("token", range(8))
def test_real_final_output_checks_every_token_not_just_aggregate_similarity(snapshot, token):
    inputs, expected = snapshot
    actual = dict(expected, x_out=expected["x_out"].clone())
    actual["x_out"][token].fill_(1000)
    assert not compare("x_out", actual, expected, inputs)[0]


@pytest.mark.parametrize("name", [name for name in MAPPINGS if name != "idx_kv_scale"])
def test_replaying_final_state_without_reset_is_not_the_frozen_input(snapshot, name):
    inputs, expected = snapshot
    initial = inputs[name].view(torch.uint8)
    final = expected[name].view(torch.uint8)
    assert torch.count_nonzero(initial != final).item() > 0, f"{name} does not exercise INOUT reset"


def test_frozen_index_scale_alone_cannot_prove_that_a_store_executed(snapshot):
    inputs, expected = snapshot
    assert torch.equal(inputs["idx_kv_scale"], expected["idx_kv_scale"])


@pytest.mark.parametrize("name,params", [(n, p) for n, p in variants.VARIANTS if p.get("poison_scale")])
def test_scale_poison_exposes_missing_store_with_the_original_golden(snapshot, name, params):
    inputs, expected = snapshot
    renamed = variants.transform_inputs(inputs, params)
    expected = variants.transform_outputs(expected, params)
    actual = dict(expected, idx_kv_scale=renamed["idx_kv_scale"])
    assert not compare("idx_kv_scale", actual, expected, renamed)[0]

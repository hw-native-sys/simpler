#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Adapt a generated Qwen decode callable to host_build_graph."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bin_manifest(cache: Path) -> dict[str, str]:
    return {path.name: _sha256(path) for path in sorted(cache.glob("incore_*.bin"))}


def verify_hbg_artifact(root: Path) -> dict:
    manifest = json.loads((root / "hbg_artifact_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "simpler-hbg-pure-artifact-v1" or manifest.get("runtime") != "host_build_graph":
        raise ValueError("unsupported HBG artifact manifest")
    if not 0 < int(manifest["graph_definition_task_count_per_layer"]) < 1024:
        raise ValueError("HBG Definition task count exceeds the supported contract")
    child = root / "next_levels" / "decode_fwd"
    paths = {
        "distributed_meta_sha256": root / "distributed_meta.json",
        "orchestration_cpp_sha256": child / "orchestration" / "decode_fwd.cpp",
        "orchestration_so_sha256": child / "orchestration" / "decode_fwd.so",
    }
    for key, path in paths.items():
        if _sha256(path) != manifest[key]:
            raise ValueError(f"HBG artifact checksum mismatch: {path.name}")
    bins = _bin_manifest(child / "cache")
    if len(bins) not in {39, 40, 41} or bins != manifest["source_incore_bins"]:
        raise ValueError("HBG in-core binary checksum mismatch")
    return manifest


def _validate_param_names(param_names: list[str]) -> None:
    required = {"out", "embed_weight", "sampled_ids_in", "sampled_ids", "next_hidden"}
    if len(param_names) not in {25, 26} or not required.issubset(param_names):
        raise RuntimeError(f"expected the Qwen decode ABI, got {param_names}")
    has_host_output = "sampled_ids_host" in param_names
    if has_host_output != (len(param_names) == 26):
        raise RuntimeError(f"sampled_ids_host does not match the {len(param_names)}-argument ABI")


def _add_sample_dependency(path: Path) -> None:
    """Order embedding, final RMSNorm, LM head, and greedy sampling tasks."""
    source = path.read_text(encoding="utf-8")
    if "hbg_token_id_offsets" in source:
        return
    embed_loop = re.search(
        r"(?m)^(?P<indent>\s*)for \(int64_t b_inline2035 = 0; "
        r"b_inline2035 < batch_inline2036; b_inline2035 \+= 1\) \{",
        source,
    )
    if embed_loop is None:
        raise RuntimeError("cannot locate generated token embedding loop")
    source = (
        source[: embed_loop.start()]
        + f"{embed_loop.group('indent')}TaskId hbg_embed_tids[16]; for (int64_t i = 0; i < 16; ++i) hbg_embed_tids[i] = TaskId::invalid();\n"
        + source[embed_loop.start() :]
    )
    token_embed_old = (
        "                CoreTaskArgs params_t0;\n"
        "                params_t0.add_input(ext_sampled_ids_in);\n"
        "                params_t0.add_output(ext_next_hidden);\n"
        "                params_t0.add_input(ext_embed_weight);\n"
        "                params_t0.add_scalar(b_inline2035);\n"
    )
    token_embed_new = (
        "                uint32_t hbg_token_id_offsets[2] = {static_cast<uint32_t>(b_inline2035), 0};\n"
        "                uint32_t hbg_token_id_shapes[2] = {1, 8};\n"
        "                TaskTensor hbg_token_id = ext_sampled_ids_in.view(hbg_token_id_shapes, hbg_token_id_offsets);\n"
        "                uint32_t hbg_embed_out_offsets[2] = {static_cast<uint32_t>(b_inline2035), 0};\n"
        "                uint32_t hbg_embed_out_shapes[2] = {1, 5120};\n"
        "                TaskTensor hbg_embed_out = ext_next_hidden.view(hbg_embed_out_shapes, hbg_embed_out_offsets);\n"
        "                CoreTaskArgs params_t0;\n"
        "                params_t0.add_input(hbg_token_id);\n"
        "                params_t0.add_output(hbg_embed_out);\n"
        "                params_t0.add_input(ext_embed_weight);\n"
        "                params_t0.add_scalar(0);\n"
    )
    if source.count(token_embed_old) != 1:
        raise RuntimeError("cannot locate generated token embed args")
    source = source.replace(token_embed_old, token_embed_new, 1)
    embed_submit = "                rt_submit_aiv_task(0, params_t0);\n"
    if source.count(embed_submit) != 1:
        raise RuntimeError("cannot locate generated token embedding submission")
    source = source.replace(
        embed_submit,
        "                TaskOutputTensors task_0_outs = rt_submit_aiv_task(0, params_t0);\n"
        "                hbg_embed_tids[b_inline2035] = task_0_outs.task_id();\n",
        1,
    )
    copy_marker = (
        "                        params_t1.add_scalar(chunk_rows_inline1463);\n"
        "                        params_t1.set_allow_early_resolve(true);\n"
    )
    if source.count(copy_marker) != 1:
        raise RuntimeError("cannot locate generated copy_hidden submission")
    copy_extra = (
        "                        TaskId hbg_embed_deps[16];\n"
        "                        uint32_t hbg_embed_dep_count = 0;\n"
        "                        for (int64_t i = 0; i < batch_inline2036; ++i) hbg_embed_deps[hbg_embed_dep_count++] = hbg_embed_tids[i];\n"
        "                        params_t1.set_dependencies(hbg_embed_deps, hbg_embed_dep_count);\n"
    )
    source = source.replace(copy_marker, copy_marker + copy_extra, 1)
    chunk_loop = re.search(
        r"(?m)^(?P<indent>\s*)for \(int64_t chunk_idx_inline\d+ = 0; "
        r"chunk_idx_inline\d+ < num_chunks_inline\d+; chunk_idx_inline\d+ \+= 1\) \{",
        source,
    )
    if chunk_loop is None:
        raise RuntimeError("cannot locate generated decode chunk loop")
    declaration = (
        f"{chunk_loop.group('indent')}TaskId hbg_final_rms_tid = TaskId::invalid();\n"
        f"{chunk_loop.group('indent')}TaskId hbg_lm_head_tid = TaskId::invalid();\n"
    )
    source = source[: chunk_loop.start()] + declaration + source[chunk_loop.start() :]
    final_rms = re.search(
        r"(?m)^(?P<indent>\s*)TaskOutputTensors (?P<task>\w+) = "
        r"rt_submit_aiv_task\(36, params_t35\);\n",
        source,
    )
    if final_rms is None:
        raise RuntimeError("cannot locate generated final_rmsnorm task")
    source = (
        source[: final_rms.end()]
        + f"{final_rms.group('indent')}hbg_final_rms_tid = {final_rms.group('task')}.task_id();\n"
        + source[final_rms.end() :]
    )
    lm_head = re.search(
        r"(?m)^(?P<indent>\s*)TaskOutputTensors (?P<task>\w+) = "
        r"rt_submit_aic_task\(37, params_t36\);\n",
        source,
    )
    if lm_head is None:
        raise RuntimeError("cannot locate generated lm_head task")
    source = (
        source[: lm_head.start()]
        + f"{lm_head.group('indent')}params_t36.set_dependencies(&hbg_final_rms_tid, 1);\n"
        + source[lm_head.start() :]
    )
    lm_head = re.search(
        r"(?m)^(?P<indent>\s*)TaskOutputTensors (?P<task>\w+) = "
        r"rt_submit_aic_task\(37, params_t36\);\n",
        source,
    )
    if lm_head is None:
        raise RuntimeError("cannot locate generated lm_head after dependency insertion")
    source = (
        source[: lm_head.end()]
        + f"{lm_head.group('indent')}hbg_lm_head_tid = {lm_head.group('task')}.task_id();\n"
        + source[lm_head.end() :]
    )
    sample = re.search(
        r"(?m)^(?P<indent>\s*)params_t37\.add_scalar\((?P<row>[^)]+)\);\n"
        r"(?P=indent)rt_submit_aiv_task\(38, params_t37\);",
        source,
    )
    if sample is None:
        raise RuntimeError("cannot locate generated greedy_sample submission")
    replacement = (
        f"{sample.group('indent')}params_t37.add_scalar({sample.group('row')});\n"
        f"{sample.group('indent')}params_t37.set_dependencies(&hbg_lm_head_tid, 1);\n"
        f"{sample.group('indent')}rt_submit_aiv_task(38, params_t37);"
    )
    source = source[: sample.start()] + replacement + source[sample.end() :]
    path.write_text(source, encoding="utf-8")


def _normalize_tensor_type(path: Path) -> int:
    source = path.read_text(encoding="utf-8")
    normalized, count = re.subn(r"\bTaskTensor\b", "Tensor", source)
    path.write_text(normalized, encoding="utf-8")
    return count


def _brace_end(source: str, open_brace: int) -> int:
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise RuntimeError("unterminated generated C++ block")


def _definition_task_count(source: str) -> int:
    start = source.find("static void decode_layer_definition")
    if start < 0:
        raise RuntimeError("cannot locate emitted layer Definition")
    open_brace = source.find("{", start)
    end = _brace_end(source, open_brace)
    body = source[start : end + 1]
    pairs: dict[int, int] = {}
    stack: list[int] = []
    for index, char in enumerate(body):
        if char == "{":
            stack.append(index)
        elif char == "}" and stack:
            pairs[stack.pop()] = index
    constants = {
        match.group(1): int(match.group(2))
        for match in re.finditer(r"\b(?:int64_t|int32_t)\s+(\w+)\s*=\s*(-?\d+)", body)
    }
    loops: list[tuple[int, int, int]] = []
    loop_pattern = re.compile(r"for\s*\([^;]+?=\s*(-?\d+)\s*;\s*\w+\s*<\s*([A-Za-z0-9_]+)\s*;[^)]*\)\s*\{")
    for match in loop_pattern.finditer(body):
        bound = constants.get(match.group(2))
        if bound is None:
            try:
                bound = int(match.group(2))
            except ValueError as error:
                raise RuntimeError(f"cannot resolve loop bound {match.group(2)}") from error
        open_loop = match.end() - 1
        loops.append((match.start(), pairs[open_loop], max(0, bound - int(match.group(1)))))
    task_count = 0
    for submit in re.finditer(r"\brt_submit_(?:dummy_task|aic_task|aiv_task|task)\s*\(", body):
        multiplier = 1
        for loop_start, loop_end, trip_count in loops:
            if loop_start < submit.start() < loop_end:
                multiplier *= trip_count
        task_count += multiplier
    return task_count


def _last_group(pattern: str, source: str) -> str:
    matches = re.findall(pattern, source)
    if not matches:
        raise RuntimeError(f"generated source does not match {pattern}")
    return matches[-1]


def _remove_layer_output_allocations(body: str, layer_hidden: str, next_normed: str) -> str:
    for name in (layer_hidden, next_normed):
        body, count = re.subn(
            rf"^[ \t]*uint32_t {re.escape(name)}_ci_shapes\[[^\n]+\n"
            rf"^[ \t]*TensorCreateInfo {re.escape(name)}_ci\([^\n]+\n",
            "",
            body,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise RuntimeError(f"cannot remove generated create-info for {name}")

    allocation = re.search(
        rf"(?P<indent>[ \t]*)TaskOutputTensors (?P<alloc>alloc_\d+) = alloc_tensors\((?P<args>[^;]*{re.escape(layer_hidden)}_ci[^;]*)\);",
        body,
    )
    if allocation is None:
        raise RuntimeError("cannot locate generated layer output allocation")
    alloc_name = allocation.group("alloc")
    arguments = [value.strip() for value in allocation.group("args").split(",")]
    filtered = [value for value in arguments if value not in (f"{layer_hidden}_ci", f"{next_normed}_ci")]
    replacement = f"{allocation.group('indent')}TaskOutputTensors {alloc_name} = alloc_tensors({', '.join(filtered)});"
    body = body[: allocation.start()] + replacement + body[allocation.end() :]
    body = re.sub(
        rf"^[ \t]*const TaskTensor& (?:{re.escape(layer_hidden)}|{re.escape(next_normed)}) = {re.escape(alloc_name)}\.get_ref\(\d+\);\n",
        "",
        body,
        flags=re.MULTILINE,
    )

    def reindex(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index < 2:
            raise RuntimeError("unexpected retained layer allocation output index")
        return f"{alloc_name}.get_ref({index - 2})"

    body = re.sub(rf"{re.escape(alloc_name)}\.get_ref\((\d+)\)", reindex, body)
    return body


def _move_layer_allocations_to_scratch(body: str) -> str:
    """Replace generated intermediate alloc_tensors calls with ScratchArena views."""
    dtype_by_ci = {
        match.group(1): match.group(2)
        for match in re.finditer(
            r"TensorCreateInfo (?P<name>[A-Za-z0-9_]+)_ci\([^;]+, DataType::(?P<dtype>[A-Z0-9_]+)\);",
            body,
        )
    }
    for alloc_name in ("alloc_2", "alloc_3"):
        match = re.search(
            rf"(?P<indent>[ \t]*)TaskOutputTensors {alloc_name} = alloc_tensors\((?P<args>[^;]+)\);",
            body,
        )
        if match is None:
            raise RuntimeError(f"cannot locate {alloc_name} for scratch conversion")
        lines = []
        for ci in (value.strip() for value in match.group("args").split(",")):
            if not ci.endswith("_ci"):
                raise RuntimeError(f"unexpected {alloc_name} argument {ci}")
            tensor_name = ci[:-3]
            dtype = dtype_by_ci.get(tensor_name)
            if dtype is None:
                raise RuntimeError(f"cannot find dtype for {ci}")
            selected_arena = "bf16_arena" if dtype == "BFLOAT16" else "fp32_arena"
            lines.append(f"{match.group('indent')}TaskTensor {tensor_name} = {selected_arena}.allocate({ci});")
        body = body[: match.start()] + "\n".join(lines) + body[match.end() :]
        body = re.sub(
            rf"^[ \t]*const TaskTensor& ([A-Za-z0-9_]+) = {alloc_name}\.get_ref\(\d+\);\n",
            "",
            body,
            flags=re.MULTILINE,
        )
    return body


def _outline_per_layer_definitions(orchestration_path: Path) -> int:
    """Outline the generated layer loop into bounded HBG Definitions."""
    source = orchestration_path.read_text(encoding="utf-8")
    loop_match = re.search(
        r"for \(int64_t (?P<layer>layer_idx_inline\d+) = 0; (?P=layer) < 40; (?P=layer) \+= 1\) \{",
        source,
    )
    if loop_match is None:
        raise RuntimeError("cannot locate generated 40-layer loop")
    layer_index = loop_match.group("layer")
    loop_open = source.find("{", loop_match.start())
    loop_close = _brace_end(source, loop_open)
    scope_start = source.find("SIMPLER_SCOPE()", loop_open, loop_close)
    if scope_start < 0:
        raise RuntimeError("generated layer loop is missing SIMPLER_SCOPE()")
    scope_open = source.find("{", scope_start, loop_close)
    scope_close = _brace_end(source, scope_open)
    if source[scope_close + 1 : loop_close].strip():
        raise RuntimeError("generated layer loop contains work outside its scope")
    prefix = source[: loop_match.start()]
    layer_body = source[scope_open + 1 : scope_close]

    cur = _last_group(r"TaskTensor (cur_inline\d+__rv_v7) =", prefix)
    normed = _last_group(r"TaskTensor (normed_inline\d+__rv_v5) =", prefix)
    prev_out = _last_group(r"TaskId (prev_out_tid_inline\d+)\[1\]", prefix)
    prev_normed = _last_group(r"TaskId (prev_normed_tid_inline\d+)\[1\]", prefix)
    chunk_row0 = _last_group(r"int64_t (chunk_row0_inline\d+) =", prefix)
    chunk_rows = _last_group(r"int64_t (chunk_rows_inline\d+) =", prefix)
    chunk_rows_i32 = _last_group(r"int32_t (chunk_rows_i32_inline\d+) =", prefix)
    layer_hidden = _last_group(r"TensorCreateInfo (layer_next_hidden_inline\d+)_ci", layer_body)
    next_normed = _last_group(r"TensorCreateInfo (next_normed_inline\d+)_ci", layer_body)
    next_gamma = _last_group(r"int64_t (next_gamma_idx_inline\d+) =", layer_body)
    hidden_base = _last_group(r"int64_t (layer_hidden_base_inline\d+) =", layer_body)
    inter_base = _last_group(r"int64_t (layer_inter_base_inline\d+) =", layer_body)
    cache_base = _last_group(r"int64_t (layer_cache_base_token_rows_inline\d+) =", layer_body)
    attention_cache_base = _last_group(r"int64_t (cache_base_inline\d+) =", layer_body)
    q_norm_view = _last_group(r"TaskTensor (q_norm_w_inline\d+) =", layer_body)
    k_norm_view = _last_group(r"TaskTensor (k_norm_w_inline\d+) =", layer_body)

    layer_body = _remove_layer_output_allocations(layer_body, layer_hidden, next_normed)
    layer_body = _move_layer_allocations_to_scratch(layer_body)
    for index, external in enumerate(
        (
            "ext_input_rms_weight",
            "ext_wq",
            "ext_wk",
            "ext_wv",
            "ext_q_norm_weight",
            "ext_k_norm_weight",
            "ext_seq_lens",
            "ext_block_table",
            "ext_slot_mapping",
            "ext_rope_cos",
            "ext_rope_sin",
            "ext_k_cache",
            "ext_v_cache",
            "ext_wo",
            "ext_w_gate",
            "ext_w_up",
            "ext_w_down",
            "ext_post_rms_weight",
        )
    ):
        layer_body = layer_body.replace(f"orch_args.tensor({index}).ref()", external)

    for name in (next_gamma, hidden_base, inter_base, cache_base, attention_cache_base):
        layer_body = re.sub(rf"^[ \t]*int64_t {re.escape(name)} = [^\n]+\n", "", layer_body, flags=re.MULTILINE)
    for view, external in ((q_norm_view, "ext_q_norm_weight"), (k_norm_view, "ext_k_norm_weight")):
        layer_body = re.sub(
            rf"^[ \t]*uint32_t {re.escape(view)}_offsets\[2\].*?^[ \t]*TaskTensor {re.escape(view)} = [^\n]+\n",
            "",
            layer_body,
            flags=re.MULTILINE | re.DOTALL,
        )
        layer_body = layer_body.replace(view, external)
    scalar_sources = (hidden_base, inter_base, cache_base, attention_cache_base, layer_index, next_gamma)
    for name in scalar_sources:
        layer_body = re.sub(
            rf"(params_[A-Za-z0-9_]+\.add_scalar\(){re.escape(name)}(\);)",
            r"\g<1>0\g<2>",
            layer_body,
        )
    layer_body = re.sub(
        r"(int64_t t__tmp_v\d+ = \(int64_t\)ext_k_cache\.shapes\[0\];)",
        lambda match: match.group(1).replace("ext_k_cache.shapes[0]", "(ext_k_cache.shapes[0] * 40)"),
        layer_body,
        count=1,
    )
    if layer_index in layer_body:
        raise RuntimeError("per-layer Definition retains host-side layer-index computation")
    carry_start = re.search(rf"^[ \t]*TaskTensor {re.escape(cur.split('__')[0])}__ssa_v8 =", layer_body, re.MULTILINE)
    if carry_start is None:
        raise RuntimeError("cannot locate generated layer carry assignments")
    carry_end_matches = list(re.finditer(rf"^[ \t]*{re.escape(normed)} = [^\n]+\n", layer_body, re.MULTILINE))
    if not carry_end_matches:
        raise RuntimeError("cannot locate final generated norm carry assignment")
    carry_end = carry_end_matches[-1].end()
    layer_body = layer_body[: carry_start.start()] + layer_body[carry_end:]

    externals = (
        "ext_input_rms_weight",
        "ext_wq",
        "ext_wk",
        "ext_wv",
        "ext_q_norm_weight",
        "ext_k_norm_weight",
        "ext_seq_lens",
        "ext_block_table",
        "ext_slot_mapping",
        "ext_rope_cos",
        "ext_rope_sin",
        "ext_k_cache",
        "ext_v_cache",
        "ext_wo",
        "ext_w_gate",
        "ext_w_up",
        "ext_w_down",
        "ext_post_rms_weight",
    )
    boundary_names = (
        cur,
        normed,
        layer_hidden,
        next_normed,
        *externals,
        "score_transfer",
        "probability_transfer",
        "pv_transfer",
        "ffts_workspace",
        "bf16_scratch",
        "fp32_scratch",
    )
    definition = [
        "class ScratchArena {",
        "public:",
        "    explicit ScratchArena(const TaskTensor& storage) : storage_(storage) {}",
        "    TaskTensor allocate(const TensorCreateInfo& create_info) {",
        "        always_assert(create_info.dtype == storage_.dtype);",
        "        const uint64_t element_size = get_element_size(storage_.dtype);",
        "        const uint64_t alignment = 1024 / element_size;",
        "        cursor_ = (cursor_ + alignment - 1) / alignment * alignment;",
        "        const uint64_t elements = create_info.buffer_size_bytes() / element_size;",
        "        const uint64_t aligned = (elements + alignment - 1) / alignment * alignment;",
        "        always_assert((cursor_ + aligned) * element_size <= storage_.buffer.size);",
        "        TaskTensor tensor;",
        "        init_tensor_from_create_info(tensor, create_info, reinterpret_cast<void *>(static_cast<uintptr_t>(storage_.buffer.addr)), storage_.buffer.size);",
        "        tensor.owner_task_id = storage_.owner_task_id;",
        "        tensor.start_offset = storage_.start_offset + cursor_;",
        "        cursor_ += aligned;",
        "        return tensor;",
        "    }",
        "private:",
        "    const TaskTensor& storage_;",
        "    uint64_t cursor_{0};",
        "};",
        "",
        "static void decode_layer_definition(const GraphTaskArgs& orch_args) {",
    ]
    definition.extend(
        f"    const TaskTensor& {name} = orch_args.tensor({index}).ref();" for index, name in enumerate(boundary_names)
    )
    definition.extend(
        [
            f"    int64_t {chunk_row0} = 0;",
            f"    int64_t {chunk_rows} = 16;",
            f"    int32_t {chunk_rows_i32} = 16;",
            f"    TaskId {prev_out}[1] = {{{cur}.owner_task_id}};",
            f"    TaskId {prev_normed}[1] = {{{normed}.owner_task_id}};",
            "    TaskId scratch_ready[1] = {TaskId::invalid()};",
            "    ScratchArena bf16_arena(bf16_scratch);",
            "    ScratchArena fp32_arena(fp32_scratch);",
            layer_body,
            "}",
            "",
        ]
    )

    storage = [
        "                uint32_t hbg_hidden_shapes[2] = {16, 5120};",
        "                TensorCreateInfo hbg_hidden_ci(hbg_hidden_shapes, 2, DataType::FLOAT32);",
        "                uint32_t hbg_norm_shapes[2] = {16, 5120};",
        "                TensorCreateInfo hbg_norm_ci(hbg_norm_shapes, 2, DataType::BFLOAT16);",
        "                TaskOutputTensors hbg_layer_storage = alloc_tensors(hbg_hidden_ci, hbg_norm_ci, hbg_hidden_ci, hbg_norm_ci);",
        "                uint32_t hbg_bf16_scratch_shapes[1] = {1048576};",
        "                TensorCreateInfo hbg_bf16_scratch_ci(hbg_bf16_scratch_shapes, 1, DataType::BFLOAT16);",
        "                uint32_t hbg_fp32_scratch_shapes[1] = {2097152};",
        "                TensorCreateInfo hbg_fp32_scratch_ci(hbg_fp32_scratch_shapes, 1, DataType::FLOAT32);",
        "                TaskOutputTensors hbg_scratch_storage = alloc_tensors(hbg_bf16_scratch_ci, hbg_fp32_scratch_ci);",
        "                const TaskTensor& hbg_bf16_scratch = hbg_scratch_storage.get_ref(0);",
        "                const TaskTensor& hbg_fp32_scratch = hbg_scratch_storage.get_ref(1);",
    ]
    replacement = [
        *storage,
        "                auto hbg_layer_view = [](const TaskTensor& tensor, int64_t layer, int64_t rows) {",
        "                    uint32_t offsets[2] = {static_cast<uint32_t>(layer * rows), 0};",
        "                    uint32_t shapes[2] = {static_cast<uint32_t>(rows), tensor.shapes[1]};",
        "                    return tensor.view(shapes, offsets);",
        "                };",
        f"                for (int64_t {layer_index} = 0; {layer_index} < 40; {layer_index} += 1) {{",
        "                    SIMPLER_SCOPE() {",
    ]
    replacement.extend(
        [
            f"                        const TaskTensor& {layer_hidden} = hbg_layer_storage.get_ref(({layer_index} & 1) * 2);",
            f"                        const TaskTensor& {next_normed} = hbg_layer_storage.get_ref(({layer_index} & 1) * 2 + 1);",
            f"                        uint32_t hbg_q_norm_offsets[2] = {{static_cast<uint32_t>({layer_index}), 0}};",
            "                        uint32_t hbg_norm_shapes[2] = {1, 128};",
            "                        TaskTensor hbg_q_norm = ext_q_norm_weight.view(hbg_norm_shapes, hbg_q_norm_offsets);",
            f"                        uint32_t hbg_k_norm_offsets[2] = {{static_cast<uint32_t>({layer_index}), 0}};",
            "                        TaskTensor hbg_k_norm = ext_k_norm_weight.view(hbg_norm_shapes, hbg_k_norm_offsets);",
            f"                        TaskTensor hbg_graph_normed = {normed};",
            f"                        hbg_graph_normed.owner_task_id = {prev_normed}[0];",
            f"                        TaskTensor hbg_input_rms = hbg_layer_view(ext_input_rms_weight, std::min<int64_t>({layer_index} + 1, 39), 1);",
            f"                        TaskTensor hbg_wq = hbg_layer_view(ext_wq, {layer_index}, 5120);",
            f"                        TaskTensor hbg_wk = hbg_layer_view(ext_wk, {layer_index}, 5120);",
            f"                        TaskTensor hbg_wv = hbg_layer_view(ext_wv, {layer_index}, 5120);",
            f"                        TaskTensor hbg_k_cache = hbg_layer_view(ext_k_cache, {layer_index}, ext_k_cache.shapes[0] / 40);",
            f"                        TaskTensor hbg_v_cache = hbg_layer_view(ext_v_cache, {layer_index}, ext_v_cache.shapes[0] / 40);",
            f"                        TaskTensor hbg_wo = hbg_layer_view(ext_wo, {layer_index}, 5120);",
            f"                        TaskTensor hbg_wgate = hbg_layer_view(ext_w_gate, {layer_index}, 5120);",
            f"                        TaskTensor hbg_wup = hbg_layer_view(ext_w_up, {layer_index}, 5120);",
            f"                        TaskTensor hbg_wdown = hbg_layer_view(ext_w_down, {layer_index}, 17408);",
            f"                        TaskTensor hbg_post_rms = hbg_layer_view(ext_post_rms_weight, {layer_index}, 1);",
            "                        GraphTaskArgs layer_args;",
        ]
    )
    replacement.extend(
        [
            f"                        layer_args.add_input({cur});",
            "                        layer_args.add_input(hbg_graph_normed);",
            f"                        layer_args.add_inout({layer_hidden});",
            f"                        layer_args.add_inout({next_normed});",
        ]
    )
    caller_names = {
        0: "hbg_input_rms",
        1: "hbg_wq",
        2: "hbg_wk",
        3: "hbg_wv",
        4: "hbg_q_norm",
        5: "hbg_k_norm",
        11: "hbg_k_cache",
        12: "hbg_v_cache",
        13: "hbg_wo",
        14: "hbg_wgate",
        15: "hbg_wup",
        16: "hbg_wdown",
        17: "hbg_post_rms",
    }
    for index, name in enumerate(externals):
        method = "add_inout" if index in (11, 12) else "add_input"
        replacement.append(f"                        layer_args.{method}({caller_names.get(index, name)});")
    replacement.extend(
        [
            "                        layer_args.add_inout(score_transfer);",
            "                        layer_args.add_inout(probability_transfer);",
            "                        layer_args.add_inout(pv_transfer);",
            "                        layer_args.add_inout(ffts_workspace);",
            "                        layer_args.add_inout(hbg_bf16_scratch);",
            "                        layer_args.add_inout(hbg_fp32_scratch);",
            "                        GraphSubmitResult hbg_layer_result = rt_submit_graph(+decode_layer_definition, layer_args);",
            f"                        TaskTensor hbg_layer_output = {layer_hidden};",
            "                        hbg_layer_output.owner_task_id = hbg_layer_result.task_id;",
            f"                        TaskTensor hbg_normed_output = {next_normed};",
            "                        hbg_normed_output.owner_task_id = hbg_layer_result.task_id;",
            f"                        {cur} = hbg_layer_output;",
            f"                        {normed} = hbg_normed_output;",
            "                    }",
            "                }",
        ]
    )
    source = source[: loop_match.start()] + "\n".join(replacement) + source[loop_close + 1 :]
    entry_marker = '__attribute__((visibility("default")))\nvoid aicpu_orchestration_entry'
    entry_index = source.find(entry_marker)
    if entry_index < 0:
        raise RuntimeError("cannot locate generated entry for layer Definition insertion")
    source = source[:entry_index] + "\n".join(definition) + source[entry_index:]
    orchestration_path.write_text(source, encoding="utf-8")
    return 1


def _adapt_child_callable(output_dir: Path, external_argument_count: int) -> dict[str, str | bool | int]:
    """Rebind the generated chip callable and outline its decoder layer."""
    child = output_dir / "next_levels" / "decode_fwd"
    config_path = child / "kernel_config.py"
    config = config_path.read_text(encoding="utf-8")
    old_runtime = '\t"runtime": "tensormap_and_ringbuffer",'
    new_runtime = '\t"runtime": "host_build_graph",'
    if config.count(old_runtime) != 1:
        raise RuntimeError("expected exactly one generated TMR child runtime binding")
    config_path.write_text(config.replace(old_runtime, new_runtime), encoding="utf-8")
    orchestration_path = child / "orchestration" / "decode_fwd.cpp"
    definition_count = _outline_per_layer_definitions(orchestration_path)
    definition_record_replay = True
    _add_sample_dependency(orchestration_path)
    _normalize_tensor_type(orchestration_path)
    for kernel_path in sorted((child / "kernels").rglob("*.cpp")):
        _normalize_tensor_type(kernel_path)

    return {
        "child_runtime": "host_build_graph",
        "graph_definition_record_replay": definition_record_replay,
        "graph_definition_boundary_tensors": 28,
        "graph_definition_count": definition_count,
        "graph_definition_invocations_per_frame": 40,
        "graph_definition_boundary_scalars": 0,
        "graph_definition_task_count_per_layer": _definition_task_count(orchestration_path.read_text(encoding="utf-8")),
        "adapter": f"qwen3-14b-{external_argument_count}-arg-per-layer-hbg-v1",
        "tensor_abi": "native HBG Tensor",
    }


def main(argv=None) -> int:
    args = _parse_args(argv)
    artifact_source = args.artifact_source.resolve()
    source_manifest_path = artifact_source / "manifest.json"
    if source_manifest_path.is_file():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_decode = artifact_source / source_manifest["programs"]["decode"]["path"]
    else:
        source_manifest = None
        source_decode = artifact_source
    metadata_path = source_decode / "distributed_meta.json"
    child = source_decode / "next_levels" / "decode_fwd"
    for required in (metadata_path, child / "kernel_config.py", child / "orchestration" / "decode_fwd.cpp"):
        if not required.is_file():
            raise FileNotFoundError(required)

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    shutil.copytree(source_decode, output_dir)

    metadata_path = output_dir / "distributed_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    param_names = [param["name"].split("__ssa_", 1)[0] for param in metadata["params"]]
    _validate_param_names(param_names)
    metadata["distributed_config"].update(
        {"device_ids": [args.device_id], "runtime": "host_build_graph", "aicpu_thread_num": 0}
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    child_output_dir = output_dir / "next_levels" / "decode_fwd"
    source_bins = _bin_manifest(child_output_dir / "cache")
    if len(source_bins) not in {39, 40, 41}:
        raise RuntimeError(f"expected 39, 40, or 41 Qwen in-core binaries, got {len(source_bins)}")
    source_cpp_sha = _sha256(child_output_dir / "orchestration" / "decode_fwd.cpp")
    child_adapter = _adapt_child_callable(output_dir, len(param_names))
    source_bin_bytes = {name: (child_output_dir / "cache" / name).read_bytes() for name in source_bins}
    from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415

    compile_and_assemble(child_output_dir, args.platform)
    assembled_bins = _bin_manifest(child_output_dir / "cache")
    changed = sorted(
        name for name in set(source_bins) | set(assembled_bins) if source_bins.get(name) != assembled_bins.get(name)
    )
    # The source artifact is the frozen TMR payload. HBG assembly also emits
    # fresh binaries while compiling the orchestration, sometimes changing the
    # content-addressed names. Remove that temporary payload and restore the
    # source bytes after assembly, retaining the change list as provenance.
    cache_dir = child_output_dir / "cache"
    for path in cache_dir.glob("incore_*.bin"):
        if path.name not in source_bin_bytes:
            path.unlink()
    for name, payload in source_bin_bytes.items():
        (cache_dir / name).write_bytes(payload)
    if _bin_manifest(child_output_dir / "cache") != source_bins:
        raise RuntimeError("failed to restore frozen in-core binaries after HBG assembly")

    manifest = {
        "schema": "simpler-hbg-pure-artifact-v1",
        "runtime": "host_build_graph",
        "platform": args.platform,
        "device_id": args.device_id,
        "batch_size": 16,
        "max_seq_len": 4096,
        "layers": 40,
        "sampled_ids_pad": 8,
        "external_argument_count": len(param_names),
        "incore_bin_count": len(source_bins),
        "source_artifact": str(source_decode),
        "source_artifact_manifest_sha256": _sha256(source_manifest_path) if source_manifest is not None else None,
        "source_distributed_meta_sha256": _sha256(source_decode / "distributed_meta.json"),
        "source_orchestration_cpp_sha256": source_cpp_sha,
        "source_incore_bins": source_bins,
        "incore_bins_identical_to_tmr": True,
        "assembly_changed_incore_bins": changed,
        "incore_bins_restored_after_assembly": True,
        "orchestration_cpp_sha256": _sha256(child_output_dir / "orchestration" / "decode_fwd.cpp"),
        "orchestration_so_sha256": _sha256(child_output_dir / "orchestration" / "decode_fwd.so"),
        "distributed_meta_sha256": _sha256(metadata_path),
        "output_dir": str(output_dir),
        **child_adapter,
    }
    (output_dir / "hbg_artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

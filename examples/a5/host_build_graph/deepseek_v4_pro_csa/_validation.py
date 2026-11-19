# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frozen PyPTO-Lib CSA comparison primitives; provenance is in manifest.json."""

from __future__ import annotations

from collections.abc import Callable

import torch


def _valid_prefix(
    actual: torch.Tensor,
    expected: torch.Tensor,
    valid_rows: int | None,
    valid_axis: int,
    zero_tail: bool,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Restrict a comparison to the leading ``valid_rows`` along ``valid_axis``.

    Packed kernels carry a padded token buffer where only the leading rows are
    active; the inactive tail would otherwise dilute a ratio-based verdict.
    Returns ``(actual, expected, error)``; ``error`` is non-empty only when
    ``zero_tail`` is set and the dropped tail is not all zeros.
    """
    if valid_rows is None:
        return actual, expected, ""
    total = actual.shape[valid_axis]
    if not 0 <= valid_rows <= total:
        return actual, expected, (f"    valid_rows={valid_rows} out of range for axis {valid_axis} of length {total}")
    if zero_tail:
        tail = actual.narrow(valid_axis, valid_rows, total - valid_rows)
        tail_nonzero = int(tail.count_nonzero().item())
        if tail_nonzero:
            return actual, expected, (f"    inactive tail contains {tail_nonzero} nonzero values")
    return (
        actual.narrow(valid_axis, 0, valid_rows),
        expected.narrow(valid_axis, 0, valid_rows),
        "",
    )


def _nonfinite_error(actual: torch.Tensor, expected: torch.Tensor) -> str:
    """Describe non-finite values on either side of a comparison."""
    actual_nan_count = int(torch.isnan(actual).sum().item())
    actual_inf_count = int(torch.isinf(actual).sum().item())
    expected_nan_count = int(torch.isnan(expected).sum().item())
    expected_inf_count = int(torch.isinf(expected).sum().item())
    if not (actual_nan_count or actual_inf_count or expected_nan_count or expected_inf_count):
        return ""
    return (
        "    illegal values in comparison: "
        f"actual: NaN={actual_nan_count} Inf={actual_inf_count}; "
        f"expected: NaN={expected_nan_count} Inf={expected_inf_count}"
    )


def ratio_allclose(
    atol: float | None = None,
    rtol: float | None = None,
    max_error_ratio: float = 0.005,
    max_show: int = 10,
    valid_rows: int | None = None,
    valid_axis: int = 0,
    zero_tail: bool = False,
    ignore_nan: bool = False,
) -> Callable:
    """Return an allclose-style comparator that tolerates a bounded outlier ratio.

    Mirrors ``torch.allclose``'s per-point tolerance rule but, instead of
    requiring every point to pass, allows up to ``max_error_ratio`` of points
    to exceed tolerance:

        tolerance = atol + rtol * |expected|
        pass iff (count of points where |actual - expected| > tolerance) / numel
                 <= max_error_ratio

    Useful for quantized kernels where a small fraction of points may diverge
    from the FP reference due to INT8 round-off, while the bulk of the output
    stays within a tight per-point tolerance.

    NaN / Inf in ``actual`` or ``expected`` always fail (hard check,
    independent of the ratio).

    Upstream reference: ``compare()`` in cann-recipes-infer ``ops/pypto_python/example/compare.py``.

    Args:
        atol: Absolute tolerance. If ``None``, falls back to ``validate_golden``'s atol.
        rtol: Relative tolerance. If ``None``, falls back to ``validate_golden``'s rtol.
        max_error_ratio: Fraction of points permitted to exceed tolerance
            (default 0.5%). Set to 0.0 for strict allclose semantics.
        max_show: Maximum number of mismatched points printed on failure.
        valid_rows: Compare only the leading ``valid_rows`` entries along
            ``valid_axis``. ``None`` (default) compares the whole tensor, ``0``
            compares nothing and passes. Use it for packed buffers whose
            inactive tail would otherwise dilute the error ratio.
        valid_axis: Axis ``valid_rows`` slices (default 0). Pass 1 when a
            leading rank axis precedes the token axis.
        zero_tail: Additionally require the dropped tail to be all zeros. Only
            meaningful with ``valid_rows``; catches a kernel writing past the
            active token count.
        ignore_nan: Ignore the positions where *expected* is ``NaN``, dropping
            them from both the comparison and the error-ratio
            denominator. A pure ``pl.Out`` parameter the kernel writes only
            conditionally leaves the rest of the buffer undefined — the runtime
            neither uploads the host placeholder nor zero-fills the device
            buffer, so those bytes are allocator residue, and a zero-filled
            golden asserts a value the kernel never promised. The golden fills
            them with ``NaN`` to say it defines nothing there. ``NaN`` / ``Inf``
            in *actual* still fails inside the care region, and an entirely
            ``NaN`` golden fails rather than passing on an empty comparison.
            Unlike ``valid_rows`` the mask is per-element, so it expresses a
            write set that depends on runtime data (slot mappings, per-request
            conditions) rather than a leading prefix; the two compose.

    Example — attention output with INT8 activation quant::

        compare_fn = {
            "attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        }

    Example — packed prefill output, active prefix only::

        compare_fn = {
            "x_out": ratio_allclose(atol=1e-4, rtol=1e-2, valid_rows=num_tokens, zero_tail=True),
        }
    """
    if max_error_ratio < 0.0 or max_error_ratio > 1.0:
        raise ValueError(f"max_error_ratio must be in [0, 1], got {max_error_ratio}")

    def cmp(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        actual_outputs: dict[str, torch.Tensor],
        expected_outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        rtol: float,
        atol: float,
    ) -> tuple[bool, str]:
        eff_atol = atol if (cmp.atol_override is None) else cmp.atol_override
        eff_rtol = rtol if (cmp.rtol_override is None) else cmp.rtol_override

        actual, expected, prefix_error = _valid_prefix(actual, expected, valid_rows, valid_axis, zero_tail)
        if prefix_error:
            return False, prefix_error
        if actual.numel() == 0:
            return True, ""

        actual_f = actual.cpu().to(torch.float32)
        expected_f = expected.cpu().to(torch.float32)

        care = None
        n_ignored = 0
        if ignore_nan:
            care = ~torch.isnan(expected_f)
            n_ignored = int((~care).sum().item())
            if not bool(care.any()):
                return False, (f"    ignore_nan: golden is entirely NaN ({n_ignored} pts) -- nothing compared")
            # Zero both sides at the ignored points: diff 0, finite tolerance,
            # never the reported max, and dropped from the denominator below.
            zero = torch.zeros((), dtype=actual_f.dtype)
            actual_f = torch.where(care, actual_f, zero)
            expected_f = torch.where(care, expected_f, zero)

        nonfinite_error = _nonfinite_error(actual_f, expected_f)
        if nonfinite_error:
            return False, nonfinite_error

        diff_abs = (actual_f - expected_f).abs()
        tolerance = eff_atol + eff_rtol * expected_f.abs()
        bad_mask = diff_abs > tolerance
        error_count = int(bad_mask.sum().item())
        numel = actual_f.numel() if care is None else int(care.sum().item())
        threshold = round(max_error_ratio * numel)

        max_diff, flat_max_pos = torch.max(diff_abs.flatten(), dim=0)
        max_pos = torch.unravel_index(flat_max_pos, actual_f.shape)
        max_pos = tuple(int(i.item()) for i in max_pos)
        max_tol = float(tolerance[max_pos].item())

        if error_count <= threshold:
            return True, ""

        bad_indices = torch.where(bad_mask.flatten())[0]
        flat_actual = actual_f.flatten()
        flat_expected = expected_f.flatten()
        flat_tol = tolerance.flatten()
        flat_diff = diff_abs.flatten()
        n_show = min(max_show, bad_indices.numel())
        idx = bad_indices[:n_show]
        lines = [
            (
                f"    [{i.item()}] actual={flat_actual[i].item():.8g}, "
                f"expected={flat_expected[i].item():.8g}, "
                f"diff={flat_diff[i].item():.4g}, tol={flat_tol[i].item():.4g}"
            )
            for i in idx
        ]
        skipped = f"    {n_ignored}/{n_ignored + numel} NaN golden pts ignored\n" if n_ignored else ""
        return False, (
            f"    ratio_allclose fail: error_count={error_count}/{numel} "
            f"(ratio={error_count / numel:.4%}, allowed<={max_error_ratio:.4%}, "
            f"threshold={threshold} pts)\n"
            f"{skipped}"
            f"    atol={eff_atol} rtol={eff_rtol}\n"
            f"    max abs diff={max_diff.item():.6g} at {max_pos} (tol={max_tol:.6g})\n"
            f"    first {n_show} mismatches:\n" + "\n".join(lines)
        )

    cmp.atol_override = atol
    cmp.rtol_override = rtol
    cmp.__name__ = (
        f"ratio_allclose(atol={atol}, rtol={rtol}, max_error_ratio={max_error_ratio}, "
        f"valid_rows={valid_rows}, valid_axis={valid_axis}, zero_tail={zero_tail}, "
        f"ignore_nan={ignore_nan})"
    )
    return cmp


def _mapped_pool_compare(
    mapping_name: str,
    *,
    mapping_shape: tuple[int, ...],
    block_size: int,
    leading_rank_axis: bool,
    pool_name: str,
    mapped_compare: Callable,
    comparator_name: str,
) -> Callable:
    """Apply ``mapped_compare`` to mapped rows and preserve every other row."""
    if not mapping_shape or any(dim <= 0 for dim in mapping_shape):
        raise ValueError(f"mapping_shape must contain positive dimensions, got {mapping_shape}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if leading_rank_axis and len(mapping_shape) < 2:
        raise ValueError(
            "leading_rank_axis requires mapping_shape to include a rank axis "
            f"and at least one mapped-item axis, got {mapping_shape}"
        )
    integer_dtypes = (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    )

    def compare(actual: torch.Tensor, expected: torch.Tensor, **kwargs) -> tuple[bool, str]:  # noqa: PLR0912
        if actual.shape != expected.shape:
            return False, (
                f"    {pool_name} shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}"
            )

        block_axis = 2 if leading_rank_axis else 1
        minimum_rank = block_axis + 1
        if actual.ndim < minimum_rank or actual.shape[block_axis] != block_size:
            layout = "[ranks, blocks, block_size, ...]" if leading_rank_axis else "[blocks, block_size, ...]"
            return False, (
                f"    expected block-major {pool_name} layout {layout} with "
                f"block_size={block_size}, got {tuple(actual.shape)}"
            )

        mapping = kwargs.get("inputs", {}).get(mapping_name)
        if mapping is None:
            return False, f"    compare_fn misconfigured: missing input '{mapping_name}'"
        if mapping.dtype not in integer_dtypes:
            return False, f"    '{mapping_name}' must have an integer dtype, got {mapping.dtype}"
        if tuple(mapping.shape) != mapping_shape:
            return False, (f"    '{mapping_name}' must have shape {mapping_shape}, got {tuple(mapping.shape)}")

        actual_cpu = actual.cpu()
        expected_cpu = expected.cpu()
        mapping = mapping.cpu().to(torch.int64)
        if leading_rank_axis:
            rank_count = mapping_shape[0]
            if actual.shape[0] != rank_count:
                return False, (f"    leading rank count of {pool_name} must be {rank_count}, got {actual.shape[0]}")
            row_count = actual.shape[1] * block_size
            actual_rows = actual_cpu.reshape(rank_count, row_count, -1)
            expected_rows = expected_cpu.reshape(rank_count, row_count, -1)
            mapping_rows = mapping.reshape(rank_count, -1)
        else:
            rank_count = 1
            row_count = actual.shape[0] * block_size
            actual_rows = actual_cpu.reshape(1, row_count, -1)
            expected_rows = expected_cpu.reshape(1, row_count, -1)
            mapping_rows = mapping.reshape(1, -1)

        invalid_negative = mapping_rows < -1
        if invalid_negative.any().item():
            first = invalid_negative.nonzero(as_tuple=False)[0]
            rank = int(first[0].item())
            item = int(first[1].item())
            value = int(mapping_rows[rank, item].item())
            location = f"[{rank}, {item}]" if leading_rank_axis else f"[{item}]"
            return False, (f"    '{mapping_name}'{location}={value} is invalid; only -1 is a negative sentinel")

        valid = mapping_rows >= 0
        out_of_range = valid & (mapping_rows >= row_count)
        if out_of_range.any().item():
            first = out_of_range.nonzero(as_tuple=False)[0]
            rank = int(first[0].item())
            item = int(first[1].item())
            value = int(mapping_rows[rank, item].item())
            location = f"[{rank}, {item}]" if leading_rank_axis else f"[{item}]"
            return False, (f"    '{mapping_name}'{location}={value} is outside physical row range [0, {row_count})")

        written_rows = torch.zeros((rank_count, row_count), dtype=torch.bool)
        for rank in range(rank_count):
            rank_mapping = mapping_rows[rank, valid[rank]]
            if rank_mapping.numel() > 1:
                unique_rows, counts = torch.unique(rank_mapping, return_counts=True)
                duplicates = counts > 1
                if duplicates.any().item():
                    duplicate_row = int(unique_rows[duplicates][0].item())
                    if leading_rank_axis:
                        return False, (
                            f"    '{mapping_name}' contains duplicate physical row {duplicate_row} on rank {rank}"
                        )
                    return False, (f"    '{mapping_name}' contains duplicate physical row {duplicate_row}")
            written_rows[rank, rank_mapping] = True

        equal_rows = (actual_rows == expected_rows).all(dim=-1)
        stray_rows = ~written_rows & ~equal_rows
        if stray_rows.any().item():
            first = stray_rows.nonzero(as_tuple=False)[0]
            rank = int(first[0].item())
            row = int(first[1].item())
            changed_values = int((actual_rows[rank, row] != expected_rows[rank, row]).count_nonzero().item())
            rank_detail = f" rank={rank}" if leading_rank_axis else ""
            return False, (
                f"    unmapped physical {pool_name} row changed:{rank_detail} "
                f"row={row} changed_values={changed_values} mapping='{mapping_name}'"
            )

        if not written_rows.any().item():
            return True, ""

        mapped_actual = actual_rows[written_rows]
        mapped_expected = expected_rows[written_rows]
        for label, rows in (("actual", mapped_actual), ("expected", mapped_expected)):
            if torch.is_floating_point(rows):
                nonfinite = ~torch.isfinite(rows)
                if nonfinite.any().item():
                    return False, (
                        f"    {label} mapped rows in {pool_name} from '{mapping_name}' "
                        f"contain {int(nonfinite.count_nonzero().item())} non-finite value(s)"
                    )

        ok, detail = mapped_compare(mapped_actual, mapped_expected, **kwargs)
        if ok:
            return True, ""
        return False, f"    mapped rows in {pool_name} from '{mapping_name}':\n{detail}"

    compare.__name__ = comparator_name
    return compare


def mapped_pool_ratio_allclose(
    mapping_name: str,
    *,
    mapping_shape: tuple[int, ...],
    block_size: int,
    leading_rank_axis: bool = False,
    pool_name: str = "pool",
    atol: float | None = None,
    rtol: float | None = None,
    max_error_ratio: float = 0.005,
) -> Callable:
    """Compare mapped rows of a block-major pool and preserve all other rows.

    ``mapping_shape`` makes the mapping contract explicit instead of relying on
    model globals captured by a caller.  A non-ranked pool has layout
    ``[blocks, block_size, ...]``.  With ``leading_rank_axis=True``, the layout
    is ``[ranks, blocks, block_size, ...]`` and duplicate mappings are checked
    independently for every rank.

    Only allocator-mapped rows use the ratio-based numerical comparison and
    floating-point finiteness checks.  Every unmapped row must remain exactly
    equal to its golden snapshot, which detects writes outside the mapping.
    """
    mapped_compare = ratio_allclose(
        atol=atol,
        rtol=rtol,
        max_error_ratio=max_error_ratio,
    )
    comparator_name = (
        f"mapped_pool_ratio_allclose(mapping={mapping_name}, shape={mapping_shape}, "
        f"block_size={block_size}, leading_rank_axis={leading_rank_axis}, "
        f"atol={atol}, rtol={rtol}, max_error_ratio={max_error_ratio})"
    )
    return _mapped_pool_compare(
        mapping_name,
        mapping_shape=mapping_shape,
        block_size=block_size,
        leading_rank_axis=leading_rank_axis,
        pool_name=pool_name,
        mapped_compare=mapped_compare,
        comparator_name=comparator_name,
    )


def ratio_reldiff(
    diff_thd: float = 0.01,
    pct_thd: float = 0.05,
    max_diff_hd: float = float("inf"),
    max_show: int = 10,
    valid_rows: int | None = None,
    valid_axis: int = 0,
    zero_tail: bool = False,
) -> Callable:
    """Relative-diff comparator with bad-point ratio and single-point cap.

    Algorithm::

        a = |actual - expected|
        b = max(|actual|, |expected|, (1 / 2^14) / diff_thd) + 1e-9
        rdiff = a if a < diff_thd else a / b
        error_count = count(rdiff > diff_thd)
        pass iff error_count / numel <= pct_thd
                 AND max(rdiff over bad points) < max_diff_hd

    The denominator floor ``(1 / 2^14) / diff_thd`` keeps rdiff well-defined
    for near-zero values (capped via the ``a < diff_thd`` early-return).
    NaN / Inf in ``actual`` or ``expected`` always fail.

    Upstream reference: ``data_compare()`` in cann-recipes-infer.

    Args:
        diff_thd: Per-point relative-difference threshold.
        pct_thd: Allowed fraction of points exceeding ``diff_thd``.
        max_diff_hd: Hard cap on worst per-point rdiff. Defaults to ``+inf``
            (no cap); pass an explicit value for a single-point catastrophic
            failure check.
        max_show: Maximum mismatched points to print on failure.
        valid_rows: Compare only the leading ``valid_rows`` entries along
            ``valid_axis``. ``None`` (default) compares the whole tensor, ``0``
            compares nothing and passes. Use it for packed buffers whose
            inactive tail would otherwise dilute the error ratio.
        valid_axis: Axis ``valid_rows`` slices (default 0). Pass 1 when a
            leading rank axis precedes the token axis.
        zero_tail: Additionally require the dropped tail to be all zeros. Only
            meaningful with ``valid_rows``; catches a kernel writing past the
            active token count.
    """
    if not 0.0 < diff_thd:
        raise ValueError(f"diff_thd must be > 0, got {diff_thd}")
    if not 0.0 <= pct_thd <= 1.0:
        raise ValueError(f"pct_thd must be in [0, 1], got {pct_thd}")
    if not 0.0 < max_diff_hd:
        raise ValueError(f"max_diff_hd must be > 0, got {max_diff_hd}")

    def cmp(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        actual_outputs: dict[str, torch.Tensor],
        expected_outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        rtol: float,
        atol: float,
    ) -> tuple[bool, str]:
        actual, expected, prefix_error = _valid_prefix(actual, expected, valid_rows, valid_axis, zero_tail)
        if prefix_error:
            return False, prefix_error
        if actual.numel() == 0:
            return True, ""

        actual_f = actual.cpu().to(torch.float32)
        expected_f = expected.cpu().to(torch.float32)

        nonfinite_error = _nonfinite_error(actual_f, expected_f)
        if nonfinite_error:
            return False, nonfinite_error

        diff_abs = (actual_f - expected_f).abs()
        small_value_floor = (1.0 / (1 << 14)) / diff_thd
        denom = (
            torch.maximum(
                torch.maximum(actual_f.abs(), expected_f.abs()),
                torch.full_like(actual_f, small_value_floor),
            )
            + 1e-9
        )
        rdiff = torch.where(diff_abs < diff_thd, diff_abs, diff_abs / denom)

        bad_mask = rdiff > diff_thd
        error_count = int(bad_mask.sum().item())
        numel = actual_f.numel()
        pct_threshold = round(pct_thd * numel)

        # Worst single-point rdiff among bad points (0 if no bad points).
        if error_count > 0:
            worst_rdiff = float(rdiff[bad_mask].max().item())
        else:
            worst_rdiff = 0.0

        passed = (error_count <= pct_threshold) and (worst_rdiff < max_diff_hd)
        if passed:
            return True, ""

        bad_indices = torch.where(bad_mask.flatten())[0]
        flat_actual = actual_f.flatten()
        flat_expected = expected_f.flatten()
        flat_abs = diff_abs.flatten()
        flat_rdiff = rdiff.flatten()
        n_show = min(max_show, bad_indices.numel())
        idx = bad_indices[:n_show]
        lines = [
            (
                f"    [{i.item()}] actual={flat_actual[i].item():.8g}, "
                f"expected={flat_expected[i].item():.8g}, "
                f"abs_diff={flat_abs[i].item():.4g}, "
                f"rdiff={flat_rdiff[i].item():.4g}"
            )
            for i in idx
        ]
        reasons = []
        if error_count > pct_threshold:
            reasons.append(
                f"error_count={error_count}/{numel} "
                f"(ratio={error_count / numel:.4%}, allowed<={pct_thd:.4%}, "
                f"threshold={pct_threshold} pts)"
            )
        if worst_rdiff >= max_diff_hd:
            reasons.append(f"worst rdiff={worst_rdiff:.4g} >= max_diff_hd={max_diff_hd:.4g}")
        return False, (
            f"    ratio_reldiff fail: {' AND '.join(reasons)}\n"
            f"    diff_thd={diff_thd} pct_thd={pct_thd} max_diff_hd={max_diff_hd}\n"
            f"    first {n_show} mismatches:\n" + "\n".join(lines)
        )

    cmp.__name__ = (
        f"ratio_reldiff(diff_thd={diff_thd}, pct_thd={pct_thd}, max_diff_hd={max_diff_hd}, "
        f"valid_rows={valid_rows}, valid_axis={valid_axis}, zero_tail={zero_tail})"
    )
    return cmp

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unused failing builders cannot prevent a selected callable from running."""

import pytest

from tests.st.a5.tensormap_and_ringbuffer.ccu_deferred_completion.test_ccu_domains import (
    _check,
    _compile_root,
    _domain,
    _new_worker,
    _results,
    _submit,
)
from tests.st.a5.tensormap_and_ringbuffer.ccu_deferred_completion.test_ccu_prepare_failure import (
    _compile_rejecting_builder,
)


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
def test_unused_failing_builder_isolated_from_selected_callable(st_platform, st_device_ids, tmp_path):
    worker = _new_worker(st_platform, st_device_ids)
    good = worker.register(_compile_root(st_platform, tmp_path, 0))
    bad = worker.register(_compile_rejecting_builder(st_platform, tmp_path))
    domains = []
    try:
        worker.init()
        outputs = _results(worker)

        def run_selected(seed):
            def orchestration(orch, _args, cfg):
                domain = _domain(orch, "selected", good)
                if domains:
                    assert domain is domains[0]
                else:
                    domains.append(domain)
                _submit(orch, cfg, good, domain, outputs, seed)

            worker.run(orchestration)
            _check(outputs, seed)
            assert not (tmp_path / "rank0-accepted").exists()
            assert not (tmp_path / "rank1-rejected").exists()

        run_selected(0)
        run_selected(1)
        original = domains[0]
        original_ref = original.ref

        def request_bad(orch, _args, _cfg):
            with pytest.raises(RuntimeError, match="Worker control broadcast"):
                _domain(orch, "selected", bad)
            assert original.released and not original.freed
            assert worker.live_domains == {}
            with pytest.raises(RuntimeError, match="release prepared CCU domains"):
                worker.unregister(bad)

        worker.run(request_bad)
        assert (tmp_path / "rank0-accepted").is_file()
        assert (tmp_path / "rank1-rejected").is_file()
        assert original.freed
        worker.unregister(bad)
        domains.clear()

        def recover(orch, _args, cfg):
            replacement = _domain(orch, "selected", good)
            assert replacement.ref != original_ref
            domains.append(replacement)
            _submit(orch, cfg, good, replacement, outputs, 4)

        worker.run(recover)
        _check(outputs, 4)
        domains[0].release()
        assert domains[0].freed and worker.live_domains == {}
        worker.unregister(good)
    finally:
        worker.close()

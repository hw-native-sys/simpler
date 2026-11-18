#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Percounter entry points reusing the common workload and golden definitions."""

from copy import deepcopy

from simpler_setup import SceneTestCase, scene_test
from tests.st.a5.host_build_graph.graph_execution.test_graph_execution import (
    TestGraphExecutionHostBuildGraphA5 as _Graph,
)
from tests.st.a5.host_build_graph.graph_execution.test_graph_execution_aic_aiv import (
    TestGraphExecutionAicAivHostBuildGraphA5 as _GraphPure,
)
from tests.st.a5.host_build_graph.graph_execution.test_graph_execution_mix_spmd import (
    TestGraphExecutionMixSpmdHostBuildGraphA5 as _GraphSpmd,
)
from tests.st.a5.host_build_graph.graph_predicated_dispatch.test_graph_predicated_dispatch import (
    TestGraphPredicatedDispatchA5 as _GraphPredicate,
)
from tests.st.a5.host_build_graph.predicated_dispatch.test_predicated_dispatch import (
    TestPredicatedDispatchA5 as _Predicate,
)
from tests.st.a5.host_build_graph.single_block_mix.test_single_block_mix import (
    TestSingleBlockMixHostBuildGraphA5 as _Mix,
)
from tests.st.a5.host_build_graph.spmd_mixed_waves.test_spmd_mixed_waves import TestSpmdMixedWaves as _MixedWaves
from tests.st.a5.host_build_graph.spmd_multiblock_aic.test_spmd_multiblock_aic import TestSpmdMultiblockAic as _SpmdAic
from tests.st.a5.host_build_graph.spmd_releases_spmd.test_spmd_releases_spmd import (
    TestSpmdReleasesSpmd as _ReleasesSpmd,
)
from tests.st.a5.host_build_graph.spmd_sync_start_mix_aiv.test_spmd_sync_start_mix_aiv import (
    TestSpmdSyncStartMixAiv as _TwoVectorSync,
)


@scene_test(level=2, runtime="percounter")
class TestPercounterMix(SceneTestCase):
    CALLABLE = deepcopy(_Mix.CALLABLE)
    CASES = deepcopy(_Mix.CASES)
    RTOL = _Mix.RTOL
    ATOL = _Mix.ATOL
    generate_args = _Mix.generate_args
    compute_golden = _Mix.compute_golden


@scene_test(level=2, runtime="percounter")
class TestPercounterSpmdAic(SceneTestCase):
    CALLABLE = deepcopy(_SpmdAic.CALLABLE)
    CASES = deepcopy(_SpmdAic.CASES)
    RTOL = _SpmdAic.RTOL
    ATOL = _SpmdAic.ATOL
    generate_args = _SpmdAic.generate_args
    compute_golden = _SpmdAic.compute_golden


@scene_test(level=2, runtime="percounter")
class TestPercounterMixedWaves(SceneTestCase):
    CALLABLE = deepcopy(_MixedWaves.CALLABLE)
    CASES = deepcopy(_MixedWaves.CASES)
    RTOL = _MixedWaves.RTOL
    ATOL = _MixedWaves.ATOL
    generate_args = _MixedWaves.generate_args
    compute_golden = _MixedWaves.compute_golden


@scene_test(level=2, runtime="percounter")
class TestPercounterReleasesSpmd(SceneTestCase):
    CALLABLE = deepcopy(_ReleasesSpmd.CALLABLE)
    CASES = deepcopy(_ReleasesSpmd.CASES)
    RTOL = _ReleasesSpmd.RTOL
    ATOL = _ReleasesSpmd.ATOL
    generate_args = _ReleasesSpmd.generate_args
    compute_golden = _ReleasesSpmd.compute_golden


@scene_test(level=2, runtime="percounter")
class TestPercounterTwoVectorSync(SceneTestCase):
    CALLABLE = deepcopy(_TwoVectorSync.CALLABLE)
    CASES = deepcopy(_TwoVectorSync.CASES)
    RTOL = _TwoVectorSync.RTOL
    ATOL = _TwoVectorSync.ATOL
    generate_args = _TwoVectorSync.generate_args
    compute_golden = _TwoVectorSync.compute_golden


@scene_test(level=2, runtime="percounter")
class TestPercounterGraph(SceneTestCase):
    CALLABLE = deepcopy(_Graph.CALLABLE)
    CASES = deepcopy(_Graph.CASES)
    RTOL = _Graph.RTOL
    ATOL = _Graph.ATOL
    generate_args = _Graph.generate_args
    compute_golden = _Graph.compute_golden


@scene_test(level=2, runtime="percounter")
class TestPercounterGraphPure(SceneTestCase):
    CALLABLE = deepcopy(_GraphPure.CALLABLE)
    CASES = deepcopy(_GraphPure.CASES)
    RTOL = _GraphPure.RTOL
    ATOL = _GraphPure.ATOL
    generate_args = _GraphPure.generate_args
    compute_golden = _GraphPure.compute_golden


@scene_test(level=2, runtime="percounter")
class TestPercounterGraphSpmd(SceneTestCase):
    CALLABLE = deepcopy(_GraphSpmd.CALLABLE)
    CASES = deepcopy(_GraphSpmd.CASES)
    RTOL = _GraphSpmd.RTOL
    ATOL = _GraphSpmd.ATOL
    generate_args = _GraphSpmd.generate_args
    compute_golden = _GraphSpmd.compute_golden
    compare_outputs = _GraphSpmd.compare_outputs


@scene_test(level=2, runtime="percounter")
class TestPercounterPredicate(SceneTestCase):
    CALLABLE = deepcopy(_Predicate.CALLABLE)
    CASES = deepcopy(_Predicate.CASES)
    for _case in CASES:
        _case.get("config", {}).pop("aicpu_thread_num", None)
    RTOL = _Predicate.RTOL
    ATOL = _Predicate.ATOL
    generate_args = _Predicate.generate_args
    compute_golden = _Predicate.compute_golden


@scene_test(level=2, runtime="percounter")
class TestPercounterGraphPredicate(SceneTestCase):
    CALLABLE = deepcopy(_GraphPredicate.CALLABLE)
    CASES = deepcopy(_GraphPredicate.CASES)
    RTOL = _GraphPredicate.RTOL
    ATOL = _GraphPredicate.ATOL
    generate_args = _GraphPredicate.generate_args
    compute_golden = _GraphPredicate.compute_golden


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)

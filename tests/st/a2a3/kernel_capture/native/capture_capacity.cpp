/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "capture_capacity_fixture.h"
using namespace simpler::capture_test;
int main(int argc, char **argv) {
    CaptureFixture fixture(argc, argv);
    for (auto &model : fixture.models) {
        check(fixture.prepare(model), "prepare-capture");
        fixture.unchanged(model);
        fixture.record(model);
    }
    for (const auto &model : fixture.models)
        fixture.unchanged(model);
    fixture.report("three-models-grant-two");
    for (size_t i = 0; i < fixture.models.size(); ++i)
        fixture.replay(fixture.models[i], static_cast<int>(i));
    const auto before_budget = fixture.report("before-refusal");
    const auto before = launch_counts();
    require(before.core > 0 && before.cpu > 0, "launch-observer-reached");
    const int rc = fixture.prepare(fixture.models[0], workspace_budget * 2);
    std::printf("workspace-refusal rc=%d accepted=%d\n", rc, fixture.accepted);
    require(rc == SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE, "typed-workspace-refusal");
    require(fixture.accepted == 0 && launch_counts() == before, "refusal-before-kernel-submission");
    check(simpler_abort_run_plan_ctx(fixture.context, 0, fixture.epoch), "abort-only-refused-plan");
    const auto after_budget = fixture.report("after-refusal-abort");
    require(
        after_budget.reserved_bytes <= before_budget.reserved_bytes &&
            after_budget.live_blocked == before_budget.live_blocked,
        "refusal-budget-balanced"
    );
    std::printf(
        "refusal-launch-counts core_before=%llu core_after=%llu cpu_before=%llu cpu_after=%llu\n",
        static_cast<unsigned long long>(before.core), static_cast<unsigned long long>(launch_counts().core),
        static_cast<unsigned long long>(before.cpu), static_cast<unsigned long long>(launch_counts().cpu)
    );
    require(
        simpler_unregister_callable(fixture.context, 0) == PTO_RUNTIME_ERR_INVALID_STATE, "refusal-keeps-live-callable"
    );
    for (size_t i = 0; i < fixture.models.size(); ++i)
        fixture.replay(fixture.models[i], static_cast<int>(i + 10));
    for (auto &model : fixture.models)
        fixture.release(model);
    fixture.report("released-three");
    fixture.poison(fixture.models[0]);
    check(fixture.prepare(fixture.models[0]), "readmit-small-capture");
    fixture.record(fixture.models[0]);
    fixture.replay(fixture.models[0], 20);
    fixture.release(fixture.models[0]);
    fixture.close();
    std::printf(
        "CAPTURE_WORKSPACE_CAPACITY_PASS runtime=%s budget=%llu models=3 grants=2 offset_views=2 refused=%d "
        "recovered=1\n",
        argv[6], static_cast<unsigned long long>(workspace_budget), rc
    );
}

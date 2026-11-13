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
    constexpr size_t maximum_models = 9;
    constexpr uint64_t request_bytes = 32ULL << 20;
    static_assert(maximum_models * request_bytes > workspace_budget);
    CaptureFixture fixture(argc, argv, maximum_models);
    size_t accepted_models = 0;
    int refused = 0;
    for (size_t i = 0; i < maximum_models; ++i) {
        const auto before = fixture.report("before-cumulative-prepare");
        const auto launches = launch_counts();
        const int rc = fixture.prepare(fixture.models[i], request_bytes);
        std::printf(
            "cumulative-prepare index=%zu request=%llu rc=%d accepted=%d\n", i,
            static_cast<unsigned long long>(request_bytes), rc, fixture.accepted
        );
        if (rc == SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE) {
            require(accepted_models >= 3, "at-least-three-live-models");
            require(launches.core > 0 && launches.cpu > 0, "launch-observer-reached");
            require(fixture.accepted == 0 && launch_counts() == launches, "refusal-before-kernel-submission");
            check(simpler_abort_run_plan_ctx(fixture.context, 0, fixture.epoch), "abort-only-refused-plan");
            const auto after = fixture.report("cumulative-refusal-aborted");
            require(
                after.reserved_bytes <= before.reserved_bytes && after.live_blocked == before.live_blocked,
                "refusal-budget-balanced"
            );
            refused = rc;
            break;
        }
        check(rc, "prepare-cumulative");
        fixture.unchanged(fixture.models[i]);
        fixture.record(fixture.models[i]);
        ++accepted_models;
        const auto after = fixture.report("retained-cumulative-generation");
        require(after.reserved_bytes > before.reserved_bytes, "retained-budget-increased");
    }
    require(refused != 0 && accepted_models < maximum_models, "bounded-cumulative-refusal");
    for (size_t i = 0; i <= accepted_models; ++i)
        fixture.unchanged(fixture.models[i]);
    require(
        simpler_unregister_callable(fixture.context, 0) == PTO_RUNTIME_ERR_INVALID_STATE, "refusal-keeps-live-callable"
    );
    for (size_t i = 0; i < accepted_models; ++i)
        fixture.replay(fixture.models[i], static_cast<int>(i + 1));
    fixture.release(fixture.models[0]);
    fixture.report("one-occupied-model-released");
    // The refused target keeps the same addresses, shape and runtime config;
    // prepare assigns only a new native call identity after aborting the old plan.
    auto &retry = fixture.models[accepted_models];
    check(fixture.prepare(retry, request_bytes), "same-pressure-readmitted");
    fixture.unchanged(retry);
    fixture.record(retry);
    fixture.replay(retry, 30);
    for (size_t i = 1; i < accepted_models; ++i)
        fixture.replay(fixture.models[i], static_cast<int>(i + 40));
    for (auto &model : fixture.models)
        if (model.model != nullptr) fixture.release(model);
    fixture.close();
    std::printf(
        "CAPTURE_CUMULATIVE_CAPACITY_PASS runtime=%s budget=%llu request=%llu accepted=%zu "
        "maximum_models=%zu grants=2 refused=%d same_pressure_recovered=1\n",
        argv[6], static_cast<unsigned long long>(workspace_budget), static_cast<unsigned long long>(request_bytes),
        accepted_models, maximum_models, refused
    );
}

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

#include "orchestration_api.h"

extern "C" void args_dump_capture_orchestration(const ChipTaskArgs &args) {
    CoreTaskArgs first;
    first.add_inout(args.tensor(0).ref());
    first.add_scalar(args.scalar<int32_t>(0));
    first.dump();
    rt_submit_aiv_task(0, first);

    // The second task overwrites the same backing after consuming the first result.
    CoreTaskArgs second;
    second.add_inout(args.tensor(0).ref());
    second.add_scalar(int32_t{7});
    second.dump();
    rt_submit_aiv_task(0, second);
}

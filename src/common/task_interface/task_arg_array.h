/**
 * TaskArgArray - Contiguous, resizable array of TaskArg
 *
 * Wraps std::vector<TaskArg> and exposes a raw data pointer so that
 * the array can be handed to C APIs (e.g. ctypes CDLL calls) that
 * expect a contiguous TaskArg[].
 */

#pragma once

#include "task_arg.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

static_assert(std::is_trivially_copyable<TaskArg>::value,
              "TaskArg must be trivially copyable for contiguous array access via data_ptr()");

struct TaskArgArray {
    std::vector<TaskArg> args;

    void append(const TaskArg& arg) { args.push_back(arg); }
    void clear() { args.clear(); }
    size_t size() const { return args.size(); }

    TaskArg& get(size_t idx) {
        if (idx >= args.size())
            throw std::out_of_range("TaskArgArray index out of range");
        return args[idx];
    }

    const TaskArg& get(size_t idx) const {
        if (idx >= args.size())
            throw std::out_of_range("TaskArgArray index out of range");
        return args[idx];
    }

    /// Raw address of the contiguous TaskArg buffer (for C interop)
    uintptr_t data_ptr() const {
        return reinterpret_cast<uintptr_t>(args.data());
    }
};

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
#pragma once

#include <atomic>
#include <initializer_list>
#include <memory>
#include <vector>

#include "runtime_c_api.h"
#include "host/workspace_manager.h"

// Host operations are serialized by the owning context. The RTS notification
// only publishes destruction; it never drains streams or drops resource refs.
class CapturedGraphResources {
public:
    // Cleanup runs only after model destruction and an explicit drain. It must
    // be idempotent: a resource can be retained by more than one graph. Failure
    // keeps its ownership and dependencies available for retry.
    struct Retirement {
        std::shared_ptr<void> resource;
        int (*release)(void *);
    };

private:
    struct Graph {
        explicit Graph(void *handle) :
            model(handle) {}
        void *model;
        std::atomic<bool> destroyed{false};
        std::vector<std::shared_ptr<void>> resources;
        std::vector<Retirement> retirements;
        std::vector<void *> streams;
        std::vector<int32_t> callable_ids;
        WorkspaceManager *workspace{nullptr};
    };

public:
    CapturedGraphResources() = default;
    CapturedGraphResources(const CapturedGraphResources &) = delete;
    CapturedGraphResources &operator=(const CapturedGraphResources &) = delete;
    ~CapturedGraphResources() {
        // Unretired references have no stop proof. Their callback storage and
        // resources survive terminal abandonment; explicit retirement frees them.
        for (auto &graph : graphs_)
            (void)graph.release();
    }

    template <typename Register>
    int retain(
        void *model, std::initializer_list<std::shared_ptr<void>> resources, Register &&register_callback,
        WorkspaceManager *workspace = nullptr, const std::vector<void *> &blocks = {},
        std::initializer_list<Retirement> retirements = {}, std::initializer_list<void *> streams = {},
        std::initializer_list<int32_t> callable_ids = {}
    ) {
        if (model == nullptr || (workspace == nullptr && !blocks.empty())) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        for (const auto &resource : resources)
            if (resource == nullptr) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        for (const auto &retirement : retirements)
            if (!retirement.resource || !retirement.release) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        for (void *stream : streams)
            if (stream == nullptr) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        for (auto &graph : graphs_) {
            if (graph->model == model && !graph->destroyed.load(std::memory_order_acquire)) {
                const size_t resource_count = graph->resources.size();
                const size_t retirement_count = graph->retirements.size();
                const size_t stream_count = graph->streams.size();
                const size_t callable_count = graph->callable_ids.size();
                const auto rollback = [&] {
                    graph->resources.resize(resource_count);
                    graph->retirements.resize(retirement_count);
                    graph->streams.resize(stream_count);
                    graph->callable_ids.resize(callable_count);
                };
                try {
                    append(*graph, resources, retirements, streams, callable_ids);
                    const int rc = retain_workspace(*graph, workspace, blocks);
                    if (rc != 0) {
                        rollback();
                        return rc;
                    }
                } catch (...) {
                    rollback();
                    throw;
                }
                return graph->destroyed.load(std::memory_order_acquire) ? PTO_RUNTIME_ERR_INVALID_STATE : 0;
            }
        }
        // A recycled RTS handle names a new incarnation. Older destroyed graphs
        // can still own resources awaiting a drain, so they remain separate.
        auto graph = std::make_unique<Graph>(model);
        append(*graph, resources, retirements, streams, callable_ids);
        // Store before taking workspace references: vector growth must not strand
        // a reference whose token was never made reachable for retirement.
        graphs_.push_back(std::move(graph));
        Graph *record = graphs_.back().get();
        int retain_rc = 0;
        try {
            retain_rc = retain_workspace(*record, workspace, blocks);
        } catch (...) {
            graphs_.pop_back();
            throw;
        }
        if (retain_rc != 0) {
            graphs_.pop_back();
            return retain_rc;
        }
        const int rc = register_callback(model, &notify_destroyed, record);
        if (rc != 0) {
            // A refused registration installs no callback. No captured operation
            // may refer to these resources before retain succeeds.
            if (record->workspace != nullptr) record->workspace->release_capture(record);
            graphs_.pop_back();
            return rc;
        }
        return record->destroyed.load(std::memory_order_acquire) ? PTO_RUNTIME_ERR_INVALID_STATE : 0;
    }

    bool has_live_model() const {
        for (const auto &graph : graphs_)
            if (!graph->destroyed.load(std::memory_order_acquire)) return true;
        return false;
    }
    // Destruction is not a release proof. Keep each identity protected until
    // every reader drains and all fallible retirements succeed. A negative ID
    // denotes a legacy attachment whose callable set was not recorded.
    bool holds_callable(int32_t callable_id) const {
        for (const auto &graph : graphs_)
            for (const int32_t held : graph->callable_ids)
                if (held < 0 || held == callable_id) return true;
        return false;
    }
    bool empty() const { return graphs_.empty(); }
    bool has_streams() const {
        for (const auto &graph : graphs_)
            if (!graph->streams.empty()) return true;
        return false;
    }
    template <typename Drain>
    int drain_and_retire(Drain &&drain) {
        if (has_live_model()) return PTO_RUNTIME_ERR_INVALID_STATE;
        for (const auto &graph : graphs_)
            for (void *stream : graph->streams) {
                const int rc = drain(stream);
                if (rc != 0) return rc;
            }
        return retire_after_drain();
    }

    // The caller has finished model destruction and proved all queued users and
    // consumers retired. The notification alone supplies none of those proofs.
    int retire_after_drain() {
        if (has_live_model()) return PTO_RUNTIME_ERR_INVALID_STATE;
        for (const auto &graph : graphs_) {
            for (auto &retirement : graph->retirements) {
                if (retirement.release == nullptr) continue;
                const int rc = retirement.release(retirement.resource.get());
                if (rc != 0) return rc;
                retirement.release = nullptr;
            }
        }
        for (const auto &graph : graphs_)
            if (graph->workspace != nullptr) graph->workspace->release_capture(graph.get());
        graphs_.clear();
        return 0;
    }

private:
    // The context owns the manager and must outlive explicit retirement. Terminal
    // abandonment leaks the records above and never calls a destroyed manager.
    static int retain_workspace(Graph &graph, WorkspaceManager *workspace, const std::vector<void *> &blocks) {
        if (workspace == nullptr) return 0;
        if (graph.workspace != nullptr && graph.workspace != workspace) return PTO_RUNTIME_ERR_INVALID_STATE;
        if (!workspace->retain_capture(&graph, blocks)) return PTO_RUNTIME_ERR_INVALID_STATE;
        graph.workspace = workspace;
        return 0;
    }
    static void notify_destroyed(void *opaque) {
        static_cast<Graph *>(opaque)->destroyed.store(true, std::memory_order_release);
    }
    static void append(
        Graph &graph, std::initializer_list<std::shared_ptr<void>> resources,
        std::initializer_list<Retirement> retirements, std::initializer_list<void *> streams,
        std::initializer_list<int32_t> callable_ids
    ) {
        // Reserve before mutation: failure must not install a partial cleanup set.
        graph.callable_ids.reserve(graph.callable_ids.size() + callable_ids.size());
        graph.streams.reserve(graph.streams.size() + streams.size());
        graph.resources.reserve(graph.resources.size() + resources.size());
        graph.retirements.reserve(graph.retirements.size() + retirements.size());
        for (const int32_t callable_id : callable_ids) {
            bool held = false;
            for (const int32_t existing : graph.callable_ids)
                held = held || existing == callable_id;
            if (!held) graph.callable_ids.push_back(callable_id);
        }
        for (void *stream : streams) {
            bool held = false;
            for (void *existing : graph.streams)
                held = held || existing == stream;
            if (!held) graph.streams.push_back(stream);
        }
        for (const auto &retirement : retirements) {
            bool held = false;
            for (const auto &existing : graph.retirements)
                held = held || (existing.resource.get() == retirement.resource.get() &&
                                existing.release == retirement.release &&
                                !existing.resource.owner_before(retirement.resource) &&
                                !retirement.resource.owner_before(existing.resource));
            if (!held) graph.retirements.push_back(retirement);
        }
        for (const auto &resource : resources) {
            bool held = false;
            for (const auto &existing : graph.resources)
                held = held || (!existing.owner_before(resource) && !resource.owner_before(existing));
            if (!held) graph.resources.push_back(resource);
        }
    }
    std::vector<std::unique_ptr<Graph>> graphs_;
};

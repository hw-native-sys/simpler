/**
 * Unit tests for PTO2 Scope mechanism — scope stack management.
 *
 * Tests scope_begin/scope_end operations, nesting, ring ID mapping,
 * and max depth enforcement.
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include "pto_runtime2_types.h"

// =============================================================================
// Scope stack helper — minimal simulation of orchestrator scope state
// =============================================================================

struct ScopeStack {
    static constexpr int32_t MAX_DEPTH = PTO2_MAX_SCOPE_DEPTH;

    PTO2TaskSlotState** scope_tasks;
    int32_t scope_tasks_size;
    int32_t scope_tasks_capacity;
    int32_t* scope_begins;
    int32_t scope_stack_top;

    ScopeStack() {
        scope_tasks_capacity = 1024;
        scope_tasks = (PTO2TaskSlotState**)calloc(scope_tasks_capacity, sizeof(PTO2TaskSlotState*));
        scope_begins = (int32_t*)calloc(MAX_DEPTH, sizeof(int32_t));
        scope_tasks_size = 0;
        scope_stack_top = -1;  // No scope open
    }

    ~ScopeStack() {
        free(scope_tasks);
        free(scope_begins);
    }

    void scope_begin() {
        scope_stack_top++;
        scope_begins[scope_stack_top] = scope_tasks_size;
    }

    void scope_add_task(PTO2TaskSlotState* slot) {
        scope_tasks[scope_tasks_size++] = slot;
    }

    int scope_end() {
        int begin = scope_begins[scope_stack_top];
        int count = scope_tasks_size - begin;
        scope_tasks_size = begin;
        scope_stack_top--;
        return count;
    }

    int current_depth() const { return scope_stack_top + 1; }

    uint8_t current_ring_id() const {
        // Ring ID maps from scope depth (capped at PTO2_MAX_RING_DEPTH - 1)
        if (scope_stack_top < 0) return 0;
        return (scope_stack_top < PTO2_MAX_RING_DEPTH)
                   ? (uint8_t)scope_stack_top
                   : (uint8_t)(PTO2_MAX_RING_DEPTH - 1);
    }
};

// =============================================================================
// Push / Pop
// =============================================================================

TEST(ScopeTest, PushPop) {
    ScopeStack ss;
    EXPECT_EQ(ss.current_depth(), 0);

    ss.scope_begin();
    EXPECT_EQ(ss.current_depth(), 1);

    int count = ss.scope_end();
    EXPECT_EQ(count, 0);  // No tasks added
    EXPECT_EQ(ss.current_depth(), 0);
}

// =============================================================================
// Nested scopes
// =============================================================================

TEST(ScopeTest, NestedScopes) {
    ScopeStack ss;
    PTO2TaskSlotState slots[10]{};

    // Outer scope
    ss.scope_begin();
    ss.scope_add_task(&slots[0]);
    ss.scope_add_task(&slots[1]);

    // Inner scope
    ss.scope_begin();
    ss.scope_add_task(&slots[2]);
    ss.scope_add_task(&slots[3]);
    ss.scope_add_task(&slots[4]);

    EXPECT_EQ(ss.current_depth(), 2);

    // End inner scope — should return 3 tasks
    int inner_count = ss.scope_end();
    EXPECT_EQ(inner_count, 3);
    EXPECT_EQ(ss.current_depth(), 1);

    // End outer scope — should return 2 tasks
    int outer_count = ss.scope_end();
    EXPECT_EQ(outer_count, 2);
    EXPECT_EQ(ss.current_depth(), 0);
}

// =============================================================================
// Ring ID mapping from scope depth
// =============================================================================

TEST(ScopeTest, RingIdMapping) {
    ScopeStack ss;

    // Before any scope, ring_id = 0
    EXPECT_EQ(ss.current_ring_id(), 0u);

    ss.scope_begin();
    EXPECT_EQ(ss.current_ring_id(), 0u);  // depth=1 → ring 0

    ss.scope_begin();
    EXPECT_EQ(ss.current_ring_id(), 1u);  // depth=2 → ring 1

    ss.scope_begin();
    EXPECT_EQ(ss.current_ring_id(), 2u);  // depth=3 → ring 2

    ss.scope_begin();
    EXPECT_EQ(ss.current_ring_id(), 3u);  // depth=4 → ring 3

    // Beyond MAX_RING_DEPTH, stays at max
    ss.scope_begin();
    EXPECT_EQ(ss.current_ring_id(), (uint8_t)(PTO2_MAX_RING_DEPTH - 1));

    // Clean up
    for (int i = 0; i < 5; i++) ss.scope_end();
}

// =============================================================================
// Max depth
// =============================================================================

TEST(ScopeTest, MaxDepth) {
    ScopeStack ss;
    // Push up to max scope depth
    for (int i = 0; i < PTO2_MAX_SCOPE_DEPTH; i++) {
        ss.scope_begin();
    }
    EXPECT_EQ(ss.current_depth(), PTO2_MAX_SCOPE_DEPTH);

    // Pop all
    for (int i = 0; i < PTO2_MAX_SCOPE_DEPTH; i++) {
        ss.scope_end();
    }
    EXPECT_EQ(ss.current_depth(), 0);
}

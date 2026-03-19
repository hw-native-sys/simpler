# Multi-TensorMap Rewrite

Date: 2026-03-19

## Background

The old multi-tensormap direction had two persistent problems:

- fallback history did not share the same lifecycle model as owner history
- owner and fallback logic were drifting toward separate implementations

The rewrite fixes both by making producer retirement the only lifecycle
source and forcing owner/fallback to share one shard implementation.

## Goals

- Keep same-ring owner history on a ring-local fast path
- Support cross-ring `INOUT` and external tensors through fallback storage
- Bind stale/cleanup semantics to real producer retirement
- Keep `sync_tensormap()` interface unchanged
- Force owner and fallback to share one core implementation

## Tensor Model

`Tensor.ring_id` means tensor owner ring.

- `ring_id in [0, PTO2_MAX_RING_DEPTH)`: internal tensor
- `ring_id == TENSOR_RING_ID_NONE`: external tensor

Submit rules:

- internal `OUTPUT` must satisfy `tensor.ring_id == submit_ring`
- `INOUT` and `INPUT` must not rewrite owner ring at submit time
- external tensors stay external; runtime must not silently assign an
  internal owner ring

## Entry Model

Each tensormap entry stores:

- `producer_task_id`: the real producer task
- `tensor_owner_ring`: the tensor owner ring, or `TENSOR_RING_ID_NONE`
- `storage_domain`: `OWNER_MAP` or `FALLBACK_MAP`
- overlap metadata: address, version, shape, offsets
- `with_alloc`: whether this history entry came from runtime allocation

The entry does not store a separate fallback lifecycle key.

Two derived values drive lifecycle handling:

- `producer_ring = producer_task_id.ring()`
- `producer_local = producer_task_id.local()`

## Shared Shard Core

Owner and fallback both use the same template:

```cpp
template <int32_t NumCleanupDomains, bool BreakOnStale>
struct TensorMapShardImpl;
```

Concrete instances:

- `OwnerTensorMapShard = TensorMapShardImpl<1, true>`
- `FallbackTensorMapShard = TensorMapShardImpl<PTO2_MAX_RING_DEPTH, false>`

This keeps one method body for:

- `init`
- `destroy`
- `lookup`
- `insert`
- `remove_entry`
- `cleanup_range`

Differences are expressed only through template parameters and entry
metadata, not through specialized method bodies.

## Cleanup Domains

`cleanup_domain` is a shard-local concept, not a stored field.

For owner shards:

- there is exactly one cleanup domain
- every entry maps to cleanup domain `0`

For fallback shard:

- there is one cleanup domain per producer ring
- an entry maps to `producer_task_id.ring()`

This is why fallback mirrors `last_task_alive[ring]` for every producer
ring instead of maintaining a fake global frontier.

## Routing Rules

### Lookup

- internal tensor: query owner shard first, then fallback shard
- external tensor: query fallback shard only

### Insert

- internal `OUTPUT`: owner shard of the submit ring
- same-ring internal `INOUT`: owner shard of the submit ring
- cross-ring internal `INOUT`: fallback shard
- external `OUTPUT` / `INOUT`: fallback shard

### Remove

`remove_entry()` routes by `storage_domain`:

- `OWNER_MAP`: remove from the owner shard indexed by `tensor_owner_ring`
- `FALLBACK_MAP`: remove from fallback shard

## Cleanup Semantics

Stale is defined only by producer retirement.

Shared validity rule:

```cpp
entry.producer_task_id.local() >=
    shard.last_task_alives[cleanup_domain_of(entry)]
```

Lookup behavior:

- owner shards may `break` on first stale entry because each owner shard is
  a single lifecycle domain
- fallback shard must continue scanning because its bucket chains mix
  producer rings

Cleanup behavior:

- `sync_tensormap()` reads real `last_task_alive` values from shared memory
- owner shard `R` cleans retired range on domain `0`
- fallback shard cleans retired range on domain `R`

No fallback-private lifecycle frontier exists.

## Main Invariants

Owner shard entry:

- `storage_domain == OWNER_MAP`
- `tensor_owner_ring == producer_task_id.ring()`

Fallback shard entry:

- `storage_domain == FALLBACK_MAP`
- cleanup is driven only by `producer_task_id.ring()`

Global invariant:

- owner and fallback share one core implementation
- differences must not grow into two independent algorithms

## Current Implementation Notes

The committed implementation also keeps two important user-facing choices:

- `sync_tensormap(uint8_t ring_id, int32_t sm_last_task_alive)` stays
  unchanged
- `with_alloc` follows allocation semantics, not `PTOParamType` alone

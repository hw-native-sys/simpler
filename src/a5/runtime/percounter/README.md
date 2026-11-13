# Percounter payload preparation

Each AICore lane claims one list entry at a time. The compile-time constant
`PERCOUNTER_PENDING_CAPACITY` controls how many ordinary candidates it retains;
the default is 1, and setting it to 4 enables a four-candidate window without
changing the scheduling code. A candidate can prepare its `DispatchPayload` once its argument
metadata is published, before its producers complete. Preparation reads tensor
descriptors and scalar arguments, not tensor contents or dispatch predicates.
Fanin readiness and predicate evaluation still gate kernel execution.

The host allocates `PERCOUNTER_PENDING_CAPACITY + 2` 512-byte GM payloads per
lane and seeds each payload's context pointers. Ordinary candidates own the
first `PERCOUNTER_PENDING_CAPACITY` slots; the next slot belongs to the lane's
ordinary MIX subtask and the last to its `sync_start` block. Host and device
derive this layout from the same constant and must be rebuilt together when
it changes. Moving a candidate in the local window retains its payload slot.
Slots are written back before the lane prepares or executes other work;
execution invalidates cached input data.

Each cluster's AIV0 announces a claimed ordinary MIX through a separate SSBUF
preparation record, including its task, block and participant mask. Named peers
prepare their own MIX payload without waiting for its producers or partners.
This record does not reserve a core and does not participate in the MIX/cohort
priority checks. A peer can continue ordinary work or execute its cohort while
the announced MIX remains unready.

In CPU simulation, concurrent preparation-field reads use acquire loads and
publication uses release stores.
These accesses remain atomic if a reader overlaps record reuse and rejects its
snapshot; reading a newer field also orders the opening generation before the
reader's closing generation check.
Initialization uses ordinary stores before the cluster-ready word is published.

Only the existing ready rendezvous admits MIX execution. Its fanin check,
cohort-hold checks, staged mask and execution-order checks remain authoritative.
A late peer prepares its payload before publishing arrival. An ordinary MIX
reuses the payload prepared from its announcement; a MIX cohort uses the
separate cohort slot. Different blocks and subtasks never share mutable block
contexts. Preparing a Graph body starts only after the AICPU publishes its
materialized metadata and order entry.

The allocation is bounded by lane count rather than task or SPMD block count.
It does not expand all task payloads on the Host or move payload construction
to the AICPU. The default uses 1536 bytes per lane; capacity 4 uses 3072 bytes.

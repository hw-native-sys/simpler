# Mixed asynchronous conditions and peer Graph tasks

Every AIV producer registers actual SDMA completion records interleaved with notification counters from all other ranks. AIC members of mixed tasks register counters. Consumers depend only on the producer or Graph EXIT and verify every record, all received data, notification payloads, and guards.

Two instances use distinct tensor views. Graph cases reuse one definition with parameter handles for the communication context and tag; the notification publisher stays outside the Graph. Deferred registration gates and a peer readiness exchange require all conditions to be registered before delayed notifications are sent. Counter-ready cases instead start with satisfied counters and transfer 4 MiB per peer.

The matrix covers two/four ranks, diagnostics off/on, ordinary AIV, two-block SPMD/sync-start, and MIX masks 3/5/6/7. The retained domain is read back after each run to check input data, notification windows and guards before release. This functional test does not measure arithmetic/DMA overlap or performance.

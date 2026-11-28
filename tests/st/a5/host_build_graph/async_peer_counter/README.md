# Deferred peer notification progress

Two ranks exchange payloads and notification counters in a retained communication domain. The business oracle checks complete payloads, consumer results, execution counts, notification values, registration counts, and neighboring guards.

- Epoch reuse runs notification values `1`, `2`, then resets the window and runs `1` again. Independent payload tags `101`, `102`, and `103` make stale data distinguishable from fresh data even after the notification value repeats.
- Saturation submits `C + 1` Cube waiters, where `C` is the accessible Cube count from `rt_available_cluster_count()`. The publisher may send data only after the receiver confirms that every waiter registered its condition.
- Both cases run with diagnostics disabled and at Chip Swimlane L2. Numerical and progress checks are identical; enabled diagnostics additionally check calendar execution coverage and match each defer registration with its deferred DONE.

The receiver registers its main waits on Cube cores. A Vector gate checks that all registrations arrived, then a Vector sender publishes READY to the other rank. Each receiver consumer depends on the READY send and its deferred waiter. The source's actual payload transfer and main notifications use the same Vector kernel in both cases.

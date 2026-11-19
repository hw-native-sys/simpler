# Deferred peer notification progress

Two ranks exchange payloads and notification counters in a retained communication domain. The business oracle checks complete payloads, consumer results, execution counts, notification values, registration counts, and neighboring guards.

- Epoch reuse runs notification values `1`, `2`, then resets the window and runs `1` again. Independent payload tags `101`, `102`, and `103` make stale data distinguishable from fresh data even after the notification value repeats.
- Saturation submits `C + 1` AIV waiters, where `C` comes from `rt_available_aiv_count()`. The publisher may send data only after the receiver confirms that every waiter registered its condition.
- Both cases run with diagnostics disabled and enabled. Numerical and progress checks are identical; enabled diagnostics additionally audit actual native execution and dependency records.

The registration gate is a bounded AIC test fixture; waiters and publisher use AIVs. Its polling does not consume the execution resources whose release is being tested. It is not an implementation of runtime WaitCheck.

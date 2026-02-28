# Codestyle Rules

1. **Avoid plan specific comments** such as Phase 1, Step 1, or Gap #3 which reflect planning details but don't aid code comprehension.
2. Use `enum class` preferentially for basic enumeration usage. Use `enum` only when implementing bitmask patterns or when bitwise operations are required.

    **Good:**
    ```cpp
    enum class CoreType : int { AIC = 0, AIV = 1 };
    CoreType type = CoreType::AIC;
    ```

    **Bad (unless implementing bitmask):**
    ```cpp
    enum CoreType { AIC = 0, AIV = 1 };  // Avoid this for basic enums
    ```

3. Prefer `volatile` decorator on struct members rather than volatile pointer casts unless necessary.
4. Avoid using pointer arithmetic with hardcoded offsets when `offsetof` is available.
5. **Never use `std::this_thread::yield()` or `sched_yield()` in AICPU spin-wait loops.** On the Ascend AICPU, yielding to the OS scheduler introduces unacceptable latency for tight spin-waits (ticket locks, CAS retries, etc.). Use an empty loop body or a bare architecture hint (`__asm__ volatile("yield")`) instead.

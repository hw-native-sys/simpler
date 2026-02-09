# Developer Guidelines

## Directory Ownership

Each developer role has a designated working directory. Stay within your assigned area unless explicitly requested by the user.

### Platform Developer
- **Working directory**: `src/platform/`
- Write platform-specific logic and abstractions here

### Runtime Developer
- **Working directory**: `src/runtime/`
- Write runtime logic including host, aicpu, aicore, and common modules here

### Codegen Developer
- **Working directory**: `examples/`
- Write code generation examples and kernel implementations here

## Important Rules

1. **Do not modify directories outside your assigned area** unless the user explicitly requests it
2. Create new subdirectories under your assigned directory as needed
3. When in doubt, ask the user before making changes to other areas
4. **Avoid including private information in documentation** such as usernames, absolute paths with usernames, or other personally identifiable information. Use relative paths or generic placeholders instead
5. **Avoid plan specific comments** such as Phase 1, Step 1, or Gap #3 which reflect planning details but don't aid code comprehension. This rule *does not apply* to commit info

## Coding Standards

1. Use `enum class` preferentially for basic enumeration usage. Use `enum` only when implementing bitmask patterns or when bitwise operations are required.

    **Good:**
    ```cpp
    enum class CoreType : int { AIC = 0, AIV = 1 };
    CoreType type = CoreType::AIC;
    ```

    **Bad (unless implementing bitmask):**
    ```cpp
    enum CoreType { AIC = 0, AIV = 1 };  // Avoid this for basic enums
    ```

2. Prefer `volatile` decorator on struct members rather than volatile pointer casts unless necessary.
3. Avoid using pointer arithmetic with hardcoded offsets when `offsetof` is available.

## Terminology (Based on Ascend NPU Architecture)

### Hardware Units

- **AIC** = **AICore-CUBE**: Matrix computation unit for tensor operations (matmul, convolution)
- **AIV** = **AICore-VECTOR**: Vector computation unit for element-wise operations (add, mul, activation)
- **AICPU**: Control processor for task scheduling and data movement (not a worker type - acts as scheduler)

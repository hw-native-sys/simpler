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

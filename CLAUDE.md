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

1. **Read `.ai-instruction/` directory first** before starting any work to understand codestyle rules and terminology guidelines
2. **Do not modify directories outside your assigned area** unless the user explicitly requests it
3. Create new subdirectories under your assigned directory as needed
4. When in doubt, ask the user before making changes to other areas
5. **Avoid including private information in documentation or code** such as usernames, absolute paths with usernames, or other personally identifiable information. Use relative paths or generic placeholders instead

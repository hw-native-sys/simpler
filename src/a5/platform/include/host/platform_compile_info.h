/**
 * Platform Compile Info Interface
 *
 * Minimal interface: platform only declares its identity.
 * Each platform (a5, a5sim, ...) implements get_platform() to return
 * its name. Runtime code uses this to decide which toolchain to use.
 */

#ifndef PLATFORM_COMPILE_INFO_H
#define PLATFORM_COMPILE_INFO_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get the platform name.
 *
 * @return Platform identifier string (e.g., "a5", "a5sim")
 */
const char* get_platform(void);

#ifdef __cplusplus
}
#endif

#endif /* PLATFORM_COMPILE_INFO_H */

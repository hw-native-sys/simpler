/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__aarch64__)
#define DLSYM_GLIBC_VERSION "GLIBC_2.17"
#elif defined(__x86_64__)
#define DLSYM_GLIBC_VERSION "GLIBC_2.2.5"
#else
#error "The stock capture fixture requires an AArch64 or x86-64 Linux host."
#endif

typedef void *(*lookup_t)(void *, const char *);
typedef int (*destroy_t)(void *);
static _Atomic(lookup_t) system_lookup;
static _Atomic(destroy_t) sdk_destroy;
static _Atomic(uintptr_t) saved_model;
static _Atomic(unsigned) matching_lookups, injected, real_calls, blocked_calls;
static _Atomic(int) last_real_rc;
static _Thread_local int armed, injected_here;
static char provider[PATH_MAX];

enum { synthetic_refusal = 923701, blocked_repeat = 923702, refused_recovery = -1 };

int stock_capture_configure(const char *path) {
    if (provider[0] || !realpath(path, provider)) return -1;
    return 0;
}

int stock_capture_arm(void) {
    if (!provider[0] || armed || atomic_load(&injected) || atomic_load(&saved_model)) return -1;
    armed = 1;
    return 0;
}

void stock_capture_disarm(void) { armed = 0; }

uint64_t stock_capture_stat(unsigned key) {
    switch (key) {
    case 0:
        return atomic_load(&matching_lookups);
    case 1:
        return atomic_load(&injected);
    case 2:
        return atomic_load(&real_calls);
    case 3:
        return atomic_load(&saved_model);
    case 4:
        return (uint32_t)atomic_load(&last_real_rc);
    case 5:
        return atomic_load(&blocked_calls);
    default:
        abort();
    }
}

int stock_capture_recover(uint64_t expected_model) {
    destroy_t fn = atomic_load(&sdk_destroy);
    if (armed || !injected_here || !fn || !expected_model || atomic_load(&injected) != 1 ||
        expected_model != atomic_load(&saved_model) || atomic_load(&blocked_calls))
        return refused_recovery;
    unsigned expected_calls = 0;
    if (!atomic_compare_exchange_strong(&real_calls, &expected_calls, 1)) return refused_recovery;
    const int rc = fn((void *)(uintptr_t)expected_model);
    const int saved_errno = errno;
    atomic_store(&last_real_rc, rc);
    dprintf(
        STDERR_FILENO, "STOCK_CAPTURE_DESTROY rescue model=%p rc=%d sdk_called=1\n", (void *)(uintptr_t)expected_model,
        rc
    );
    errno = saved_errno;
    return rc;
}

static int dispatch_destroy(void *model) {
    destroy_t fn = atomic_load(&sdk_destroy);
    if (!fn) abort();
    if (armed) {
        armed = 0;
        if (!model || atomic_exchange(&injected, 1)) abort();
        atomic_store(&saved_model, (uintptr_t)model);
        injected_here = 1;
        dprintf(
            STDERR_FILENO, "STOCK_CAPTURE_DESTROY synthetic model=%p rc=%d sdk_called=0\n", model, synthetic_refusal
        );
        return synthetic_refusal;
    }
    if (model && (uintptr_t)model == atomic_load(&saved_model)) {
        atomic_fetch_add(&blocked_calls, 1);
        dprintf(STDERR_FILENO, "STOCK_CAPTURE_DESTROY unexpected repeat model=%p sdk_called=0\n", model);
        return blocked_repeat;
    }
    return fn(model);
}

void *dlsym(void *handle, const char *name) {
    lookup_t lookup = atomic_load(&system_lookup);
    if (!lookup) {
        lookup = (lookup_t)dlvsym(RTLD_NEXT, "dlsym", DLSYM_GLIBC_VERSION);
        if (!lookup) abort();
        atomic_store(&system_lookup, lookup);
    }
    if (handle == RTLD_DEFAULT || handle == RTLD_NEXT || !provider[0] || strcmp(name, "aclmdlRIDestroy") != 0)
        return lookup(handle, name);
    void *address = lookup(handle, name);
    if (!address) return address;
    Dl_info origin;
    char resolved[PATH_MAX];
    int saved_errno = errno;
    if (!dladdr(address, &origin) || !realpath(origin.dli_fname, resolved) || strcmp(resolved, provider) != 0) {
        errno = saved_errno;
        return address;
    }
    destroy_t fn = (destroy_t)address;
    destroy_t original = atomic_load(&sdk_destroy);
    if (original && original != fn) abort();
    atomic_store(&sdk_destroy, fn);
    atomic_fetch_add(&matching_lookups, 1);
    dprintf(STDERR_FILENO, "STOCK_CAPTURE_LOOKUP symbol=aclmdlRIDestroy provider=%s address=%p\n", resolved, address);
    errno = saved_errno;
    return (void *)dispatch_destroy;
}

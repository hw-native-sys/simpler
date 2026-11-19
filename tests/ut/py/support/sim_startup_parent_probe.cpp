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

#include <pthread.h>
#include <semaphore.h>
#include <cerrno>
#include <cstdio>
sem_t go, done;
pthread_key_t key;
int background_rc = -1;
void *noop(void *) { return nullptr; }
void *background(void *) {
    sem_wait(&go);
    pthread_t child;
    background_rc = pthread_create(&child, nullptr, noop, nullptr);
    if (background_rc == 0) pthread_join(child, nullptr);
    sem_post(&done);
    return nullptr;
}
void *trigger(void *) {
    pthread_setname_np(pthread_self(), "sim-aic-0");
    pthread_setspecific(key, reinterpret_cast<void *>(1));
    return nullptr;
}
int main() {
    sem_init(&go, 0, 0);
    sem_init(&done, 0, 0);
    pthread_key_create(&key, nullptr);
    pthread_t bg, first, peer;
    if (pthread_create(&bg, nullptr, background, nullptr)) return 2;
    if (pthread_create(&first, nullptr, trigger, nullptr)) return 3;
    pthread_join(first, nullptr);
    sem_post(&go);
    sem_wait(&done);
    int rc = pthread_create(&peer, nullptr, noop, nullptr);
    if (rc == 0) pthread_join(peer, nullptr);
    pthread_join(bg, nullptr);
    std::printf("background=%d target=%d\n", background_rc, rc);
    return background_rc == 0 && rc == EAGAIN ? 0 : 1;
}

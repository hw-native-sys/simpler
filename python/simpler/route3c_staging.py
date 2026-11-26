"""Give a published dependent frame one bounded opportunity to stage."""

import mmap
import os
import struct
import time

PRED = 8
SUCC = 16
GRACE_NS = 400_000
COUNTS = struct.Struct("<QQ")


def mapping(path):
    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    try:
        return mmap.mmap(fd, 0, access=mmap.ACCESS_WRITE)
    finally:
        os.close(fd)


class StagingPriority:
    def __init__(self, path, rank):
        self.state = mapping(path)
        self.offset = 64 + rank * COUNTS.size
        self.checked = 0
        self.waited = 0
        self.predecessor = None
        self.deadline_ns = 0

    def defer_observation(self, run, stages):
        if not self.state[0]:
            return False
        predecessor, successor = struct.unpack_from("<QQ", self.state, PRED)
        if predecessor != run or successor <= run:
            return False
        if any(value.identity[1] == successor for value in stages):
            return False
        now = time.monotonic_ns()
        if self.predecessor != run:
            self.predecessor = run
            self.deadline_ns = now + GRACE_NS
            self.checked += 1
            COUNTS.pack_into(self.state, self.offset, self.checked, self.waited)
        if now < self.deadline_ns:
            return True
        if self.deadline_ns:
            self.waited += 1
            self.deadline_ns = 0
            COUNTS.pack_into(self.state, self.offset, self.checked, self.waited)
        return False

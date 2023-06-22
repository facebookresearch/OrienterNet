# Copyright (c) Meta Platforms, Inc. and affiliates.

import time


class Timer:
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.duration = time.time() - self.tstart
        if self.name is not None:
            print("[%s] Elapsed: %s" % (self.name, self.duration))

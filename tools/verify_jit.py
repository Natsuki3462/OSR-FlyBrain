
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from brain.circuit import Circuit
from brain.lif import LIFNetwork


def main():
    c = Circuit()
    pre, post, w = c.edge_arrays()
    a = LIFNetwork(c.n, pre, post, w, seed=0, backend="numpy")
    b = LIFNetwork(c.n, pre, post, w, seed=0, backend="numba")
    print(f"backend: numpy={a.backend} numba={b.backend}")

    a.noise = 0.0
    b.noise = 0.0
    a.baseline[:] = 0.02
    b.baseline[:] = 0.02

    rng = np.random.default_rng(7)
    steps = 4000
    drives = []
    for t in range(steps):
        d = np.zeros(c.n, dtype=np.float32)
        if t % 250 < 2:
            d[rng.choice(c.n, 60, replace=False)] = 3.0
        drives.append(d)

    sa, sb = [], []
    for d in drives[:300]:
        a.step(d)
        b.step(d)
    a.reset()
    b.reset()
    a.noise = 0.0
    b.noise = 0.0
    a.baseline[:] = 0.02
    b.baseline[:] = 0.02

    t0 = time.perf_counter()
    for d in drives:
        sa.append(a.step(d))
    ta = time.perf_counter() - t0
    t0 = time.perf_counter()
    for d in drives:
        sb.append(b.step(d))
    tb = time.perf_counter() - t0

    dv = np.abs(a.v - b.v).max()
    da = np.abs(a.adapt - b.adapt).max()
    dr = np.abs(a.rate - b.rate).max()
    same = all(np.array_equal(x, y) for x, y in zip(sa, sb))
    na = sum(x.size for x in sa)
    nb = sum(x.size for x in sb)
    print(f"放电 {na} vs {nb}  序列完全一致: {same}")
    print(f"max|dv|={dv:.3e}  max|dadapt|={da:.3e}  max|drate|={dr:.3e}")
    print(f"numpy {ta * 1e3 / steps * 1000:.2f} us/步 -> {1000 / (ta * 1e3 / steps * 1000):.1f}x")
    print(f"numba {tb * 1e3 / steps * 1000:.2f} us/步 -> {1000 / (tb * 1e3 / steps * 1000):.1f}x")
    print(f"加速 {ta / tb:.1f}x")
    return 0 if same and dv < 1e-5 else 1


if __name__ == "__main__":
    sys.exit(main())

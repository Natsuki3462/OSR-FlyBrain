
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from brain.circuit import Circuit
from brain.lif import build_from_circuit

WINDOW_MS = 25


def trial(net, idx, gf_mask, amp=3.0):
    net.reset()
    net.baseline[:] = 0.02
    net.noise = 0.03
    d = np.zeros(net.n, dtype=np.float32)
    d[idx] = amp
    for t in range(WINDOW_MS):
        s = net.step(d if t < 2 else None)
        if s.size and gf_mask[s].any():
            return t
    return None


def run(path, trials=20, stim=60, seed0=0):
    c = Circuit(path or None)
    net = build_from_circuit(c, seed=seed0)
    gf = np.asarray(c.by_role.get("escape_gf", []), dtype=np.int32)
    gfm = np.zeros(c.n, dtype=bool)
    gfm[gf] = True
    loom = np.asarray(c.indices(role="loom_lc4")
                      + c.indices(role="loom_lplc2"), dtype=np.int32)
    sensory = set(loom.tolist()) | set(gf.tolist())
    other = np.asarray([i for i in range(c.n) if i not in sensory], dtype=np.int32)

    def hit_rate(pool, label):
        rng = np.random.default_rng(7)
        hits, lats = 0, []
        k = min(stim, pool.size)
        for _ in range(trials):
            pick = rng.choice(pool, size=k, replace=False)
            t = trial(net, pick, gfm)
            if t is not None:
                hits += 1
                lats.append(t)
        med = float(np.median(lats)) if lats else float("nan")
        print(f"  {label:<10} {hits:>2}/{trials}  延迟中位 {med:.0f} ms")
        return hits

    print(f"{os.path.basename(path or 'circuit.json')}: "
          f"{c.n} 神经元 / {len(c.edges)} 突触")
    a = hit_rate(loom, "loom")
    b = hit_rate(other, "非感觉")
    ratio = (a / b) if b else float("inf")
    print(f"  比值 {ratio:.1f}x  "
          f"{'[通过]' if (a >= trials * 0.9 and b <= trials * 0.4 and ratio > 2) else '[未通过]'}")
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", nargs="*", default=[""])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--stim", type=int, default=60)
    a = ap.parse_args()
    for p in a.circuit:
        run(p, a.trials, a.stim)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

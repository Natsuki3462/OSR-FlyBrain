from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from brain.circuit import Circuit
from brain.lif import build_from_circuit

BAR = " .:-=+*#%@"


def spark(v, vmax):
    if vmax <= 0:
        return BAR[0]
    i = int(min(1.0, v / vmax) * (len(BAR) - 1))
    return BAR[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, default=1500)
    ap.add_argument("--loom-at", type=int, default=500)
    ap.add_argument("--loom-ms", type=int, default=2)
    ap.add_argument("--amp", type=float, default=3.0)
    ap.add_argument("--stim", type=int, default=60,
                    help="参与刺激的感觉神经元数量（0=全部）")
    ap.add_argument("--baseline", type=float, default=0.02)
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--every", type=int, default=25)
    a = ap.parse_args()

    c = Circuit()
    net = build_from_circuit(c, seed=0)
    net.baseline[:] = a.baseline
    net.noise = a.noise

    roles = [r for r in c.roles if r != "partner"]
    groups = {r: c.indices(role=r) for r in roles}
    print(c.describe())
    print()

    head = f"{'t(ms)':>6} {'spikes':>7} " + " ".join(f"{r[:9]:>9}" for r in roles)
    print(head)
    print("-" * len(head))

    loom = c.indices(role="loom_lc4") + c.indices(role="loom_lplc2")
    if a.stim and a.stim < len(loom):
        loom = np.random.default_rng(0).choice(loom, size=a.stim, replace=False)
    drive = np.zeros(c.n, dtype=np.float32)
    if len(loom):
        drive[np.asarray(loom, dtype=np.int32)] = a.amp

    t0 = time.perf_counter()
    peak = {r: 0.0 for r in roles}
    gf_hits = []
    for t in range(a.ms):
        active = a.loom_at <= t < a.loom_at + a.loom_ms
        spikes = net.step(drive if active else None)
        if spikes.size:
            for s in spikes:
                if s in groups.get("escape_gf", []):
                    gf_hits.append(t)
        if t % a.every == 0:
            cells = []
            for r in roles:
                v = net.group_rate(groups[r])
                peak[r] = max(peak[r], v)
                cells.append(f"{spark(v, 0.02):>9}")
            print(f"{t:>6} {len(spikes):>7} " + " ".join(cells))
    el = time.perf_counter() - t0

    print()
    print(f"仿真 {a.ms} ms，耗时 {el * 1000:.0f} ms，实时倍率 {a.ms / (el * 1000):.1f}x")
    if gf_hits:
        first = gf_hits[0]
        print(f"巨纤维首次放电：t={first} ms（loom 起始 {a.loom_at} ms，"
              f"延迟 {first - a.loom_at} ms），共 {len(gf_hits)} 次")
    else:
        print("巨纤维未放电")
    top = sorted(peak.items(), key=lambda kv: -kv[1])[:5]
    print("峰值放电率：" + "  ".join(f"{k}={v:.3f}" for k, v in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())

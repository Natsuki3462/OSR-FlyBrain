
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from brain.circuit import Circuit
from brain.lif import build_from_circuit


def drive_test(net, idx, amp=3.0, ms=200, baseline=0.02, noise=0.03):
    net.reset()
    net.baseline[:] = baseline
    net.noise = noise
    d = np.zeros(net.n, dtype=np.float32)
    d[idx] = amp
    hit = 0
    for _ in range(ms):
        hit += int(net.step(d).size)
    return hit


def roles_of(c, ids):
    m = {v: k for k, v in c.id_to_index().items()}
    out = Counter()
    idx2role = {i: nb.get("role", "") for i, nb in enumerate(c.neurons)}
    for i in ids:
        out[idx2role.get(int(i), "?")] += 1
    return out


def diagnose(c, role):
    net = build_from_circuit(c)
    net.baseline[:] = 0.02
    net.noise = 0.03
    idx = np.asarray(c.by_role.get(role, []), dtype=np.int32)
    if idx.size == 0:
        print(f"[{role}] 电路里没有这个角色")
        return

    pre, post, w = c.edge_arrays()
    print(f"[{role}] {idx.size} 个神经元")
    for i in idx:
        m = post == i
        e = float(w[m & (w > 0)].sum())
        ih = float(-w[m & (w < 0)].sum())
        ups = pre[m]
        src = roles_of(c, ups)
        top = ", ".join(f"{k}×{v}" for k, v in src.most_common(5))
        print(f"  #{i:<5} 兴奋 {e:5.2f}  抑制 {ih:5.2f}  "
              f"上游 {ups.size:3d}  净 {e - ih:+.2f}")
        print(f"          来源: {top}")


    net.reset()
    net.baseline[:] = 0.02
    net.noise = 0.03
    sp = 0
    for _ in range(400):
        sp += int(net.step(None).size)
    print(f"  自发（400 ms，无驱动）: {sp} 个尖峰")

    hits = drive_test(net, idx, amp=3.0)
    print(f"  直接驱动 {idx.size} 个（amp=3.0, 200 ms）: {hits} 个尖峰")


    ups = np.unique(pre[np.isin(post, idx)])
    if ups.size:
        k = min(60, ups.size)
        sub = ups[:k]
        hits2 = drive_test(net, sub, amp=3.0)
        print(f"  驱动上游 {k} 个同步（amp=3.0, 200 ms）: {hits2} 个尖峰")
    else:
        print("  没有任何上游")


def all_rates(c):
    net = build_from_circuit(c)
    net.baseline[:] = 0.02
    net.noise = 0.03
    for _ in range(300):
        net.step(None)
    print(f"{'角色':<14}{'神经元':>7}{'自发率':>10}{'最大':>9}")
    print("-" * 42)
    for role in c.roles:
        idx = np.asarray(c.by_role[role], dtype=np.int32)
        r = net.rate[idx]
        print(f"{role:<14}{idx.size:>7}{r.mean():>10.5f}{r.max():>9.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="")
    ap.add_argument("--role", default="")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    c = Circuit(a.circuit or None)
    print(f"电路 {c.n} 神经元 / {len(c.edges)} 突触\n")
    if a.all or not a.role:
        all_rates(c)
    if a.role:
        diagnose(c, a.role)
    return 0


if __name__ == "__main__":
    sys.exit(main())

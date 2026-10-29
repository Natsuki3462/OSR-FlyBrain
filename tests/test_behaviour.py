from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from brain.circuit import Circuit
from brain.lif import LIFNetwork, build_from_circuit

PASS = 0
FAIL = 0

BASE = 0.02
NOISE = 0.03
AMP = 0.30
SEEDS = 8


def ck(cond, msg, detail=None):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK]   {msg}")
    else:
        FAIL += 1
        extra = f"  -> {detail}" if detail is not None else ""
        print(f"  [FAIL] {msg}{extra}")


def fresh(circuit, baseline=BASE, noise=NOISE, seed=0, **kw):
    net = build_from_circuit(circuit, seed=seed, **kw)
    net.baseline[:] = baseline
    net.noise = noise
    return net


def drive_vector(n, idx, amp):
    v = np.zeros(n, dtype=np.float32)
    if len(idx):
        v[np.asarray(idx, dtype=np.int32)] = amp
    return v


def gf_latency(net, gf, idx, steps=60, pulse=2, amp=3.0):
    d = drive_vector(net.n, idx, amp)
    watch = np.asarray(gf, dtype=np.int32)
    for t in range(steps):
        spikes = net.step(d if t < pulse else None)
        if spikes.size and np.isin(spikes, watch).any():
            return t
    return None


def main():
    t0 = time.time()
    print("载入电路")
    c = Circuit()
    print("  " + c.describe().replace("\n", "\n  "))
    print(f"  载入耗时 {time.time() - t0:.2f}s\n")

    gf = c.indices(role="escape_gf")
    lc4 = c.indices(role="loom_lc4")
    lp = c.indices(role="loom_lplc2")
    loom = np.concatenate([lc4, lp])
    steer_l = c.indices(role="steer_left")
    steer_r = c.indices(role="steer_right")
    wind = c.indices(role="wind")

    print("[1] 电路完整性")
    ck(c.n > 300, f"神经元数量合理（{c.n}）", c.n)
    ck(len(c.edges) > 3000, f"突触数量合理（{len(c.edges)}）", len(c.edges))
    ck(len(gf) == 2, f"巨纤维 DNp01 有 2 个（{len(gf)}）", len(gf))
    ck(len(lc4) == 104, f"LC4 有 104 个（{len(lc4)}）", len(lc4))
    ck(len(lp) == 210, f"LPLC2 有 210 个（{len(lp)}）", len(lp))
    ck(len(steer_l) == 2 and len(steer_r) == 2,
       f"左右转向群各 2 个（{len(steer_l)}/{len(steer_r)}）")
    ck(len(wind) > 0, f"风感 JO 神经元存在（{len(wind)}）", len(wind))

    print("\n[2] 静息状态：无输入时不得自发发放")
    net = fresh(c, baseline=0.0, noise=0.0, seed=1)
    net.run(1000)
    ck(net.spike_count == 0, "零驱动下完全不发放", net.spike_count)
    quiet = 0
    for s in range(SEEDS):
        n2 = fresh(c, seed=100 + s)
        n2.run(300)
        quiet += n2.spike_count
    ck(quiet < 200, f"生理静息下活动极低（{SEEDS} 次共 {quiet} 尖峰）", quiet)

    print("\n[3] 刺激 → 巨纤维响应")
    lats = []
    for s in range(SEEDS):
        net = fresh(c, seed=2 + s)
        lat = gf_latency(net, gf, loom)
        if lat is not None:
            lats.append(lat)
    ck(len(lats) == SEEDS,
       f"{len(lats)}/{SEEDS} 次 loom 均触发巨纤维", lats)
    if lats:
        med = float(np.median(lats))
        ck(med < 25, f"巨纤维响应延迟中位 {med:.0f} ms（阈值 <25 ms）", med)

    print("\n[4] 选择性：loom 局部同步刺激 vs 等规模非感觉刺激")
    stim = 60
    exclude = set(int(i) for i in loom) | set(int(i) for i in gf)
    pool = [i for i in range(c.n) if i not in exclude]
    hit_loom = hit_rand = 0
    lat_loom = []
    for s in range(20):
        rng = np.random.default_rng(1000 + s)
        sub = rng.choice(loom, size=stim, replace=False)
        lat = gf_latency(fresh(c, seed=40 + s), gf, sub)
        if lat is not None:
            hit_loom += 1
            lat_loom.append(lat)
        sub2 = rng.choice(pool, size=stim, replace=False)
        if gf_latency(fresh(c, seed=40 + s), gf, sub2) is not None:
            hit_rand += 1
    ck(hit_loom >= 18,
       f"{stim} 个 loom 神经元触发巨纤维 {hit_loom}/20（要求 >=18）", hit_loom)
    ck(hit_rand <= 8,
       f"{stim} 个非感觉神经元触发 {hit_rand}/20（要求 <=8）", hit_rand)
    ck(hit_loom > hit_rand * 2,
       f"loom 触发率是非感觉群的 {hit_loom / max(1, hit_rand):.1f} 倍（要求 >2）",
       (hit_loom, hit_rand))
    if lat_loom:
        ck(float(np.median(lat_loom)) < 25,
           f"loom 响应延迟中位 {np.median(lat_loom):.0f} ms（阈值 <25）")

    print("\n[5] 不应期")
    net = LIFNetwork(1, np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                     np.array([], dtype=np.float64), adapt_gain=0.0, seed=4)
    net.noise = 0.0
    times = [t for t in range(40)
             if net.step(np.array([100.0], dtype=np.float32)).size]
    gaps = [b - a for a, b in zip(times, times[1:])]
    ck(bool(gaps) and min(gaps) >= 2,
       f"饱和驱动下最短间隔 {min(gaps) if gaps else None} ms（应 >=2）", gaps)

    print("\n[6] 突触符号与量级")
    pre, post, w = c.edge_arrays()
    ck((w < 0).any() and (w > 0).any(),
       f"存在兴奋与抑制两类突触（+{int((w > 0).sum())}/-{int((w < 0).sum())}）")
    net = LIFNetwork(2, np.array([0], dtype=np.int32), np.array([1], dtype=np.int32),
                     np.array([-0.8], dtype=np.float64), adapt_gain=0.0, seed=0)
    net.noise = 0.0
    neg_at = None
    for t in range(10):
        net.step(np.array([50.0, 0.0], dtype=np.float32) if t == 0 else None)
        if neg_at is None and net.v[1] < 0:
            neg_at = t
    ck(neg_at == 4, f"抑制性输入在第 {neg_at} 步到达（应为 4 ms）", neg_at)

    print("\n[7] 实时性能（内部 1 kHz）")
    net = fresh(c, seed=5)
    net.run(100)
    steps = 2000
    t0 = time.perf_counter()
    net.run(steps)
    el = time.perf_counter() - t0
    rtf = (steps / 1000.0) / el
    ck(rtf > 1.0, f"实时倍率 {rtf:.1f}x（{c.n} 神经元 / {len(c.edges)} 突触）", rtf)
    print(f"         单步 {el / steps * 1000:.3f} ms，可用预算 1.000 ms")

    print("\n" + "-" * 56)
    print(f"结果：{PASS} 项通过，{FAIL} 项失败")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

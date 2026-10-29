
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from brain.circuit import Circuit
from brain.lif import LIFNetwork, build_from_circuit


def timed(fn, reps):

    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e6


def bench_net(net, steps, baseline=0.02, noise=0.03, warmup=500):
    net.reset()
    net.baseline[:] = baseline
    net.noise = noise
    for _ in range(warmup):
        net.step(None)
    net.reset()
    net.baseline[:] = baseline
    net.noise = noise
    t0 = time.perf_counter()
    spikes = 0
    for _ in range(steps):
        spikes += net.step(None).size
    el = time.perf_counter() - t0
    return el / steps * 1e6, spikes / steps


def profile():
    c = Circuit()
    net = build_from_circuit(c, seed=0)
    net.baseline[:] = 0.02
    net.noise = 0.03
    for _ in range(1000):
        net.step(None)

    n = net.n
    reps = 20000
    v = net.v
    ad = net.adapt
    r = net.refr
    act = net.active
    dl = net._dl

    rows = []
    rows.append(("v -= v*leak", timed(lambda: np.subtract(v, v * net.leak, out=v), reps)))
    rows.append(("v += baseline", timed(lambda: np.add(v, net.baseline, out=v), reps)))
    rows.append(("noise randn", timed(
        lambda: np.add(v, net.rng.standard_normal(n, dtype=np.float32) * 0.03, out=v), reps // 4)))
    rows.append(("adapt decay", timed(lambda: np.subtract(ad, ad * net.adapt_alpha, out=ad), reps)))
    rows.append(("v -= adapt", timed(lambda: np.subtract(v, ad, out=v), reps)))
    rows.append(("refr.any()", timed(lambda: r.any(), reps)))
    rows.append(("refr mask+apply", timed(
        lambda: np.subtract(r, 1, out=r, where=r > 0), reps)))
    rows.append(("flatnonzero(th)", timed(lambda: np.flatnonzero(v >= net.v_th), reps)))
    rows.append(("active[:]=0", timed(lambda: act.fill(0.0), reps)))
    rows.append(("dl shift", timed(lambda: dl.__setitem__((net._dl_pos + 1) % (net.inh_delay + 1), 0.0), reps)))
    rows.append(("v -= dl", timed(lambda: np.subtract(v, dl[0], out=v), reps)))
    rows.append(("rate EMA", timed(
        lambda: np.add(net.rate, (act - net.rate) * net.rate_alpha, out=net.rate), reps)))

    pre_e, post_e, w_e = net.pre_e, net.post_e, net.w_e
    pre_i, post_i, w_i = net.pre_i, net.post_i, net.w_i

    def bin_e():
        np.bincount(post_e, weights=w_e * act[pre_e], minlength=n)

    def bin_i():
        np.bincount(post_i, weights=w_i * act[pre_i], minlength=n)

    rows.append(("bincount E (%.0f edges)" % pre_e.size, timed(bin_e, reps // 2)))
    rows.append(("bincount I (%.0f edges)" % pre_i.size, timed(bin_i, reps // 2)))

    per_step, rate_hz = bench_net(net, 2000)

    print(f"网络：{n} 神经元 / {len(c.edges)} 突触，平均放电 {rate_hz:.2f} 个/步")
    print(f"实测整步：{per_step:.2f} us  ->  {1000.0 / per_step:.1f}x 实时\n")
    print(f"{'操作':<28}{'us/次':>9}{'占整步':>9}")
    print("-" * 46)
    total = 0.0
    for name, us in rows:
        total += us
        print(f"{name:<28}{us:>9.2f}{us / per_step * 100:>8.1f}%")
    print("-" * 46)
    print(f"{'各项之和':<28}{total:>9.2f}{total / per_step * 100:>8.1f}%")
    print(f"{'整步实测':<28}{per_step:>9.2f}{100.0:>8.1f}%")
    return 0


def synth(n, per_neur=28, gain_e=3.2, gain_i=1.8, seed=0):

    rng = np.random.default_rng(seed)
    m = n * per_neur
    pre = rng.integers(0, n, m, dtype=np.int32)
    post = rng.integers(0, n, m, dtype=np.int32)
    exc = rng.random(m) < 0.5
    tot_e = np.bincount(post[exc], minlength=n).astype(np.float32)
    tot_i = np.bincount(post[~exc], minlength=n).astype(np.float32)
    w = np.empty(m, dtype=np.float32)
    w[exc] = gain_e / np.maximum(tot_e[post[exc]], 1.0)
    w[~exc] = -gain_i / np.maximum(tot_i[post[~exc]], 1.0)
    return pre, post, w


def scale(sizes, steps=1500, drive=0.06, backend="auto"):
    print(f"{'神经元':>8}{'突触':>10}{'us/步':>10}{'实时倍率':>11}{'放电/步':>10}")
    print("-" * 52)
    for n in sizes:
        pre, post, w = synth(n)
        net = LIFNetwork(n, pre, post, w, seed=0, backend=backend)
        net.baseline[:] = drive
        us, sp = bench_net(net, steps, baseline=drive, warmup=200)
        print(f"{n:>8}{len(pre):>10}{us:>10.1f}{1000.0 / us:>10.1f}x{sp:>10.2f}")
    return 0


def workload(ms=20000):

    from body.world import World

    c = Circuit()
    net = build_from_circuit(c, seed=0)
    net.baseline[:] = 0.02
    net.noise = 0.03
    world = World(c, mode="demo", seed=0)

    drives = [world.step(t).copy() for t in range(ms)]
    for d in drives[:500]:
        net.step(d)

    net.reset()
    net.baseline[:] = 0.02
    net.noise = 0.03
    t0 = time.perf_counter()
    spikes = 0
    for d in drives:
        spikes += net.step(d).size
    el = time.perf_counter() - t0
    us = el / ms * 1e6

    t0 = time.perf_counter()
    for t in range(ms):
        world.step(t)
    wus = (time.perf_counter() - t0) / ms * 1e6

    print(f"真实工作负载 {ms} ms（{len(world.events)} 次事件）")
    print(f"  net.step   {us:8.2f} us/步   放电 {spikes / ms:.2f} 个/步")
    print(f"  world.step {wus:8.2f} us/步")
    print(f"  合计       {us + wus:8.2f} us/步 -> {1000.0 / (us + wus):.1f}x 实时")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--scale", nargs="*", type=int, default=None)
    ap.add_argument("--workload", type=int, default=0)
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "numpy", "numba"])
    a = ap.parse_args()
    if a.workload:
        return workload(a.workload)
    if a.scale:
        return scale(a.scale, backend=a.backend)
    return profile()


if __name__ == "__main__":
    sys.exit(main())


from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from brain.circuit import Circuit
from brain.lif import build_from_circuit
from body.mapping import LOOM_SYNC_MIN
from body.world import World

DOWNSTREAM = ("jo_down", "wing_a", "wing_b", "wing_c",
              "escape_gf", "backward", "forward", "steer_left", "steer_right")


def measure(path, ms=3000, seed=0, amp=3.0, stim=20):
    c = Circuit(path or None)
    net = build_from_circuit(c, seed=seed)
    net.baseline[:] = 0.02
    net.noise = 0.03
    world = World(c, mode="wind", seed=seed, amp=amp, stim=stim)

    roles = sorted({n.get("role", "") for n in c.neurons})
    idx = {r: np.asarray(c.indices(role=r), dtype=np.int32) for r in roles}


    slot_of = np.full(c.n, -1, dtype=np.int32)
    for k, r in enumerate(roles):
        if idx[r].size:
            slot_of[idx[r]] = k

    loom = idx.get("loom_lc4", np.zeros(0, np.int32))
    loom = np.concatenate([loom, idx.get("loom_lplc2", np.zeros(0, np.int32))])
    loom_f = np.zeros(c.n, dtype=np.float32)
    loom_f[loom] = 1.0

    gf = idx.get("escape_gf", np.zeros(0, np.int32))
    gf_mask = np.zeros(c.n, dtype=bool)
    gf_mask[gf] = True

    counts = {r: 0 for r in roles}
    peak_sync = 0
    gf_frames = 0
    wind_events = 0
    last_ev = -1

    for t in range(ms):
        spikes = net.step(world.step(t))
        if spikes.size:
            hit = slot_of[spikes]
            hit = hit[hit >= 0]
            if hit.size:
                for k, v in enumerate(np.bincount(hit, minlength=len(roles))):
                    if v:
                        counts[roles[k]] += int(v)
            if gf_mask[spikes].any():
                gf_frames += 1
        if loom.size:
            s = int(np.dot(net.active, loom_f))
            if s > peak_sync:
                peak_sync = s
        if len(world.events) > last_ev:
            last_ev = len(world.events)
            wind_events += 1

    return {
        "n": c.n,
        "counts": counts,
        "peak_sync": peak_sync,
        "gf_frames": gf_frames,
        "wind_events": wind_events,
        "ms": ms,
    }


def report(res, label):
    print(f"\n=== {label}（{res['n']} 神经元 / {res['ms']} ms / "
          f"{res['wind_events']} 次风事件）===")
    print(f"  {'群':<12}{'尖峰':>8}{'每事件':>10}")
    for r in ("wind",) + DOWNSTREAM:
        v = res["counts"].get(r, 0)
        if r in res["counts"] or v:
            per = v / max(1, res["wind_events"])
            print(f"  {r:<12}{v:>8}{per:>10.1f}")
    other = sum(v for r, v in res["counts"].items() if r not in ("wind",) + DOWNSTREAM)
    print(f"  {'(其他)':<12}{other:>8}")
    print(f"  loom 同步峰值 {res['peak_sync']}（门控阈值 {LOOM_SYNC_MIN}）"
          f" → {'安全' if res['peak_sync'] < LOOM_SYNC_MIN else '会误判为逃逸'}")
    print(f"  巨纤维放电步数 {res['gf_frames']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="data/circuit.json")
    ap.add_argument("--baseline", default="data/circuit_571_backup.json",
                    help="对照电路（无 jo_down 层），传空串跳过")
    ap.add_argument("--ms", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", type=float, default=3.0)
    ap.add_argument("--stim", type=int, default=20)
    a = ap.parse_args()

    res = measure(a.circuit, a.ms, a.seed, a.amp, a.stim)
    report(res, a.circuit)

    if a.baseline:
        base = measure(a.baseline, a.ms, a.seed, a.amp, a.stim)
        report(base, a.baseline + "（对照）")
        print("\n--- 对照差异 ---")
        for r in ("wind",) + DOWNSTREAM:
            x = base["counts"].get(r, 0)
            y = res["counts"].get(r, 0)
            if x or y:
                print(f"  {r:<12}{x:>8} → {y:>8}")

    ok = (res["counts"].get("wind", 0) > 0
          and res["counts"].get("jo_down", 0) > 0
          and any(res["counts"].get(r, 0) > 0
                  for r in ("wing_a", "wing_b", "wing_c", "escape_gf", "backward")))
    print("\n结论:", "通路贯通" if ok else "仍未贯通")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

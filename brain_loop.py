from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from body.emu import EmuDevice, AXES6
from body.mapping import SixAxisMapper
from body.world import World
from brain.circuit import Circuit
from brain.lif import build_from_circuit

OUT_HZ = 50
STEPS_PER_FRAME = 1000 // OUT_HZ
LOOM_WINDOW = 25


def run(ms, scenario, seed, stim, amp, realtime, web, port, out_path,
        feedback=0.0):
    c = Circuit()
    net = build_from_circuit(c, seed=seed)
    net.baseline[:] = 0.02
    net.noise = 0.03

    world = World(c, mode=scenario, seed=seed, stim=stim, amp=amp)
    mapper = SixAxisMapper(c)
    emu = EmuDevice()
    gf = np.asarray(c.by_role.get("escape_gf", []), dtype=np.int32)
    asc = np.asarray(c.by_role.get("ascending", []), dtype=np.int32)
    loom = np.asarray(list(c.indices(role="loom_lc4"))
                      + list(c.indices(role="loom_lplc2")), dtype=np.int32)
    fb = np.zeros(c.n, dtype=np.float32)
    fb_buf = np.zeros(c.n, dtype=np.float32)

    gf_mask = np.zeros(c.n, dtype=bool)
    gf_mask[gf] = True
    has_gf = bool(gf.size)
    loom_f = np.zeros(c.n, dtype=np.float32)
    loom_f[loom] = 1.0

    frames = []
    t_start = time.perf_counter()
    next_frame = 0
    gf_any = False
    sync_buf = [0] * LOOM_WINDOW
    sync_pos = 0

    for t in range(ms):
        drive = world.step(t)
        if feedback > 0.0 and asc.size:
            np.add(drive, fb, out=fb_buf)
            drive = fb_buf
        spikes = net.step(drive)
        if spikes.size and has_gf and gf_mask[spikes].any():
            gf_any = True
        if loom.size:
            sync_buf[sync_pos] = int(np.dot(net.active, loom_f))
            sync_pos = (sync_pos + 1) % LOOM_WINDOW

        if t >= next_frame:
            axes = mapper.update(net, gf_any, dt_ms=STEPS_PER_FRAME,
                                 loom_sync=max(sync_buf),
                                 vision_steer=world.vision_steer)


            world.set_axes(axes, mapper.diag.get("escape", 0.0))
            gf_any = False
            emu.send6(axes)
            frames.append({
                "t": t,
                "ax": axes,
                "d": {k: round(v, 4) for k, v in mapper.diag.items()},
                "pred": round(world.pred_d, 2),
            })
            next_frame += STEPS_PER_FRAME
            if feedback > 0.0 and asc.size:
                disp = (abs(axes["L0"] - 0.5) + abs(axes["L1"] - 0.5)
                        + abs(axes["L2"] - 0.5))
                lvl = min(0.6, feedback * (1.0 + 2.5 * disp))
                fb[:] = 0.0
                fb[asc] = lvl

        if realtime:
            target = t_start + (t + 1) / 1000.0
            lag = target - time.perf_counter()
            if lag > 0:
                time.sleep(lag)

    elapsed = time.perf_counter() - t_start
    emu.stop()

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"frames": frames, "events": world.events,
                       "ms": ms, "hz": OUT_HZ}, f, separators=(",", ":"))

    return frames, world.events, emu, elapsed


def summarize(frames, events, emu, elapsed, ms):
    print(f"仿真 {ms} ms，耗时 {elapsed * 1000:.0f} ms，实时倍率 {ms / (elapsed * 1000):.1f}x")
    print(f"TCode 指令 {emu.tx_count} 条，输出帧 {len(frames)}")
    print(f"刺激事件 {len(events)} 次")
    if not frames:
        return
    print()
    print("六轴范围（最小 → 最大）：")
    for a in AXES6 + ["V0"]:
        vals = [f["ax"][a] for f in frames]
        lo, hi = min(vals), max(vals)
        bar_lo = int(lo * 30)
        bar_hi = int(hi * 30)
        bar = ["-"] * 30
        for i in range(bar_lo, min(30, bar_hi + 1)):
            bar[i] = "#"
        print(f"  {a}  {lo:.3f} → {hi:.3f}  [{''.join(bar)}]")
    moving = [a for a in AXES6 + ["V0"]
              if max(f["ax"][a] for f in frames) - min(f["ax"][a] for f in frames) > 0.02]
    print()
    print("有动作的轴：" + (" ".join(moving) if moving else "（无）"))
    esc_n = sum(1 for f in frames if f["d"].get("escape_hit", 0) > 0)
    syncs = [f["d"].get("loom_sync", 0) for f in frames]
    print(f"逃逸触发 {esc_n} 次 / {len(frames)} 帧，"
          f"loom 同步峰值 {max(syncs) if syncs else 0:.0f}")
    print()
    print("最后 6 条 TCode：")
    for line in emu.lines[-6:]:
        print("  " + line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, default=8000)
    ap.add_argument("--scenario", default="live",
                    choices=["live", "demo", "quiet", "wind", "flow"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stim", type=int, default=60)
    ap.add_argument("--amp", type=float, default=3.0)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--web", action="store_true")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--feedback", type=float, default=0.06,
                    help="本体感觉回流强度：轴偏移 → 上行神经元驱动；0=关闭")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if not a.web:
        frames, events, emu, el = run(a.ms, a.scenario, a.seed, a.stim, a.amp,
                                      a.realtime, False, a.port, a.out, a.feedback)
        summarize(frames, events, emu, el, a.ms)
        return 0

    from body.server import serve
    return serve(a.ms, a.scenario, a.seed, a.stim, a.amp, a.port, a.out,
                 a.feedback)


if __name__ == "__main__":
    sys.exit(main())

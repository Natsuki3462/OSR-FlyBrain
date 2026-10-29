
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from body.emu import EmuDevice
from body.mapping import SixAxisMapper
from body.world import World
from brain.circuit import Circuit
from brain.lif import build_from_circuit

MS = 20000
FRAME = 20


def main():
    c = Circuit()
    net = build_from_circuit(c, seed=0)
    net.baseline[:] = 0.02
    net.noise = 0.03
    world = World(c, mode="demo", seed=0)
    mapper = SixAxisMapper(c)
    emu = EmuDevice()
    gf = np.asarray(c.by_role.get("escape_gf", []), dtype=np.int32)
    gfm = np.zeros(c.n, dtype=bool)
    gfm[gf] = True
    loom = np.asarray(c.indices(role="loom_lc4")
                      + c.indices(role="loom_lplc2"), dtype=np.int32)
    loom_f = np.zeros(c.n, dtype=np.float32)
    loom_f[loom] = 1.0

    for t in range(300):
        net.step(world.step(t))
    net.reset()
    net.baseline[:] = 0.02
    net.noise = 0.03

    t = dict(world=0.0, net=0.0, mapper=0.0, emu=0.0, frames=0.0, sync=0.0)
    sync_buf = [0] * 25
    sync_pos = 0
    gf_any = False
    nf = 0
    p = time.perf_counter
    for i in range(MS):
        a = p()
        drive = world.step(i)
        b = p()
        spikes = net.step(drive)
        d0 = p()
        t["world"] += b - a
        t["net"] += d0 - b
        if spikes.size and gfm[spikes].any():
            gf_any = True
        e0 = p()
        sync_buf[sync_pos] = int(np.dot(net.active, loom_f))
        sync_pos = (sync_pos + 1) % 25
        e1 = p()
        t["sync"] += e1 - e0
        if i % FRAME == 0:
            m0 = p()
            axes = mapper.update(net, gf_any, dt_ms=FRAME, loom_sync=max(sync_buf))
            m1 = p()
            emu.send6(axes)
            m2 = p()
            frames = {"t": i, "ax": axes,
                      "d": {k: round(v, 4) for k, v in mapper.diag.items()}}
            m3 = p()
            t["mapper"] += m1 - m0
            t["emu"] += m2 - m1
            t["frames"] += m3 - m2
            nf += 1
            gf_any = False
    emu.stop()

    tot = sum(t.values())
    print(f"{MS} ms，{nf} 帧，总计 {tot * 1000:.0f} ms -> {MS / (tot * 1000):.1f}x")
    for k, v in sorted(t.items(), key=lambda kv: -kv[1]):
        per = v / nf * 1e6 if k in ("mapper", "emu", "frames") else v / MS * 1e6
        unit = "帧" if k in ("mapper", "emu", "frames") else "步"
        print(f"  {k:<8}{v * 1000:>8.0f} ms{v / tot * 100:>7.1f}%{per:>10.1f} us/{unit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

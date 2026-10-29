from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from body.emu import AXES6, CENTER, EmuDevice, build_line, encode, magnitude
from body.mapping import LOOM_SYNC_MIN, SixAxisMapper
from body.world import World
from brain.circuit import Circuit
from brain.lif import build_from_circuit

PASS = 0
FAIL = 0


def ck(cond, msg, detail=None):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK]   {msg}")
    else:
        FAIL += 1
        extra = f"  -> {detail}" if detail is not None else ""
        print(f"  [FAIL] {msg}{extra}")


def main():
    print("[1] TCode 编码")
    ck(magnitude(0.5) == "5000", f"0.5 -> {magnitude(0.5)}")
    ck(magnitude(0.05) == "0500", f"0.05 -> {magnitude(0.05)}（前导零必须保留）")
    ck(magnitude(0.9999) == "9999", f"0.9999 -> {magnitude(0.9999)}")
    ck(magnitude(1.0) == "9999", f"1.0 钳到 {magnitude(1.0)}")
    ck(magnitude(-5) == "0000", f"负数钳到 {magnitude(-5)}")
    ck(encode("L0", 0.5) == "L05000", f"encode -> {encode('L0', 0.5)}")
    ck(build_line({"L0": 0.5, "R2": 0.25}) == "L05000 R22500",
       f"多轴 -> {build_line({'L0': 0.5, 'R2': 0.25})}")

    print("\n[2] 模拟器行为")
    emu = EmuDevice()
    ck(all(abs(emu.state[a] - CENTER) < 1e-9 for a in AXES6), "初始在中位")
    emu.send6({"L0": 0.9})
    ck(abs(emu.state["L0"] - 0.9) < 1e-9, "设置 L0=0.9 生效")
    n = emu.tx_count
    emu.send6({"L0": 0.9})
    ck(emu.tx_count == n, "同值不重复发送")
    emu.send6({"L0": 2.0, "L1": -1.0})
    ck(emu.state["L0"] == 1.0 and emu.state["L1"] == 0.0, "越界值被钳位")
    emu.stop()
    ck(all(abs(emu.state[a] - CENTER) < 1e-9 for a in AXES6), "DSTOP 回中位")

    print("\n[3] 闭环：脑 → 六轴")
    c = Circuit()
    net = build_from_circuit(c, seed=0)
    net.baseline[:] = 0.02
    net.noise = 0.03
    world = World(c, mode="demo", seed=0)
    mapper = SixAxisMapper(c)
    emu = EmuDevice()
    gf = np.asarray(c.by_role.get("escape_gf", []), dtype=np.int32)
    asc = np.asarray(c.by_role.get("ascending", []), dtype=np.int32)
    loom = np.asarray(list(c.indices(role="loom_lc4"))
                      + list(c.indices(role="loom_lplc2")), dtype=np.int32)
    fb = np.zeros(c.n, dtype=np.float32)

    frames = []
    nf = 0
    gf_any = False
    sync_buf = [0] * 25
    sync_pos = 0
    esc_hits = 0
    for t in range(9000):
        s = net.step(world.step(t) + fb)
        if s.size and gf.size and np.isin(s, gf).any():
            gf_any = True
        sync_buf[sync_pos] = int(net.active[loom].sum())
        sync_pos = (sync_pos + 1) % 25
        if t >= nf:
            ax = mapper.update(net, gf_any, dt_ms=20, loom_sync=max(sync_buf))
            if mapper.diag.get("escape_hit"):
                esc_hits += 1
            gf_any = False
            emu.send6(ax)
            frames.append(ax)
            nf += 20
            disp = abs(ax["L0"] - .5) + abs(ax["L1"] - .5) + abs(ax["L2"] - .5)
            fb[:] = 0.0
            fb[asc] = min(0.6, 0.06 * (1 + 2.5 * disp))

    ck(len(frames) > 400, f"输出帧数 {len(frames)}（50 Hz × 9 s）", len(frames))
    ck(all(0.0 <= v <= 1.0 for f in frames for v in f.values()),
       "所有轴值都在 [0,1] 内")
    moved = {a: (min(f[a] for f in frames), max(f[a] for f in frames))
             for a in AXES6}
    active = [a for a, (lo, hi) in moved.items() if hi - lo > 0.05]
    ck(len(active) >= 6,
       f"有动作的轴 {len(active)} 个：{' '.join(active)}", moved)
    print("         范围：" + "  ".join(
        f"{a}={moved[a][0]:.2f}~{moved[a][1]:.2f}" for a in AXES6))
    ck(esc_hits >= 1, f"逃逸被触发 {esc_hits} 次", esc_hits)
    ck(emu.tx_count > 100, f"发出 TCode 指令 {emu.tx_count} 条", emu.tx_count)
    ck(all(line.replace(" ", "").replace("DSTOP", "").isalnum()
           or line == "DSTOP" for line in emu.lines), "指令行格式合法")

    print("\n[4] 转向只用变化量（慢基线）")
    m2 = SixAxisMapper(c)
    net2 = build_from_circuit(c, seed=3)
    net2.baseline[:] = 0.0
    net2.noise = 0.0
    net2.rate[c.indices(role="steer_right")] = 0.05
    for _ in range(400):
        m2.update(net2, False, dt_ms=20)
    first = m2.axes["L2"]
    for _ in range(2000):
        m2.update(net2, False, dt_ms=20)
    later = m2.axes["L2"]
    ck(abs(later - 0.5) < abs(first - 0.5) * 0.5,
       f"恒定转向输入会被慢基线吃掉（{first:.3f} → {later:.3f}）",
       (first, later))

    print("\n[5] 前进通路：持续气流 → DNp09 → L1 前进")
    net3 = build_from_circuit(c, seed=5)
    net3.baseline[:] = 0.02
    net3.noise = 0.03
    w3 = World(c, mode="flow", seed=5)
    m3 = SixAxisMapper(c)
    fw = np.asarray(c.by_role.get("forward", []), dtype=np.int32)
    fw_mask = np.zeros(c.n, dtype=bool)
    fw_mask[fw] = True
    loom_f = np.zeros(c.n, dtype=np.float32)
    loom_f[loom] = 1.0
    fw_sp = 0
    sync_max = 0
    l1_hi = 0.5
    for t in range(6000):
        s = net3.step(w3.step(t))
        if s.size and fw.size:
            fw_sp += int(fw_mask[s].sum())
        v = int(np.dot(net3.active, loom_f))
        if v > sync_max:
            sync_max = v
        if t % 20 == 0:
            ax = m3.update(net3, False, dt_ms=20, loom_sync=0)
            if ax["L1"] > l1_hi:
                l1_hi = ax["L1"]
    ck(fw_sp > 0, f"气流驱动下 forward 放电 {fw_sp} 次", fw_sp)
    ck(l1_hi > 0.62, f"L1 前进到 {l1_hi:.3f}（>0.62 才算真的在往前）", l1_hi)
    ck(sync_max < LOOM_SYNC_MIN,
       f"气流引起的 loom 群活动 {sync_max} < 逃逸门控 {LOOM_SYNC_MIN}", sync_max)

    print("\n[6] 风感通路：JO → jo_down → DNp02")
    jd = np.asarray(c.by_role.get("jo_down", []), dtype=np.int32)
    wing = np.asarray(c.by_role.get("wing_a", []), dtype=np.int32)
    gf = np.asarray(c.by_role.get("escape_gf", []), dtype=np.int32)
    ck(jd.size > 0, f"jo_down 中继层存在（{jd.size} 个）", jd.size)


    def _count(mode, seed, ms=20000):
        net = build_from_circuit(c, seed=seed)
        net.baseline[:] = 0.02
        net.noise = 0.03
        w = World(c, mode=mode, seed=seed)
        pools = {"jo_down": jd, "wing_a": wing, "escape_gf": gf}
        cnt = {k: 0 for k in pools}
        for t in range(ms):
            s = net.step(w.step(t))
            if s.size:
                for k, pool in pools.items():
                    if pool.size:
                        cnt[k] += int(np.isin(s, pool).sum())
        return cnt

    seeds = (0, 1, 2)
    quiet = {"jo_down": 0, "wing_a": 0, "escape_gf": 0}
    windy = {"jo_down": 0, "wing_a": 0, "escape_gf": 0}
    for sd in seeds:
        for src, dst in ((_count("quiet", sd), quiet), (_count("wind", sd), windy)):
            for k in dst:
                dst[k] += src[k]

    ck(windy["jo_down"] > quiet["jo_down"] * 2.0,
       f"风让 jo_down 从 {quiet['jo_down']} 涨到 {windy['jo_down']} 个尖峰（>2 倍）",
       (quiet, windy))
    ck(windy["wing_a"] > quiet["wing_a"],
       f"风让 wing_a 从 {quiet['wing_a']} 涨到 {windy['wing_a']} 个尖峰"
       f"（弱通路，只要求方向正确）", (quiet, windy))
    ck(windy["escape_gf"] <= quiet["escape_gf"] + 3,
       f"风不会伪装成逃逸（escape_gf {quiet['escape_gf']} → {windy['escape_gf']}）",
       (quiet, windy))

    print("\n" + "-" * 56)
    print(f"结果：{PASS} 项通过，{FAIL} 项失败")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

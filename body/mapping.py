from __future__ import annotations

import math

import numpy as np

from .emu import CENTER

AXES6 = ["L0", "L1", "L2", "R0", "R1", "R2"]

REF = {
    "backward": 0.0026,
    "forward": 0.0010,
    "steer_left": 0.0026,
    "steer_right": 0.0024,
    "groom": 0.0013,
    "wing_a": 0.0011,
    "wing_b": 0.0013,
    "wing_c": 0.0005,
}

STEER_REF = 0.0018
STEER_BASELINE_MS = 8000.0
DWELL_MS = 80.0
ESCAPE_MS = 220.0
LOOM_SYNC_MIN = 35


WING_HZ_REST = 200.0
WING_HZ_ESCAPE = 45.0
WING_HZ_MAX = 250.0


ESCAPE_FWD = 0.45


def _clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else (hi if v > hi else v)


class Latch:

    def __init__(self, on_th, off_th, dwell_ms):
        self.on_th = on_th
        self.off_th = off_th
        self.dwell = dwell_ms
        self.state = False
        self._pending = None
        self._since = 0.0

    def update(self, value, dt_ms):
        want = self.state
        if not self.state and value >= self.on_th:
            want = True
        elif self.state and value <= self.off_th:
            want = False
        if want == self.state:
            self._pending = None
            return self.state
        if self._pending is not want:
            self._pending = want
            self._since = 0.0
        self._since += dt_ms
        if self._since >= self.dwell:
            self.state = want
            self._pending = None
        return self.state


class SixAxisMapper:

    def __init__(self, circuit, *, steer_gain=1.0, fwd_gain=1.0, back_gain=1.0,
                 wing_gain=1.0, groom_gain=1.0,
                 steer_tau_ms=STEER_BASELINE_MS,
                 dwell_ms=DWELL_MS, escape_ms=ESCAPE_MS,
                 smooth_ms=110.0, loom_sync_min=LOOM_SYNC_MIN):
        self.c = circuit
        self.steer_gain = float(steer_gain)
        self.fwd_gain = float(fwd_gain)
        self.back_gain = float(back_gain)
        self.wing_gain = float(wing_gain)
        self.groom_gain = float(groom_gain)
        self.steer_tau = max(1.0, float(steer_tau_ms))
        self.escape_ms = float(escape_ms)
        self.smooth_tau = max(1.0, float(smooth_ms))
        self.loom_sync_min = int(loom_sync_min)

        self.steer_base = 0.0
        self.escape_timer = 0.0
        self.groom_phase = 0.0
        self.back_latch = Latch(0.6, 0.20, dwell_ms)
        self.axes = {a: CENTER for a in AXES6}
        self.axes["V0"] = 0.0
        self.diag = {}


        roles = list(REF) + ["steer_left", "steer_right"]
        idx_list, starts, sizes = [], [], []
        pos = 0
        for role in roles:
            ii = np.asarray(circuit.by_role.get(role, []), dtype=np.int32)
            if ii.size == 0:
                continue
            starts.append(pos)
            idx_list.append(ii)
            sizes.append(ii.size)
            pos += ii.size
        self._roles = [r for r in roles
                       if len(circuit.by_role.get(r, [])) > 0]
        self._all_idx = (np.concatenate(idx_list) if idx_list
                         else np.zeros(0, dtype=np.int32))
        self._starts = np.asarray(starts, dtype=np.intp)
        self._sizes = np.asarray(sizes, dtype=np.float64)
        self._refs = np.asarray(
            [REF.get(r, 1.0) for r in self._roles], dtype=np.float64)
        self._i_steer_l = self._roles.index("steer_left") if "steer_left" in self._roles else -1
        self._i_steer_r = self._roles.index("steer_right") if "steer_right" in self._roles else -1

    def _group_means(self, net):
        if self._all_idx.size == 0:
            return np.zeros(0, dtype=np.float64)
        vals = net.rate[self._all_idx]
        sums = np.add.reduceat(vals, self._starts)
        return sums / self._sizes

    def _norm(self, net, role):
        idx = self.c.by_role.get(role)
        if idx is None or len(idx) == 0:
            return 0.0
        ref = REF.get(role)
        if not ref:
            return 0.0
        return float(net.rate[idx].mean()) / ref

    def update(self, net, gf_fired, dt_ms=1.0, loom_sync=0):
        dt = max(1e-3, float(dt_ms))
        steer_alpha = 1.0 - math.exp(-dt / self.steer_tau)
        smooth = 1.0 - math.exp(-dt / self.smooth_tau)

        means = self._group_means(net)
        r = {k: 0.0 for k in REF}
        for i, role in enumerate(self._roles):
            if role in REF:
                r[role] = float(means[i] / self._refs[i])

        if self._i_steer_l >= 0 and self._i_steer_r >= 0:
            steer_now = float((means[self._i_steer_r]
                               - means[self._i_steer_l]) / STEER_REF)
        else:
            steer_now = 0.0
        self.steer_base += (steer_now - self.steer_base) * steer_alpha
        steer = _clamp((steer_now - self.steer_base) * self.steer_gain, -1.0, 1.0)

        fwd = _clamp(r["forward"] * self.fwd_gain, 0.0, 1.0)
        back = _clamp(r["backward"] * self.back_gain, 0.0, 1.0)
        back_on = self.back_latch.update(back, dt)

        wing = (r["wing_a"] + r["wing_b"] + r["wing_c"]) / 3.0
        wing_n = _clamp(wing * self.wing_gain, 0.0, 1.0)

        groom = _clamp(r["groom"] * self.groom_gain, 0.0, 1.0)
        self.groom_phase = (self.groom_phase + dt * (0.6 + 2.4 * groom)) % 6283.185
        groom_wave = math.sin(self.groom_phase / 1000.0) * groom

        escape_hit = bool(gf_fired) and int(loom_sync) >= self.loom_sync_min
        if escape_hit:
            self.escape_timer = self.escape_ms
        esc = 0.0
        if self.escape_timer > 0.0:
            self.escape_timer -= dt
            esc = _clamp(self.escape_timer / self.escape_ms, 0.0, 1.0)

        ax = {}
        ax["L0"] = _clamp(0.5 + 0.20 * fwd + 0.40 * wing_n - 0.48 * esc)


        if back_on and esc < 0.3:
            l1 = 0.5 - 0.42 * back
        else:
            l1 = 0.5 + 0.35 * fwd + ESCAPE_FWD * esc
        ax["L1"] = _clamp(l1)
        ax["L2"] = _clamp(0.5 + 0.40 * steer)
        ax["R0"] = _clamp(0.5 + 0.28 * steer)
        ax["R1"] = _clamp(0.5 + 0.35 * groom_wave)
        ax["R2"] = _clamp(0.5 + 0.40 * wing_n - 0.20 * esc)
        wing_hz = WING_HZ_REST * wing_n + WING_HZ_ESCAPE * esc
        ax["V0"] = _clamp(wing_hz / WING_HZ_MAX)
        for k in ax:
            target = _clamp(ax[k])
            prev = self.axes.get(k, CENTER)
            ax[k] = round(prev + (target - prev) * smooth, 4)

        self.axes = ax
        self.diag = {"steer": steer, "fwd": fwd, "back": back,
                     "wing": wing_n, "groom": groom, "escape": esc,
                     "steer_now": steer_now, "steer_base": self.steer_base,
                     "escape_hit": 1.0 if escape_hit else 0.0,
                     "loom_sync": float(loom_sync)}
        return ax

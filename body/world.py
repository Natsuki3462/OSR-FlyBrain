from __future__ import annotations

import math

import numpy as np

_EMPTY_I32 = np.zeros(0, dtype=np.int32)

FLOW_UPSTREAM = 12


PRED_D0 = 40.0
PRED_PX0 = 9.0
PRED_V = 0.012
PRED_D_MAX = 60.0
PRED_D_MIN = 2.5
LOOM_DIST = 12.0
LOOM_CADENCE = 15.0
FLY_V_CRUISE = 0.020
FLY_V_ESCAPE = 0.160
FLY_TURN = 0.004
FLY_DEADBAND = 0.08
FLY_BACK_SCALE = 0.20


PRED_GAP_MIN = 10000.0
PRED_GAP_MAX = 50000.0
PRED_GONE = 52.0
PRED_RETREAT_V = 0.030
PRED_MAX_ACTIVE = 10000.0


STIM = {
    "loom": (3.0, 6.0),
    "food": (3.2, 500.0),
    "wind": (3.0, 220.0),
    "flow": (2.2, 1500.0),
    "wing": (2.6, 600.0),
    "groom": (2.4, 1400.0),
    "backward": (3.0, 500.0),
}


class World:


    DIRTY_CAP = 640

    def __init__(self, circuit, *, mode="demo", seed=0, stim=60,
                 amp=3.0, pulse_ms=2, quiet_ms=900, interval_ms=1500,
                 asc_hz=20.0, asc_amp=1.5,
                 flow_hz=6.0, flow_amp=2.0, flow_pulse_ms=2,
                 predator=None):
        self.c = circuit
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.stim = int(stim)
        self.amp = float(amp)
        self.pulse = int(pulse_ms)
        self.interval = float(interval_ms)
        self.loom = np.asarray(
            circuit.indices(role="loom_lc4")
            + circuit.indices(role="loom_lplc2"), dtype=np.int32)
        self.wind = np.asarray(circuit.indices(role="wind"), dtype=np.int32)
        self.asc = np.asarray(circuit.indices(role="ascending"), dtype=np.int32)
        self.asc_hz = float(asc_hz)
        self.asc_amp = float(asc_amp)
        self.asc_poisson = self.asc_hz * len(self.asc) / 1000.0


        self.flow = _EMPTY_I32
        fw = np.asarray(circuit.by_role.get("forward", []), dtype=np.int32)
        if fw.size:
            pre, post, _w = circuit.edge_arrays()
            if post.size:
                ups = np.unique(pre[np.isin(post, fw)])
                cset = {int(i) for i in ups
                        if circuit.neurons[int(i)].get("role", "")
                        not in ("loom_lc4", "loom_lplc2")}
                pick = np.asarray(sorted(cset), dtype=np.int32)
                if pick.size == 0:
                    pick = ups.astype(np.int32)
                self.flow = pick[:FLOW_UPSTREAM]


        self._stim_idx = {
            "loom": self.loom,
            "wind": self.wind,
            "flow": self.flow,
            "food": np.asarray(circuit.by_role.get("forward", []),
                               dtype=np.int32),
            "backward": np.asarray(circuit.by_role.get("backward", []),
                                   dtype=np.int32),
            "groom": np.asarray(circuit.by_role.get("groom", []),
                                dtype=np.int32),
            "wing": np.asarray(
                list(circuit.indices(role="wing_a"))
                + list(circuit.indices(role="wing_b"))
                + list(circuit.indices(role="wing_c")), dtype=np.int32),
        }
        self._manual = {}

        self.drive = np.zeros(circuit.n, dtype=np.float32)
        self._active = _EMPTY_I32
        self._until = 0
        self._next = float(quiet_ms)
        self.events = []

        self._dirty = np.zeros(self.DIRTY_CAP, dtype=np.int32)
        self._nd = 0

        if self.asc.size and self.asc_poisson > 0.0:
            pool = 1 << 14
            self._pois = self.rng.poisson(self.asc_poisson, pool).astype(np.int32)


            size = int(self.asc.size)
            nblk = max(1, pool // size)
            self._rsel = np.concatenate(
                [self.rng.permutation(size) for _ in range(nblk)]
            ).astype(np.int32)
            self._sel_pool = self.asc[self._rsel]
            self._blk = size
            self._mask = pool - 1
        else:
            self._pois = None
            self._rsel = None
            self._blk = 0
            self._mask = 0
        self._pp = 0
        self._rp = 0
        self.flow_hz = float(flow_hz)
        self.flow_amp = float(flow_amp)
        self.flow_pulse = int(flow_pulse_ms)
        self._flow_next = 0.0
        self._flow_until = 0


        self.pred_active = ((mode in ("demo", "live")) if predator is None
                            else bool(predator))
        self.loom_dist = LOOM_DIST
        self.pred_d = math.hypot(PRED_PX0, PRED_D0)
        self._px = PRED_PX0
        self._py = -PRED_D0
        self._heading = 0.0
        self._last_axes = None
        self._last_esc = 0.0
        self._pred_v = PRED_V
        self._near = False
        self._last_t = None
        self._next_loom = 0.0
        self._active_w = _EMPTY_I32
        self._until_w = 0


        self.pred_on = False
        self.pred_state = "wait" if self.pred_active else "off"
        self._leaving = False
        self._spawn_t = 0.0
        self._next_appear = 0.0

    def set_axes(self, axes, esc=0.0):

        if axes:
            self._last_axes = axes
            self._last_esc = float(esc) if esc else 0.0

    def _pick(self, pool):
        k = min(self.stim, pool.size)
        return self.rng.choice(pool, size=k, replace=False)

    def _mark(self, idx):
        k = int(idx.size)
        nd = self._nd
        if nd + k > self.DIRTY_CAP:
            self.drive[:] = 0.0
            nd = 0
        self._dirty[nd:nd + k] = idx
        self._nd = nd + k

    def trigger(self, kind, t_ms, dur_ms=None, amp=None):

        cfg = STIM.get(kind)
        if cfg is None:
            return False
        def_amp, def_dur = cfg
        end = float(t_ms) + (float(dur_ms) if dur_ms is not None else def_dur)
        self._manual[kind] = (end, float(amp) if amp is not None else def_amp)
        return True

    def _apply_manual(self, d, t_ms):
        if not self._manual:
            return
        for kind in list(self._manual):
            end, amp = self._manual[kind]
            if t_ms >= end:
                del self._manual[kind]
                continue
            idx = self._stim_idx.get(kind)
            if idx is not None and idx.size:
                d[idx] = amp
                self._mark(idx)

    def _spawn(self, t_ms):

        self._spawn_t = float(t_ms)
        self._px = PRED_PX0 * (0.4 + 1.2 * self.rng.random())
        self._py = -PRED_D0
        self._heading = 0.0
        self._last_axes = None
        self._last_esc = 0.0
        self._near = False
        self._leaving = False
        self._active = _EMPTY_I32
        self._until = 0
        self._next_loom = 0.0
        self.pred_d = math.hypot(self._px, self._py)
        self._pred_v = PRED_V * (0.7 + 0.6 * self.rng.random())
        self.pred_on = True
        self.pred_state = "active"
        span = PRED_GAP_MAX - PRED_GAP_MIN
        self._next_appear = (float(t_ms) + PRED_GAP_MIN
                             + span * self.rng.random())
        self.events.append((int(t_ms), "pred_appear", 0))

    def _retreat(self, dt):

        old = max(self.pred_d, 1e-6)
        nd = min(PRED_D_MAX, old + PRED_RETREAT_V * dt)
        s = nd / old
        self._px *= s
        self._py *= s
        self.pred_d = nd

    def _update_predator(self, dt):

        ax = self._last_axes
        if ax is not None:
            l1 = float(ax.get("L1", 0.5))
            l2 = float(ax.get("L2", 0.5))
            self._heading += FLY_TURN * (l2 - 0.5) * dt

            eff = l1 - 0.5
            if abs(eff) < FLY_DEADBAND:
                eff = 0.0
            elif eff < 0.0:
                eff *= FLY_BACK_SCALE

            df = (FLY_V_CRUISE * eff + FLY_V_ESCAPE * self._last_esc) * dt
            h = self._heading

            self._px -= math.sin(h) * df
            self._py -= math.cos(h) * df

        d = math.hypot(self._px, self._py)
        if d > 1e-6:
            step = self._pred_v * dt
            if step >= d:

                self._px *= 0.02
                self._py *= 0.02
            else:
                self._px -= self._px / d * step
                self._py -= self._py / d * step

        d = math.hypot(self._px, self._py)
        if d > PRED_D_MAX:
            s = PRED_D_MAX / d
            self._px *= s
            self._py *= s
            d = PRED_D_MAX
        elif d < PRED_D_MIN:

            s = PRED_D_MIN / max(d, 1e-6)
            self._px *= s
            self._py *= s
            d = PRED_D_MIN
        self.pred_d = d
        return d

    def step(self, t_ms):
        d = self.drive
        if self._nd:
            d[self._dirty[:self._nd]] = 0.0
            self._nd = 0

        if self._pois is not None:
            k = int(self._pois[self._pp])
            self._pp = (self._pp + 1) & self._mask
            if k:
                size = self._blk
                if k > size:
                    k = size
                pos = self._rp
                if pos % size + k > size:
                    pos = (pos // size + 1) * size
                    if pos + size > self._sel_pool.size:
                        pos = 0
                self._rp = pos + k
                sel = self._sel_pool[pos:pos + k]
                d[sel] = self.asc_amp
                self._mark(sel)


        self._apply_manual(d, t_ms)

        if self.mode == "quiet":
            return d


        if self.flow.size:
            if t_ms >= self._flow_next:
                self._flow_until = t_ms + self.flow_pulse
                self._flow_next = t_ms + max(
                    1, int(1000.0 / self.flow_hz * (0.7 + 0.6 * self.rng.random())))
            if t_ms < self._flow_until:
                d[self.flow] = self.flow_amp
                self._mark(self.flow)

        if self.mode == "flow":
            return d


        if self._last_t is None:
            dt = 1.0
        else:
            dt = float(t_ms - self._last_t)
            if dt <= 0.0 or dt > 100.0:
                dt = 1.0
        self._last_t = t_ms


        if self.pred_active:
            if not self.pred_on and t_ms >= self._next_appear:
                self._spawn(t_ms)

            near = False
            if self.pred_on:
                if self._leaving:
                    self.pred_state = "leave"
                    self._retreat(dt)
                    if self.pred_d >= PRED_GONE:
                        self.pred_on = False
                        self._leaving = False
                        self.pred_state = "wait"
                        self.events.append((int(t_ms), "pred_gone", 0))
                else:
                    self.pred_state = "active"
                    dnow = self._update_predator(dt)
                    near = dnow < self.loom_dist
                    if near != self._near:
                        self.events.append(
                            (int(t_ms), "loom_near" if near else "loom_far",
                             int(self.loom.size)))
                        self._near = near
                        if not near:

                            self._leaving = True
                            self.pred_state = "leave"
                    if not self._leaving and (
                            t_ms - self._spawn_t > PRED_MAX_ACTIVE):

                        self._leaving = True
                        self.pred_state = "leave"

                if near and not self._leaving:
                    if t_ms >= self._next_loom:
                        self._active = (self._pick(self.loom) if self.loom.size
                                        else _EMPTY_I32)
                        self._until = t_ms + self.pulse
                        self._next_loom = t_ms + LOOM_CADENCE * (
                            0.85 + 0.3 * self.rng.random())
                    if t_ms < self._until and self._active.size:
                        d[self._active] = self.amp
                        self._mark(self._active)


        if self.mode in ("demo", "live", "wind"):
            if t_ms >= self._next:
                self._next = t_ms + self.interval * (
                    0.7 + 0.6 * self.rng.random())
                if self.mode == "wind" or (
                        self.wind.size and self.rng.random() < 0.25):
                    self._active_w = (self._pick(self.wind) if self.wind.size
                                      else _EMPTY_I32)
                    self._until_w = t_ms + self.pulse
                    self.events.append(
                        (int(t_ms), "wind", int(self._active_w.size)))
            if t_ms < self._until_w and self._active_w.size:
                d[self._active_w] = self.amp
                self._mark(self._active_w)

        return d

from __future__ import annotations

import os

import numpy as np

from . import lif_jit

DT_MS = 1.0
TAU_M_MS = 20.0
V_TH = 1.0
REFRACT_MS = 2.0
INH_DELAY_MS = 4.0
RATE_TAU_MS = 120.0
ADAPT_TAU_MS = 200.0
ADAPT_GAIN = 0.28

_EMPTY_I64 = np.zeros(0, dtype=np.int64)


def _resolve_backend(backend):

    backend = (backend or "auto").lower()
    env = os.environ.get("OSR_LIF_BACKEND", "").lower()
    if env:
        backend = env
    if backend == "numpy":
        return False
    if backend == "numba":
        if not lif_jit.HAVE_NUMBA:
            raise RuntimeError("backend='numba' 但 numba 未安装")
        return True
    return lif_jit.HAVE_NUMBA


class LIFNetwork:

    def __init__(self, n, pre, post, weight, *, dt_ms=DT_MS, tau_ms=TAU_M_MS,
                 v_th=V_TH, refract_ms=REFRACT_MS, inh_delay_ms=INH_DELAY_MS,
                 rate_tau_ms=RATE_TAU_MS, adapt_tau_ms=ADAPT_TAU_MS,
                 adapt_gain=ADAPT_GAIN, seed=0, backend="auto"):
        self.n = int(n)
        self.dt = float(dt_ms)
        self.leak = self.dt / float(tau_ms)
        self.v_th = float(v_th)
        self.refr_steps = max(0, int(round(refract_ms / self.dt)) - 1)
        self.inh_delay = max(0, int(round(inh_delay_ms / self.dt)))
        self.rate_alpha = self.dt / float(rate_tau_ms)
        self.adapt_alpha = self.dt / float(adapt_tau_ms)
        self.adapt_gain = float(adapt_gain)

        pre = np.asarray(pre, dtype=np.int32)
        post = np.asarray(post, dtype=np.int32)
        w = np.asarray(weight, dtype=np.float32)
        if pre.size != post.size or pre.size != w.size:
            raise ValueError("pre/post/weight length mismatch")
        if pre.size and (pre.max() >= self.n or post.max() >= self.n or pre.min() < 0):
            raise ValueError("edge index out of range")

        exc = w > 0
        self.pre_e = pre[exc]
        self.post_e = post[exc]
        self.w_e = w[exc].astype(np.float32)
        self.pre_i = pre[~exc]
        self.post_i = post[~exc]
        self.w_i = (-w[~exc]).astype(np.float32)

        self.v = np.zeros(self.n, dtype=np.float32)
        self.refr = np.zeros(self.n, dtype=np.int32)
        self.adapt = np.zeros(self.n, dtype=np.float32)
        self.active = np.zeros(self.n, dtype=np.float32)
        self.rate = np.zeros(self.n, dtype=np.float32)
        self.baseline = np.zeros(self.n, dtype=np.float32)
        self.noise = 0.0
        self.rng = np.random.default_rng(seed)

        self._dl = np.zeros((self.inh_delay + 1, self.n), dtype=np.float32)
        self._dl_pos = 0
        self.steps = 0
        self.spike_count = 0

        self.backend = "numba" if _resolve_backend(backend) else "numpy"
        self._jit = self.backend == "numba"
        self._act_idx = np.zeros(self.n, dtype=np.int32)
        self._drive_buf = np.zeros(self.n, dtype=np.float32)
        self._noise_pos = 0
        self._pool_steps = 0
        self._noise_flat = np.zeros(0, dtype=np.float32)
        self._dl_flat = self._dl.reshape(-1)
        self._csr_e = lif_jit.build_csr(self.pre_e, self.post_e, self.w_e, self.n)
        self._csr_i = lif_jit.build_csr(self.pre_i, self.post_i, self.w_i, self.n)
        if self._jit:
            self._make_noise_pool()

    def _make_noise_pool(self):
        steps = lif_jit.noise_pool_steps(self.n)
        self._pool_steps = steps
        self._noise_flat = self.rng.standard_normal(
            steps * self.n, dtype=np.float32)
        self._noise_pos = 0

    def reset(self):
        self.v[:] = 0.0
        self.refr[:] = 0
        self.adapt[:] = 0.0
        self.active[:] = 0.0
        self.rate[:] = 0.0
        self._dl[:] = 0.0
        self._dl_pos = 0
        self._noise_pos = 0
        self.steps = 0
        self.spike_count = 0

    def step(self, drive=None):
        if self._jit:
            return self._step_jit(drive)
        return self._step_numpy(drive)

    def _step_numpy(self, drive=None):
        v = self.v
        v -= v * self.leak
        v += self.baseline
        if self.noise:
            v += self.rng.standard_normal(self.n, dtype=np.float32) * self.noise
        if drive is not None:
            v += drive

        ad = self.adapt
        ad -= ad * self.adapt_alpha
        v -= ad

        r = self.refr
        if r.any():
            blocked = r > 0
            v[blocked] = 0.0
            r[blocked] -= 1

        idx = np.flatnonzero(v >= self.v_th)
        if idx.size:
            v[idx] = 0.0
            r[idx] = self.refr_steps
            ad[idx] += self.adapt_gain

        self.active[:] = 0.0
        if idx.size:
            self.active[idx] = 1.0
            v += np.bincount(self.post_e,
                             weights=self.w_e * self.active[self.pre_e],
                             minlength=self.n)
            self._dl[self._dl_pos] = np.bincount(
                self.post_i, weights=self.w_i * self.active[self.pre_i],
                minlength=self.n)
        else:
            self._dl[self._dl_pos] = 0.0
        self._dl_pos = (self._dl_pos + 1) % (self.inh_delay + 1)
        v -= self._dl[self._dl_pos]

        self.rate += (self.active - self.rate) * self.rate_alpha
        self.steps += 1
        self.spike_count += int(idx.size)
        return idx

    def _step_jit(self, drive=None):
        if drive is None:
            has_drive = False
            d = self._drive_buf
        else:
            if drive.dtype == np.float32 and drive.ndim == 1 and drive.shape[0] == self.n:
                d = drive
            else:
                self._drive_buf[:] = drive
                d = self._drive_buf
            has_drive = True

        if self.noise:
            has_noise = True
            if self._noise_pos >= self._pool_steps:
                self._noise_pos = 0
            off = self._noise_pos * self.n
            self._noise_pos += 1
        else:
            has_noise = False
            off = 0

        ie, pe, we = self._csr_e
        ii, pi, wi = self._csr_i
        k, pos = lif_jit.step_kernel(
            self.v, self.refr, self.adapt, self.rate, self.active,
            self._act_idx,
            ie, pe, we, ii, pi, wi,
            self._dl_flat, self._dl_pos, self.inh_delay + 1,
            d, self.baseline,
            self._noise_flat, off, self.noise,
            has_drive, has_noise,
            self.leak, self.v_th, self.refr_steps,
            self.rate_alpha, self.adapt_alpha, self.adapt_gain,
        )
        self._dl_pos = pos
        self.steps += 1
        self.spike_count += k
        if k == 0:
            return _EMPTY_I64
        return self._act_idx[:k].astype(np.int64)

    def run(self, steps, drive=None, on_step=None):
        out = []
        for t in range(int(steps)):
            idx = self.step(drive)
            if on_step is not None:
                on_step(t, idx)
            elif idx.size:
                out.append((t, idx))
        return out

    def group_rate(self, index):
        idx = np.asarray(index, dtype=np.int32)
        if idx.size == 0:
            return 0.0
        return float(self.rate[idx].mean())


def build_from_circuit(circuit, **kw):
    pre, post, w = circuit.edge_arrays()
    return LIFNetwork(circuit.n, pre, post, w, **kw)

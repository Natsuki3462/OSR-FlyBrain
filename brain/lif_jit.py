
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]

        def deco(f):
            return f

        return deco


@njit(cache=True)
def step_kernel(v, refr, adapt, rate, active, act_idx,
                indptr_e, post_e, w_e,
                indptr_i, post_i, w_i,
                dl_flat, dl_pos, nslots,
                drive, baseline, noise_flat, noise_off, noise_amp,
                has_drive, has_noise,
                leak, v_th, refr_steps,
                rate_alpha, adapt_alpha, adapt_gain):
    n = v.shape[0]
    base = dl_pos * n
    nxt = dl_pos + 1
    if nxt >= nslots:
        nxt = 0
    nb = nxt * n


    k = 0
    for i in range(n):
        vi = v[i] - v[i] * leak + baseline[i]
        ai = adapt[i] - adapt[i] * adapt_alpha
        adapt[i] = ai
        vi -= ai
        if has_noise:
            vi += noise_flat[noise_off + i] * noise_amp
        if has_drive:
            vi += drive[i]
        if refr[i] > 0:
            vi = 0.0
            refr[i] -= 1
        if vi >= v_th:
            vi = 0.0
            refr[i] = refr_steps
            adapt[i] += adapt_gain
            active[i] = 1.0
            rate[i] = rate[i] + (1.0 - rate[i]) * rate_alpha
            act_idx[k] = i
            k += 1
        else:
            active[i] = 0.0
            rate[i] = rate[i] + (0.0 - rate[i]) * rate_alpha
        vi -= dl_flat[nb + i]
        dl_flat[nb + i] = 0.0
        v[i] = vi


    for q in range(k):
        p = act_idx[q]
        for j in range(indptr_e[p], indptr_e[p + 1]):
            v[post_e[j]] += w_e[j]
        for j in range(indptr_i[p], indptr_i[p + 1]):
            dl_flat[base + post_i[j]] += w_i[j]

    return k, nxt


def build_csr(pre, post, w, n):

    order = np.argsort(pre, kind="stable")
    pre_s = pre[order]
    indptr = np.searchsorted(pre_s, np.arange(n + 1)).astype(np.int64)
    return indptr, post[order], w[order]


def noise_pool_steps(n, budget=4_000_000):
    return int(min(1024, max(32, budget // max(1, n))))

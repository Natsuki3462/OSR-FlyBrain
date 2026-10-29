from __future__ import annotations

import asyncio

from .osr.osr import AXES6, CENTER, PRECISION, build_line, encode, magnitude
from .osr.osr_emu import D2_LABELS, EMU_AXES, EmuDevice as _EmuDevice

VIB_AXES = ["V0"]

LABELS = {
    "L0": "上下", "L1": "前后", "L2": "左右",
    "R0": "旋转", "R1": "滚转", "R2": "俯仰",
    "V0": "振动",
}


def home_state() -> dict:
    from .osr.osr_emu import _home_state
    return _home_state()


class EmuDevice:

    name = "osr6-emu"

    def __init__(self, verbose: bool = False, delay: float = 0.0,
                 print_hz: float = 5.0):
        self._loop = asyncio.new_event_loop()
        self._dev = _EmuDevice(verbose=verbose, delay=delay, print_hz=print_hz)

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    @property
    def state(self) -> dict:
        return self._dev.state

    @property
    def lines(self) -> list:
        return self._dev.lines

    @property
    def tx_count(self) -> int:
        return self._dev.tx_count

    @property
    def supported(self) -> set:
        return self._dev.supported

    @property
    def alive(self) -> bool:
        return self._dev.alive

    def describe(self) -> str:
        return self._dev.describe()

    def snapshot(self) -> dict:
        return self._dev.snapshot()

    def send6(self, axes: dict) -> int:

        dev = self._dev
        if dev.delay:
            before = dev.tx_count
            self._run(dev.send6(axes))
            return dev.tx_count - before

        supported = dev.supported
        state = dev.state
        changed = {}
        for k, v in axes.items():
            k = str(k).upper().strip()
            if k not in supported:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v != v:
                continue
            v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
            if abs(state.get(k, -1.0) - v) > 1e-6:
                changed[k] = v
        if not changed:
            return 0
        dev.lines.append(build_line(changed))
        dev.tx_count += 1
        state.update(changed)
        dev._maybe_print()
        return 1

    def stop(self):
        self._run(self._dev.stop())

    def close(self):
        try:
            self._run(self._dev.close())
        finally:
            if not self._loop.is_closed():
                self._loop.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


__all__ = [
    "AXES6", "VIB_AXES", "CENTER", "PRECISION", "LABELS", "EMU_AXES",
    "D2_LABELS", "EmuDevice", "build_line", "encode", "magnitude",
    "home_state",
]



from __future__ import annotations

import asyncio
import threading
import time

from .emu import EmuDevice
from .osr.osr import AXES6, CENTER, DeviceNotFound, OSR6, build_line

__all__ = ["RealDevice", "LinkManager", "list_ports_info"]


class RealDevice:


    name = "osr6-serial"

    def __init__(self, port=None, udp=None, ble=None, baud=None,
                 identify_timeout=12.0):
        kwargs = {}
        if baud:
            kwargs["baud"] = int(baud)
        self._dev = OSR6(port=port, udp=udp, ble=ble, **kwargs)
        self._loop = asyncio.new_event_loop()
        self.lines: list[str] = []
        self.tx_count = 0
        self.state: dict[str, float] = {}
        self.info: dict = {}
        self.endpoint = ""
        self.transport = ""
        self.supported: set[str] = set(AXES6)
        self._last: dict[str, float] = {}
        self._timeout = identify_timeout


    def connect(self) -> dict:

        async def _c():
            await self._dev.connect()
            return await self._dev.identify()

        info = self._loop.run_until_complete(
            asyncio.wait_for(_c(), timeout=self._timeout))
        d0 = (info["raw"].get("D0") or "").strip()
        d1 = (info["raw"].get("D1") or "").strip()
        if "TCode" not in d1:
            self._hard_close()
            raise DeviceNotFound(
                f"{self._dev.transport}:{self._dev.endpoint} 的应答不是 TCode"
                f"（D1 = {d1!r}）。可能连错了设备，或者控制器还没启动完。")

        self.info = info
        self.transport = self._dev.transport or ""
        self.endpoint = self._dev.endpoint or ""
        self.supported = set(self._dev.supported) or set(AXES6)
        self.state = {k: (CENTER if k[0] in "LR" else 0.0)
                      for k in self.supported}
        self._last = {}
        return {
            "device": d0,
            "tcode": d1,
            "axes": sorted(self.supported),
            "transport": self.transport,
            "endpoint": self.endpoint,
        }

    def _hard_close(self):

        for attr in ("_ser", "_sock"):
            try:
                h = getattr(self._dev, attr, None)
                if h:
                    h.close()
            except Exception:
                pass
            setattr(self._dev, attr, None)
        self._dev.transport = None


    def send6(self, axes: dict) -> int:

        changed = {}
        for k, v in axes.items():
            k = str(k).upper().strip()
            if k not in self.supported:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v != v:
                continue
            v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
            if abs(self.state.get(k, -1.0) - v) > 1e-6:
                changed[k] = v
        if not changed:
            return 0
        line = build_line(changed)
        try:
            self._loop.run_until_complete(self._dev._write(line))
        except Exception:
            return 0
        self.lines.append(line)
        if len(self.lines) > 400:
            del self.lines[:-200]
        self.tx_count += 1
        self.state.update(changed)
        return 1

    @property
    def alive(self) -> bool:
        return self._dev.transport is not None

    def describe(self) -> str:
        return f"OSR6 (TCode v0.3) [{self.transport}:{self.endpoint}]"

    def stop(self):
        try:
            self._loop.run_until_complete(
                asyncio.wait_for(self._dev.stop(home=True), timeout=3.0))
        except Exception:
            pass

    def close(self):
        try:
            self._loop.run_until_complete(
                asyncio.wait_for(self._dev.close(), timeout=3.0))
        except Exception:
            pass
        finally:
            if not self._loop.is_closed():
                self._loop.close()


def list_ports_info() -> list[dict]:

    try:
        from .osr.osr import USB_HINTS, find_serial_port, list_serial_ports
        hint = find_serial_port()
        out = []
        for p in list_serial_ports():
            blob = " ".join(str(x) for x in (
                p.device, p.name, p.description, p.hwid) if x).upper()
            out.append({
                "device": p.device,
                "name": p.name or "",
                "description": p.description or "",
                "likely": bool(any(h.upper() in blob for h in USB_HINTS)),
                "pick": p.device == hint,
            })
        out.sort(key=lambda x: (not x["pick"], not x["likely"], x["device"]))
        return out
    except Exception:
        return []


class LinkManager:


    def __init__(self):
        self._lock = threading.RLock()
        self.target: str | None = None
        self.status = "idle"
        self.detail = ""
        self.error = ""
        self.endpoint = ""
        self.device_name = ""
        self.tcode = ""
        self.axes: list[str] = []
        self.since = 0.0
        self._dev = None


    def connect(self, target: str, port: str | None = None,
                baud: int | None = None) -> dict:
        target = (target or "").lower()
        if target not in ("emu", "serial"):
            return {"ok": False, "error": f"未知目标 {target!r}"}

        with self._lock:
            if self.target == target and self.status == "connected":
                return {"ok": True, "note": "已连接", "link": self.info()}
            self.disconnect()
            self.target = target
            self.status = "connecting"
            self.error = ""
            self.detail = "正在连接…"

        try:
            if target == "emu":
                dev = EmuDevice()
                info = {"device": "osr-emu", "tcode": "SR6 (本地模拟)",
                        "axes": sorted(dev.supported),
                        "transport": "emu", "endpoint": "local"}
            else:
                dev = RealDevice(port=port or None, baud=baud)
                info = dev.connect()

            with self._lock:
                self._dev = dev
                self.status = "connected"
                self.detail = info.get("tcode") or ""
                self.device_name = info.get("device") or ""
                self.tcode = info.get("tcode") or ""
                self.axes = info.get("axes") or []
                self.endpoint = (f"{info.get('transport')}:{info.get('endpoint')}"
                                 if target == "serial" else "local")
                self.since = time.time()
                return {"ok": True, "link": self.info()}
        except Exception as e:
            msg = str(e) or e.__class__.__name__
            with self._lock:
                self._dev = None
                self.status = "error"
                self.error = msg
                self.detail = "连接失败"
                self.endpoint = ""
            return {"ok": False, "error": msg, "link": self.info()}

    def disconnect(self) -> dict:
        with self._lock:
            dev, self._dev = self._dev, None
            was = self.target
            self.target = None
            self.status = "idle"
            self.detail = ""
            self.error = ""
            self.endpoint = ""
            self.device_name = ""
            self.tcode = ""
            self.axes = []
            self.since = 0.0
        if dev is not None:
            try:
                dev.stop()
            except Exception:
                pass
            try:
                dev.close()
            except Exception:
                pass
        return {"ok": True, "was": was, "link": self.info()}


    def send6(self, axes: dict) -> tuple[int, str]:

        dev = self._dev
        if dev is None:
            return 0, ""
        try:
            n = dev.send6(axes)
            line = dev.lines[-1] if dev.lines else ""
            return n, line
        except Exception as e:
            with self._lock:
                self.status = "error"
                self.error = f"发送失败：{e}"
            return 0, ""

    @property
    def connected(self) -> bool:
        return self.status == "connected" and self._dev is not None

    @property
    def tx_count(self) -> int:
        dev = self._dev
        return dev.tx_count if dev is not None else 0

    def info(self) -> dict:
        return {
            "target": self.target,
            "status": self.status,
            "connected": self.connected,
            "detail": self.detail,
            "error": self.error,
            "endpoint": self.endpoint,
            "device": self.device_name,
            "tcode": self.tcode,
            "axes": self.axes,
            "uptime_s": (round(time.time() - self.since, 1)
                         if self.since else 0.0),
            "tx": self.tx_count,
        }

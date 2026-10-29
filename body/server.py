from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, SimpleQueue

import numpy as np

from .device import LinkManager, list_ports_info
from .emu import AXES6, build_line
from .mapping import SixAxisMapper
from .world import STIM, World
from brain.circuit import Circuit
from brain.lif import build_from_circuit

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")
STEPS_PER_FRAME = 20
FRAME_MS = 20
LOOM_WINDOW = 25


def build_brain_payload(circuit) -> dict:

    pos = np.zeros((circuit.n, 3), dtype=np.float64)
    vox = np.zeros((circuit.n, 3), dtype=np.int32)
    ok = np.zeros(circuit.n, dtype=bool)
    for i, nb in enumerate(circuit.neurons):
        p = nb.get("pos")
        if p and len(p) == 3:
            pos[i] = p
            vox[i] = [int(round(v)) for v in p]
            ok[i] = True
    center = np.zeros(3)
    span = 1.0
    if ok.any():
        center = pos[ok].mean(axis=0)
        pos -= center
        span = float(np.abs(pos[ok]).max()) or 1.0
        pos /= span

    edges = []
    for a, b, w in circuit.edges:
        edges.append([int(a), int(b), 1 if w > 0 else 0])

    return {
        "n": circuit.n,
        "pos": [[round(float(v), 4) for v in p] for p in pos],


        "vox": vox.tolist(),
        "role": [nb.get("role", "") for nb in circuit.neurons],
        "type": [nb.get("type", "") for nb in circuit.neurons],
        "side": [nb.get("side", "") for nb in circuit.neurons],
        "edges": edges,
        "meta": {
            "source": circuit.meta.get("source", ""),
            "synapses": len(circuit.edges),
            "groups": {r: len(v) for r, v in circuit.by_role.items() if r},
            "center": [round(float(v), 1) for v in center],
            "span": round(float(span), 1),
        },
    }


class SimRunner(threading.Thread):

    def __init__(self, ms, scenario, seed, stim, amp, link: LinkManager,
                 feedback=0.06):
        super().__init__(daemon=True)
        self.ms = ms
        self.scenario = scenario
        self.seed = seed
        self.stim = stim
        self.amp = amp
        self.feedback = feedback
        self.link = link

        self.c = Circuit()
        self.net = build_from_circuit(self.c, seed=seed)
        self.net.baseline[:] = 0.02
        self.net.noise = 0.03
        self.world = World(self.c, mode=scenario, seed=seed,
                           stim=stim, amp=amp)
        self.mapper = SixAxisMapper(self.c)

        self.gf = np.asarray(self.c.by_role.get("escape_gf", []), dtype=np.int32)
        self.asc = np.asarray(self.c.by_role.get("ascending", []), dtype=np.int32)
        self.loom = np.asarray(list(self.c.indices(role="loom_lc4"))
                               + list(self.c.indices(role="loom_lplc2")),
                               dtype=np.int32)
        self.fb = np.zeros(self.c.n, dtype=np.float32)
        self.gf_mask = np.zeros(self.c.n, dtype=bool)
        self.gf_mask[self.gf] = True
        self.loom_f = np.zeros(self.c.n, dtype=np.float32)
        self.loom_f[self.loom] = 1.0
        self.drive_buf = np.zeros(self.c.n, dtype=np.float32)

        self.cum = np.zeros(self.c.n, dtype=np.int32)
        self.brain = build_brain_payload(self.c)

        self.lock = threading.Lock()
        self.stim_q = SimpleQueue()
        self.frame = {"t": 0, "ax": dict(self.mapper.axes), "d": {},
                      "tc": "", "tx": 0, "spikes": 0, "active": []}
        self.hist = []
        self.events = []
        self.done = False
        self.running = True

    def snapshot(self):
        with self.lock:
            return {
                "frame": dict(self.frame),
                "hist": self.hist[-240:],
                "cum": self.cum.tolist(),
                "events": self.events[-12:],
                "link": self.link.info(),
                "ms": self.ms,
                "hz": 1000 // FRAME_MS,
                "neurons": self.c.n,
                "synapses": len(self.c.edges),
                "scenario": self.scenario,
            }

    def push_stim(self, kind):

        if kind not in STIM:
            return False
        self.stim_q.put(kind)
        return True

    def run(self):
        episode = 0
        while self.running:
            self._episode(episode)
            if not self.running:
                break
            episode += 1
            self.net.reset()
            self.net.baseline[:] = 0.02
            self.net.noise = 0.03
            self.world = World(self.c, mode=self.scenario,
                               seed=self.seed + episode * 977,
                               stim=self.stim, amp=self.amp)
            self.mapper = SixAxisMapper(self.c)
            self.fb[:] = 0.0
            with self.lock:
                self.events = []
        with self.lock:
            self.done = True

    def _episode(self, episode):
        t0 = time.perf_counter()
        nf = 0
        gf_any = False
        sync_buf = [0] * LOOM_WINDOW
        sync_pos = 0

        frame_set: set[int] = set()

        for t in range(self.ms):
            if not self.running:
                break

            while True:
                try:
                    kind = self.stim_q.get_nowait()
                except Empty:
                    break
                self.world.trigger(kind, t)
            drive = self.world.step(t)
            if self.feedback > 0.0 and self.asc.size:
                np.add(drive, self.fb, out=self.drive_buf)
                drive = self.drive_buf
            spikes = self.net.step(drive)
            if spikes.size:
                self.cum[spikes] += 1
                frame_set.update(spikes.tolist())
            if spikes.size and self.gf_mask[spikes].any():
                gf_any = True
            if self.loom.size:
                sync_buf[sync_pos] = int(np.dot(self.net.active, self.loom_f))
                sync_pos = (sync_pos + 1) % LOOM_WINDOW

            if t >= nf:
                loom_sync = max(sync_buf)
                axes = self.mapper.update(self.net, gf_any, dt_ms=FRAME_MS,
                                          loom_sync=loom_sync)

                self.world.set_axes(axes, self.mapper.diag.get("escape", 0.0))
                gf_any = False

                _n, line = self.link.send6(axes)
                preview = build_line({a: axes[a] for a in AXES6
                                      if a in axes})
                with self.lock:
                    self.frame = {
                        "t": t,
                        "ep": episode,
                        "ax": dict(axes),
                        "d": {k: round(v, 4) for k, v in self.mapper.diag.items()},
                        "tc": line,
                        "tc_preview": preview,
                        "sent": bool(line),
                        "tx": self.link.tx_count,
                        "spikes": len(frame_set),
                        "active": sorted(frame_set),
                        "pred_d": round(float(self.world.pred_d), 2),
                        "pred_on": bool(self.world.pred_on),
                        "pred_state": self.world.pred_state,
                        "threat": bool(self.world.pred_on
                                       and self.world.pred_d
                                       < self.world.loom_dist),
                    }
                    self.hist.append([axes[a] for a in AXES6])
                    if len(self.hist) > 2000:
                        del self.hist[:1000]
                    self.events = list(self.world.events[-12:])
                frame_set = set()
                nf += STEPS_PER_FRAME
                if self.feedback > 0.0 and self.asc.size:
                    disp = (abs(axes["L0"] - 0.5) + abs(axes["L1"] - 0.5)
                            + abs(axes["L2"] - 0.5))
                    lvl = min(0.6, self.feedback * (1.0 + 2.5 * disp))
                    self.fb[:] = 0.0
                    self.fb[self.asc] = lvl
                target = t0 + (nf / 1000.0)
                lag = target - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)


class Handler(BaseHTTPRequestHandler):

    runner: SimRunner = None
    brain_json: str = "{}"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return {}

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._file(os.path.join(WEB, "sim.html"),
                              "text/html; charset=utf-8")
        if path.startswith("/vendor/"):
            name = os.path.basename(path)
            return self._file(os.path.join(WEB, "vendor", name),
                              "application/javascript; charset=utf-8")
        if path == "/state":
            return self._send(200, json.dumps(self.runner.snapshot()))
        if path == "/brain":
            return self._send(200, self.brain_json)
        if path == "/api/link":
            return self._send(200, json.dumps(self.runner.link.info()))
        if path == "/api/ports":
            return self._send(200, json.dumps({"ports": list_ports_info()}))
        if path == "/stop":
            self.runner.running = False
            return self._send(200, '{"ok":true}')
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/link":
            body = self._body()
            target = body.get("target")
            if target in (None, "", "none", "off"):
                return self._send(200, json.dumps(self.runner.link.disconnect()))
            r = self.runner.link.connect(
                target, port=body.get("port"), baud=body.get("baud"))
            return self._send(200, json.dumps(r))
        if path == "/api/stim":
            body = self._body()
            kind = str(body.get("kind") or "")
            ok = self.runner.push_stim(kind)
            return self._send(200, json.dumps(
                {"ok": bool(ok), "kind": kind, "kinds": list(STIM)}))
        return self._send(404, "not found", "text/plain")

    def _file(self, fp, ctype):
        if not os.path.isfile(fp):
            return self._send(404, f"missing {os.path.basename(fp)}", "text/plain")
        with open(fp, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))

        self.send_header("Cache-Control",
                         "max-age=86400" if "/vendor/" in fp.replace("\\", "/")
                         else "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


class _Server(ThreadingHTTPServer):

    allow_reuse_address = (os.name != "nt")
    daemon_threads = True


def serve(ms, scenario, seed, stim, amp, port, out_path, feedback=0.06):
    link = LinkManager()
    runner = SimRunner(ms, scenario, seed, stim, amp, link, feedback)
    Handler.runner = runner
    Handler.brain_json = json.dumps(runner.brain)
    try:
        httpd = _Server(("127.0.0.1", port), Handler)
    except OSError as e:
        print(f"[端口 {port} 已被占用] {e}")
        print("  换一个端口：start.bat <ms> <port> <scenario>")
        print("  或者先关掉正在跑的那个窗口")
        return 2
    runner.start()
    print(f"脑-体闭环已启动：http://127.0.0.1:{port}/")
    print(f"  {runner.c.n} 神经元 / {len(runner.c.edges)} 突触，"
          f"脑 1 kHz，输出 50 Hz，{ms} ms 一轮")
    print("  未连接设备 —— 在页面上点「连接大脑模型」才开始下发指令")
    print("  Ctrl+C 停止")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止中…")
    finally:
        runner.running = False
        link.disconnect()
        httpd.server_close()
    return 0

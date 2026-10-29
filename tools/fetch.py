from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.request

CHUNK = 256 << 10
UA = "OSR-FlyBrain-fetch/1.0"


def _open(url, start, end):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Range": f"bytes={start}-{end}",
    })
    return urllib.request.urlopen(req, timeout=25)


def _size(url):
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def _worker(url, dest, start, end, state, lock):
    pos = start
    if os.path.exists(dest):
        pos += os.path.getsize(dest)
    if pos > end:
        return
    fails = 0
    while pos <= end:
        stop = min(pos + CHUNK - 1, end)
        try:
            with _open(url, pos, stop) as r:
                data = r.read()
            if not data:
                raise IOError("empty chunk")
            with open(dest, "ab") as f:
                f.write(data)
            with lock:
                state["done"] += len(data)
            pos += len(data)
            fails = 0
        except Exception:
            fails += 1
            if fails > 200:
                raise
            time.sleep(min(0.5 * fails, 5.0))


def fetch(url, out, parts=4):
    total = _size(url)
    seg = (total + parts - 1) // parts
    ranges = []
    for i in range(parts):
        a = i * seg
        b = min(a + seg - 1, total - 1)
        if a <= b:
            ranges.append((a, b))

    state = {"done": 0}
    lock = threading.Lock()
    errors = []

    def run(i, a, b):
        part = f"{out}.seg{i}"
        try:
            _worker(url, part, a, b, state, lock)
        except Exception as exc:
            errors.append(f"seg{i}: {type(exc).__name__} {exc}")

    threads = []
    for i, (a, b) in enumerate(ranges):
        t = threading.Thread(target=run, args=(i, a, b), daemon=True)
        t.start()
        threads.append(t)

    t0 = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(2.0)
        pct = state["done"] / total * 100
        speed = state["done"] / max(1e-6, time.time() - t0) / 1024
        print(f"\r  {pct:5.1f}%  {state['done'] / 1048576:5.1f}/"
              f"{total / 1048576:.1f} MB  {speed:5.0f} KB/s", end="", flush=True)
    for t in threads:
        t.join()
    print()

    if errors:
        print("失败:", "; ".join(errors))
        return False

    with open(out, "wb") as dst:
        for i in range(len(ranges)):
            part = f"{out}.seg{i}"
            with open(part, "rb") as src:
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    dst.write(buf)
            os.remove(part)

    got = os.path.getsize(out)
    if got != total:
        print(f"大小不符: {got} != {total}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://storage.googleapis.com/"
                    "flywire-data/codex/data/fafb/783")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("files", nargs="*", default=[
        "classification.csv.gz", "coordinates.csv.gz",
        "connections.csv.gz", "consolidated_cell_types.csv.gz"])
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    ok = True
    for name in a.files:
        dst = os.path.join(a.out, name)
        url = f"{a.base}/{name}"
        try:
            total = _size(url)
        except Exception as exc:
            print(f"{name}: 无法访问 ({exc})")
            ok = False
            continue
        if os.path.exists(dst) and os.path.getsize(dst) == total and not a.force:
            print(f"{name}: 已存在 {total} 字节，跳过")
            continue
        for seg in range(a.parts):
            p = f"{dst}.seg{seg}"
            if os.path.exists(p) and a.force:
                os.remove(p)
        print(f"{name}: {total / 1048576:.1f} MB")
        if not fetch(url, dst, a.parts):
            ok = False
            break
        print(f"  -> {os.path.getsize(dst)} 字节")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

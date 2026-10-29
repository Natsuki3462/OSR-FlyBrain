







const PORT = process.env.CDP_PORT || 9223;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getPage() {
  for (let i = 0; i < 60; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/list`);
      const list = await r.json();
      const p = list.find((t) => t.type === "page");
      if (p) return p;
    } catch (e) {  }
    await sleep(300);
  }
  throw new Error("CDP 没就绪");
}

async function main() {
  const url = process.argv[2] || "http://127.0.0.1:8099/";
  const out = process.argv[3] || "shot.png";
  const wait = parseInt(process.argv[4] || "8000", 10);
  const W = parseInt(process.argv[5] || "1440", 10);
  const H = parseInt(process.argv[6] || "900", 10);
  const preJs = process.argv[7] || "";

  const page = await getPage();
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 0;
  const pending = new Map();

  ws.addEventListener("message", (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
    }
  });

  const send = (method, params = {}) => new Promise((res, rej) => {
    const mid = ++id;
    pending.set(mid, (m) => (m.error ? rej(new Error(method + ": " + JSON.stringify(m.error))) : res(m.result)));
    ws.send(JSON.stringify({ id: mid, method, params }));
  });

  await new Promise((res, rej) => {
    ws.addEventListener("open", res, { once: true });
    ws.addEventListener("error", rej, { once: true });
  });

  await send("Page.enable");
  await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", {
    width: W, height: H, deviceScaleFactor: 1, mobile: false,
  });
  await send("Page.navigate", { url });
  await sleep(wait);

  if (preJs) {
    await send("Runtime.evaluate", {
      expression: preJs, returnByValue: true, awaitPromise: true,
    }).catch((e) => console.error("preJs:", e.message));
    await sleep(900);
  }

  const errs = await send("Runtime.evaluate", {
    expression: "JSON.stringify(window.__ERR__ || null)", returnByValue: true,
  }).catch(() => null);

  const shot = await send("Page.captureScreenshot", { format: "png" });
  const fs = require("fs");
  fs.writeFileSync(out, Buffer.from(shot.data, "base64"));
  console.log("saved", out, fs.statSync(out).size, "bytes");
  if (errs && errs.result && errs.result.value && errs.result.value !== "null") {
    console.log("page errors:", errs.result.value);
  }
  ws.close();
}

main().then(() => process.exit(0)).catch((e) => {
  console.error("FAILED:", e.message);
  process.exit(1);
});

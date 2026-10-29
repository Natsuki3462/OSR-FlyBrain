




const PORT = process.env.CDP_PORT || 9223;
const VW = parseInt(process.env.CDP_W || "1440", 10);
const VH = parseInt(process.env.CDP_H || "900", 10);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getPage() {
  for (let i = 0; i < 60; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const p = list.find((t) => t.type === "page");
      if (p) return p;
    } catch (e) {}
    await sleep(300);
  }
  throw new Error("CDP 没就绪");
}

async function main() {
  const url = process.argv[2];
  const expr = process.argv[3];
  const wait = parseInt(process.argv[4] || "6000", 10);

  const page = await getPage();
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 0;
  const pending = new Map();
  ws.addEventListener("message", (ev) => {
    let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  });
  const send = (method, params = {}) => new Promise((res, rej) => {
    const mid = ++id;
    pending.set(mid, (m) => (m.error ? rej(new Error(JSON.stringify(m.error))) : res(m.result)));
    ws.send(JSON.stringify({ id: mid, method, params }));
  });
  await new Promise((res, rej) => {
    ws.addEventListener("open", res, { once: true });
    ws.addEventListener("error", rej, { once: true });
  });

  await send("Page.enable");
  await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", {
    width: VW, height: VH, deviceScaleFactor: 1, mobile: false,
  });
  await send("Page.navigate", { url });
  await sleep(wait);

  const r = await send("Runtime.evaluate", {
    expression: expr, returnByValue: true, awaitPromise: true,
  });
  console.log(JSON.stringify(r.result && r.result.value !== undefined
    ? r.result.value : r, null, 2));
  ws.close();
}

main().then(() => process.exit(0)).catch((e) => {
  console.error("FAILED:", e.message); process.exit(1);
});

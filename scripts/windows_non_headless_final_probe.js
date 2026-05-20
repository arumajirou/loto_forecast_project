const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const chromePath = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const port = 9227;
const targetUrl = "http://localhost:8501";
const screenshotPath = "C:\\Temp\\16_windows_non_headless_final.png";
const fontCheckPath = "C:\\Temp\\17_windows_non_headless_font_check.png";
const jsonPath = "C:\\Temp\\windows_runtime_final.json";
const profileDir = "C:\\Temp\\codex-chrome-final-js";

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForJson(url, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        return await response.json();
      }
      lastError = new Error(`HTTP ${response.status} for ${url}`);
    } catch (error) {
      lastError = error;
    }
    await sleep(500);
  }
  throw lastError || new Error(`Timed out waiting for ${url}`);
}

async function send(socket, state, method, params = {}) {
  const id = ++state.id;
  socket.send(JSON.stringify({ id, method, params }));
  return await new Promise((resolve, reject) => {
    state.pending.set(id, { resolve, reject });
  });
}

async function createCdpSession(wsUrl) {
  const socket = new WebSocket(wsUrl);
  const state = {
    id: 0,
    pending: new Map(),
    events: [],
  };

  socket.onmessage = (event) => {
    const message = JSON.parse(event.data.toString());
    if (message.id && state.pending.has(message.id)) {
      const { resolve, reject } = state.pending.get(message.id);
      state.pending.delete(message.id);
      if (message.error) {
        reject(new Error(JSON.stringify(message.error)));
      } else {
        resolve(message);
      }
      return;
    }
    state.events.push(message);
  };

  await new Promise((resolve, reject) => {
    socket.onopen = resolve;
    socket.onerror = reject;
  });

  return { socket, state };
}

async function runtimeEval(session, expression) {
  const response = await send(session.socket, session.state, "Runtime.evaluate", {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  return response.result.result.value;
}

function ensureDir(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

async function main() {
  ensureDir(screenshotPath);
  ensureDir(fontCheckPath);
  ensureDir(jsonPath);

  const chrome = spawn(
    chromePath,
    [
      "--new-window",
      "--no-first-run",
      "--no-default-browser-check",
      `--remote-debugging-address=127.0.0.1`,
      `--remote-debugging-port=${port}`,
      "--remote-allow-origins=*",
      `--user-data-dir=${profileDir}`,
      "--window-position=40,40",
      "--window-size=1400,980",
      targetUrl,
    ],
    {
      detached: false,
      stdio: ["ignore", "ignore", "ignore"],
    },
  );

  try {
    const version = await waitForJson(`http://127.0.0.1:${port}/json/version`, 45000);
    const targets = await waitForJson(`http://127.0.0.1:${port}/json/list`, 45000);
    const pageTarget =
      targets.find((target) => target.type === "page" && String(target.url || "").startsWith(targetUrl)) ||
      targets.find((target) => target.type === "page");

    if (!pageTarget) {
      throw new Error("No page target found");
    }

    const session = await createCdpSession(pageTarget.webSocketDebuggerUrl);
    await send(session.socket, session.state, "Page.enable");
    await send(session.socket, session.state, "Runtime.enable");
    await send(session.socket, session.state, "Network.enable");

    const readyDeadline = Date.now() + 60000;
    let ready = null;
    while (Date.now() < readyDeadline) {
      ready = JSON.parse(
        await runtimeEval(
          session,
          `(() => JSON.stringify({
            title: document.title,
            h1: document.querySelector("h1")?.textContent ?? "",
            hasDb: (document.body?.innerText ?? "").includes("DB接続"),
            hasHeading: (document.body?.innerText ?? "").includes("ロト予測")
          }))()`,
        ),
      );
      if (ready.h1.includes("ロト予測") && ready.hasDb) {
        break;
      }
      await sleep(1000);
    }

    if (!ready || !ready.h1.includes("ロト予測") || !ready.hasDb) {
      throw new Error(`Dashboard did not hydrate in Windows Chrome: ${JSON.stringify(ready)}`);
    }

    const evaluation = JSON.parse(
      await runtimeEval(
        session,
        `(() => JSON.stringify({
          title: document.title,
          htmlLang: document.documentElement.lang || null,
          metaDescription: document.head.querySelector('meta[name="description"]')?.content ?? null,
          h1: document.querySelector("h1")?.textContent ?? null,
          sidebarTitle: document.querySelector(".ops-sidebar-title")?.textContent ?? null,
          hostLabel: Array.from(document.querySelectorAll("label, div, span, p")).find((el) => el.textContent?.trim() === "ホスト")?.textContent?.trim() ?? null,
          bodyTextHasJapanese: (document.body?.innerText ?? "").includes("ロト予測"),
          bodyTextHasDbLabel: (document.body?.innerText ?? "").includes("DB接続"),
          bodyTextSnippet: (document.body?.innerText ?? "").slice(0, 240),
          appFontFamily: getComputedStyle(document.querySelector(".stApp")).fontFamily,
          ua: navigator.userAgent,
          platform: navigator.platform,
          webdriver: navigator.webdriver,
          devicePixelRatio: window.devicePixelRatio,
          h1Rect: document.querySelector("h1")?.getBoundingClientRect() ?? null,
          sidebarRect: document.querySelector(".ops-sidebar-title")?.getBoundingClientRect() ?? null,
          notoCheck: document.fonts ? document.fonts.check('16px "OpsNotoSansJP"', "ロト予測") : null,
          notoFallbackCheck: document.fonts ? document.fonts.check('16px "Noto Sans JP"', "ロト予測") : null
        }))()`,
      ),
    );

    const fullShot = await send(session.socket, session.state, "Page.captureScreenshot", {
      format: "png",
      captureBeyondViewport: true,
    });
    fs.writeFileSync(screenshotPath, Buffer.from(fullShot.result.data, "base64"));

    const h1Rect = evaluation.h1Rect || { x: 320, y: 90, width: 700, height: 180 };
    const sidebarRect = evaluation.sidebarRect || { x: 20, y: 40, width: 260, height: 120 };
    const clip = {
      x: Math.max(0, Math.min(sidebarRect.x || sidebarRect.left || 20, h1Rect.x || h1Rect.left || 320) - 24),
      y: Math.max(0, Math.min(sidebarRect.y || sidebarRect.top || 40, h1Rect.y || h1Rect.top || 90) - 24),
      width: Math.max(
        500,
        Math.max(
          (h1Rect.x || h1Rect.left || 320) + (h1Rect.width || 700),
          (sidebarRect.x || sidebarRect.left || 20) + (sidebarRect.width || 260),
        ) -
          Math.max(0, Math.min(sidebarRect.x || sidebarRect.left || 20, h1Rect.x || h1Rect.left || 320) - 24) +
          24,
      ),
      height: 420,
      scale: 1,
    };

    const clipShot = await send(session.socket, session.state, "Page.captureScreenshot", {
      format: "png",
      clip,
    });
    fs.writeFileSync(fontCheckPath, Buffer.from(clipShot.result.data, "base64"));

    const payload = {
      generatedAt: new Date().toISOString(),
      targetUrl,
      chromeBinary: chromePath,
      port,
      process: {
        pid: chrome.pid,
      },
      browserVersion: version,
      selectedTarget: pageTarget,
      evaluation,
      screenshots: {
        full: screenshotPath,
        fontCheck: fontCheckPath,
      },
      clip,
    };
    fs.writeFileSync(jsonPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");

    session.socket.close();
  } finally {
    try {
      process.kill(chrome.pid);
    } catch {}
  }
}

main().catch((error) => {
  fs.writeFileSync(
    "C:\\Temp\\windows_non_headless_final_probe.log",
    `${error.stack || String(error)}\n`,
    "utf8",
  );
  process.exit(1);
});

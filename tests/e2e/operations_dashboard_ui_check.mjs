import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import net from "node:net";
import { setTimeout as delay } from "node:timers/promises";

import { chromium } from "playwright";

const APP_PATH =
  "${PROJECT_ROOT}/src/loto_forecast/api/streamlit/operations_dashboard.py";
const PROJECT_ROOT = "${PROJECT_ROOT}";

async function pickFreePort() {
  return await new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close((err) => {
        if (err) {
          reject(err);
          return;
        }
        resolve(address.port);
      });
    });
    server.on("error", reject);
  });
}

async function waitForHealth(baseUrl, timeoutMs = 90000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${baseUrl}/_stcore/health`);
      if (response.ok) {
        return;
      }
    } catch {
      // keep polling until the server is ready
    }
    await delay(500);
  }
  throw new Error(`Streamlit server did not become ready: ${baseUrl}`);
}

async function withServer(run) {
  if (process.env.BASE_URL) {
    await run(process.env.BASE_URL);
    return;
  }

  const port = await pickFreePort();
  const baseUrl = `http://127.0.0.1:${port}`;
  const server = spawn(
    "python",
    [
      "-m",
      "streamlit",
      "run",
      APP_PATH,
      "--server.headless",
      "true",
      "--server.port",
      String(port),
      "--browser.gatherUsageStats",
      "false",
    ],
    {
      cwd: PROJECT_ROOT,
      env: { ...process.env, MPLCONFIGDIR: "/tmp/matplotlib" },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  let serverLog = "";
  server.stdout.on("data", (chunk) => {
    serverLog += chunk.toString();
  });
  server.stderr.on("data", (chunk) => {
    serverLog += chunk.toString();
  });

  try {
    await waitForHealth(baseUrl);
    await run(baseUrl);
  } catch (error) {
    error.message += `\n\nServer log:\n${serverLog.slice(-4000)}`;
    throw error;
  } finally {
    server.kill("SIGTERM");
    await Promise.race([
      new Promise((resolve) => server.once("exit", resolve)),
      delay(5000).then(() => server.kill("SIGKILL")),
    ]);
  }
}

async function injectAxe(page) {
  await page.evaluate(async () => {
    if (window.axe) {
      return;
    }
    await new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.2/axe.min.js";
      script.onload = resolve;
      script.onerror = () => reject(new Error("axe load failed"));
      document.head.appendChild(script);
    });
  });
}

await withServer(async (baseUrl) => {
  const browser = await chromium.launch({ headless: true });
  const desktop = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await desktop.goto(baseUrl, { waitUntil: "networkidle", timeout: 90000 });

  await desktop.getByRole("heading", { name: "ロト予測 運用ダッシュボード" }).waitFor();
  const passwordValue = await desktop.getByRole("textbox", { name: "パスワード" }).inputValue();
  assert.equal(passwordValue, "", "password input should not expose a stored password");

  const bodyText = await desktop.locator("body").innerText();
  assert.equal(bodyText.includes(".git/"), false, "project tree should not expose hidden repo internals");

  const menuVisibility = await desktop.evaluate(() => {
    const el = document.querySelector("#MainMenu");
    if (!el) {
      return "missing";
    }
    return window.getComputedStyle(el).visibility;
  });
  assert.ok(menuVisibility === "hidden" || menuVisibility === "missing");

  await injectAxe(desktop);
  const axeResult = await desktop.evaluate(async () => {
    const result = await window.axe.run(document, {
      runOnly: { type: "tag", values: ["wcag2a", "wcag2aa"] },
    });
    return result.violations.map((violation) => violation.id);
  });
  assert.ok(Array.isArray(axeResult), "axe should return a violation id list");

  const mobile = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await mobile.goto(baseUrl, { waitUntil: "networkidle", timeout: 90000 });
  const mobileMetrics = await mobile.evaluate(() => {
    const heading = document.querySelector("h1");
    const rect = heading?.getBoundingClientRect();
    return {
      fontSize: Number.parseFloat(window.getComputedStyle(heading).fontSize),
      right: rect?.right ?? 0,
      viewport: window.innerWidth,
    };
  });
  assert.ok(mobileMetrics.fontSize <= 60, `mobile h1 font size too large: ${mobileMetrics.fontSize}`);
  assert.ok(mobileMetrics.right <= mobileMetrics.viewport + 4, "mobile h1 should not overflow horizontally");

  await mobile.close();
  await desktop.close();
  await browser.close();
  console.log(`operations_dashboard_ui_check passed for ${baseUrl}`);
});

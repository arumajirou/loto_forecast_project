import fs from "node:fs/promises";
import path from "node:path";

import { chromium } from "playwright";

function getArg(name, fallback = null) {
  const idx = process.argv.indexOf(name);
  if (idx === -1 || idx === process.argv.length - 1) {
    return fallback;
  }
  return process.argv[idx + 1];
}

const targetUrl = getArg("--url", "http://127.0.0.1:8501");
const screenshotPath = getArg("--screenshot");
const jsonPath = getArg("--json");
const width = Number(getArg("--width", "1280"));
const height = Number(getArg("--height", "800"));
const deviceScaleFactor = Number(getArg("--device-scale-factor", "1"));
const mobile = getArg("--mobile", "false") === "true";

if (!screenshotPath || !jsonPath) {
  console.error("Usage: node scripts/linux_browser_probe.mjs --screenshot <path> --json <path> [--url <url>]");
  process.exit(1);
}

const sample = "ロト予測 運用ダッシュボード";

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({
  viewport: { width, height },
  deviceScaleFactor,
  isMobile: mobile,
  hasTouch: mobile,
});

const response = await page.goto(targetUrl, { waitUntil: "domcontentloaded", timeout: 90000 });
await page.waitForSelector("h1", { timeout: 90000 });
await page.waitForTimeout(2000);

const evaluation = await page.evaluate((sampleText) => {
  const app = document.querySelector(".stApp");
  const metaDescription = document.head.querySelector('meta[name="description"]')?.content ?? null;
  const metaCharset = document.head.querySelector('meta[charset]')?.getAttribute("charset") ?? null;
  const htmlLang = document.documentElement.lang || null;
  const bodyText = document.body?.innerText ?? "";
  const h1 = document.querySelector("h1")?.textContent ?? null;
  const bodyStyle = document.body ? getComputedStyle(document.body).fontFamily : null;
  const appStyle = app ? getComputedStyle(app).fontFamily : null;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  ctx.font = `32px ${appStyle || bodyStyle || "sans-serif"}`;
  const sampleWidth = ctx.measureText(sampleText).width;
  ctx.font = "32px monospace";
  const monoWidth = ctx.measureText(sampleText).width;
  return {
    title: document.title,
    htmlLang,
    metaCharset,
    metaDescription,
    h1,
    bodyTextHasJapanese: bodyText.includes("ロト予測"),
    bodyTextSnippet: bodyText.slice(0, 200),
    bodyFontFamily: bodyStyle,
    appFontFamily: appStyle,
    ua: navigator.userAgent,
    platform: navigator.platform,
    webdriver: navigator.webdriver,
    viewport: { width: window.innerWidth, height: window.innerHeight, devicePixelRatio: window.devicePixelRatio },
    sampleWidth,
    monoWidth,
    fontsReady: document.fonts ? document.fonts.status : null,
    notoCheck: document.fonts ? document.fonts.check('16px "Noto Sans JP"', sampleText) : null,
    meiryoCheck: document.fonts ? document.fonts.check('16px "Meiryo"', sampleText) : null,
    yuGothicCheck: document.fonts ? document.fonts.check('16px "Yu Gothic"', sampleText) : null,
    bizCheck: document.fonts ? document.fonts.check('16px "BIZ UDGothic"', sampleText) : null,
    screenshotPath: null,
  };
}, sample);

await fs.mkdir(path.dirname(screenshotPath), { recursive: true });
await fs.mkdir(path.dirname(jsonPath), { recursive: true });
await page.screenshot({ path: screenshotPath, fullPage: true });

const payload = {
  targetUrl,
  screenshotPath,
  jsonPath,
  response: response
    ? {
        status: response.status(),
        headers: await response.allHeaders(),
      }
    : null,
  evaluation,
};

await fs.writeFile(jsonPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
await browser.close();

console.log(JSON.stringify(payload, null, 2));

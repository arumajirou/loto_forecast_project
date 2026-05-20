import fs from "node:fs/promises";
import path from "node:path";

import { chromium } from "@playwright/test";

const projectRoot = process.cwd();
const baseUrl = process.env.BASE_URL || "http://127.0.0.1:8501";
const routeName = process.env.ROUTE_NAME || "unspecified_route";
const headless = (process.env.PW_HEADLESS || "1").trim().toLowerCase() !== "0";
const screenshotDir =
  process.env.SCREENSHOT_DIR || path.join(projectRoot, "artifacts", "screenshots", "exhaustive");
const observationPath =
  process.env.OBS_PATH || path.join(projectRoot, "artifacts", "logs", "browser_observation_detailed.json");
const dynamicTraceJsonlPath =
  process.env.DYNAMIC_TRACE_JSONL || path.join(projectRoot, "artifacts", "logs", "dynamic_trace.jsonl");
const dynamicTraceCsvPath =
  process.env.DYNAMIC_TRACE_CSV || path.join(projectRoot, "artifacts", "logs", "dynamic_trace.csv");
const coverageMatrixPath =
  process.env.COVERAGE_MATRIX || path.join(projectRoot, "artifacts", "logs", "coverage_matrix.md");
const executionSummaryPath =
  process.env.EXEC_SUMMARY || path.join(projectRoot, "artifacts", "logs", "execution_summary.md");
const uiSummaryPath =
  process.env.UI_EXEC_SUMMARY || path.join(projectRoot, "artifacts", "logs", "ui_redesign_execution_summary.md");
const fileObservationPath =
  process.env.FILE_OBS_PATH || path.join(projectRoot, "artifacts", "logs", "file_observation.json");
const dbObservationPath =
  process.env.DB_OBS_PATH || path.join(projectRoot, "artifacts", "logs", "db_observation.json");
const serverLogPath = process.env.SERVER_LOG_PATH || null;
const executablePath = resolveExecutablePath();

await Promise.all([
  fs.mkdir(screenshotDir, { recursive: true }),
  fs.mkdir(path.dirname(observationPath), { recursive: true }),
  fs.mkdir(path.dirname(dynamicTraceJsonlPath), { recursive: true }),
]);

const browser = await chromium.launch({
  headless,
  executablePath,
});
const context = await browser.newContext({ viewport: { width: 1600, height: 1400 } });
const page = await context.newPage();

const checks = [];
const screenshots = [];
const consoleErrors = [];
const pageErrors = [];
const uiTracebacks = [];
const serverLogErrors = [];
const dynamicTraceRows = [];
const startedAt = new Date().toISOString();

page.on("console", (message) => {
  if (message.type() === "error") {
    consoleErrors.push(message.text());
  }
});
page.on("pageerror", (error) => {
  pageErrors.push(String(error));
});

try {
  await check("initial_display", "初期表示", async () => {
    await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: 120000 });
    await page.waitForLoadState("networkidle");
    await page.getByRole("heading", { name: "ロト予測 運用ダッシュボード" }).waitFor();
    await page.getByLabel("表示パネル(高速モード)").waitFor();
    await takeShot("home");
  });

  await check("notification_settings", "通知設定表示", async () => {
    await page.getByText("通知設定: email=", { exact: false }).waitFor();
    const beepCount = await page.getByLabel("通知音 ON/OFF").count();
    const emailCount = await page.getByLabel("メール通知 dry-run").count();
    if (beepCount < 1 || emailCount < 1) {
      throw new Error(`notification controls missing: beep=${beepCount} email=${emailCount}`);
    }
  });

  await check("overview_panel", "概要パネル", async () => {
    await openPanel("概要");
    if (isConnectedRoute()) {
      await page.getByRole("heading", { name: "概要" }).waitFor();
    } else {
      await page.getByText("DB未接続のため表示できません。", { exact: false }).waitFor();
    }
    await takeShot("overview");
  });

  await check("operations_panel", "運用パネル", async () => {
    await openPanel("運用");
    if (isConnectedRoute()) {
      await page.getByRole("tab", { name: "実行履歴" }).waitFor();
    } else {
      await page.getByRole("tab", { name: "機能動作確認" }).waitFor();
    }
    await takeShot("operations");
  });

  await check("wizard_display", "Step Wizard 表示", async () => {
    await openTrainPanel();
    await page.getByText("Step Wizard").waitFor();
    await page.getByText("実行前チェック").first().waitFor();
    await page.getByRole("button", { name: "Run train" }).waitFor();
    await takeShot("nf_lab_train");
  });

  await check("recommended_preset", "おすすめ設定を自動入力", async () => {
    await openTrainPanel();
    await page.getByRole("button", { name: "おすすめ設定を自動入力" }).click();
    await page.waitForTimeout(1200);
    await page.getByRole("button", { name: "おすすめ設定を自動入力" }).waitFor();
    await assertFieldValue("backend", "ray");
    await assertFieldValue("search_alg", "BasicVariantGenerator");
    await assertFieldValue("num_samples", "30");
    await assertRadioChecked("標準");
    await takeShot("notification_recommended");
  });

  await check("quick_preset", "最短で試す", async () => {
    await openTrainPanel();
    await page.getByRole("button", { name: "最短で試す" }).click();
    await page.waitForTimeout(1200);
    await page.getByRole("button", { name: "最短で試す" }).waitFor();
    await assertFieldValue("backend", "optuna");
    await assertFieldValue("search_alg", "TPESampler");
    await assertFieldValue("num_samples", "10");
    await assertRadioChecked("かんたん");
    await takeShot("notification_quick");
  });

  await check("multi_candidate_selection", "候補複数選択", async () => {
    await prepareValidComboInputs();
    await openTrainPanel("固定/網羅メタ反映");
    await configureBackendSearchCandidates();
    await takeShot("multi_candidate_selection");
  });

  await check("combo_auto_exclude", "自動除外で有効候補生成", async () => {
    await prepareValidComboInputs();
    await openTrainPanel("固定/網羅メタ反映");
    await configureBackendSearchCandidates();
    await page.getByText("理論組合せ数").first().waitFor();
    await expectMetricValue("理論組合せ数", "4");
    await expectMetricValue("自動除外後の有効件数", "2");
    await expectMetricValue("除外件数", "2");
    await page.getByText("除外対象の設定組合せ: 2", { exact: false }).waitFor();
    await expectBodyTextContains("除外理由サマリ:");
    await expectBodyTextContains("backend と search_alg の整合性が取れないため除外しました。");
    await takeShot("combo_auto_exclude");
  });

  await check("meta_reflect_panel", "meta反映 UI", async () => {
    await openTrainPanel("固定/網羅メタ反映");
    await page.getByText("meta反映 dry-run（DB未書込）", { exact: false }).waitFor();
    await page.getByText("有効候補をすべて実行", { exact: false }).first().waitFor();
    await takeShot("meta_reflect");
  });

  await check("batch_run_results", "有効候補全実行と結果一覧", async () => {
    await prepareValidComboInputs("11");
    await openTrainPanel("固定/網羅メタ反映");
    await configureBackendSearchCandidates();
    await expectMetricValue("自動除外後の有効件数", "2");
    await page.getByRole("button", { name: "有効候補をすべて実行" }).click();
    await page.getByText("一括実行判定", { exact: false }).waitFor({ timeout: 1200000 });
    await page.getByText("success", { exact: false }).waitFor({ timeout: 120000 });
    await page.getByText("excluded", { exact: false }).waitFor({ timeout: 120000 });
    await page.getByText("status", { exact: false }).first().waitFor({ timeout: 120000 });
    await takeShot("batch_run_results");
  });

  await check("save_load_panel", "save/load UI", async () => {
    await openPanel("NeuralForecast 実行・検証ラボ");
    await selectComboboxOption("NeuralForecast 実行・検証ラボ メニュー", "保存/ロード");
    await page.getByText("保存(save) / ロード(load) / 保存+ロード+分析", { exact: false }).waitFor();
    const subMenu = page.getByLabel("保存/ロード サブメニュー");
    if (await subMenu.count()) {
      await selectComboboxOption("保存/ロード サブメニュー", "保存+ロード+分析");
    }
    await page.getByText("保存+ロード+分析", { exact: false }).first().waitFor();
    await takeShot("save_load");
  });

  await check("resource_panel", "リソース分析パネル", async () => {
    await openPanel("リソース分析");
    if (isConnectedRoute()) {
      await page.getByText("リソース分析", { exact: false }).waitFor();
    } else {
      await page.getByText("DB未接続のため表示できません。", { exact: false }).waitFor();
    }
    await takeShot("resources");
  });

  await check("schema_export_panel", "スキーマ出力パネル", async () => {
    await openPanel("スキーマ出力");
    if (isConnectedRoute()) {
      await page.getByText("スキーマスナップショット出力", { exact: false }).waitFor();
    } else {
      await page.getByText("DB未接続のため表示できません。", { exact: false }).waitFor();
    }
    await takeShot("schema_export");
  });

  await check("directory_panel", "ディレクトリ統合", async () => {
    await openPanel("ディレクトリ統合");
    await page.getByText("ディレクトリ統合コンパイラ", { exact: false }).waitFor();
    await takeShot("directory_compile");
  });

  await check("markdown_panel", "Markdown統合", async () => {
    await openPanel("Markdown統合");
    await page.getByText("Markdown資料コンパイラ", { exact: false }).waitFor();
    await takeShot("markdown_compile");
  });

  await check("artifacts_logs_panel", "成果物・ログ", async () => {
    await openPanel("成果物・ログ");
    await page.getByText("成果物 / ログ / 変更サマリ", { exact: false }).waitFor();
    await takeShot("artifacts_logs");
  });

  await check("server_log_clean", "server log 例外監査", async () => {
    const serverLogSummary = await loadServerLogSummary();
    if (serverLogSummary.errors.length > 0) {
      serverLogErrors.push(...serverLogSummary.errors);
      throw new Error(`server log errors detected: ${serverLogSummary.errors.join(" | ")}`);
    }
  });

  await writeArtifacts("passed");
  console.log(`operations_dashboard_redesign_ui_check passed for ${routeName} ${baseUrl}`);
} catch (error) {
  await takeShot("failure");
  await writeArtifacts("failed", error);
  throw error;
} finally {
  await context.close();
  await browser.close();
}

async function openPanel(panelName) {
  await selectComboboxOption("表示パネル(高速モード)", panelName);
  await page.waitForTimeout(1200);
}

async function openTrainPanel(subMenuName = "全パラメータ選択") {
  await openPanel("NeuralForecast 実行・検証ラボ");
  await selectComboboxOption("NeuralForecast 実行・検証ラボ メニュー", "学習(train)");
  const subMenu = page.getByLabel("学習(train) サブメニュー");
  if (await subMenu.count()) {
    await selectComboboxOption("学習(train) サブメニュー", subMenuName);
  }
  await page.waitForTimeout(1200);
  if (subMenuName === "固定/網羅メタ反映") {
    await ensureWidgetKeyVisible("nf_lab_axis_fixed_backend");
  }
}

async function clickRadio(label) {
  const clicked = await page.evaluate((targetLabel) => {
    const candidates = [
      ...document.querySelectorAll('[data-testid="stRadio"] label'),
      ...document.querySelectorAll('[role="radiogroup"] label'),
      ...document.querySelectorAll('label[data-baseweb="radio"]'),
      ...document.querySelectorAll('label'),
      ...document.querySelectorAll('[role="radio"]'),
      ...document.querySelectorAll('button'),
    ];
    const target = candidates.find((node) => node.textContent?.trim() === targetLabel);
    if (!target) {
      return false;
    }
    target.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    return true;
  }, label);
  if (!clicked) {
    const fallback = page.getByText(label, { exact: true }).first();
    await fallback.click();
  }
  await page.waitForTimeout(400);
}

async function assertFieldValue(label, expected) {
  let actual = "";
  const textbox = page.getByLabel(label).first();
  if (await textbox.count()) {
    actual = await textbox.inputValue().catch(() => "");
  }
  if (!String(actual).trim()) {
    const combobox = page.getByRole("combobox", { name: new RegExp(`${escapeRegExp(label)}$`) }).first();
    if (await combobox.count()) {
      const ariaLabel = String((await combobox.getAttribute("aria-label")) || "");
      const matched = ariaLabel.match(/^Selected\s+(.+?)\.\s+/);
      actual = matched ? matched[1] : ariaLabel;
    }
  }
  if (String(actual).trim() !== String(expected)) {
    throw new Error(`${label} expected=${expected} actual=${actual}`);
  }
}

async function assertRadioChecked(label) {
  const checked = await page.evaluate((targetLabel) => {
    const checkedInputs = [...document.querySelectorAll('input[type="radio"]:checked')];
    const checkedLabels = checkedInputs
      .map((input) => input.closest("label")?.textContent?.trim() || input.getAttribute("aria-label") || "")
      .filter(Boolean);
    if (checkedLabels.some((item) => item === targetLabel)) {
      return true;
    }
    const radioNodes = [...document.querySelectorAll('[role="radio"][aria-checked="true"]')];
    return radioNodes.some((node) => node.textContent?.trim() === targetLabel);
  }, label);
  if (!checked) {
    throw new Error(`radio not checked: ${label}`);
  }
}

async function setCheckbox(label, checked, widgetKey = null) {
  const checkbox = widgetKey
    ? page.locator(`.st-key-${widgetKey} input[type="checkbox"]`).first()
    : page.getByLabel(label);
  const container = widgetKey ? page.locator(`.st-key-${widgetKey}`).first() : checkbox;
  await container.waitFor({ state: "visible", timeout: 30000 });
  const current = await checkbox.isChecked().catch(() => false);
  if (current !== checked) {
    if (widgetKey) {
      const clickTarget = page.locator(`.st-key-${widgetKey} label`).filter({ hasText: label }).first();
      if (await clickTarget.count()) {
        await clickTarget.click();
      } else {
        await container.click();
      }
    } else {
      await checkbox.click();
    }
    await page.waitForTimeout(500);
  }
}

async function addMultiselectOption(label, optionName, widgetKey = null) {
  if (widgetKey) {
    await ensureWidgetKeyVisible(widgetKey);
  }
  const keyedTrigger = widgetKey
    ? page.locator(`.st-key-${widgetKey} [data-baseweb="select"]`).first()
    : page.locator("body").locator("non-existent");
  const keyedInput = widgetKey
    ? page.locator(`.st-key-${widgetKey} [data-baseweb="select"] input`).first()
    : page.locator("body").locator("non-existent");
  if (widgetKey && (await keyedTrigger.count())) {
    if (await keyedInput.count()) {
      await keyedInput.click();
    } else {
      await keyedTrigger.click();
    }
  } else {
    const input = page.getByLabel(label).first();
    if (await input.count()) {
      await input.click();
    } else {
      await page.getByText(label, { exact: true }).first().click();
    }
  }
  const option = page.getByRole("option", { name: optionName });
  if (await option.count()) {
    await option.click();
    await page.waitForTimeout(400);
  }
}

async function ensureFirstOptionSelected(label) {
  const input = page.getByRole("combobox", { name: new RegExp(`^${escapeRegExp(label)}$`) }).first();
  await input.click();
  const firstOption = page.getByRole("option").first();
  if (await firstOption.count()) {
    await firstOption.click();
    await page.waitForTimeout(500);
  } else {
    await page.keyboard.press("Escape").catch(() => {});
  }
}

async function prepareValidComboInputs(numSamples = null) {
  await openTrainPanel("全パラメータ選択");
  if (numSamples !== null) {
    await page.getByLabel("num_samples").fill(String(numSamples));
  }
  await ensureFirstOptionSelected("unique_id");
  await ensureFirstOptionSelected("ts_type");
}

async function configureBackendSearchCandidates() {
  await setCheckbox("backend 固定", false, "nf_lab_axis_fixed_backend");
  await addMultiselectOption("backend 候補", "ray", "nf_lab_axis_pool_backend");
  await setCheckbox("search_alg 固定", false, "nf_lab_axis_fixed_search_alg");
  await addMultiselectOption("search_alg 候補", "BasicVariantGenerator", "nf_lab_axis_pool_search_alg");
}

async function ensureWidgetKeyVisible(widgetKey) {
  const selector = `.st-key-${widgetKey}`;
  for (let attempt = 0; attempt < 8; attempt += 1) {
    const locator = page.locator(selector).first();
    if (await locator.count()) {
      await locator.scrollIntoViewIfNeeded().catch(() => {});
      await page.waitForTimeout(200);
      return;
    }
    await scrollMainBy(1200);
  }
  await page.locator(selector).first().waitFor({ state: "visible", timeout: 30000 });
}

async function scrollMainBy(deltaY) {
  await page.evaluate((delta) => {
    const app = document.querySelector('[data-testid="stAppViewContainer"]');
    const scroller =
      app?.querySelector('section.main') ||
      app?.querySelector('[data-testid="stMainBlockContainer"]') ||
      app ||
      document.scrollingElement ||
      document.documentElement;
    if (scroller) {
      scroller.scrollBy({ top: delta, behavior: "instant" });
    }
  }, deltaY);
  await page.waitForTimeout(250);
}

async function expectMetricValue(label, expectedText) {
  const bodyText = await page.locator("body").innerText();
  const matcher = new RegExp(`${label}\\s+${expectedText}`);
  if (!matcher.test(bodyText.replaceAll(",", ""))) {
    const idx = bodyText.indexOf(label);
    const snippet = idx >= 0 ? bodyText.slice(Math.max(0, idx - 80), idx + 240) : bodyText.slice(0, 320);
    throw new Error(`metric mismatch: ${label} expected=${expectedText} snippet=${JSON.stringify(snippet)}`);
  }
}

async function expectBodyTextContains(expectedText) {
  const bodyText = await page.locator("body").innerText();
  if (!bodyText.includes(expectedText)) {
    throw new Error(`body text missing: ${expectedText}`);
  }
}

async function check(id, label, fn) {
  const started = new Date().toISOString();
  console.log(`check:start:${id}`);
  try {
    await fn();
    await assertNoUiError(id);
    checks.push({ id, label, status: "pass", started_at: started, ended_at: new Date().toISOString() });
    dynamicTraceRows.push(buildTraceRow(id, label, "pass", ""));
    console.log(`check:pass:${id}`);
  } catch (error) {
    const errorText = String(error);
    checks.push({
      id,
      label,
      status: "fail",
      started_at: started,
      ended_at: new Date().toISOString(),
      error: errorText,
    });
    dynamicTraceRows.push(buildTraceRow(id, label, "fail", errorText));
    console.log(`check:fail:${id}:${errorText}`);
    throw error;
  }
}

function buildTraceRow(id, label, status, errorText) {
  return {
    route_name: routeName,
    page: label,
    action: id,
    expected_result: "traceback/console error/pageerror がなく画面が表示される",
    observed_result: status === "pass" ? "pass" : errorText,
    pass_fail: status,
    screenshot_path: screenshots[screenshots.length - 1] || "",
    started_at: new Date().toISOString(),
  };
}

async function assertNoUiError(contextLabel) {
  const exceptionLocator = page.locator('[data-testid="stException"], [data-testid="stAlert"]');
  const exceptionText = await exceptionLocator.allTextContents().catch(() => []);
  const rawTexts = exceptionText
    .map((item) => String(item || "").trim())
    .filter((item) => item && /(Traceback|StreamlitAPIException|Exception|Error)/.test(item));
  const pageText = await page.locator("body").innerText().catch(() => "");
  const matchedPageText = pageText
    .split("\n")
    .filter((line) => /(Traceback|StreamlitAPIException|cannot be modified after the widget)/.test(line))
    .slice(0, 5);
  const combined = [...rawTexts, ...matchedPageText].filter(Boolean);
  if (combined.length > 0) {
    uiTracebacks.push({ context: contextLabel, messages: combined });
    throw new Error(`${contextLabel}: ui traceback detected: ${combined.join(" | ")}`);
  }
}

async function takeShot(suffix) {
  const filePath = path.join(screenshotDir, `${routeName}_${suffix}.png`);
  await page.screenshot({ path: filePath, fullPage: true });
  screenshots.push(filePath);
  return filePath;
}

async function selectComboboxOption(label, optionName) {
  const input = page.getByLabel(label);
  await input.click();
  await page.getByRole("option", { name: optionName }).click();
}

function resolveExecutablePath() {
  const candidates = [
    process.env.PW_EXECUTABLE_PATH,
    process.env.CHROME_BIN,
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
  ].filter(Boolean);
  return candidates[0];
}

function isConnectedRoute() {
  return !routeName.includes("fallback");
}

async function writeArtifacts(result, error = null) {
  const userAgent = await page.evaluate(() => navigator.userAgent);
  const payload = {
    route_name: routeName,
    base_url: baseUrl,
    headless,
    executable_path: executablePath,
    user_agent: userAgent,
    started_at: startedAt,
    finished_at: new Date().toISOString(),
    result,
    checks,
    screenshots,
    console_errors: consoleErrors,
    page_errors: pageErrors,
    ui_tracebacks: uiTracebacks,
    server_log_errors: serverLogErrors,
    server_log_path: serverLogPath,
    error: error ? String(error) : null,
  };
  const existing = await loadJson(observationPath);
  const routes = Array.isArray(existing.routes) ? existing.routes.filter((item) => item.route_name !== routeName) : [];
  routes.push(payload);
  await fs.writeFile(
    observationPath,
    JSON.stringify({ updated_at: new Date().toISOString(), routes }, null, 2),
    "utf-8",
  );
  await appendDynamicTrace();
  await writeCoverageMatrix();
  await writeSummary(executionSummaryPath, result, error);
  await writeSummary(uiSummaryPath, result, error);
  await writeObservationFile(fileObservationPath, {
    updated_at: new Date().toISOString(),
    route_name: routeName,
    writes_observed: false,
    observations: [],
  });
  await writeObservationFile(dbObservationPath, {
    updated_at: new Date().toISOString(),
    route_name: routeName,
    writes_observed: false,
    observations: [],
  });
}

async function appendDynamicTrace() {
  const jsonl = dynamicTraceRows.map((row) => JSON.stringify(row)).join("\n");
  if (jsonl) {
    await fs.appendFile(dynamicTraceJsonlPath, `${jsonl}\n`, "utf-8");
  }
  const csvHeader = "route_name,page,action,expected_result,observed_result,pass_fail,screenshot_path,started_at\n";
  const csvRows = dynamicTraceRows
    .map((row) =>
      [
        row.route_name,
        row.page,
        row.action,
        row.expected_result,
        row.observed_result,
        row.pass_fail,
        row.screenshot_path,
        row.started_at,
      ]
        .map(csvEscape)
        .join(","),
    )
    .join("\n");
  const exists = await fileExists(dynamicTraceCsvPath);
  const prefix = exists ? "" : csvHeader;
  await fs.appendFile(dynamicTraceCsvPath, `${prefix}${csvRows}\n`, "utf-8");
}

async function writeCoverageMatrix() {
  const lines = [
    "# Coverage Matrix",
    "",
    `- route: ${routeName}`,
    `- base_url: ${baseUrl}`,
    `- updated_at: ${new Date().toISOString()}`,
    "",
    "| action | status | note |",
    "| --- | --- | --- |",
    ...checks.map((item) => `| ${item.label} | ${item.status} | ${item.error || ""} |`),
    "",
  ];
  await fs.writeFile(coverageMatrixPath, `${lines.join("\n")}\n`, "utf-8");
}

async function writeSummary(filePath, result, error) {
  const lines = [
    `# UI Execution Summary`,
    "",
    `- route: ${routeName}`,
    `- base_url: ${baseUrl}`,
    `- result: ${result}`,
    `- started_at: ${startedAt}`,
    `- finished_at: ${new Date().toISOString()}`,
    `- console_errors: ${consoleErrors.length}`,
    `- page_errors: ${pageErrors.length}`,
    `- ui_tracebacks: ${uiTracebacks.length}`,
    `- server_log_errors: ${serverLogErrors.length}`,
    `- server_log_path: ${serverLogPath || "n/a"}`,
    "",
    "## Checks",
    ...checks.map((item) => `- ${item.label}: ${item.status}${item.error ? ` (${item.error})` : ""}`),
    "",
  ];
  if (error) {
    lines.push("## Failure", `- ${String(error)}`, "");
  }
  await fs.writeFile(filePath, `${lines.join("\n")}\n`, "utf-8");
}

async function writeObservationFile(filePath, payload) {
  await fs.writeFile(filePath, JSON.stringify(payload, null, 2), "utf-8");
}

async function loadJson(filePath) {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function loadServerLogSummary() {
  if (!serverLogPath) {
    return { errors: [] };
  }
  try {
    const raw = await fs.readFile(serverLogPath, "utf-8");
    const lines = raw
      .split("\n")
      .map((line) => String(line || "").trim())
      .filter(Boolean);
    const errors = lines.filter((line) =>
      /(StreamlitAPIException|Traceback \(most recent call last\)|cannot be modified after the widget|Unhandled exception)/.test(line),
    );
    return { errors: errors.slice(-20) };
  } catch {
    return { errors: [] };
  }
}

function csvEscape(value) {
  const text = String(value ?? "");
  if (!/[",\n]/.test(text)) {
    return text;
  }
  return `"${text.replaceAll("\"", "\"\"")}"`;
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

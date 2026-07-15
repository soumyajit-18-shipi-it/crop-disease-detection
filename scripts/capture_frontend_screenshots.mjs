import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const CHROME_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const APP_URL = "http://127.0.0.1:5173/";
const DEBUG_PORT = 9333;
const VIEWPORT = { width: 1600, height: 900 };
const PROJECT_ROOT = process.cwd();
const OUTPUT_DIR = path.join(PROJECT_ROOT, "docs", "screenshots");
const PROFILE_DIR = path.join("C:\\tmp", `leaflight-screenshots-${process.pid}`);

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function waitForJson(url, timeoutMs = 20_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return response.json();
    } catch {
      // Chrome may still be starting.
    }
    await delay(200);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

class CdpClient {
  constructor(socket) {
    this.socket = socket;
    this.nextId = 1;
    this.pending = new Map();
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (!message.id || !this.pending.has(message.id)) return;
      const { resolve, reject } = this.pending.get(message.id);
      this.pending.delete(message.id);
      if (message.error) reject(new Error(message.error.message));
      else resolve(message.result);
    });
  }

  static async connect(url) {
    const socket = new WebSocket(url);
    await new Promise((resolve, reject) => {
      socket.addEventListener("open", resolve, { once: true });
      socket.addEventListener("error", reject, { once: true });
    });
    return new CdpClient(socket);
  }

  send(method, params = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  close() {
    this.socket.close();
  }
}

async function waitForExpression(client, expression, timeoutMs = 20_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const response = await client.send("Runtime.evaluate", {
      expression,
      returnByValue: true,
    });
    if (response.result?.value) return;
    await delay(200);
  }
  throw new Error(`Timed out waiting for page condition: ${expression}`);
}

async function saveScreenshot(client, filename) {
  await client.send("Runtime.evaluate", {
    expression: "window.scrollTo(0, 0)",
    returnByValue: true,
  });
  await delay(250);
  const screenshot = await client.send("Page.captureScreenshot", {
    format: "png",
    fromSurface: true,
    captureBeyondViewport: false,
  });
  const output = path.join(OUTPUT_DIR, filename);
  await writeFile(output, Buffer.from(screenshot.data, "base64"));
  return output;
}

await mkdir(OUTPUT_DIR, { recursive: true });
await mkdir(PROFILE_DIR, { recursive: true });

const chrome = spawn(
  CHROME_PATH,
  [
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--remote-allow-origins=*",
    `--remote-debugging-port=${DEBUG_PORT}`,
    `--user-data-dir=${PROFILE_DIR}`,
    `--window-size=${VIEWPORT.width},${VIEWPORT.height}`,
    "about:blank",
  ],
  { stdio: "ignore", windowsHide: true },
);

let client;
try {
  const targets = await waitForJson(`http://127.0.0.1:${DEBUG_PORT}/json/list`);
  const target = targets.find((item) => item.type === "page");
  if (!target) throw new Error("Chrome did not expose a page target");

  client = await CdpClient.connect(target.webSocketDebuggerUrl);
  await client.send("Page.enable");
  await client.send("Runtime.enable");
  await client.send("Emulation.setDeviceMetricsOverride", {
    ...VIEWPORT,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await client.send("Page.navigate", { url: APP_URL });
  await waitForExpression(
    client,
    "document.readyState === 'complete' && document.querySelector('.app-shell') !== null",
  );
  await waitForExpression(client, "document.querySelectorAll('.history-panel li').length > 0");
  await delay(500);

  const scanPath = await saveScreenshot(client, "leaflight-scan.png");

  await client.send("Runtime.evaluate", {
    expression: `(() => {
      const button = [...document.querySelectorAll('nav button')]
        .find((item) => item.textContent.trim() === 'Dashboard');
      if (!button) return false;
      button.click();
      return true;
    })()`,
    returnByValue: true,
  });
  await waitForExpression(
    client,
    "document.querySelector('.dashboard-layout') !== null && document.querySelector('.chart-panel svg') !== null",
  );
  await delay(750);

  const dashboardPath = await saveScreenshot(client, "leaflight-dashboard.png");
  console.log(JSON.stringify({ scanPath, dashboardPath, viewport: VIEWPORT }, null, 2));
} finally {
  client?.close();
  chrome.kill();
}

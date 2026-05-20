param(
  [int]$Port = 9226,
  [string]$TargetUrl = "http://localhost:8501",
  [string]$ScreenshotPath = "C:\Temp\16_windows_non_headless_final.png",
  [string]$FontCheckPath = "C:\Temp\17_windows_non_headless_font_check.png",
  [string]$JsonPath = "C:\Temp\windows_runtime_final.json",
  [string]$ProfileDir = "C:\Temp\codex-chrome-final",
  [int]$WindowWidth = 1400,
  [int]$WindowHeight = 980,
  [int]$WindowLeft = 40,
  [int]$WindowTop = 40,
  [int]$TimeoutSec = 60
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System;
using System.Runtime.InteropServices;

public struct RECT {
  public int Left;
  public int Top;
  public int Right;
  public int Bottom;
}

public static class Win32 {
  [DllImport("user32.dll")]
  public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);

  [DllImport("user32.dll")]
  public static extern bool SetForegroundWindow(IntPtr hWnd);

  [DllImport("user32.dll")]
  public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow);
}
"@

function Ensure-ParentDirectory {
  param([string]$PathValue)
  $parent = Split-Path -Parent $PathValue
  if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
  }
}

function Wait-HttpJson {
  param(
    [string]$Uri,
    [int]$TimeoutSeconds
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      return Invoke-RestMethod -Uri $Uri -TimeoutSec 2
    } catch {
      Start-Sleep -Milliseconds 500
    }
  }
  throw "Timed out waiting for $Uri"
}

function Receive-CdpMessage {
  param(
    [System.Net.WebSockets.ClientWebSocket]$Socket,
    [byte[]]$Buffer
  )

  $stream = New-Object System.IO.MemoryStream
  while ($true) {
    $segment = [System.ArraySegment[byte]]::new($Buffer)
    $result = $Socket.ReceiveAsync($segment, [Threading.CancellationToken]::None).GetAwaiter().GetResult()
    if ($result.MessageType -eq [System.Net.WebSockets.WebSocketMessageType]::Close) {
      throw "CDP websocket closed unexpectedly"
    }
    $stream.Write($Buffer, 0, $result.Count)
    if ($result.EndOfMessage) {
      break
    }
  }
  $json = [Text.Encoding]::UTF8.GetString($stream.ToArray())
  $stream.Dispose()
  return $json
}

function Send-CdpCommand {
  param(
    [System.Net.WebSockets.ClientWebSocket]$Socket,
    [int]$Id,
    [string]$Method,
    [hashtable]$Params,
    [System.Collections.Generic.List[object]]$Events,
    [byte[]]$Buffer
  )

  $payload = @{
    id = $Id
    method = $Method
    params = $Params
  } | ConvertTo-Json -Depth 30 -Compress
  $bytes = [Text.Encoding]::UTF8.GetBytes($payload)
  $segment = [System.ArraySegment[byte]]::new($bytes)
  $Socket.SendAsync(
    $segment,
    [System.Net.WebSockets.WebSocketMessageType]::Text,
    $true,
    [Threading.CancellationToken]::None
  ).GetAwaiter().GetResult()

  while ($true) {
    $raw = Receive-CdpMessage -Socket $Socket -Buffer $Buffer
    $msg = $raw | ConvertFrom-Json
    $msgIdProperty = $msg.PSObject.Properties["id"]
    if ($null -ne $msgIdProperty -and [int]$msgIdProperty.Value -eq $Id) {
      return $msg
    }
    $Events.Add($msg)
  }
}

function Invoke-RuntimeEval {
  param(
    [System.Net.WebSockets.ClientWebSocket]$Socket,
    [int]$Id,
    [string]$Expression,
    [System.Collections.Generic.List[object]]$Events,
    [byte[]]$Buffer
  )

  return Send-CdpCommand -Socket $Socket -Id $Id -Method "Runtime.evaluate" -Params @{
    expression = $Expression
    returnByValue = $true
    awaitPromise = $true
  } -Events $Events -Buffer $Buffer
}

function Get-CdpValue {
  param([object]$Message)

  $resultProperty = $Message.PSObject.Properties["result"]
  if ($null -eq $resultProperty) {
    $errorProperty = $Message.PSObject.Properties["error"]
    if ($null -ne $errorProperty) {
      throw ("CDP command failed: " + ($errorProperty.Value | ConvertTo-Json -Compress))
    }
    throw "CDP message did not contain result"
  }

  $innerResult = $resultProperty.Value.PSObject.Properties["result"]
  if ($null -eq $innerResult) {
    throw "CDP result payload did not contain Runtime result"
  }

  $valueProperty = $innerResult.Value.PSObject.Properties["value"]
  if ($null -eq $valueProperty) {
    throw "CDP runtime result did not contain value"
  }

  return $valueProperty.Value
}

function Wait-ChromeWindow {
  param(
    [System.Diagnostics.Process]$Process,
    [int]$TimeoutSeconds
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    $Process.Refresh()
    if ($Process.MainWindowHandle -ne 0) {
      return $Process.MainWindowHandle
    }
    Start-Sleep -Milliseconds 500
  }
  throw "Timed out waiting for Chrome main window"
}

function Capture-WindowBitmap {
  param(
    [IntPtr]$Hwnd,
    [string]$PathValue
  )

  $rect = New-Object RECT
  if (-not [Win32]::GetWindowRect($Hwnd, [ref]$rect)) {
    throw "GetWindowRect failed"
  }
  $width = [Math]::Max(1, $rect.Right - $rect.Left)
  $height = [Math]::Max(1, $rect.Bottom - $rect.Top)
  $bitmap = [System.Drawing.Bitmap]::new($width, $height)
  $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
  $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, [System.Drawing.Size]::new($width, $height))
  Ensure-ParentDirectory -PathValue $PathValue
  $bitmap.Save($PathValue, [System.Drawing.Imaging.ImageFormat]::Png)
  $graphics.Dispose()
  $bitmap.Dispose()
  return @{
    left = $rect.Left
    top = $rect.Top
    right = $rect.Right
    bottom = $rect.Bottom
    width = $width
    height = $height
  }
}

function Crop-Bitmap {
  param(
    [string]$SourcePath,
    [string]$TargetPath,
    [int]$X,
    [int]$Y,
    [int]$Width,
    [int]$Height
  )

  $source = [System.Drawing.Bitmap]::new($SourcePath)
  $safeX = [Math]::Max(0, [Math]::Min($source.Width - 1, $X))
  $safeY = [Math]::Max(0, [Math]::Min($source.Height - 1, $Y))
  $safeWidth = [Math]::Max(1, [Math]::Min($Width, $source.Width - $safeX))
  $safeHeight = [Math]::Max(1, [Math]::Min($Height, $source.Height - $safeY))
  $rect = [System.Drawing.Rectangle]::new($safeX, $safeY, $safeWidth, $safeHeight)
  $cropped = $source.Clone($rect, $source.PixelFormat)
  Ensure-ParentDirectory -PathValue $TargetPath
  $cropped.Save($TargetPath, [System.Drawing.Imaging.ImageFormat]::Png)
  $cropped.Dispose()
  $source.Dispose()
  return @{
    x = $safeX
    y = $safeY
    width = $safeWidth
    height = $safeHeight
  }
}

Ensure-ParentDirectory -PathValue $ScreenshotPath
Ensure-ParentDirectory -PathValue $FontCheckPath
Ensure-ParentDirectory -PathValue $JsonPath

$chromePath = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$args = @(
  "--new-window",
  "--no-first-run",
  "--no-default-browser-check",
  "--remote-debugging-address=127.0.0.1",
  "--remote-debugging-port=$Port",
  "--remote-allow-origins=*",
  "--user-data-dir=$ProfileDir",
  "--window-position=$WindowLeft,$WindowTop",
  "--window-size=$WindowWidth,$WindowHeight",
  $TargetUrl
)

$process = Start-Process -FilePath $chromePath -ArgumentList $args -PassThru

try {
  $version = Wait-HttpJson -Uri "http://127.0.0.1:$Port/json/version" -TimeoutSeconds $TimeoutSec
  $targets = Wait-HttpJson -Uri "http://127.0.0.1:$Port/json/list" -TimeoutSeconds $TimeoutSec
  $pageTarget = $targets | Where-Object { $_.type -eq "page" -and $_.url -like "http://localhost:8501*" } | Select-Object -First 1
  if ($null -eq $pageTarget) {
    $pageTarget = $targets | Where-Object { $_.type -eq "page" } | Select-Object -First 1
  }
  if ($null -eq $pageTarget) {
    throw "No page target found for Chrome"
  }

  $socket = [System.Net.WebSockets.ClientWebSocket]::new()
  $socket.ConnectAsync([Uri]$pageTarget.webSocketDebuggerUrl, [Threading.CancellationToken]::None).GetAwaiter().GetResult()
  $buffer = New-Object byte[] 65536
  $events = [System.Collections.Generic.List[object]]::new()
  $nextId = 1

  [void](Send-CdpCommand -Socket $socket -Id $nextId -Method "Page.enable" -Params @{} -Events $events -Buffer $buffer)
  $nextId++
  [void](Send-CdpCommand -Socket $socket -Id $nextId -Method "Runtime.enable" -Params @{} -Events $events -Buffer $buffer)
  $nextId++
  [void](Send-CdpCommand -Socket $socket -Id $nextId -Method "Network.enable" -Params @{} -Events $events -Buffer $buffer)
  $nextId++

  $readyExpression = @'
(() => {
  const h1 = document.querySelector("h1")?.textContent ?? "";
  const body = document.body?.innerText ?? "";
  return JSON.stringify({
    ready: h1.includes("ロト予測") && body.includes("DB接続"),
    h1,
    hasJapanese: body.includes("ロト予測") || body.includes("DB接続")
  });
})()
'@

  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  $readyPayload = $null
  while ((Get-Date) -lt $deadline) {
    $readyResult = Invoke-RuntimeEval -Socket $socket -Id $nextId -Expression $readyExpression -Events $events -Buffer $buffer
    $nextId++
    $readyPayload = (Get-CdpValue -Message $readyResult) | ConvertFrom-Json
    if ($readyPayload.ready) {
      break
    }
    Start-Sleep -Milliseconds 1000
  }
  if (-not $readyPayload.ready) {
    throw "Dashboard did not reach ready state in non-headless Chrome"
  }

  $evalExpression = @'
(() => {
  const h1El = document.querySelector("h1");
  const sidebarTitleEl = document.querySelector(".ops-sidebar-title");
  const hostLabelEl = Array.from(document.querySelectorAll("label, div, p, span")).find((el) => el.textContent?.trim() === "ホスト");
  const h1Rect = h1El ? h1El.getBoundingClientRect() : null;
  const sidebarRect = sidebarTitleEl ? sidebarTitleEl.getBoundingClientRect() : null;
  const hostRect = hostLabelEl ? hostLabelEl.getBoundingClientRect() : null;
  const app = document.querySelector(".stApp");
  return {
    title: document.title,
    htmlLang: document.documentElement.lang || null,
    metaDescription: document.head.querySelector('meta[name="description"]')?.content ?? null,
    h1: h1El?.textContent ?? null,
    sidebarTitle: sidebarTitleEl?.textContent ?? null,
    hostLabel: hostLabelEl?.textContent?.trim() ?? null,
    bodyTextHasJapanese: (document.body?.innerText ?? "").includes("ロト予測"),
    bodyTextHasDbLabel: (document.body?.innerText ?? "").includes("DB接続"),
    bodyTextSnippet: (document.body?.innerText ?? "").slice(0, 240),
    appFontFamily: app ? getComputedStyle(app).fontFamily : null,
    ua: navigator.userAgent,
    platform: navigator.platform,
    webdriver: navigator.webdriver,
    screenX: window.screenX,
    screenY: window.screenY,
    outerWidth: window.outerWidth,
    outerHeight: window.outerHeight,
    innerWidth: window.innerWidth,
    innerHeight: window.innerHeight,
    devicePixelRatio: window.devicePixelRatio,
    h1Rect,
    sidebarRect,
    hostRect,
    notoCheck: document.fonts ? document.fonts.check('16px "OpsNotoSansJP"', "ロト予測") : null,
    notoFallbackCheck: document.fonts ? document.fonts.check('16px "Noto Sans JP"', "ロト予測") : null
  };
})()
'@

  $evalResult = Invoke-RuntimeEval -Socket $socket -Id $nextId -Expression $evalExpression -Events $events -Buffer $buffer
  $nextId++
  $evaluation = (Get-CdpValue -Message $evalResult) | ConvertFrom-Json

  $windowHandle = Wait-ChromeWindow -Process $process -TimeoutSeconds $TimeoutSec
  [void][Win32]::ShowWindowAsync($windowHandle, 9)
  [void][Win32]::SetForegroundWindow($windowHandle)
  Start-Sleep -Milliseconds 1500

  $windowRect = Capture-WindowBitmap -Hwnd $windowHandle -PathValue $ScreenshotPath

  $cropX = 20
  $cropY = 120
  $cropWidth = [Math]::Min([int]$windowRect.width - 40, 1100)
  $cropHeight = [Math]::Min([int]$windowRect.height - 140, 500)

  if ($null -ne $evaluation.h1Rect -and $null -ne $evaluation.sidebarRect) {
    $browserTop = [Math]::Max(0, [int]$evaluation.outerHeight - [int]$evaluation.innerHeight)
    $left = [Math]::Max(0, [Math]::Min([int]$evaluation.sidebarRect.left, [int]$evaluation.h1Rect.left) - 24)
    $top = [Math]::Max(0, [Math]::Min([int]$evaluation.sidebarRect.top, [int]$evaluation.h1Rect.top) + $browserTop - 24)
    $right = [Math]::Min([int]$windowRect.width, [Math]::Max([int]$evaluation.h1Rect.right, [int]$evaluation.sidebarRect.right) + 40)
    $bottom = [Math]::Min([int]$windowRect.height, [Math]::Max([int]$evaluation.h1Rect.bottom, [int]$evaluation.sidebarRect.bottom) + $browserTop + 220)
    $cropX = [Math]::Max(0, $left)
    $cropY = [Math]::Max(0, $top)
    $cropWidth = [Math]::Max(1, $right - $cropX)
    $cropHeight = [Math]::Max(1, $bottom - $cropY)
  }

  $cropRect = Crop-Bitmap -SourcePath $ScreenshotPath -TargetPath $FontCheckPath -X $cropX -Y $cropY -Width $cropWidth -Height $cropHeight

  $responseEvents = @($events | Where-Object { $_.method -eq "Network.responseReceived" } | ForEach-Object {
    @{
      url = $_.params.response.url
      status = $_.params.response.status
      mimeType = $_.params.response.mimeType
    }
  })

  $payload = [ordered]@{
    generatedAt = (Get-Date).ToString("o")
    targetUrl = $TargetUrl
    chromeBinary = $chromePath
    launchArgs = $args
    process = @{
      id = $process.Id
      mainWindowHandle = [int64]$windowHandle
    }
    browserVersion = $version
    selectedTarget = $pageTarget
    evaluation = $evaluation
    windowRect = $windowRect
    cropRect = $cropRect
    screenshots = @{
      full = $ScreenshotPath
      fontCheck = $FontCheckPath
    }
    responseEvents = $responseEvents
  }

  $payload | ConvertTo-Json -Depth 50 | Set-Content -Encoding UTF8 -Path $JsonPath

  if ($socket.State -eq [System.Net.WebSockets.WebSocketState]::Open) {
    $socket.CloseAsync(
      [System.Net.WebSockets.WebSocketCloseStatus]::NormalClosure,
      "done",
      [Threading.CancellationToken]::None
    ).GetAwaiter().GetResult()
  }
  $socket.Dispose()
} finally {
  if ($null -ne $process -and -not $process.HasExited) {
    Start-Sleep -Milliseconds 1000
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
  }
}

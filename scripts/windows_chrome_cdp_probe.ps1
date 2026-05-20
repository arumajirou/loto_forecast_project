param(
  [Parameter(Mandatory = $true)]
  [int]$Port,

  [Parameter(Mandatory = $true)]
  [string]$TargetUrl,

  [Parameter(Mandatory = $true)]
  [string]$ScreenshotPath,

  [Parameter(Mandatory = $true)]
  [string]$JsonPath,

  [int]$ViewportWidth = 1280,
  [int]$ViewportHeight = 800,
  [bool]$Mobile = $false,
  [int]$DeviceScaleFactor = 1,
  [int]$WaitAfterNavigateMs = 8000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Receive-CdpMessage {
  param(
    [System.Net.WebSockets.ClientWebSocket]$Socket,
    [byte[]]$Buffer
  )

  $stream = New-Object System.IO.MemoryStream
  while ($true) {
    $segment = New-Object System.ArraySegment[byte] -ArgumentList @(,$Buffer)
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
  } | ConvertTo-Json -Depth 20 -Compress
  $bytes = [Text.Encoding]::UTF8.GetBytes($payload)
  $segment = New-Object System.ArraySegment[byte] -ArgumentList @(,$bytes)
  $Socket.SendAsync(
    $segment,
    [System.Net.WebSockets.WebSocketMessageType]::Text,
    $true,
    [Threading.CancellationToken]::None
  ).GetAwaiter().GetResult()

  while ($true) {
    $raw = Receive-CdpMessage -Socket $Socket -Buffer $Buffer
    $msg = $raw | ConvertFrom-Json -Depth 50
    if ($null -ne $msg.id -and [int]$msg.id -eq $Id) {
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

$version = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/json/version"
$targets = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/json/list"
$pageTarget = $targets | Where-Object { $_.type -eq "page" } | Select-Object -First 1
if ($null -eq $pageTarget) {
  throw "No page target found on port $Port"
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
[void](Send-CdpCommand -Socket $socket -Id $nextId -Method "Log.enable" -Params @{} -Events $events -Buffer $buffer)
$nextId++
[void](Send-CdpCommand -Socket $socket -Id $nextId -Method "Emulation.setDeviceMetricsOverride" -Params @{
  width = $ViewportWidth
  height = $ViewportHeight
  deviceScaleFactor = $DeviceScaleFactor
  mobile = $Mobile
} -Events $events -Buffer $buffer)
$nextId++

$nav = Send-CdpCommand -Socket $socket -Id $nextId -Method "Page.navigate" -Params @{ url = $TargetUrl } -Events $events -Buffer $buffer
$nextId++
Start-Sleep -Milliseconds $WaitAfterNavigateMs

$evalExpression = @'
(() => {
  const app = document.querySelector(".stApp");
  const metaDescription = document.head.querySelector('meta[name="description"]')?.content ?? null;
  const metaCharset = document.head.querySelector('meta[charset]')?.getAttribute("charset") ?? null;
  const htmlLang = document.documentElement.lang || null;
  const bodyText = document.body?.innerText ?? "";
  const h1 = document.querySelector("h1")?.textContent ?? null;
  const bodyStyle = document.body ? getComputedStyle(document.body).fontFamily : null;
  const appStyle = app ? getComputedStyle(app).fontFamily : null;
  const sample = "ロト予測 運用ダッシュボード";
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  ctx.font = "32px " + (appStyle || bodyStyle || "sans-serif");
  const sampleWidth = ctx.measureText(sample).width;
  ctx.font = "32px monospace";
  const monoWidth = ctx.measureText(sample).width;
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
    notoCheck: document.fonts ? document.fonts.check('16px "Noto Sans JP"', sample) : null,
    meiryoCheck: document.fonts ? document.fonts.check('16px "Meiryo"', sample) : null,
    yuGothicCheck: document.fonts ? document.fonts.check('16px "Yu Gothic"', sample) : null,
    bizCheck: document.fonts ? document.fonts.check('16px "BIZ UDGothic"', sample) : null
  };
})()
'@

$evalResult = Invoke-RuntimeEval -Socket $socket -Id $nextId -Expression $evalExpression -Events $events -Buffer $buffer
$nextId++
$shot = Send-CdpCommand -Socket $socket -Id $nextId -Method "Page.captureScreenshot" -Params @{
  format = "png"
  captureBeyondViewport = $true
} -Events $events -Buffer $buffer

$shotBytes = [Convert]::FromBase64String($shot.result.data)
$screenshotDir = Split-Path -Parent $ScreenshotPath
$jsonDir = Split-Path -Parent $JsonPath
if ($screenshotDir) {
  New-Item -ItemType Directory -Force -Path $screenshotDir | Out-Null
}
if ($jsonDir) {
  New-Item -ItemType Directory -Force -Path $jsonDir | Out-Null
}
[System.IO.File]::WriteAllBytes($ScreenshotPath, $shotBytes)

$responseEvents = @($events | Where-Object { $_.method -eq "Network.responseReceived" } | ForEach-Object {
  @{
    url = $_.params.response.url
    status = $_.params.response.status
    mimeType = $_.params.response.mimeType
  }
})
$failedRequests = @($events | Where-Object { $_.method -eq "Network.loadingFailed" } | ForEach-Object {
  @{
    requestId = $_.params.requestId
    errorText = $_.params.errorText
    canceled = $_.params.canceled
  }
})

$payload = [ordered]@{
  port = $Port
  targetUrl = $TargetUrl
  screenshotPath = $ScreenshotPath
  jsonPath = $JsonPath
  browserVersion = $version
  selectedTarget = $pageTarget
  navigate = $nav.result
  evaluation = $evalResult.result.result.value | ConvertFrom-Json -Depth 20
  responseEvents = $responseEvents
  failedRequests = $failedRequests
  eventCount = $events.Count
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

$payload | ConvertTo-Json -Depth 20

$out = "\\wsl.localhost\Ubuntu-24.04\mnt\e\env\ts\codex\loto_forecast_project\artifacts\logs\windows_detached_write_test.txt"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $out) | Out-Null
"windows detached powershell ok" | Set-Content -Encoding UTF8 -Path $out

@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Temp\windows_non_headless_final_probe.ps1" -Port 9226 -TargetUrl http://localhost:8501 -ScreenshotPath "C:\Temp\16_windows_non_headless_final.png" -FontCheckPath "C:\Temp\17_windows_non_headless_font_check.png" -JsonPath "C:\Temp\windows_runtime_final.json" > "C:\Temp\windows_non_headless_final_probe.log" 2>&1
endlocal

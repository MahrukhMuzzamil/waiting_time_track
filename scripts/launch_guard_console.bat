@echo off
setlocal

set "SCRIPT_ROOT=%~dp0"
set "PROJECT_ROOT=%SCRIPT_ROOT%.."

cd /d "%PROJECT_ROOT%"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\start_guard_console.ps1" %*

endlocal



@echo off
echo [COD-AI] Starting data collection...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "Set-Location -LiteralPath '\\wsl.localhost\Ubuntu-24.04\home\user\code\Game projects\COD projects\COD-AI'; & 'C:\Users\t0mwh\AppData\Local\ZombieOps\venv\Scripts\python.exe' collect.py"
echo.
echo [COD-AI] Collection finished. Press any key to close.
pause >nul

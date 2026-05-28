@echo off
echo [COD-AI] Starting ghost (dry-run)...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "Set-Location -LiteralPath '\\wsl.localhost\Ubuntu-24.04\home\user\code\Game projects\COD projects\COD-AI'; & 'C:\Users\t0mwh\AppData\Local\ZombieOps\venv\Scripts\python.exe' ghost.py --dry-run"
echo.
echo [COD-AI] Ghost stopped. Press any key to close.
pause >nul

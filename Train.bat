@echo off
echo [COD-AI] Starting training...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "Set-Location -LiteralPath '\\wsl.localhost\Ubuntu-24.04\home\user\code\Game projects\COD projects\COD-AI'; & 'C:\Users\t0mwh\AppData\Local\ZombieOps\venv\Scripts\python.exe' train.py"
echo.
echo [COD-AI] Training finished. Press any key to close.
pause >nul

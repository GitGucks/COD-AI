@echo off
set "PYTHON=C:\Users\t0mwh\AppData\Local\ZombieOps\venv\Scripts\python.exe"
set "DIR=\\wsl.localhost\Ubuntu-24.04\home\user\code\Game projects\COD projects\COD-AI"

echo [COD-AI] Starting ghost (dry-run)...
echo         Add --live to send real controller input.
cd /d "%DIR%"
"%PYTHON%" ghost.py --dry-run %*

echo.
echo [COD-AI] Ghost stopped. Press any key to close.
pause >nul

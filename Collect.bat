@echo off
set "PYTHON=C:\Users\t0mwh\AppData\Local\ZombieOps\venv\Scripts\python.exe"
set "DIR=\\wsl.localhost\Ubuntu-24.04\home\user\code\Game projects\COD projects\COD-AI"

echo [COD-AI] Starting data collection...
cd /d "%DIR%"
"%PYTHON%" collect.py %*

echo.
echo [COD-AI] Collection finished. Press any key to close.
pause >nul

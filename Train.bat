@echo off
set "PYTHON=C:\Users\t0mwh\AppData\Local\ZombieOps\venv\Scripts\python.exe"
set "DIR=\\wsl.localhost\Ubuntu-24.04\home\user\code\Game projects\COD projects\COD-AI"

echo [COD-AI] Starting training...
cd /d "%DIR%"
"%PYTHON%" train.py %*

echo.
echo [COD-AI] Training finished. Press any key to close.
pause >nul

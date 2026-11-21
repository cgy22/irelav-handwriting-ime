@echo off
chcp 65001 >nul
cd /d "%~dp0"
start "" pythonw.exe "app\desktop.py"
exit /b 0

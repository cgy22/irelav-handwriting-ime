@echo off
chcp 65001 >nul
cd /d F:\mnist
echo 正在启动手写输入法服务...
echo 浏览器访问: http://127.0.0.1:8765
python app\server.py
pause

@echo off
chcp 65001 >nul
cd /d F:\mnist
REM 生产入口：启动 HTTP 服务 + 原生面板 + 焦点监听 + 全局热键
REM 必须用 pythonw（无控制台窗口），且不能直接打开网页 —— 网页模式没有系统钩子
if not exist "dist\TonhTzeIME\native\ime_shell.exe" (
  echo [!] 还没有构建原生外壳，先运行: python tools\build_shell_cs.py
  pause
  exit /b 1
)
start "" pythonw.exe "app\production.py"
exit /b 0

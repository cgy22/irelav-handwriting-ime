# -*- coding: utf-8 -*-
"""生产外壳：把"网页版手写板"变成桌面输入法。

启动顺序
--------
1. 起本地 HTTP 服务（复用 app/server.py，端口 8765）
2. 用 Edge 的 app 模式开一个 Chromium 窗口（无地址栏、无标签页）
3. 去掉系统标题栏 + 加圆角 + 置顶  -> 看起来是一个现代浮层
4. 注册全局热键 Ctrl+Alt+H（显示/隐藏面板）
5. 监听前台窗口：焦点进入可编辑控件时自动弹出手写板
6. 把识别结果 / 方向键注入到"上次获得焦点"的那个窗口

关键取舍
--------
* 界面 100% 由 Chromium 渲染（HTML/CSS），不出现任何原生控件；
  原生 API 只负责窗口外观和输入注入，用户看不到。
* 注入用 SendInput：先把焦点还给目标窗口，发完再把面板拉回前台。
  这是浏览器内输入框唯一可靠的方式（原生 IME 需要 TSF，成本高得多）。
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import platform_win as pw  # noqa: E402
import server as srv  # noqa: E402

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
PANEL_W, PANEL_H = 640, 470
PORT = int(os.environ.get("IME_PORT", "8765"))
RUNTIME = os.path.join(ROOT, ".runtime")

state = {"panel": 0, "target": 0, "proc": None, "visible": True, "auto_show": True}
_log_lock = threading.Lock()


def log(msg: str) -> None:
    os.makedirs(RUNTIME, exist_ok=True)
    with _log_lock:
        with open(os.path.join(RUNTIME, "ime.log"), "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


# ---------------------------------------------------------------- 窗口
def _edge_path() -> str | None:
    for p in EDGE_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _wait_window(timeout: float = 20.0) -> int:
    """等 Chromium 窗口出现：按标题匹配，避免抓到用户已有的 Edge 窗口。"""
    end = time.time() + timeout
    while time.time() < end:
        hwnd = _find_panel()
        if hwnd:
            return hwnd
        time.sleep(0.25)
    return 0


def _find_panel() -> int:
    """按标题找我们的面板窗口（避免抓到用户已有的 Edge 窗口）。"""
    found: list[int] = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

    def cb(hwnd, _):
        if not pw.user32.IsWindowVisible(hwnd):
            return True
        if pw.window_title(hwnd).startswith("TonhTze"):
            found.append(int(hwnd))
            return False
        return True

    pw.user32.EnumWindows(WNDENUMPROC(cb), 0)
    return found[0] if found else 0


def _place_panel(hwnd: int) -> None:
    """按 DIP 尺寸放到屏幕右下角。

    尺寸必须乘 DPI 缩放：窗口是物理像素，而网页的 CSS 像素是 DIP。
    200% 缩放下如果只给 640 物理像素，网页只会拿到 320 CSS 像素，界面会被放大一倍。
    """
    try:
        scale = pw.window_dpi(hwnd) / 96.0
        w = int(round(PANEL_W * scale))
        h = int(round(PANEL_H * scale))
        rect = wt.RECT()
        pw.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)  # SPI_GETWORKAREA
        x = max(0, rect.right - w - 24)
        y = max(0, rect.bottom - h - 24)
        pw.move_window(hwnd, x, y, w, h)
        log(f"面板尺寸 {w}x{h} (DPI {pw.window_dpi(hwnd)}) @ ({x},{y})")
    except Exception:
        pass


def launch_panel() -> None:
    edge = _edge_path()
    url = f"http://127.0.0.1:{PORT}"

    # 优先用自带的原生外壳（WinForms + WebView2 + Mica）
    native = os.path.join(ROOT, "dist", "TonhTzeIME", "native", "ime_shell.exe")
    if os.path.exists(native):
        state["proc"] = subprocess.Popen([native, url])
        log(f"已启动原生外壳 pid={state['proc'].pid}")
        hwnd = _wait_window()
        if hwnd:
            state["panel"] = hwnd
            pw.register_own_pid(pw._pid_of(hwnd))
            pw.strip_frame(hwnd, topmost=True)
            pw.set_window_icon(hwnd, os.path.join(HERE, "static", "icon.ico"))
            _place_panel(hwnd)
            pw.hide_window(hwnd)
            state["visible"] = False
            log(f"原生面板 hwnd={hwnd} 已就绪（Mica + 圆角 + 置顶），托盘常驻、按焦点弹出")
            return
        log("原生外壳未出现窗口，退回 Edge")

    if edge:
        profile = os.path.join(RUNTIME, "edge-profile")
        os.makedirs(profile, exist_ok=True)
        args = [
            edge,
            f"--app={url}",
            f"--window-size={PANEL_W},{PANEL_H}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-features=Translate,msEdgeTranslate",
            "--disable-session-crashed-bubble",
            "--hide-crash-restore-bubble",
        ]
        state["proc"] = subprocess.Popen(args)
        log(f"已启动 Edge app 模式 pid={state['proc'].pid}")
    else:
        import webbrowser

        webbrowser.open(url)
        log("未找到 Edge，退回默认浏览器（无边框样式不可用）")

    hwnd = _wait_window()
    if hwnd:
        state["panel"] = hwnd
        pw.register_own_pid(pw._pid_of(hwnd))
        pw.strip_frame(hwnd, topmost=True)
        pw.set_window_icon(hwnd, os.path.join(HERE, "static", "icon.ico"))
        _place_panel(hwnd)
        log(f"面板窗口 hwnd={hwnd} 已美化（无边框+圆角+置顶）")

        def _reapply():
            # Chromium 启动过程中会替换窗口句柄，这里每次都重新定位一遍，
            # 确保美化和定位作用在"最终那个"窗口上。
            for _ in range(5):
                time.sleep(1.0)
                try:
                    h = _find_panel() or state["panel"]
                    if not h:
                        continue
                    state["panel"] = h
                    pw.register_own_pid(pw._pid_of(h))
                    pw.strip_frame(h, topmost=True)
                    pw.set_window_icon(h, os.path.join(HERE, "static", "icon.ico"))
                    _place_panel(h)
                except Exception:
                    pass

        threading.Thread(target=_reapply, daemon=True).start()
    else:
        log("未等到面板窗口，样式化跳过")


# ---------------------------------------------------------------- 注入
def _restore_panel() -> None:
    if state["panel"]:
        pw.set_foreground(state["panel"])


def _restore_later(delay: float = 0.35) -> None:
    """延迟把焦点还给面板。

    SendInput 只是把按键放进目标程序的输入队列，立刻抢焦点会让后半截字符
    跑到面板网页里（实测：13 个字符只有前 5 个落进记事本）。
    """
    def _f():
        time.sleep(delay)
        try:
            _restore_panel()
        except Exception:
            pass

    threading.Thread(target=_f, daemon=True).start()


def inject_text(text: str, enter: bool = False):
    target = state["target"]
    if not target or pw.is_our_window(target):
        target = pw.get_foreground()
    if not target or pw.is_our_window(target):
        return False, "请先把光标放进要输入的输入框"
    prev = pw.get_foreground()
    pw.set_foreground(target)
    # 等焦点真的切过去再粘贴，否则第一次 Ctrl+V 会落进面板自己
    for _ in range(10):
        if pw.get_foreground() == target:
            break
        time.sleep(0.03)
    time.sleep(0.06)
    ok = pw.paste_text(text)
    if not ok:
        ok = pw.type_text(text)   # 剪贴板不可用时退回逐字符
    if enter:
        time.sleep(0.02)
        pw.press_key("enter")
    log(f"inject_text len={len(text)} target={target} prev={prev} ok={ok}")
    if prev != target:
        _restore_later()
    return ok, "" if ok else "SendInput 被目标程序拒绝"


def inject_key(key: str):
    target = state["target"] or pw.get_foreground()
    if not target or pw.is_our_window(target):
        return False, "请先点一下目标输入框"
    prev = pw.get_foreground()
    pw.set_foreground(target)
    time.sleep(0.04)
    ok = pw.press_key(key)
    log(f"inject_key {key} target={target} ok={ok}")
    if prev != target:
        _restore_later(0.25)
    return ok, "" if ok else f"未知按键: {key}"


def do_drag() -> None:
    pw.start_drag(state["panel"])


def do_window(payload) -> bool:
    """窗口控制：hide / show / close / move(dx,dy) / topmost。"""
    action = str(payload.get("action") or "")
    hwnd = state["panel"]
    if action == "hide":
        pw.hide_window(hwnd)
        state["visible"] = False
    elif action == "show":
        pw.show_window(hwnd, focus=False)
        state["visible"] = True
    elif action == "close":
        log("用户关闭面板，退出")
        os._exit(0)
    elif action == "move":
        # 拖动：面板是 NOACTIVATE 的，不能靠系统标题栏拖动，这里按位移手动移动
        dx = int(payload.get("dx") or 0)
        dy = int(payload.get("dy") or 0)
        if hwnd and (dx or dy):
            rect = wt.RECT()
            pw.user32.GetWindowRect(hwnd, ctypes.byref(rect))
            pw.move_window(hwnd, rect.left + dx, rect.top + dy)
        return True
    elif action == "topmost":
        pw.set_foreground(hwnd)
    return True


# ---------------------------------------------------------------- 焦点
def on_focus(hwnd: int, editable: bool) -> None:
    """输入法的生命周期：焦点进输入框就弹，离开就收。"""
    if not hwnd or pw.is_our_window(hwnd):
        return
    if editable:
        state["target"] = hwnd
        if state["auto_show"] and not state["visible"] and state["panel"]:
            pw.show_window(state["panel"], focus=False)
            state["visible"] = True
            log(f"焦点进入输入框 {hwnd}，弹出手写板")
    else:
        # 焦点跑到非输入区域：收起面板（托盘里可以再唤出）
        if state["visible"] and state["panel"] and not state.get("pinned"):
            pw.hide_window(state["panel"])
            state["visible"] = False
            log("焦点离开输入框，收起手写板")


def _watch_panel_alive() -> None:
    """外壳被关掉/杀进程时，Python 侧也要退出，别留一个看不见的服务。"""
    while True:
        time.sleep(2.0)
        hwnd = state.get("panel")
        if hwnd and not pw.user32.IsWindow(hwnd):
            log("面板窗口已消失，退出")
            os._exit(0)


def toggle_panel() -> None:
    if not state["panel"]:
        launch_panel()
        state["visible"] = True
        return
    if state["visible"]:
        pw.hide_window(state["panel"])
        state["visible"] = False
    else:
        pw.show_window(state["panel"])
        state["visible"] = True


# ---------------------------------------------------------------- 主流程
def _kill_stale_listener() -> None:
    """端口被上一份实例占着时，先把它清理掉，避免出现"页面连的是旧服务、
    功能钩子却是空的"这种鬼状态（症状就是"窗口控制不可用（网页模式）"）。"""
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True
        ).stdout
        pids = set()
        for line in out.splitlines():
            if f":{PORT} " in line and "LISTENING" in line:
                parts = line.split()
                if parts and parts[-1].isdigit():
                    pids.add(parts[-1])
        for pid in pids:
            r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                               capture_output=True, text=True).stdout
            if "python" in r.lower() or "ime_shell" in r.lower():
                log(f"清理占用端口 {PORT} 的旧进程 pid={pid}")
                subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
    except Exception:
        pass


def main() -> None:
    pw.set_process_dpi_aware()
    os.makedirs(RUNTIME, exist_ok=True)
    log("===== TonhTze 手写输入法 启动 =====")

    srv.HOOKS["key"] = inject_key
    srv.HOOKS["send"] = inject_text
    srv.HOOKS["drag"] = do_drag
    srv.HOOKS["window"] = do_window

    _kill_stale_listener()
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), srv.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    log(f"HTTP 服务已监听 127.0.0.1:{PORT}")

    pw.register_own_pid(os.getpid())
    launch_panel()

    pw.start_hotkeys(
        [(1, pw.MOD_CONTROL | pw.MOD_ALT, 0x48),      # Ctrl+Alt+H 显示/隐藏
         (2, pw.MOD_CONTROL | pw.MOD_ALT, 0x51)],     # Ctrl+Alt+Q 退出
        {1: toggle_panel, 2: lambda: os._exit(0)},
    )
    pw.FocusWatcher(on_focus).start()
    threading.Thread(target=_watch_panel_alive, daemon=True).start()
    log("热键 Ctrl+Alt+H 显示/隐藏，Ctrl+Alt+Q 退出")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

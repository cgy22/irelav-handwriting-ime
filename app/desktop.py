# -*- coding: utf-8 -*-
"""生产外壳：把"网页版手写板"变成桌面输入法。

启动顺序
--------
1. 起本地 HTTP 服务（复用 app/server.py，端口 8765）
2. 启动自带的原生外壳（C# + WebView2），窗口无边框、圆角、Mica、不进任务栏
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

import win32_helper as pw  # noqa: E402
import server as srv  # noqa: E402

# 面板逻辑尺寸（DIP）与窗口标题；标题必须精确匹配，否则会命中同名的资源管理器窗口
PANEL_W, PANEL_H = 720, 560
PANEL_TITLE = "TonhTze AI · Handwriting IME"
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


def _wait_window(timeout: float = 25.0) -> int:
    """等 Chromium 窗口出现：按标题匹配，避免抓到用户已有的 Edge 窗口。"""
    end = time.time() + timeout
    while time.time() < end:
        hwnd = _find_panel()
        if hwnd:
            return hwnd
        time.sleep(0.25)
    return 0


def _find_panel() -> int:
    """精确找到我们的面板窗口。

    不能用"标题以 TonhTze 开头"这种模糊匹配 —— 用户打开的
    `TonhTzeIME` 文件夹（资源管理器窗口）会被误判成面板，被改样式、被隐藏，
    表现得就像"启动程序把文件资源管理器弄坏了"。
    """
    found: list[int] = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

    def cb(hwnd, _):
        if not pw.user32.IsWindowVisible(hwnd):
            return True
        if pw.window_title(hwnd) != PANEL_TITLE:
            return True
        cls = pw.window_class(hwnd)
        if not (cls.startswith("WindowsForms10") or cls == "Chrome_WidgetWin_1"):
            return True
        found.append(int(hwnd))
        return False

    pw.user32.EnumWindows(WNDENUMPROC(cb), 0)
    return found[0] if found else 0


GEOM_PATH = None  # 在 main() 里指向 .runtime/window.json


def _screen_work_area():
    rect = wt.RECT()
    pw.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)  # SPI_GETWORKAREA
    return rect


def _load_geom():
    """读取上次的窗口几何（物理像素）；越界或损坏则忽略。"""
    try:
        if GEOM_PATH and os.path.exists(GEOM_PATH):
            import json as _json
            with open(GEOM_PATH, "r", encoding="utf-8") as f:
                g = _json.load(f)
            wa = _screen_work_area()
            w, h = int(g["w"]), int(g["h"])
            x, y = int(g["x"]), int(g["y"])
            if w < 320 or h < 240:
                return None
            # 至少有一部分落在屏幕里
            if x + w < 80 or y + h < 80 or x > wa.right - 80 or y > wa.bottom - 80:
                return None
            return x, y, w, h
    except Exception:
        pass
    return None


def _save_geom() -> None:
    try:
        hwnd = state.get("panel")
        if not hwnd or not GEOM_PATH:
            return
        r = wt.RECT()
        pw.user32.GetWindowRect(hwnd, ctypes.byref(r))
        import json as _json
        os.makedirs(os.path.dirname(GEOM_PATH), exist_ok=True)
        with open(GEOM_PATH, "w", encoding="utf-8") as f:
            _json.dump({"x": r.left, "y": r.top,
                        "w": r.right - r.left, "h": r.bottom - r.top}, f)
    except Exception:
        pass


def _place_panel(hwnd: int) -> None:
    """把面板放到上次的位置；没有记录就按默认尺寸停到右下角。

    尺寸用 DIP 定义、按 DPI 换算成物理像素，否则 200% 缩放下网页只拿到一半宽度。
    """
    try:
        saved = _load_geom()
        if saved:
            x, y, w, h = saved
            pw.move_window(hwnd, x, y, w, h)
            log(f"面板恢复上次位置 ({x},{y}) {w}x{h}")
            return
        scale = pw.window_dpi(hwnd) / 96.0
        w = int(round(PANEL_W * scale))
        h = int(round(PANEL_H * scale))
        wa = _screen_work_area()
        x = max(0, wa.right - w - 24)
        y = max(0, wa.bottom - h - 24)
        pw.move_window(hwnd, x, y, w, h)
        log(f"面板默认尺寸 {w}x{h} (DPI {pw.window_dpi(hwnd)}) @ ({x},{y})")
    except Exception:
        pass


EDGE_PATHS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def _edge() -> str | None:
    for p in EDGE_PATHS:
        if os.path.exists(p):
            return p
    return None


def launch_panel() -> None:
    """启动面板外壳：优先用自绘的 WinForms+WebView2 外壳（无边框、无标题栏），
    失败或不存在时才回落到 Chromium app 窗口。"""
    url = f"http://127.0.0.1:{PORT}"

    native = None
    for c in (os.path.join(ROOT, "native", "ime_shell.exe"),
              os.path.join(ROOT, "dist", "TonhTzeIME", "native", "ime_shell.exe")):
        if os.path.exists(c):
            native = c
            break

    if native and os.environ.get("IME_SHELL", "native") != "edge":
        state["proc"] = subprocess.Popen([native, url + "?shell=1"])
        log(f"已启动原生外壳 pid={state['proc'].pid}")
        hwnd = _wait_window(timeout=20.0)
        if hwnd:
            state["panel"] = hwnd
            pw.register_own_pid(pw._pid_of(hwnd))
            _style_panel(hwnd)
            state["visible"] = True
            log(f"面板 hwnd={hwnd} 已就绪（原生外壳，常驻置顶）")
            return
        log("[!] 原生外壳未出现窗口，回落到 Chromium 外壳")
        try:
            state["proc"].kill()
        except Exception:
            pass

    _launch_chromium(url)


def _launch_chromium(url: str) -> None:
    """回落方案：Chromium app 窗口（自带标题栏，但渲染稳定）。"""
    edge = _edge()
    if not edge:
        log("[x] 既没有原生外壳也找不到 Edge")
        raise SystemExit(2)
    profile = os.path.join(RUNTIME, "edge-profile")
    os.makedirs(profile, exist_ok=True)
    state["proc"] = subprocess.Popen([
        edge, f"--app={url}?shell=1", f"--window-size={PANEL_W},{PANEL_H}",
        f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
    ])
    log(f"已启动 Chromium 外壳 pid={state['proc'].pid}")
    hwnd = _wait_window(timeout=25.0)
    if not hwnd:
        log("[x] 界面窗口未出现")
        os._exit(3)
    state["panel"] = hwnd
    pw.register_own_pid(pw._pid_of(hwnd))
    _style_panel(hwnd)
    state["visible"] = True
    log(f"面板 hwnd={hwnd} 已就绪（Chromium 外壳，常驻置顶）")


def _apply_frame_styles(hwnd: int) -> None:
    """只改窗口样式（可重复调用）。"""
    style = pw.user32.GetWindowLongW(hwnd, -16)
    style &= ~0x00C00000          # 去掉 WS_CAPTION（标题栏）
    style |= 0x00040000           # 保留可缩放边缘
    pw.user32.SetWindowLongW(hwnd, -16, style)

    ex = pw.user32.GetWindowLongW(hwnd, -20)
    ex |= 0x00000080              # WS_EX_TOOLWINDOW：不进任务栏/Alt-Tab
    ex |= 0x08000000              # WS_EX_NOACTIVATE
    ex &= ~0x00040000             # 去掉 WS_EX_APPWINDOW
    pw.user32.SetWindowLongW(hwnd, -20, ex)

    pw.user32.SetWindowPos(hwnd, ctypes.c_void_p(-1), 0, 0, 0, 0,
                           0x0001 | 0x0002 | 0x0010 | 0x0020 | 0x0040)


def _style_panel(hwnd: int) -> None:
    """无边框 + 置顶 + 不进任务栏 + 不抢焦点。"""
    try:
        style = pw.user32.GetWindowLongW(hwnd, -16)
        style &= ~0x00C00000                     # WS_CAPTION
        style |= 0x00040000                      # WS_THICKFRAME（保留边缘缩放）
        pw.user32.SetWindowLongW(hwnd, -16, style)

        ex = pw.user32.GetWindowLongW(hwnd, -20)
        ex |= 0x00000080                         # WS_EX_TOOLWINDOW
        ex |= 0x08000000                         # WS_EX_NOACTIVATE（点击不抢焦点）
        ex &= ~0x00040000                        # 去掉 WS_EX_APPWINDOW
        pw.user32.SetWindowLongW(hwnd, -20, ex)

        pw.user32.SetWindowPos(hwnd, ctypes.c_void_p(-1), 0, 0, 0, 0,
                               0x0001 | 0x0002 | 0x0010 | 0x0020 | 0x0040)
        pref = ctypes.c_int(2)                   # DWMWCP_ROUND
        try:
            dwm = ctypes.WinDLL("dwmapi")
            dwm.DwmSetWindowAttribute(ctypes.c_void_p(hwnd), 33,
                                      ctypes.byref(pref), ctypes.sizeof(pref))
        except Exception:
            pass
        _place_panel(hwnd)
        pw.show_window(hwnd, focus=False)

        # Edge 在启动后还会重新套一遍自己的窗口边框，所以要重试
        def _retry():
            for delay in (0.8, 1.6, 3.0):
                time.sleep(delay)
                try:
                    _apply_frame_styles(hwnd)
                except Exception:
                    pass
        threading.Thread(target=_retry, daemon=True).start()
    except Exception as e:
        log(f"样式化失败: {e}")


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
    clip = pw.set_clipboard_text(text)
    ok = clip and pw.paste_text(text)
    if not ok:
        ok = pw.type_text(text)   # 剪贴板不可用时退回逐字符
    log(f"  clipboard={clip} paste={ok} fg_after={pw.get_foreground()} target={target}")
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


VK_LBUTTON = 0x01


def _drag_loop(hwnd: int) -> None:
    """跟着鼠标移动窗口。

    用"按下时的光标位置"和"按下时的窗口位置"做绝对位移，不做累加：
    每一帧都是 win_start + (cursor - cursor_start)，不会像逐帧 delta 那样漂移。
    """
    try:
        pw.user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        pw.user32.GetCursorPos.argtypes = [ctypes.POINTER(wt.POINT)]
        start = wt.POINT()
        pw.user32.GetCursorPos(ctypes.byref(start))
        rect = wt.RECT()
        pw.user32.GetWindowRect(hwnd, ctypes.byref(rect))
        while pw.user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000:
            cur = wt.POINT()
            pw.user32.GetCursorPos(ctypes.byref(cur))
            nx = rect.left + (cur.x - start.x)
            ny = rect.top + (cur.y - start.y)
            pw.move_window(hwnd, nx, ny)
            time.sleep(0.008)
        _save_geom()
        # 拖动结束把焦点还给目标输入框
        target = state.get("target")
        if target and not pw.is_our_window(target):
            pw.set_foreground(target)
    except Exception as e:
        log(f"拖动异常: {e}")


def do_drag() -> None:
    if state.get("panel"):
        threading.Thread(target=_drag_loop, args=(state["panel"],), daemon=True).start()


def do_window(payload) -> bool:
    """窗口控制：hide / show / close / move(dx,dy) / topmost。"""
    action = str(payload.get("action") or "")
    hwnd = state["panel"]
    if action == "hide":
        pw.hide_window(hwnd)
        state["visible"] = False
    elif action == "show":
        _place_panel(hwnd)
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
    elif action == "refocus":
        # 把焦点还给最后那个输入框：无论是拖动还是点面板，都不该让目标丢焦点
        target = state.get("target")
        if target and not pw.is_our_window(target) and pw.user32.IsWindow(target):
            if pw.get_foreground() != target:
                pw.set_foreground(target)
                log(f"焦点还给 {target} ({pw.window_class(target)})")
    elif action == "topmost":
        pw.set_foreground(hwnd)
    return True


# ---------------------------------------------------------------- 焦点
def on_focus(hwnd: int, editable: bool) -> None:
    """只记录"最后获得焦点的可编辑窗口"，不再自动显示/隐藏面板。

    自动显隐在这台机器上会和"不抢焦点"打架，改成常驻置顶后逻辑更简单可靠：
    用户点哪，就把哪当输入目标；面板自己永远在，不会因为焦点变化而消失。
    """
    if not hwnd or pw.is_our_window(hwnd):
        return
    if editable:
        if state.get("target") != hwnd:
            state["target"] = hwnd
            log(f"输入目标 -> {hwnd} ({pw.window_class(hwnd)})")


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
    global GEOM_PATH
    os.makedirs(RUNTIME, exist_ok=True)
    GEOM_PATH = os.path.join(RUNTIME, "window.json")
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

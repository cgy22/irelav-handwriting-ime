# -*- coding: utf-8 -*-
"""Windows 系统集成层（全部隐藏，不出现任何原生控件）。

设计原则
--------
* **界面永远是 Chromium 渲染的 HTML/CSS**，这里只做三件"看不见"的事：
  1. 把窗口变成无边框 + 圆角 + 置顶的浮层（DWM/Win32 一次性样式）
  2. 监听"焦点进入可编辑控件"→ 通知上层弹出手写板
  3. 把识别结果/按键注入到目标窗口（SendInput）
* 所有调用都包了 try/except：任何一步失败，程序退化成"普通网页版"，不会崩。

为什么不用 WinUI 3 / WPF
-------------------------
本机只有 .NET Core 3.1 运行时，没有 .NET SDK（`dotnet --list-sdks` 为空），
WinUI 3 还需要 Windows App SDK。为了一个输入法面板引入几百 MB 的工具链不划算，
而且每次打包都要重新编译。用 Chromium 渲染 + 隐藏的 Win32 集成，外观一样现代，
但改样式只改 CSS、打包只是复制文件。
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import threading
import time
from typing import Callable, Optional

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
try:
    dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
except OSError:  # 极老的系统
    dwmapi = None

# ---------------------------------------------------------------- 常量
GWL_STYLE = -16
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000
WS_SYSMENU = 0x00080000
WS_MINIMIZEBOX = 0x00020000
WS_MAXIMIZEBOX = 0x00010000
HWND_TOPMOST = ctypes.c_void_p(-1)
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_SHOWWINDOW = 0x0040
SWP_NOACTIVATE = 0x0010
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWCP_ROUND = 2

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
EVENT_SYSTEM_FOREGROUND = 0x0003
WINEVENT_OUTOFCONTEXT = 0x0000
WINEVENT_SKIPOWNPROCESS = 0x0002

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN = 0x0001, 0x0002, 0x0004, 0x0008

VK = {
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "home": 0x24, "end": 0x23, "backspace": 0x08, "enter": 0x0D,
    "tab": 0x09, "esc": 0x1B, "delete": 0x2E, "space": 0x20,
    "pageup": 0x21, "pagedown": 0x22,
}

# 这些窗口类名可能包含可编辑控件：浏览器/编辑器/输入框宿主
_EDITABLE_HINTS = (
    "edit", "richedit", "chrome_renderwidgethost", "mozilla", "mozillawindowclass",
    "windows.ui.core.corewindow", "scintilla", "textbox", "webview", "notepad",
    "sunawtframe", "internet explorer_server", "wordpadclass",
)


def set_process_dpi_aware() -> None:
    """让本进程按物理像素工作。不声明的话，窗口坐标/尺寸都会被系统虚拟化，
    导致"设 1280 得到 640"这类诡异现象。"""
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


def window_dpi(hwnd: int) -> int:
    try:
        user32.GetDpiForWindow.argtypes = [ctypes.c_void_p]
        user32.GetDpiForWindow.restype = ctypes.c_uint
        d = int(user32.GetDpiForWindow(hwnd))
        return d if d > 0 else 96
    except Exception:
        return 96


# ---------------------------------------------------------------- 函数签名
# 64 位下如果不定 argtypes，Python int 会按 32 位传递，HWND_TOPMOST(-1)
# 会被零扩展成 0xFFFFFFFF 而不是全 1，SetWindowPos 静默失败。
user32.SetWindowPos.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_uint,
]
user32.SetWindowPos.restype = ctypes.c_bool
user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
user32.GetWindowLongW.restype = ctypes.c_long
user32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
user32.SetWindowLongW.restype = ctypes.c_long
user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
user32.GetForegroundWindow.restype = ctypes.c_void_p
user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
user32.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(wt.RECT)]
user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
user32.ReleaseCapture.restype = ctypes.c_bool


# ---------------------------------------------------------------- 结构体
class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class _INPUTUNION(ctypes.Union):
    # 必须是 32 字节：KEYBDINPUT 是 24，MOUSEINPUT 是 32，Windows 以 40 字节的 INPUT 为准。
    # 之前补成 64 导致 sizeof(INPUT)=72，SendInput 直接返回 0。
    _fields_ = [("ki", _KEYBDINPUT), ("_pad", ctypes.c_byte * 32)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("u", _INPUTUNION)]

user32.SendInput.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_int]
user32.SendInput.restype = ctypes.c_uint


# ---------------------------------------------------------------- 窗口工具
def get_foreground() -> int:
    return int(user32.GetForegroundWindow() or 0)


def window_class(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def window_title(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buf, 512)
    return buf.value


def is_editable(hwnd: int) -> bool:
    """粗略判断焦点窗口是否可能是输入框。

    浏览器把输入框画在渲染进程里，拿不到真正的 Edit 控件，
    所以这里把浏览器渲染窗口也当作"可能可编辑"——这正是手写板应该弹出来的场景。
    """
    if not hwnd:
        return False
    if is_our_window(hwnd):
        return False
    cls = window_class(hwnd).lower()
    if any(h in cls for h in _EDITABLE_HINTS):
        return True
    # UWP / WinUI 的输入框宿主
    if cls.startswith("windows.ui.core") or cls.startswith("applicationframe"):
        return True
    return False


_our_pids: set[int] = set()


def register_own_pid(pid: int) -> None:
    _our_pids.add(int(pid))


def _pid_of(hwnd: int) -> int:
    pid = wt.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def is_our_window(hwnd: int) -> bool:
    return _pid_of(hwnd) in _our_pids


def find_main_window(pid: int) -> int:
    """找出属于某个进程的第一个可见顶层窗口。"""
    found: list[int] = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

    def cb(hwnd, _):
        if _pid_of(hwnd) == pid and user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                found.append(int(hwnd))
                return False
        return True

    user32.EnumWindows(WNDENUMPROC(cb), 0)
    return found[0] if found else 0


def strip_frame(hwnd: int, topmost: bool = True) -> bool:
    """去掉标题栏、加圆角、置顶 —— 让 Chromium 窗口看起来像一个现代浮层。"""
    if not hwnd:
        return False
    try:
        style = user32.GetWindowLongW(hwnd, GWL_STYLE)
        style &= ~(WS_CAPTION | WS_SYSMENU)
        style |= WS_THICKFRAME | WS_MINIMIZEBOX
        user32.SetWindowLongW(hwnd, GWL_STYLE, style)
        # 触发一次重排，样式才会生效
        user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | 0x0020 | 0x0002 | 0x0001 | 0x0004)
        if dwmapi:
            pref = ctypes.c_int(DWMWCP_ROUND)
            dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE,
                                         ctypes.byref(pref), ctypes.sizeof(pref))
        if topmost:
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW)
        return True
    except Exception:
        return False


def move_window(hwnd: int, x: int, y: int, w: Optional[int] = None, h: Optional[int] = None) -> None:
    """移动/缩放窗口，且**绝不激活**（SWP_NOACTIVATE）——
    输入法面板一旦被激活，目标输入框的光标就丢了。"""
    flags = (0 if (w and h) else SWP_NOSIZE) | SWP_NOACTIVATE
    user32.SetWindowPos(hwnd, HWND_TOPMOST, int(x), int(y), int(w or 0), int(h or 0),
                        flags | SWP_SHOWWINDOW)


def show_window(hwnd: int, focus: bool = False) -> None:
    """显示面板。默认不激活自己——输入法弹出时不能把目标输入框的光标抢走。"""
    if not hwnd:
        return
    SW_SHOWNOACTIVATE = 4
    SW_RESTORE = 9
    user32.ShowWindow(hwnd, SW_RESTORE if focus else SW_SHOWNOACTIVATE)
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW)
    if focus:
        set_foreground(hwnd)


def hide_window(hwnd: int) -> None:
    if hwnd:
        user32.ShowWindow(hwnd, 0)  # SW_HIDE


def set_foreground(hwnd: int) -> bool:
    if not hwnd:
        return False
    if get_foreground() == hwnd:
        return True
    try:
        fg = get_foreground()
        tid_fg = user32.GetWindowThreadProcessId(fg, None) if fg else 0
        tid_me = kernel32.GetCurrentThreadId()
        attached = bool(user32.AttachThreadInput(tid_me, tid_fg, True)) if tid_fg else False
        ok = bool(user32.SetForegroundWindow(hwnd))
        if attached:
            user32.AttachThreadInput(tid_me, tid_fg, False)
        return ok
    except Exception:
        return False


def start_drag(hwnd: int) -> None:
    """让 Windows 接管拖动（无边框窗口的标准做法）。"""
    if not hwnd:
        return
    try:
        user32.ReleaseCapture()
        user32.SendMessageW(hwnd, 0x00A1, 0x0002, 0)  # WM_NCLBUTTONDOWN, HTCAPTION
    except Exception:
        pass


def set_window_icon(hwnd: int, ico_path: str) -> bool:
    """给无边框窗口设置图标（任务栏、Alt-Tab 都会用到）。"""
    if not hwnd or not os.path.exists(ico_path):
        return False
    try:
        user32.LoadImageW.restype = ctypes.c_void_p
        IMAGE_ICON, LR_LOADFROMFILE, LR_DEFAULTSIZE = 1, 0x10, 0x40
        big = user32.LoadImageW(0, ico_path, IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE)
        small = user32.LoadImageW(0, ico_path, IMAGE_ICON, 16, 16, LR_LOADFROMFILE)
        if big:
            user32.SendMessageW(hwnd, 0x0080, 1, big)   # WM_SETICON, ICON_BIG
        if small:
            user32.SendMessageW(hwnd, 0x0080, 0, small)  # WM_SETICON, ICON_SMALL
        return bool(big)
    except Exception:
        return False


# ---------------------------------------------------------------- 输入注入
def _send_inputs(items) -> bool:
    n = len(items)
    if n == 0:
        return True
    arr = (INPUT * n)()
    for i, (vk, scan, flags) in enumerate(items):
        arr[i].type = INPUT_KEYBOARD
        arr[i].u.ki.wVk = vk
        arr[i].u.ki.wScan = scan
        arr[i].u.ki.dwFlags = flags
        arr[i].u.ki.time = 0
        arr[i].u.ki.dwExtraInfo = None
    sent = user32.SendInput(n, ctypes.byref(arr), ctypes.sizeof(INPUT))
    return sent == n


def type_text(text: str) -> bool:
    """用 Unicode 扫描码输入文本（不受输入法/键盘布局影响）。"""
    data = text.encode("utf-16-le")
    items = []
    for i in range(0, len(data), 2):
        sc = int.from_bytes(data[i:i + 2], "little")
        items.append((0, sc, KEYEVENTF_UNICODE))
        items.append((0, sc, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
    ok = True
    for k in range(0, len(items), 64):  # 分批，避免个别程序丢字
        ok = _send_inputs(items[k:k + 64]) and ok
        time.sleep(0.004)
    return ok


def set_clipboard_text(text: str) -> bool:
    """把文本放进剪贴板（CF_UNICODETEXT）。"""
    try:
        user32.OpenClipboard.argtypes = [ctypes.c_void_p]
        user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
        user32.SetClipboardData.restype = ctypes.c_void_p
        kernel32.GlobalAlloc.restype = ctypes.c_void_p
        kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]

        if not user32.OpenClipboard(None):
            return False
        try:
            user32.EmptyClipboard()
            data = text.encode("utf-16-le") + b"\x00\x00"
            GMEM_MOVEABLE = 0x0002
            handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
            if not handle:
                return False
            ptr = kernel32.GlobalLock(handle)
            if not ptr:
                return False
            ctypes.memmove(ptr, data, len(data))
            kernel32.GlobalUnlock(handle)
            return bool(user32.SetClipboardData(13, handle))  # CF_UNICODETEXT
        finally:
            user32.CloseClipboard()
    except Exception:
        return False


def paste_text(text: str) -> bool:
    """把文本"粘贴"到当前焦点窗口：剪贴板 + Ctrl+V。

    比逐字符 SendInput 稳得多——跨进程时 SendInput 的 unicode 事件会被
    目标程序按物理键解析（实测 "mads 123" 会变成 "33333333"）。
    """
    if not set_clipboard_text(text):
        return False
    time.sleep(0.05)
    VK_CONTROL, VK_V = 0x11, 0x56
    ok = _send_inputs([(VK_CONTROL, 0, 0), (VK_V, 0, 0), (VK_V, 0, KEYEVENTF_KEYUP),
                       (VK_CONTROL, 0, KEYEVENTF_KEYUP)])
    time.sleep(0.03)
    return ok


def press_key(name: str) -> bool:
    vk = VK.get(name.lower())
    if vk is None:
        return False
    return _send_inputs([(vk, 0, 0), (vk, 0, KEYEVENTF_KEYUP)])


# ---------------------------------------------------------------- 焦点监听
class FocusWatcher:
    """监听前台窗口切换，回调 (hwnd, is_editable)。所有异常内部消化。"""

    def __init__(self, on_change: Callable[[int, bool], None]):
        self._cb = on_change
        self._thread: Optional[threading.Thread] = None
        self._hook = None
        self._alive = False
        self._proc = None

    def start(self) -> bool:
        try:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self._alive = True
            return True
        except Exception:
            return False

    def _run(self) -> None:
        WINEVENTPROC = ctypes.WINFUNCTYPE(
            None, wt.HANDLE, wt.DWORD, wt.HWND, wt.LONG, wt.LONG, wt.DWORD, wt.DWORD
        )

        def _handler(_hook, _event, hwnd, _o1, _o2, _t, _ms):
            try:
                self._cb(int(hwnd or 0), is_editable(int(hwnd or 0)))
            except Exception:
                pass

        self._proc = WINEVENTPROC(_handler)
        self._hook = user32.SetWinEventHook(
            EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_FOREGROUND, 0,
            self._proc, 0, 0, WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS,
        )
        msg = wt.MSG()
        while user32.GetMessageW(ctypes.byref(msg), 0, 0, 0) != 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))


def start_hotkeys(specs, callbacks) -> None:
    """在独立线程里注册全局热键并跑消息循环。

    RegisterHotKey 是线程相关的：WM_HOTKEY 只会投递到注册它的那个线程的消息队列，
    所以注册和消息循环必须在同一个线程里。
    specs: [(id, mods, vk), ...]  callbacks: {id: callable}
    """
    def _thread():
        for hid, mods, vk in specs:
            register_hotkey(hid, mods, vk)
        msg = wt.MSG()
        while user32.GetMessageW(ctypes.byref(msg), 0, 0, 0) != 0:
            if msg.message == 0x0312:  # WM_HOTKEY
                cb = callbacks.get(int(msg.wParam))
                if cb:
                    try:
                        cb()
                    except Exception:
                        pass

    threading.Thread(target=_thread, daemon=True).start()


def register_hotkey(hid: int, mods: int, vk: int) -> bool:
    try:
        return bool(user32.RegisterHotKey(0, hid, mods, vk))
    except Exception:
        return False

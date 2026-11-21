// TonhTze AI 手写输入法 —— 原生外壳
//
// 这是一个最小的 Win32 宿主窗口：无标题栏、圆角、Windows 11 Mica 材质，
// 内容全部由 WebView2 渲染（HTML/CSS 在 app/static 下）。
// 不用 WinUI 3 的原因：本机没有 .NET SDK / Windows App SDK，而 WinUI 3 的 C++
// 工程必须带 XAML 编译器与 XamlTypeInfo 骨架。这里用 Win32 宿主 + DWM 材质，
// 视觉上同样是 Win11 的原生窗口（Mica + 圆角 + 无边框），但只需 WebView2 SDK。
//
// 内部逻辑（焦点监听、按键注入、HTTP 服务）都在 app/production.py 里，
// 这个 exe 只负责"把窗口画出来、把网页装进去"。

#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0A00
#include <windows.h>
#include <windowsx.h>
#include <dwmapi.h>
#include <shellapi.h>
#include <string>
#include <wrl/client.h>
#include <wrl/implements.h>
#include "WebView2.h"

#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "WebView2LoaderStatic.lib")

using Microsoft::WRL::ComPtr;

static const wchar_t* kClass = L"TonhTzeIMEShell";
static const wchar_t* kTitle = L"TonhTze AI · Handwriting IME";
static const int kWidth = 640;
static const int kHeight = 470;

static ComPtr<ICoreWebView2Controller> g_controller;
static ComPtr<ICoreWebView2> g_webview;

#ifndef DWMWA_SYSTEMBACKDROP_TYPE
#define DWMWA_SYSTEMBACKDROP_TYPE 38
#endif
#ifndef DWMWA_WINDOW_CORNER_PREFERENCE
#define DWMWA_WINDOW_CORNER_PREFERENCE 33
#endif
#define DWMSBT_MAINWINDOW 2
#define DWMWCP_ROUND 2

static void ApplyModernChrome(HWND hwnd) {
    // 圆角
    int corner = DWMWCP_ROUND;
    DwmSetWindowAttribute(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, &corner, sizeof(corner));

    // Mica（Win11 22H2+；老系统上调用会失败，忽略即可）
    int backdrop = DWMSBT_MAINWINDOW;
    DwmSetWindowAttribute(hwnd, DWMWA_SYSTEMBACKDROP_TYPE, &backdrop, sizeof(backdrop));

    // 深色标题栏跟随系统（无标题栏时也影响边框描边）
    BOOL dark = FALSE;
    DwmSetWindowAttribute(hwnd, 20 /* DWMWA_USE_IMMERSIVE_DARK_MODE */, &dark, sizeof(dark));

    // 去掉 DWM 边框线，避免无边框窗口外圈出现一条灰边
    MARGINS margins = {-1, -1, -1, -1};
    DwmExtendFrameIntoClientArea(hwnd, &margins);
}

static void PlaceBottomRight(HWND hwnd) {
    RECT work{};
    SystemParametersInfoW(SPI_GETWORKAREA, 0, &work, 0);
    int x = work.right - kWidth - 24;
    int y = work.bottom - kHeight - 24;
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    SetWindowPos(hwnd, HWND_TOPMOST, x, y, kWidth, kHeight, SWP_SHOWWINDOW);
}

static LRESULT HitTest(HWND hwnd, POINT pt) {
    RECT rc{};
    GetWindowRect(hwnd, &rc);
    const int border = 6;
    bool left = pt.x < rc.left + border;
    bool right = pt.x >= rc.right - border;
    bool top = pt.y < rc.top + border;
    bool bottom = pt.y >= rc.bottom - border;

    if (top && left) return HTTOPLEFT;
    if (top && right) return HTTOPRIGHT;
    if (bottom && left) return HTBOTTOMLEFT;
    if (bottom && right) return HTBOTTOMRIGHT;
    if (left) return HTLEFT;
    if (right) return HTRIGHT;
    if (top) return HTTOP;
    if (bottom) return HTBOTTOM;
    return HTCLIENT;  // 顶部拖动交给网页里的 app-region: drag
}

static bool InitWebView(HWND hwnd, const std::wstring& url, const std::wstring& userDataDir) {
    auto envHandler = Microsoft::WRL::Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
        [hwnd, url](HRESULT result, ICoreWebView2Environment* env) -> HRESULT {
            if (FAILED(result) || !env) return result;
            env->CreateCoreWebView2Controller(
                hwnd,
                Microsoft::WRL::Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                    [hwnd, url](HRESULT r2, ICoreWebView2Controller* controller) -> HRESULT {
                        if (FAILED(r2) || !controller) return r2;
                        g_controller = controller;
                        g_controller->get_CoreWebView2(&g_webview);

                        RECT rc{};
                        GetClientRect(hwnd, &rc);
                        g_controller->put_Bounds(rc);

                        // 透明背景：让 Mica 从窗口透到网页后面
                        // 这个属性在 ICoreWebView2Controller2 上，需要 QueryInterface
                        ComPtr<ICoreWebView2Controller2> c2;
                        if (SUCCEEDED(g_controller.As(&c2)) && c2) {
                            COREWEBVIEW2_COLOR transparent{0, 0, 0, 0};
                            c2->put_DefaultBackgroundColor(transparent);
                        }

                        ComPtr<ICoreWebView2Settings> settings;
                        if (SUCCEEDED(g_webview->get_Settings(&settings)) && settings) {
                            settings->put_AreDefaultContextMenusEnabled(FALSE);
                            settings->put_AreDevToolsEnabled(TRUE);
                            settings->put_IsStatusBarEnabled(FALSE);
                            settings->put_IsZoomControlEnabled(FALSE);
                        }

                        g_webview->Navigate(url.c_str());
                        return S_OK;
                    })
                    .Get());
            return S_OK;
        });

    return SUCCEEDED(CreateCoreWebView2EnvironmentWithOptions(
        nullptr, userDataDir.c_str(), nullptr, envHandler.Get()));
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_NCCALCSIZE:
        // 返回 0 即"客户区 == 整个窗口"，等于去掉系统标题栏与边框
        if (wp == TRUE) return 0;
        break;
    case WM_NCHITTEST: {
        POINT pt{GET_X_LPARAM(lp), GET_Y_LPARAM(lp)};
        return HitTest(hwnd, pt);
    }
    case WM_SIZE: {
        if (g_controller) {
            RECT rc{};
            GetClientRect(hwnd, &rc);
            g_controller->put_Bounds(rc);
        }
        return 0;
    }
    case WM_ERASEBKGND:
        return 1;  // 背景由 DWM 材质负责
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, LPWSTR, int) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = kClass;
    RegisterClassExW(&wc);

    std::wstring exeDir = L".";
    {
        wchar_t path[MAX_PATH]{};
        GetModuleFileNameW(nullptr, path, MAX_PATH);
        std::wstring p(path);
        size_t pos = p.find_last_of(L"\\/");
        if (pos != std::wstring::npos) exeDir = p.substr(0, pos);
    }

    HWND hwnd = CreateWindowExW(
        WS_EX_TOPMOST | WS_EX_APPWINDOW,
        kClass, kTitle,
        WS_POPUP | WS_THICKFRAME | WS_MINIMIZEBOX | WS_CLIPCHILDREN,
        0, 0, kWidth, kHeight,
        nullptr, nullptr, hInst, nullptr);
    if (!hwnd) return 1;

    // 图标：<exe目录>\..\app\static\icon.ico
    std::wstring ico = exeDir + L"\\..\\app\\static\\icon.ico";
    HICON icon = (HICON)LoadImageW(nullptr, ico.c_str(), IMAGE_ICON, 0, 0,
                                   LR_LOADFROMFILE | LR_DEFAULTSIZE);
    if (icon) {
        SendMessageW(hwnd, WM_SETICON, ICON_BIG, (LPARAM)icon);
        SendMessageW(hwnd, WM_SETICON, ICON_SMALL, (LPARAM)icon);
    }

    ApplyModernChrome(hwnd);
    PlaceBottomRight(hwnd);

    // URL 由命令行传入；数据目录放 exe 旁边，便于整包迁移
    std::wstring url = L"http://127.0.0.1:8765";
    {
        int argc = 0;
        LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
        if (argv) {
            if (argc > 1) url = argv[1];
            LocalFree(argv);
        }
    }
    std::wstring dataDir = exeDir + L"\\..\\.runtime\\webview2";
    InitWebView(hwnd, url, dataDir);

    MSG msg{};
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    CoUninitialize();
    return 0;
}

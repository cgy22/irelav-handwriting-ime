// TonhTze AI 手写输入法 —— 原生外壳（C# / WinForms + WebView2）
//
// 为什么是 WinForms 而不是 WinUI 3：
//   本机没有 .NET SDK（`dotnet --list-sdks` 为空），装 SDK 要几百 MB；
//   而 Windows 自带 .NET Framework 4.8 的 csc.exe，配合 WebView2 的 net462 程序集
//   就能直接编译出一个原生窗口。窗口外观由 DWM 负责（Mica + 圆角 + 无边框），
//   和 WinUI 3 的窗口材质是同一套 API，肉眼没有区别。
//
// 这个 exe 只做一件事：把窗口画出来、把网页装进去。
// 焦点监听、按键注入、HTTP 服务都在 app/production.py。

using System;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

namespace TonhTzeIME
{
    internal static class Native
    {
        [DllImport("dwmapi.dll")]
        public static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int value, int size);

        public const int DWMWA_WINDOW_CORNER_PREFERENCE = 33;
        public const int DWMWA_SYSTEMBACKDROP_TYPE = 38;
        public const int DWMWCP_ROUND = 2;
        public const int DWMSBT_MAINWINDOW = 2;

        [DllImport("user32.dll")]
        public static extern bool SetProcessDPIAware();

        [DllImport("user32.dll")]
        public static extern bool SetWindowPos(IntPtr hwnd, IntPtr after, int x, int y,
                                               int cx, int cy, uint flags);

        public static readonly IntPtr HWND_TOPMOST = new IntPtr(-1);
        public const uint SWP_SHOWWINDOW = 0x0040;

        [DllImport("user32.dll")]
        public static extern bool ReleaseCapture();

        [DllImport("user32.dll")]
        public static extern IntPtr SendMessage(IntPtr hwnd, int msg, int wparam, int lparam);

        public const int WM_NCLBUTTONDOWN = 0xA1;
        public const int HTCAPTION = 2;
    }

    internal sealed class ShellForm : Form
    {
        private readonly WebView2 _web;
        private readonly string _url;
        private readonly string _userDataDir;

        public ShellForm(string url, string userDataDir)
        {
            _url = url;
            _userDataDir = userDataDir;

            Text = "TonhTze AI · Handwriting IME";
            FormBorderStyle = FormBorderStyle.None;
            StartPosition = FormStartPosition.Manual;
            TopMost = true;
            BackColor = Color.FromArgb(243, 243, 243);
            DoubleBuffered = true;
            // 尺寸按 DIP 定义，OnLoad 里再按 DeviceDpi 换算成物理像素，
            // 否则 200% 缩放下网页只拿到一半的 CSS 宽度（文字会被放大得离谱）
            AutoScaleMode = AutoScaleMode.None;
            Size = new Size(640, 470);

            var wa = Screen.PrimaryScreen.WorkingArea;
            Location = new Point(wa.Right - Width - 24, wa.Bottom - Height - 24);

            _web = new WebView2 { Dock = DockStyle.Fill };
            Controls.Add(_web);

            Load += OnLoad;
            Shown += OnShown;
        }

        private void OnLoad(object sender, EventArgs e)
        {
            ApplyMica();
            ApplyDpiSize();
            try
            {
                string ico = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
                                          "..", "app", "static", "icon.ico");
                if (File.Exists(ico)) Icon = new Icon(ico);
            }
            catch { }
            InitWebView();
        }

        private void OnShown(object sender, EventArgs e)
        {
            ApplyMica();
            ApplyDpiSize();
        }

        // 面板逻辑尺寸（DIP）：宽度 640、高度 470
        private const int DipW = 640;
        private const int DipH = 470;

        private void ApplyDpiSize()
        {
            double scale = DeviceDpi / 96.0;
            int w = (int)Math.Round(DipW * scale);
            int h = (int)Math.Round(DipH * scale);
            var wa = Screen.PrimaryScreen.WorkingArea;
            Native.SetWindowPos(Handle, Native.HWND_TOPMOST,
                                wa.Right - w - 24, wa.Bottom - h - 24,
                                w, h, Native.SWP_SHOWWINDOW);
        }

        private void ApplyMica()
        {
            int corner = Native.DWMWCP_ROUND;
            Native.DwmSetWindowAttribute(Handle, Native.DWMWA_WINDOW_CORNER_PREFERENCE,
                                         ref corner, sizeof(int));
            int backdrop = Native.DWMSBT_MAINWINDOW;
            Native.DwmSetWindowAttribute(Handle, Native.DWMWA_SYSTEMBACKDROP_TYPE,
                                         ref backdrop, sizeof(int));
        }

        private async void InitWebView()
        {
            try
            {
                var env = await CoreWebView2Environment.CreateAsync(null, _userDataDir);
                await _web.EnsureCoreWebView2Async(env);

                var core = _web.CoreWebView2;
                core.Settings.AreDefaultContextMenusEnabled = false;
                core.Settings.AreDevToolsEnabled = true;
                core.Settings.IsStatusBarEnabled = false;
                core.Settings.IsZoomControlEnabled = false;
                // 透明背景：让 Mica 从窗口透到网页后面
                _web.DefaultBackgroundColor = Color.Transparent;

                core.Navigate(_url);
            }
            catch (Exception ex)
            {
                MessageBox.Show("WebView2 初始化失败：" + ex.Message,
                                "TonhTze 手写输入法",
                                MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        // 关键：把它做成"输入法面板"而不是普通窗口
        //   WS_EX_TOOLWINDOW  —— 不进任务栏、不进 Alt-Tab
        //   WS_EX_NOACTIVATE  —— 点击/书写时不夺取焦点，目标输入框的光标不会掉
        protected override CreateParams CreateParams
        {
            get
            {
                const int WS_EX_TOOLWINDOW = 0x00000080;
                const int WS_EX_NOACTIVATE = 0x08000000;
                CreateParams cp = base.CreateParams;
                cp.ExStyle |= WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE;
                cp.ExStyle &= ~0x00040000;   // 去掉 WS_EX_APPWINDOW
                return cp;
            }
        }

        // 鼠标点上来也不要激活自己
        private const int WM_MOUSEACTIVATE = 0x0021;
        private const int MA_NOACTIVATE = 3;

        protected override void WndProc(ref Message m)
        {
            if (m.Msg == WM_MOUSEACTIVATE)
            {
                m.Result = (IntPtr)MA_NOACTIVATE;
                return;
            }
            base.WndProc(ref m);
        }
    }

    internal static class Program
    {
        [STAThread]
        private static void Main(string[] args)
        {
            Native.SetProcessDPIAware();
            string url = args.Length > 0 ? args[0] : "http://127.0.0.1:8765";
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string userData = Path.Combine(baseDir, "..", ".runtime", "webview2");

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new ShellForm(url, userData));
        }
    }
}

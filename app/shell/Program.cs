// TonhTze AI 手写输入法 —— 原生外壳（C# / WinForms + WebView2）
//
// 定位：这是"输入法面板"，不是普通程序。
//   * WS_EX_TOOLWINDOW  不进任务栏、不进 Alt-Tab
//   * WS_EX_NOACTIVATE  点击/书写时不夺取焦点，目标输入框的光标不会掉
//   * 启动不显示界面，只留托盘图标；焦点进入可编辑控件时由 Python 侧唤出
//   * 托盘菜单：显示 / 隐藏 / 退出；退出时通知 Python 一起结束
//   * Python 服务消失（被任务管理器杀掉等）时，外壳自己退出，不留孤儿窗口
//
// 界面由 WebView2 渲染（app/static），窗口材质交给 DWM（Mica + 圆角）。

using System;
using System.Drawing;
using System.IO;
using System.Net;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
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
        [DllImport("user32.dll")]
        public static extern bool ShowWindow(IntPtr hwnd, int cmd);

        public static readonly IntPtr HWND_TOPMOST = new IntPtr(-1);
        public const uint SWP_SHOWWINDOW = 0x0040;
        public const uint SWP_NOACTIVATE = 0x0010;
        public const int SW_HIDE = 0;
        public const int SW_SHOWNOACTIVATE = 4;
    }


    /// <summary>扁平化的托盘菜单渲染器（默认的 ToolStripProfessionalRenderer 是 Win7 风格）。</summary>
    internal sealed class ModernMenuRenderer : ToolStripProfessionalRenderer
    {
        private static readonly Color Accent = Color.FromArgb(22, 218, 73);
        private static readonly Color Hover = Color.FromArgb(240, 253, 243);

        public ModernMenuRenderer() : base(new ModernColorTable()) { RoundedEdges = false; }

        protected override void OnRenderMenuItemBackground(ToolStripItemRenderEventArgs e)
        {
            var g = e.Graphics;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            var r = new Rectangle(4, 1, e.Item.Width - 8, e.Item.Height - 2);
            if (e.Item.Selected)
            {
                using (var b = new SolidBrush(Hover)) g.FillRectangle(b, r);
                using (var p = new Pen(Accent, 3))
                    g.DrawLine(p, r.Left, r.Top + 3, r.Left, r.Bottom - 3);
            }
        }

        protected override void OnRenderItemText(ToolStripItemTextRenderEventArgs e)
        {
            e.TextColor = e.Item.ForeColor;
            base.OnRenderItemText(e);
        }

        protected override void OnRenderToolStripBorder(ToolStripRenderEventArgs e) { }

        private sealed class ModernColorTable : ProfessionalColorTable
        {
            public override Color ToolStripDropDownBackground => Color.White;
            public override Color ImageMarginGradientBegin => Color.White;
            public override Color ImageMarginGradientMiddle => Color.White;
            public override Color ImageMarginGradientEnd => Color.White;
            public override Color MenuBorder => Color.FromArgb(229, 229, 229);
            public override Color MenuItemSelected => Color.FromArgb(240, 253, 243);
            public override Color SeparatorDark => Color.FromArgb(238, 238, 238);
            public override Color SeparatorLight => Color.White;
        }
    }

    internal sealed class ShellForm : Form
    {
        private readonly WebView2 _web;
        private readonly string _url;
        private readonly string _userDataDir;
        private readonly Timer _healthTimer = new Timer();
        private NotifyIcon _tray;
        private int _misses;

        private const int DipW = 640;
        private const int DipH = 470;

        public ShellForm(string url, string userDataDir)
        {
            _url = url;
            _userDataDir = userDataDir;

            Text = "TonhTze AI · Handwriting IME";
            FormBorderStyle = FormBorderStyle.None;
            StartPosition = FormStartPosition.Manual;
            TopMost = true;
            ShowInTaskbar = false;
            BackColor = Color.FromArgb(243, 243, 243);
            DoubleBuffered = true;
            AutoScaleMode = AutoScaleMode.None;
            Size = new Size(DipW, DipH);

            _web = new WebView2 { Dock = DockStyle.Fill };
            Controls.Add(_web);

            Load += OnLoad;
            BuildTray();
        }


        private static ToolStripMenuItem MenuItem(string text, EventHandler onClick)
        {
            return new ToolStripMenuItem(text, null, onClick)
            {
                Padding = new Padding(12, 5, 14, 5),
                ForeColor = Color.FromArgb(50, 49, 48),
            };
        }

        /// <summary>把 WinForms 的方形菜单切成圆角（Win11 的菜单是 8px 圆角）。</summary>
        private static void RoundMenu(ContextMenuStrip menu)
        {
            try
            {
                var h = menu.Handle;
                var r = new Rectangle(0, 0, menu.Width, menu.Height);
                int d = 12;
                using (var path = new System.Drawing.Drawing2D.GraphicsPath())
                {
                    path.AddArc(r.X, r.Y, d, d, 180, 90);
                    path.AddArc(r.Right - d, r.Y, d, d, 270, 90);
                    path.AddArc(r.Right - d, r.Bottom - d, d, d, 0, 90);
                    path.AddArc(r.X, r.Bottom - d, d, d, 90, 90);
                    path.CloseFigure();
                    using (var rgn = new Region(path))
                        menu.Region = rgn;
                }
            }
            catch { }
        }

        private void BuildTray()
        {
            _tray = new NotifyIcon
            {
                Text = "TonhTze 手写输入法（点输入框自动弹出）",
                Visible = true,
            };
            try
            {
                // 优先用 exe 自带的图标（csc /win32icon 嵌入），这样托盘图标一定是我们的
                Icon exeIcon = Icon.ExtractAssociatedIcon(Application.ExecutablePath);
                if (exeIcon != null) { _tray.Icon = exeIcon; }
                else
                {
                    string ico = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
                                              "..", "app", "static", "icon.ico");
                    _tray.Icon = File.Exists(ico) ? new Icon(ico) : SystemIcons.Application;
                }
            }
            catch { _tray.Icon = SystemIcons.Application; }

            var menu = new ContextMenuStrip
            {
                Renderer = new ModernMenuRenderer(),
                ShowImageMargin = false,
                BackColor = Color.White,
                Font = new Font("Segoe UI", 9.5f),
                Padding = new Padding(4, 6, 4, 6),
            };
            menu.Items.Add(MenuItem("显示面板", (s2, e2) => ShowPanel()));
            menu.Items.Add(MenuItem("隐藏面板", (s2, e2) => HidePanel()));
            menu.Items.Add(new ToolStripSeparator());
            var quit = MenuItem("退出", (s2, e2) => QuitApp());
            quit.ForeColor = Color.FromArgb(209, 52, 56);
            menu.Items.Add(quit);
            menu.Opened += (s2, e2) => RoundMenu(menu);
            _tray.ContextMenuStrip = menu;
            _tray.DoubleClick += (s, e) => ShowPanel();
        }

        public void ShowPanel()
        {
            ApplyDpiSize();
            Native.ShowWindow(Handle, Native.SW_SHOWNOACTIVATE);
        }

        public void HidePanel()
        {
            Native.ShowWindow(Handle, Native.SW_HIDE);
        }

        private void PostJson(string path, string json)
        {
            try
            {
                var req = (HttpWebRequest)WebRequest.Create(_url + path);
                req.Method = "POST";
                req.ContentType = "application/json";
                req.Timeout = 1500;
                byte[] data = System.Text.Encoding.UTF8.GetBytes(json);
                req.ContentLength = data.Length;
                using (var st = req.GetRequestStream()) st.Write(data, 0, data.Length);
                using (var resp = req.GetResponse()) { }
            }
            catch { }
        }

        private void QuitApp()
        {
            PostJson("/api/window", "{\"action\":\"close\"}");
            _tray.Visible = false;
            Application.Exit();
        }

        private void OnLoad(object sender, EventArgs e)
        {
            ApplyMica();
            ApplyDpiSize();
            InitWebView();
            HidePanel();          // 启动不显示界面

            _healthTimer.Interval = 3000;
            _healthTimer.Tick += async (s2, e2) => await CheckServerAsync();
            _healthTimer.Start();
        }

        private Task CheckServerAsync()
        {
            return Task.Run(() =>
            {
                bool ok = false;
                try
                {
                    var req = (HttpWebRequest)WebRequest.Create(_url + "/api/state");
                    req.Timeout = 2000;
                    using (var resp = (HttpWebResponse)req.GetResponse())
                        ok = (int)resp.StatusCode == 200;
                }
                catch { ok = false; }

                if (ok) { _misses = 0; return; }
                if (++_misses >= 3)
                {
                    if (_tray != null) _tray.Visible = false;
                    Application.Exit();
                }
            });
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

        private void ApplyDpiSize()
        {
            double scale = DeviceDpi / 96.0;
            int w = (int)Math.Round(DipW * scale);
            int h = (int)Math.Round(DipH * scale);
            var wa = Screen.PrimaryScreen.WorkingArea;
            Native.SetWindowPos(Handle, Native.HWND_TOPMOST,
                                wa.Right - w - 24, wa.Bottom - h - 24,
                                w, h, Native.SWP_SHOWWINDOW | Native.SWP_NOACTIVATE);
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

        protected override CreateParams CreateParams
        {
            get
            {
                const int WS_EX_TOOLWINDOW = 0x00000080;
                const int WS_EX_NOACTIVATE = 0x08000000;
                CreateParams cp = base.CreateParams;
                cp.ExStyle |= WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE;
                cp.ExStyle &= ~0x00040000;
                return cp;
            }
        }

        private const int WM_MOUSEACTIVATE = 0x0021;
        private const int MA_NOACTIVATE = 3;
        private const int WM_EXITSIZEMOVE = 0x0232;

        protected override void WndProc(ref Message m)
        {
            if (m.Msg == WM_MOUSEACTIVATE)
            {
                m.Result = (IntPtr)MA_NOACTIVATE;
                return;
            }
            if (m.Msg == WM_EXITSIZEMOVE)
            {
                // 原生拖动会把窗口激活；等激活状态落定 250ms，再把焦点还给目标输入框
                var t = new Timer { Interval = 250 };
                t.Tick += (s2, e2) =>
                {
                    t.Stop();
                    t.Dispose();
                    PostJson("/api/window", "{\"action\":\"refocus\"}");
                };
                t.Start();
            }
            base.WndProc(ref m);
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            _tray.Visible = false;
            base.OnFormClosing(e);
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

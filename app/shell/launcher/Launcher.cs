// TonhTze AI 手写输入法 —— 单文件启动器
//
// 作用：双击这一个 exe 就能启动全部组件（Python 服务 + 面板外壳），
// 不再需要 start_ime.cmd，也不会闪黑框。
//
// 它做四件事：
//   1. 找到 pythonw.exe（PATH、配置里的路径、常见安装位置）
//   2. 用无窗口方式启动 app\desktop.py（工作目录 = 本 exe 所在目录）
//   3. 如果已经有一份在跑，就只把面板叫出来，不重复启动
//   4. 出错时弹一个说人话的对话框，而不是静默失败
//
// 注意：这仍是"启动器单文件"。真正的免安装单文件（把 Python + PyTorch 一起打包）
// 会是几百 MB，且启动更慢；需要的话另说。

using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Windows.Forms;

static class Program
{
    private const int Port = 8765;

    [STAThread]
    static int Main(string[] args)
    {
        string root = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\');
        string entry = Path.Combine(root, "app", "desktop.py");

        if (!File.Exists(entry))
        {
            Fail("找不到 app\\desktop.py。\n\n请把本程序放在 TonhTzeIME 目录下（与 app、native 同级）。");
            return 1;
        }

        // 已经在运行：只把面板叫出来
        if (ServerAlive())
        {
            try
            {
                Post("/api/window", "{\"action\":\"show\"}");
                return 0;
            }
            catch
            {
                return 0;
            }
        }

        string py = FindPython(root);
        if (py == null)
        {
            Fail("找不到 Python 运行环境。\n\n" +
                 "请任选其一：\n" +
                 "  · 把 Python 加入 PATH\n" +
                 "  · 在本程序同目录放一个 python.txt，里面写 pythonw.exe 的完整路径");
            return 2;
        }

        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = py,
                Arguments = "\"" + entry + "\"",
                WorkingDirectory = root,
                UseShellExecute = false,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden,
            };
            psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
            Process.Start(psi);
        }
        catch (Exception ex)
        {
            Fail("启动失败：" + ex.Message);
            return 3;
        }

        // 等端口起来，最多 25 秒；成功后让启动器自己退出
        for (int i = 0; i < 50; i++)
        {
            Thread.Sleep(500);
            if (ServerAlive()) return 0;
        }
        Fail("服务没有在预期时间内启动。\n\n可以手动运行 app\\desktop.py 看看报错信息。");
        return 4;
    }

    /// <summary>找 pythonw（优先无控制台版本）。</summary>
    private static string FindPython(string root)
    {
        // 0) 便携版自带运行时（dist/TonhTzeIME-portable/runtime/pythonw.exe）
        foreach (string rel in new[] { @"runtime\pythonw.exe", @"runtime\python.exe" })
        {
            string bundled = Path.Combine(root, rel);
            if (File.Exists(bundled)) return bundled;
        }

        // 1) 同目录的 python.txt
        string conf = Path.Combine(root, "python.txt");
        if (File.Exists(conf))
        {
            string p = File.ReadAllText(conf).Trim().Trim('"');
            if (File.Exists(p)) return p;
        }

        // 2) PATH 里的 pythonw / python
        foreach (string name in new[] { "pythonw.exe", "python.exe" })
        {
            string found = Which(name);
            if (found != null) return found;
        }

        // 3) 常见的按用户安装位置
        string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        foreach (string pattern in new[]
        {
            Path.Combine(local, "Programs", "Python"),
            Path.Combine(local, "Python"),
            @"C:\Python315", @"C:\Python314", @"C:\Python313", @"C:\Python312",
        })
        {
            try
            {
                if (Directory.Exists(pattern))
                {
                    string direct = Path.Combine(pattern, "pythonw.exe");
                    if (File.Exists(direct)) return direct;
                    foreach (string dir in Directory.GetDirectories(pattern))
                    {
                        string exe = Path.Combine(dir, "pythonw.exe");
                        if (File.Exists(exe)) return exe;
                    }
                }
            }
            catch { }
        }

        // 4) 本机 pythoncore 布局
        try
        {
            foreach (string dir in Directory.GetDirectories(local))
            {
                if (!dir.Contains("Python")) continue;
                foreach (string sub in Directory.GetDirectories(dir))
                {
                    string exe = Path.Combine(sub, "pythonw.exe");
                    if (File.Exists(exe)) return exe;
                }
            }
        }
        catch { }

        return null;
    }

    /// <summary>用 where 语义在 PATH 里查找可执行文件。</summary>
    private static string Which(string exe)
    {
        string path = Environment.GetEnvironmentVariable("PATH") ?? "";
        foreach (string dir in path.Split(';'))
        {
            if (string.IsNullOrWhiteSpace(dir)) continue;
            try
            {
                string full = Path.Combine(dir.Trim('"'), exe);
                if (File.Exists(full)) return full;
            }
            catch { }
        }
        return null;
    }

    private static bool ServerAlive()
    {
        try
        {
            using (var c = new TcpClient())
            {
                var task = c.ConnectAsync("127.0.0.1", Port);
                if (!task.Wait(600)) return false;
                return c.Connected;
            }
        }
        catch
        {
            return false;
        }
    }

    private static void Post(string path, string json)
    {
        var req = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:" + Port + path);
        req.Method = "POST";
        req.ContentType = "application/json";
        req.Timeout = 2000;
        byte[] body = Encoding.UTF8.GetBytes(json);
        req.ContentLength = body.Length;
        using (var s = req.GetRequestStream()) s.Write(body, 0, body.Length);
        using (req.GetResponse()) { }
    }

    private static void Fail(string msg)
    {
        MessageBox.Show(msg, "TonhTze 手写输入法", MessageBoxButtons.OK, MessageBoxIcon.Warning);
    }
}

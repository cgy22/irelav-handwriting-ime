// TonhTze AI 手写输入法 —— 单文件自解压启动器
//
// 一个 exe 里装着：Python 运行时 + numpy + Pillow + 程序本体 + 模型权重。
// 首次运行解压到 %LOCALAPPDATA%\TonhTzeIME\<版本>\，之后直接启动，秒开。
//
// 为什么不做成"每次跑到临时目录"：解压 90 MB 每次都要几秒，用户会以为卡了。
// 所以解压一次、留在本地；要彻底卸载就删掉那个目录。

using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Windows.Forms;

static class Bootstrap
{
    private const string AppVersion = "1.0.0";
    private const int Port = 8765;

    [STAThread]
    static int Main(string[] args)
    {
        string baseDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "TonhTzeIME", AppVersion);
        string marker = Path.Combine(baseDir, ".ready");
        string pyw = Path.Combine(baseDir, "runtime", "pythonw.exe");
        string entry = Path.Combine(baseDir, "app", "desktop.py");

        if (!File.Exists(marker) || !File.Exists(pyw) || !File.Exists(entry))
        {
            try
            {
                if (Directory.Exists(baseDir)) Directory.Delete(baseDir, true);
                Directory.CreateDirectory(baseDir);
                ExtractPayload(baseDir);
                File.WriteAllText(marker, DateTime.Now.ToString("o"));
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "首次解压失败：" + ex.Message + "\n\n目标目录：" + baseDir,
                    "TonhTze 手写输入法", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 1;
            }
        }

        if (ServerAlive())
        {
            try { Post("/api/window", "{\"action\":\"show\"}"); } catch { }
            return 0;
        }

        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = pyw,
                Arguments = "\"" + entry + "\"",
                WorkingDirectory = baseDir,
                UseShellExecute = false,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden,
            };
            psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
            Process.Start(psi);
        }
        catch (Exception ex)
        {
            MessageBox.Show("启动失败：" + ex.Message, "TonhTze 手写输入法",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 2;
        }

        for (int i = 0; i < 60; i++)
        {
            Thread.Sleep(500);
            if (ServerAlive()) return 0;
        }
        MessageBox.Show("服务没有在预期时间内启动。\n\n目录：" + baseDir,
            "TonhTze 手写输入法", MessageBoxButtons.OK, MessageBoxIcon.Warning);
        return 3;
    }

    /// <summary>把嵌入的 payload.zip 解压到目标目录。</summary>
    private static void ExtractPayload(string dest)
    {
        using (Stream s = Assembly.GetExecutingAssembly()
                   .GetManifestResourceStream("payload.zip"))
        {
            if (s == null) throw new InvalidOperationException("内嵌资源缺失");
            using (var zip = new ZipArchive(s, ZipArchiveMode.Read))
            {
                zip.ExtractToDirectory(dest);
            }
        }
    }

    private static bool ServerAlive()
    {
        try
        {
            using (var c = new TcpClient())
            {
                var t = c.ConnectAsync("127.0.0.1", Port);
                if (!t.Wait(600)) return false;
                return c.Connected;
            }
        }
        catch { return false; }
    }

    private static void Post(string path, string json)
    {
        var req = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:" + Port + path);
        req.Method = "POST";
        req.ContentType = "application/json";
        req.Timeout = 2000;
        byte[] body = Encoding.UTF8.GetBytes(json);
        req.ContentLength = body.Length;
        using (var st = req.GetRequestStream()) st.Write(body, 0, body.Length);
        using (req.GetResponse()) { }
    }
}

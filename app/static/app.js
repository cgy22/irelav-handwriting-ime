/* TonhTze AI 手写输入法 · 前端逻辑
 *
 * 输入：Pointer Events（手写笔 / 触控 / 鼠标统一），笔压用于笔锋。
 * 交互：写完停 1.2s 自动提交；候选字可改字；候选词只显示 dh；功能键注入到系统光标处。
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const pad = $("pad");
  const ctx = pad.getContext("2d");
  const candbar = $("candbar");
  const wordbar = $("wordbar");
  const textEl = $("text");
  const phEl = $("ph");
  const statusEl = $("status");
  const latencyEl = $("latency");
  const metaEl = $("meta");
  const padhint = $("padhint");
  const casebadge = $("casebadge");
  const collectbar = $("collectbar");
  const tcharEl = $("tchar");
  const tprogEl = $("tprog");
  const targetSel = $("targetSel");
  const splitSel = $("splitSel");
  const quotaEl = $("quota");

  // 识别概率、往返延迟这类调试信息只在 ?debug=1 时展示
  const DEBUG = new URLSearchParams(location.search).has("debug");
  // 写完立刻插入到目标输入框（无需点"发送"）
  const AUTO_INSERT = true;

  const state = {
    mode: "input",
    strokes: [],
    cur: null,
    text: "",
    lineHeights: [],
    candidates: [],
    shift: false,
    dpr: 1,
    commitTimer: null,
    wordTimer: null,
    classes: [],
    countsTrain: {},
    countsTest: {},
    counts: {},
    targetIdx: 0,
    quota: 10,
    split: "train",
    pointerId: null,
  };

  /* ------------------------------------------------------------------ 画布 */
  function resize() {
    const rect = pad.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    state.dpr = dpr;
    pad.width = Math.max(1, Math.round(rect.width * dpr));
    pad.height = Math.max(1, Math.round(rect.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    redraw();
  }

  function ptOf(e) {
    const r = pad.getBoundingClientRect();
    return {
      x: e.clientX - r.left,
      y: e.clientY - r.top,
      p: typeof e.pressure === "number" && e.pressure > 0 ? e.pressure : 0.5,
    };
  }

  const strokeWidth = (p) => 2.2 + 4.6 * Math.min(1, Math.max(0, p));

  function drawSegment(a, b) {
    ctx.strokeStyle = "#1b2430";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = strokeWidth(b.p);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.quadraticCurveTo(a.x, a.y, (a.x + b.x) / 2, (a.y + b.y) / 2);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  function clearPad() {
    ctx.clearRect(0, 0, pad.clientWidth, pad.clientHeight);
    paintWatermark();
  }

  /* 采集模式：把目标字符画成书写区水印（避免"顺手写别的"导致标签错位） */
  function paintWatermark() {
    if (state.mode !== "collect") return;
    const ch = currentTarget();
    if (!ch) return;
    const size = Math.min(pad.clientHeight * 0.8, 190);
    ctx.save();
    ctx.globalAlpha = 0.08;
    ctx.fillStyle = "#16DA49";
    ctx.font = `600 ${size}px "Microsoft YaHei", "Segoe UI", serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(ch === " " ? "␣" : ch, pad.clientWidth / 2, pad.clientHeight / 2);
    ctx.restore();
  }

  function redraw() {
    clearPad();
    const all = state.cur ? state.strokes.concat([state.cur]) : state.strokes;
    for (const s of all) {
      if (!s.length) continue;
      if (s.length === 1) {
        ctx.beginPath();
        ctx.arc(s[0].x, s[0].y, strokeWidth(s[0].p) / 2, 0, Math.PI * 2);
        ctx.fillStyle = "#1b2430";
        ctx.fill();
        continue;
      }
      for (let i = 1; i < s.length; i++) drawSegment(s[i - 1], s[i]);
    }
  }

  /* -------------------------------------------------------------- 指针输入 */
  let lastPt = null;

  pad.addEventListener("pointerdown", (e) => {
    if (e.button !== undefined && e.button > 0) return;
    e.preventDefault();
    pad.setPointerCapture(e.pointerId);
    state.pointerId = e.pointerId;
    if (state.commitTimer) {
      clearTimeout(state.commitTimer);
      state.commitTimer = null;
    }
    padhint.classList.add("hide");
    lastPt = ptOf(e);
    state.cur = [lastPt];
    redraw();
  });

  pad.addEventListener("pointermove", (e) => {
    if (!state.cur || e.pointerId !== state.pointerId) return;
    e.preventDefault();
    const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
    for (const ev of evs) {
      const p = ptOf(ev);
      if (lastPt && Math.hypot(p.x - lastPt.x, p.y - lastPt.y) < 0.7) continue;
      state.cur.push(p);
      if (lastPt) drawSegment(lastPt, p);
      lastPt = p;
    }
  });

  function endStroke() {
    if (!state.cur) return;
    if (state.cur.length) state.strokes.push(state.cur);
    state.cur = null;
    state.pointerId = null;
    lastPt = null;
    redraw();
    scheduleCommit();
  }
  pad.addEventListener("pointerup", endStroke);
  pad.addEventListener("pointercancel", endStroke);

  function scheduleCommit() {
    if (state.commitTimer) clearTimeout(state.commitTimer);
    state.commitTimer = setTimeout(() => {
      state.commitTimer = null;
      if (state.strokes.length) commit();
    }, 1200);
  }

  /* ------------------------------------------------------------------ 网络 */
  async function postJSON(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return r.json();
  }

  async function recognize() {
    const t0 = performance.now();
    const res = await postJSON("/api/recognize", {
      strokes: state.strokes.map((s) => s.map((p) => [p.x, p.y, p.p])),
      canvas: { w: pad.clientWidth, h: pad.clientHeight },
      line_heights: state.lineHeights,
      context: state.text.slice(-8),
      prefix: currentPrefix(),
    });
    if (res.error) {
      setStatus(res.error, true);
      return null;
    }
    state.candidates = res.top || [];
    renderCandidates();
    latencyEl.textContent = DEBUG
      ? `推理 ${res.ms}ms · 往返 ${(performance.now() - t0).toFixed(0)}ms`
      : "";
    if (res.case_hint === "upper") casebadge.textContent = "大小写：偏大写（本行偏高）";
    else if (res.case_hint === "lower") casebadge.textContent = "大小写：偏小写（本行偏低）";
    else casebadge.textContent = state.shift ? "大小写：Shift 强制大写" : "大小写：自动";
    return res;
  }

  /* -------------------------------------------------------------- 候选渲染 */
  function displayChar(c) {
    let ch = c.ch;
    if (state.shift) ch = ch.toUpperCase();
    return ch === " " ? "␣" : ch;
  }

  function renderCandidates() {
    candbar.innerHTML = "";
    if (!state.candidates.length) {
      candbar.innerHTML = '<div class="cand empty">书写后显示候选</div>';
      return;
    }
    state.candidates.forEach((c, i) => {
      const el = document.createElement("button");
      el.className = "cand";
      el.innerHTML = DEBUG
        ? `<span class="ch">${escapeHtml(displayChar(c))}</span><span class="p">${(c.p * 100).toFixed(0)}%</span>`
        : `<span class="ch">${escapeHtml(displayChar(c))}</span>`;
      el.title = `点击替换为 ${displayChar(c)}`;
      el.onclick = () => pickCandidate(i);
      candbar.appendChild(el);
    });
  }

  function pickCandidate(i) {
    const c = state.candidates[i];
    if (!c) return;
    if (state.mode === "collect") return;
    const ch = displayChar(c);
    if (state.text.length && state.candidates.length) {
      state.text = state.text.slice(0, -1) + ch;
      textEl.textContent = state.text;
      setStatus(`改为 "${ch}"`);
      refreshWords();
    }
  }

  /* ------------------------------------------------------------- 候选词 dh */
  function currentPrefix() {
    if (state.mode === "collect") return "";
    const m = state.text.match(/[A-Za-z\u00C0-\u024F'\u2018\u2019-]+$/);
    return m ? m[0] : "";
  }

  function refreshWords() {
    if (state.wordTimer) clearTimeout(state.wordTimer);
    state.wordTimer = setTimeout(async () => {
      const prefix = currentPrefix();
      if (!prefix) {
        wordbar.innerHTML = "";
        return;
      }
      try {
        const r = await fetch(`/api/suggest?prefix=${encodeURIComponent(prefix)}&limit=10`);
        const d = await r.json();
        if (!d.items || !d.items.length) {
          wordbar.innerHTML = "";
          return;
        }
        wordbar.innerHTML = "";
        for (const w of d.items) {
          const el = document.createElement("button");
          el.className = "word";
          el.textContent = w.dh;          // 只显示 dh，不显示释义
          el.title = `补全为 ${w.dh}`;
          el.onclick = async () => {
            const rest = w.dh.slice(prefix.length);
            if (rest) appendChar(rest);
            if (AUTO_INSERT && rest) await postJSON("/api/send", { text: rest });
            setStatus(`选词 ${w.dh}`);
            refreshWords();
          };
          wordbar.appendChild(el);
        }
      } catch {
        wordbar.innerHTML = "";
      }
    }, 120);
  }

  /* ------------------------------------------------------------------ 提交 */
  async function commit() {
    if (!state.strokes.length) return;
    setStatus("识别中…");
    const res = await recognize();
    if (!res) return;

    if (state.mode === "collect") {
      await saveSample();
      return;
    }
    if (!state.candidates.length) {
      setStatus("没有候选", true);
      return;
    }
    const ch = displayChar(state.candidates[0]);
    appendChar(ch);
    if (res.height) state.lineHeights.push(res.height);
    state.strokes = [];
    clearPad();
    if (AUTO_INSERT) {
      const sent = await postJSON("/api/send", { text: ch });
      setStatus(sent.ok ? `已插入 "${ch}"` : `已识别 "${ch}"（插入失败：${sent.error}）`, !sent.ok);
    } else {
      setStatus(`已输入 "${ch}"`);
    }
  }

  function appendChar(ch) {
    state.text += ch;
    phEl.style.display = "none";
    textEl.textContent = state.text;
    refreshWords();
  }

  /* ------------------------------------------------------------------ 采集 */
  function currentCounts() {
    return state.split === "test" ? state.countsTest : state.countsTrain;
  }

  function currentTarget() {
    return state.classes[state.targetIdx % Math.max(1, state.classes.length)] || "";
  }

  function refreshTargetUI() {
    const ch = currentTarget();
    tcharEl.textContent = ch === " " ? "␣" : ch;
    const c = currentCounts();
    const n = c[ch] || 0;
    const total = Object.values(c).reduce((a, b) => a + b, 0);
    const goal = (state.classes.length || 1) * state.quota;
    tprogEl.textContent = `${n} / ${state.quota}　总计 ${total} / ${goal}`;
    if (!state.strokes.length && !state.cur) clearPad();
  }

  function advanceTarget() {
    const total = state.classes.length || 1;
    let i = state.targetIdx;
    for (let k = 0; k < total; k++) {
      i = (i + 1) % total;
      if ((currentCounts()[state.classes[i]] || 0) < state.quota) break;
    }
    state.targetIdx = i;
    if (targetSel.options.length) targetSel.selectedIndex = i;
    refreshTargetUI();
  }

  async function saveSample() {
    const ch = currentTarget();
    if (!ch) return;
    const res = await postJSON("/api/collect", {
      label: ch,
      split: state.split,
      strokes: state.strokes.map((s) => s.map((p) => [p.x, p.y, p.p])),
      canvas: { w: pad.clientWidth, h: pad.clientHeight },
    });
    if (res.error) {
      setStatus(res.error, true);
      return;
    }
    currentCounts()[ch] = res.count;
    const pred = state.candidates.length ? state.candidates[0].ch : "?";
    const pp = state.candidates.length ? state.candidates[0].p : 0;
    const mismatch = pred !== "?" && pred.toLowerCase() !== ch.toLowerCase() && pp >= 0.55;
    setStatus(
      mismatch
        ? `⚠ 已按目标 "${ch}" 保存，但模型读成 "${pred}"（${(pp * 100).toFixed(0)}%）——写错了请点「撤销上一条」`
        : `已存 "${ch}"（${res.count}）· 模型识别为 "${pred}"`,
      mismatch
    );
    state.strokes = [];
    clearPad();
    refreshTargetUI();
    if (res.count >= state.quota) advanceTarget();
  }

  function fillTargets() {
    targetSel.innerHTML = "";
    state.classes.forEach((ch, i) => {
      const o = document.createElement("option");
      o.value = String(i);
      o.textContent = ch === " " ? "␣" : ch;
      targetSel.appendChild(o);
    });
    if (state.classes.length) {
      const first = state.classes.findIndex((c) => (currentCounts()[c] || 0) < state.quota);
      state.targetIdx = first < 0 ? 0 : first;
      targetSel.selectedIndex = state.targetIdx;
    }
    refreshTargetUI();
  }

  /* ------------------------------------------------------------------ 状态 */
  function setStatus(msg, isErr = false) {
    statusEl.textContent = msg;
    statusEl.style.color = isErr ? "var(--danger)" : "";
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
  }

  async function loadState() {
    try {
      const r = await fetch("/api/state");
      const s = await r.json();
      state.classes = s.classes && s.classes.length ? s.classes : "0123456789".split("");
      state.countsTrain = s.counts_train || {};
      state.countsTest = s.counts_test || {};
      state.counts = currentCounts();
      metaEl.textContent = s.model || "已连接";
      fillTargets();
      setStatus(`就绪 · 训练集 ${s.count_train} / 测试集 ${s.count_test}`);
    } catch {
      metaEl.textContent = "服务未连接";
      setStatus("无法连接后端，请确认服务已启动", true);
    }
  }

  /* ------------------------------------------------------------ 窗口控制 */
  // 面板是 WS_EX_NOACTIVATE 的（点了不抢焦点），所以不能用系统标题栏拖动，
  // 这里把表头上的拖动换算成位移，逐帧发给服务端移动窗口。
  const head = document.querySelector(".ime-head");
  let dragging = null;
  let lastSent = 0;
  if (head) {
    head.addEventListener("pointerdown", (e) => {
      if (e.target.closest("button")) return;
      head.setPointerCapture(e.pointerId);
      dragging = { x: e.clientX, y: e.clientY, id: e.pointerId };
      e.preventDefault();
    });
    head.addEventListener("pointermove", (e) => {
      if (!dragging || e.pointerId !== dragging.id) return;
      const now = performance.now();
      if (now - lastSent < 16) return;   // 约 60fps
      const dx = Math.round(e.clientX - dragging.x);
      const dy = Math.round(e.clientY - dragging.y);
      if (!dx && !dy) return;
      dragging.x = e.clientX;
      dragging.y = e.clientY;
      lastSent = now;
      postJSON("/api/window", { action: "move", dx, dy }).catch(() => {});
    });
    const stopDrag = (e) => {
      if (dragging && e.pointerId === dragging.id) dragging = null;
    };
    head.addEventListener("pointerup", stopDrag);
    head.addEventListener("pointercancel", stopDrag);
  }

  async function windowAction(action) {
    try {
      const res = await postJSON("/api/window", { action });
      if (res && res.ok === false) {
        setStatus("窗口控制不可用：请用 start_ime.cmd 启动（不要直接开网页）", true);
      }
    } catch {
      setStatus("窗口控制不可用：服务未以桌面外壳启动", true);
    }
  }
  $("minBtn").onclick = () => windowAction("hide");
  $("closeBtn").onclick = () => windowAction("close");

  /* -------------------------------------------------------------- 系统按键 */
  async function sendKey(key) {
    const res = await postJSON("/api/key", { key });
    if (res.error) setStatus(res.error, true);
    else setStatus(`已发送 ${key}`);
  }

  document.querySelectorAll("[data-key]").forEach((btn) => {
    btn.onclick = () => {
      const k = btn.dataset.key;
      // ⌫ 优先删本地文本，空了才把退格送给系统
      if (k === "backspace" && state.text.length) {
        state.text = state.text.slice(0, -1);
        textEl.textContent = state.text;
        if (!state.text.length) phEl.style.display = "";
        refreshWords();
        setStatus("退格");
        return;
      }
      // ↵ 先把已输入文本送到光标处，再发回车
      if (k === "enter") {
        sendEnter();
        return;
      }
      sendKey(k);
    };
  });

  async function sendEnter() {
    if (state.text) {
      await postJSON("/api/send", { text: state.text, trailing_enter: true });
      state.text = "";
      textEl.textContent = "";
      phEl.style.display = "";
      wordbar.innerHTML = "";
      setStatus("已发送并回车");
    } else {
      await postJSON("/api/key", { key: "enter" });
      setStatus("回车");
    }
  }

  $("sendBtn").onclick = async () => {
    if (!state.text) {
      setStatus("没有可发送的内容", true);
      return;
    }
    const res = await postJSON("/api/send", { text: state.text });
    if (res.error) setStatus(res.error, true);
    else setStatus(`已发送 ${state.text.length} 个字符到光标处`);
  };

  /* ------------------------------------------------------------------ 控件 */
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      collectbar.classList.toggle("hidden", state.mode !== "collect");
      padhint.textContent =
        state.mode === "collect" ? "按上方提示书写，停顿 1.2 秒自动保存" : "在此书写 · 支持手写笔 / 触控 / 鼠标";
      padhint.classList.remove("hide");
      setStatus(state.mode === "collect" ? "采集模式：写一个存一个" : "输入模式");
      refreshTargetUI();
    };
  });

  $("shiftBtn").onclick = (e) => {
    state.shift = !state.shift;
    e.currentTarget.classList.toggle("on", state.shift);
    casebadge.textContent = state.shift ? "大小写：Shift 强制大写" : "大小写：自动";
    renderCandidates();
  };

  $("spaceBtn").onclick = () => {
    appendChar(" ");
    state.lineHeights = [];
    setStatus("空格");
  };
  $("clearBtn").onclick = () => {
    state.strokes = [];
    state.cur = null;
    clearPad();
    state.text = "";
    state.lineHeights = [];
    textEl.textContent = "";
    phEl.style.display = "";
    candbar.innerHTML = '<div class="cand empty">书写后显示候选</div>';
    wordbar.innerHTML = "";
    padhint.classList.remove("hide");
    setStatus("已清空");
  };
  $("commitBtn").onclick = () => commit();
  $("copyBtn").onclick = async () => {
    try {
      await navigator.clipboard.writeText(state.text);
      setStatus("已复制到剪贴板");
    } catch {
      setStatus("复制失败（浏览器权限）", true);
    }
  };
  $("skipBtn").onclick = () => advanceTarget();
  $("undoBtn").onclick = async () => {
    const res = await postJSON("/api/undo", { split: state.split });
    if (!res.ok) {
      setStatus(res.error || "撤销失败", true);
      return;
    }
    currentCounts()[res.removed] = res.count;
    state.strokes = [];
    clearPad();
    refreshTargetUI();
    setStatus(`已撤销一条 "${res.removed}"`);
  };

  targetSel.onchange = () => {
    state.targetIdx = Number(targetSel.value);
    refreshTargetUI();
  };
  quotaEl.onchange = () => {
    state.quota = Math.max(1, Number(quotaEl.value) || 10);
    refreshTargetUI();
  };
  splitSel.onchange = () => {
    state.split = splitSel.value;
    state.counts = currentCounts();
    fillTargets();
    setStatus(state.split === "test" ? "注意：当前写入测试集，不参与训练" : "当前写入训练集");
  };

  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter") commit();
    else if (e.key === "Escape") $("clearBtn").click();
  });

  /* ------------------------------------------------------------------ 启动 */
  window.addEventListener("resize", resize);
  if (window.ResizeObserver) new ResizeObserver(resize).observe(pad);
  resize();
  loadState().then(() => {
    const q = new URLSearchParams(location.search);
    if (q.get("mode") === "collect") document.querySelector('.tab[data-mode="collect"]').click();
    const preset = q.get("text");
    if (preset) {
      state.text = preset;
      phEl.style.display = "none";
      textEl.textContent = preset;
      refreshWords();
    }
  });
})();

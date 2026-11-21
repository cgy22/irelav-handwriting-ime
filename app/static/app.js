/* TonhTze AI 手写输入法 · 前端逻辑
 *
 * 与早期版本的差别：这里没有"文本框预览"，写完直接插到目标输入框。
 * 因此界面上只保留：候选字、候选词、书写区、导航/功能键。
 *
 * 关键点：
 *   * 面板是 WS_EX_NOACTIVATE，点击不会抢焦点，所以拖动要自己按位移移动窗口
 *   * 位移必须乘 devicePixelRatio（200% 缩放下 1 CSS px = 2 物理像素）
 *   * 撤销 = 往目标发送 N 个退格（N = 上一次插入的字符数）
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const pad = $("pad");
  const ctx = pad.getContext("2d");
  const candbar = $("candbar");
  const wordbar = $("wordbar");
  const statusEl = $("status");
  const latencyEl = $("latency");
  const padhint = $("padhint");
  const casebadge = $("casebadge");
  const collectbar = $("collectbar");
  const tcharEl = $("tchar");
  const tprogEl = $("tprog");
  const targetSel = $("targetSel");
  const splitSel = $("splitSel");
  const quotaEl = $("quota");

  const DEBUG = new URLSearchParams(location.search).has("debug");

  const state = {
    mode: "input",
    strokes: [],
    cur: null,
    candidates: [],
    shift: false,
    dpr: window.devicePixelRatio || 1,
    commitTimer: null,
    wordTimer: null,
    classes: [],
    countsTrain: {},
    countsTest: {},
    targetIdx: 0,
    quota: 10,
    split: "train",
    pointerId: null,
    typed: "",        // 已插入到目标输入框的文本（用于词前缀与撤销）
    lastInsert: "",   // 最近一次插入的片段
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
    if (state.commitTimer) { clearTimeout(state.commitTimer); state.commitTimer = null; }
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

  function currentPrefix() {
    if (state.mode === "collect") return "";
    const m = state.typed.match(/[A-Za-z\u00C0-\u024F'\u2018\u2019-]+$/);
    return m ? m[0] : "";
  }

  function displayChar(c) {
    let ch = c.ch;
    if (state.shift) ch = ch.toUpperCase();
    return ch === " " ? "␣" : ch;
  }

  async function recognize() {
    const t0 = performance.now();
    const res = await postJSON("/api/recognize", {
      strokes: state.strokes.map((s) => s.map((p) => [p.x, p.y, p.p])),
      canvas: { w: pad.clientWidth, h: pad.clientHeight },
      line_heights: [],
      context: state.typed.slice(-8),
      prefix: currentPrefix(),
    });
    if (res.error) { setStatus(res.error, true); return null; }
    state.candidates = res.top || [];
    renderCandidates();
    latencyEl.textContent = DEBUG
      ? `推理 ${res.ms}ms · 往返 ${(performance.now() - t0).toFixed(0)}ms`
      : "";
    if (res.case_hint === "upper") casebadge.textContent = "大小写：偏大写";
    else if (res.case_hint === "lower") casebadge.textContent = "大小写：偏小写";
    else casebadge.textContent = state.shift ? "大小写：Shift 大写" : "大小写：自动";
    return res;
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
      el.onclick = () => chooseCandidate(i);
      candbar.appendChild(el);
    });
  }

  /* 点候选 = 替换刚才插入的那个字：先退格，再插入 */
  async function chooseCandidate(i) {
    const c = state.candidates[i];
    if (!c || state.mode === "collect") return;
    const ch = displayChar(c);
    if (state.lastInsert) {
      await postJSON("/api/key", { key: "backspace" });
      state.typed = state.typed.slice(0, -state.lastInsert.length);
    }
    await insertText(ch);
    setStatus(`改为 "${ch}"`);
  }

  async function insertText(text) {
    const res = await postJSON("/api/send", { text });
    if (!res.ok) { setStatus(`插入失败：${res.error}`, true); return false; }
    state.typed += text;
    state.lastInsert = text;
    refreshWords();
    return true;
  }

  /* ------------------------------------------------------------------ 提交 */
  async function commit() {
    if (!state.strokes.length) return;
    setStatus("识别中…");
    const res = await recognize();
    if (!res) return;

    if (state.mode === "collect") { await saveSample(); return; }
    if (!state.candidates.length) { setStatus("没有候选", true); return; }

    const ch = displayChar(state.candidates[0]);
    state.strokes = [];
    clearPad();
    if (await insertText(ch)) setStatus(`已输入 "${ch}"`);
  }

  /* ------------------------------------------------------------- 候选词 dh */
  function refreshWords() {
    if (state.wordTimer) clearTimeout(state.wordTimer);
    state.wordTimer = setTimeout(async () => {
      const prefix = currentPrefix();
      if (!prefix) { wordbar.innerHTML = ""; return; }
      try {
        const r = await fetch(`/api/suggest?prefix=${encodeURIComponent(prefix)}&limit=10`);
        const d = await r.json();
        if (!d.items || !d.items.length) { wordbar.innerHTML = ""; return; }
        wordbar.innerHTML = "";
        for (const w of d.items) {
          const el = document.createElement("button");
          el.className = "word";
          el.textContent = w.dh;   // 只显示 dh
          el.title = `补全为 ${w.dh}`;
          el.onclick = async () => {
            const rest = w.dh.slice(prefix.length);
            if (rest) await insertText(rest);
            setStatus(`选词 ${w.dh}`);
            refreshWords();
          };
          wordbar.appendChild(el);
        }
      } catch { wordbar.innerHTML = ""; }
    }, 120);
  }

  /* ------------------------------------------------------------------ 撤销 */
  async function undoLast() {
    if (!state.lastInsert) { setStatus("没有可撤销的内容"); return; }
    for (let i = 0; i < state.lastInsert.length; i++) {
      await postJSON("/api/key", { key: "backspace" });
    }
    state.typed = state.typed.slice(0, -state.lastInsert.length);
    state.lastInsert = "";
    refreshWords();
    setStatus("已撤销");
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
    tprogEl.textContent = `${n}/${state.quota}　总 ${total}/${goal}`;
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
    if (res.error) { setStatus(res.error, true); return; }
    currentCounts()[ch] = res.count;
    const pred = state.candidates.length ? state.candidates[0].ch : "?";
    const pp = state.candidates.length ? state.candidates[0].p : 0;
    const mismatch = pred !== "?" && pred.toLowerCase() !== ch.toLowerCase() && pp >= 0.55;
    setStatus(
      mismatch
        ? `⚠ 按目标 "${ch}" 保存，模型读成 "${pred}"（${(pp * 100).toFixed(0)}%）`
        : `已存 "${ch}"（${res.count}）· 识别为 "${pred}"`,
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
      fillTargets();
      setStatus("就绪");
    } catch {
      setStatus("服务未连接", true);
    }
  }

  /* ------------------------------------------------------------ 窗口控制 */
  // 面板不激活，不能靠系统标题栏拖动：把表头拖动换算成位移发给服务端。
  // dx/dy 是 CSS 像素，必须乘 devicePixelRatio 才是物理像素，
  // 否则 200% 缩放下窗口只移动鼠标位移的一半。
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
      if (now - lastSent < 16) return;
      const ratio = window.devicePixelRatio || 1;
      const dx = Math.round((e.clientX - dragging.x) * ratio);
      const dy = Math.round((e.clientY - dragging.y) * ratio);
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
      if (res && res.ok === false) setStatus("窗口控制不可用：请用 start_ime.cmd 启动", true);
    } catch {
      setStatus("窗口控制不可用：服务未以桌面外壳启动", true);
    }
  }
  $("minBtn").onclick = () => windowAction("hide");
  $("closeBtn").onclick = () => windowAction("close");

  /* -------------------------------------------------------------- 系统按键 */
  async function sendKey(key) {
    const res = await postJSON("/api/key", { key });
    if (res.error) { setStatus(res.error, true); return; }
    if (key === "backspace") {
      state.typed = state.typed.slice(0, -1);
      state.lastInsert = "";
      refreshWords();
    } else if (key === "enter") {
      state.typed = "";
      state.lastInsert = "";
      refreshWords();
    }
    setStatus("");
  }

  document.querySelectorAll("[data-key]").forEach((btn) => {
    btn.onclick = () => sendKey(btn.dataset.key);
  });

  /* ------------------------------------------------------------------ 控件 */
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      collectbar.classList.toggle("hidden", state.mode !== "collect");
      padhint.textContent =
        state.mode === "collect" ? "按提示书写，停顿 1.2 秒自动保存" : "在此书写 · 笔 / 触控 / 鼠标";
      padhint.classList.remove("hide");
      setStatus(state.mode === "collect" ? "采集模式" : "输入模式");
      refreshTargetUI();
    };
  });

  $("shiftBtn").onclick = (e) => {
    state.shift = !state.shift;
    e.currentTarget.classList.toggle("on", state.shift);
    casebadge.textContent = state.shift ? "大小写：Shift 大写" : "大小写：自动";
    renderCandidates();
  };
  $("spaceBtn").onclick = async () => {
    if (await insertText(" ")) {
      state.typed = "";        // 空格结束当前词
      state.lastInsert = " ";
      setStatus("空格");
    }
  };
  $("undoBtn2").onclick = () => undoLast();
  $("skipBtn").onclick = () => advanceTarget();
  $("undoBtn").onclick = async () => {
    const res = await postJSON("/api/undo", { split: state.split });
    if (!res.ok) { setStatus(res.error || "撤销失败", true); return; }
    currentCounts()[res.removed] = res.count;
    state.strokes = [];
    clearPad();
    refreshTargetUI();
    setStatus(`已撤销一条 "${res.removed}"`);
  };

  targetSel.onchange = () => { state.targetIdx = Number(targetSel.value); refreshTargetUI(); };
  quotaEl.onchange = () => {
    state.quota = Math.max(1, Number(quotaEl.value) || 10);
    refreshTargetUI();
  };
  splitSel.onchange = () => {
    state.split = splitSel.value;
    fillTargets();
    setStatus(state.split === "test" ? "写入测试集" : "写入训练集");
  };

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") windowAction("hide");
  });

  /* ------------------------------------------------------------------ 启动 */
  window.addEventListener("resize", resize);
  if (window.ResizeObserver) new ResizeObserver(resize).observe(pad);
  resize();
  loadState().then(() => {
    const q = new URLSearchParams(location.search);
    if (q.get("mode") === "collect") document.querySelector('.tab[data-mode="collect"]').click();
  });
})();

/* 手写输入法前端
 * 笔 / 触控 / 鼠标统一走 Pointer Events；触控笔的压力用于笔锋粗细。
 * 核心交互：书写 -> 停顿 800ms 自动提交 -> 候选条可改字。
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const pad = $("pad");
  const ctx = pad.getContext("2d", { desynchronized: true });
  const candbar = $("candbar");
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

  const state = {
    mode: "input",
    strokes: [],
    cur: null,
    text: "",
    lineHeights: [],
    candidates: [],
    lastCands: [],
    lastStrokes: [],
    shift: false,
    dpr: 1,
    commitTimer: null,
    classes: [],
    counts: {},
    countsTrain: {},
    countsTest: {},
    targetIdx: 0,
    quota: 12,
    split: "train",
  };

  /* ------------------------------------------------------------------ 画布 */
  function resize() {
    const rect = pad.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    state.dpr = dpr;
    pad.width = Math.round(rect.width * dpr);
    pad.height = Math.round(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    redraw();
  }
  window.addEventListener("resize", resize);

  function ptOf(e) {
    const r = pad.getBoundingClientRect();
    return {
      x: e.clientX - r.left,
      y: e.clientY - r.top,
      p: typeof e.pressure === "number" && e.pressure > 0 ? e.pressure : 0.5,
    };
  }

  function strokeWidth(p) {
    return 2.2 + 4.6 * Math.min(1, Math.max(0, p)); // 压力 -> 笔锋
  }

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

  // 采集模式下把目标字符画成书写区里的淡色水印：
  // 只在小标签上显示目标时，很容易"顺手写别的数字"，标签就会错位。
  function paintWatermark() {
    if (state.mode !== "collect") return;
    const ch = currentTarget();
    if (!ch) return;
    ctx.save();
    ctx.globalAlpha = 0.08;
    ctx.fillStyle = "#2f6df6";
    ctx.font = "190px 'Microsoft YaHei', 'Segoe UI', serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(ch === " " ? "␣" : ch, pad.clientWidth / 2, pad.clientHeight / 2);
    ctx.restore();
  }

  function redraw() {
    clearPad();
    for (const s of state.strokes) {
      if (s.length === 1) {
        ctx.beginPath();
        ctx.arc(s[0].x, s[0].y, strokeWidth(s[0].p) / 2, 0, Math.PI * 2);
        ctx.fillStyle = "#1b2430";
        ctx.fill();
      }
      for (let i = 1; i < s.length; i++) drawSegment(s[i - 1], s[i]);
    }
    if (state.cur && state.cur.length) {
      const s = state.cur;
      if (s.length === 1) {
        ctx.beginPath();
        ctx.arc(s[0].x, s[0].y, strokeWidth(s[0].p) / 2, 0, Math.PI * 2);
        ctx.fillStyle = "#1b2430";
        ctx.fill();
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
    if (!state.cur) return;
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

  function endStroke(e) {
    if (!state.cur) return;
    if (state.cur.length === 0) {
      state.cur = null;
      return;
    }
    state.strokes.push(state.cur);
    state.cur = null;
    lastPt = null;
    redraw();
    scheduleCommit();
  }
  pad.addEventListener("pointerup", endStroke);
  pad.addEventListener("pointercancel", endStroke);
  pad.addEventListener("pointerleave", (e) => {
    if (state.cur) endStroke(e);
  });

  function scheduleCommit() {
    if (state.commitTimer) clearTimeout(state.commitTimer);
    state.commitTimer = setTimeout(() => {
      state.commitTimer = null;
      if (state.strokes.length) commit();
    }, 1200);
  }

  /* ------------------------------------------------------------------ 识别 */
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
    });
    if (res.error) {
      setStatus(res.error, true);
      return null;
    }
    state.candidates = res.top || [];
    renderCandidates();
    latencyEl.textContent = `推理 ${res.ms}ms · 往返 ${(performance.now() - t0).toFixed(0)}ms`;
    if (res.case_hint === "upper") casebadge.textContent = "大小写：偏大写（本行偏高）";
    else if (res.case_hint === "lower") casebadge.textContent = "大小写：偏小写（本行偏低）";
    else casebadge.textContent = state.shift ? "大小写：Shift 强制大写态" : "大小写：自动";
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
      el.className = "cand" + (i === 0 ? "" : " alt");
      el.innerHTML = `<span class="ch">${escapeHtml(displayChar(c))}</span><span class="p">${(c.p * 100).toFixed(0)}%</span>`;
      el.title = `点击替换为 ${displayChar(c)}`;
      el.onclick = () => pickCandidate(i);
      candbar.appendChild(el);
    });
  }

  function displayChar(c) {
    let ch = c.ch;
    if (state.shift) ch = ch.toUpperCase(); // Shift 只把字母推向大写形态
    return ch === " " ? "␣" : ch;
  }

  /* ------------------------------------------------------------------ 提交 */
  async function commit(forceChar) {
    if (!state.strokes.length) return;
    setStatus("识别中…");
    const res = await recognize();
    if (!res) return;

    if (state.mode === "collect") {
      await saveSample();
      return;
    }

    let ch = forceChar;
    if (!ch) {
      if (!state.candidates.length) {
        setStatus("没有候选", true);
        return;
      }
      ch = displayChar(state.candidates[0]);
    }
    appendChar(ch);
    if (res.height) state.lineHeights.push(res.height);
    state.lastStrokes = state.strokes.map((s) => s.slice());
    state.lastCands = state.candidates.slice();
    state.strokes = [];
    clearPad();
    setStatus(`已输入 "${ch}"`);
  }

  function appendChar(ch) {
    state.text += ch;
    phEl.style.display = "none";
    textEl.textContent = state.text;
  }

  function pickCandidate(i) {
    const c = state.candidates[i];
    if (!c) return;
    const ch = displayChar(c);
    if (state.mode === "collect") {
      setStatus(`候选: ${ch}`);
      return;
    }
    // 有刚提交的字就替换，否则重新提交
    if (state.text.length >= 1 && state.lastStrokes.length) {
      state.text = state.text.slice(0, -1) + ch;
      textEl.textContent = state.text;
      setStatus(`改为 "${ch}"`);
    }
  }

  /* ------------------------------------------------------------------ 采集 */
  function currentTarget() {
    return state.classes[state.targetIdx % Math.max(1, state.classes.length)] || "";
  }

  function currentCounts() {
    return state.split === "test" ? state.countsTest : state.countsTrain;
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
      const ch = state.classes[i];
      if ((currentCounts()[ch] || 0) < state.quota) break;
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
    state.lastSaved = { ch, split: state.split };
    const mismatch = pred !== "?" && pred.toLowerCase() !== ch.toLowerCase() && pp >= 0.55;
    if (mismatch) {
      setStatus(`⚠ 已按目标 "${ch}" 保存，但模型读成了 "${pred}"（${(pp * 100).toFixed(0)}%）——写错了请点「撤销上一条」`, true);
    } else {
      setStatus(`已存 "${ch}"（${res.count}）· 模型识别为 "${pred}"`);
    }
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
      state.counts = state.split === "test" ? state.countsTest : state.countsTrain;
      metaEl.textContent = `${s.model} · ${state.classes.length} 类`;
      fillTargets();
      setStatus(`就绪 · 已采集训练 ${s.count_train} / 测试 ${s.count_test}`);
    } catch (e) {
      metaEl.textContent = "服务未连接";
      setStatus("无法连接后端，请确认 server.py 已启动", true);
    }
  }

  /* ------------------------------------------------------------------ 按钮 */
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      collectbar.classList.toggle("hidden", state.mode !== "collect");
      padhint.textContent =
        state.mode === "collect"
          ? "按上方提示书写，停顿 1.2 秒自动保存"
          : "在此书写 · 支持笔 / 触控 / 鼠标";
      padhint.classList.remove("hide");
      setStatus(state.mode === "collect" ? "采集模式：写一个存一个" : "输入模式");
      refreshTargetUI();
    };
  });

  $("shiftBtn").onclick = (e) => {
    state.shift = !state.shift;
    e.currentTarget.classList.toggle("on", state.shift);
    casebadge.textContent = state.shift ? "大小写：Shift 强制大写态" : "大小写：自动";
    renderCandidates();
  };

  $("spaceBtn").onclick = () => {
    appendChar(" ");
    state.lineHeights = [];
    setStatus("空格");
  };
  $("backBtn").onclick = () => {
    if (state.text.length) {
      state.text = state.text.slice(0, -1);
      textEl.textContent = state.text;
      if (!state.text.length) phEl.style.display = "";
    }
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
    const c = currentCounts();
    c[res.removed] = res.count;
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
    state.quota = Math.max(1, Number(quotaEl.value) || 12);
    refreshTargetUI();
  };
  splitSel.onchange = () => {
    state.split = splitSel.value;
    state.counts = currentCounts();
    setStatus(state.split === "test" ? "注意：当前写入的是**测试集**，不要用于训练" : "当前写入训练集");
  };

  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter") commit();
    else if (e.key === "Backspace") $("backBtn").click();
    else if (e.key === "Escape") $("clearBtn").click();
  });

  /* ------------------------------------------------------------------ 启动 */
  resize();
  loadState().then(() => {
    // 支持 ?mode=collect 直接进入采集模式（便于分享链接/截图）
    const m = new URLSearchParams(location.search).get("mode");
    if (m === "collect") document.querySelector('.tab[data-mode="collect"]').click();
  });
  setInterval(loadState, 15000);
})();

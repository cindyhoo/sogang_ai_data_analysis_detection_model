/* 공통 도구: 다국어, API 호출, 포맷, 토스트, trainer 상태 배지, Job 폴링 */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* --- 다국어 (사전은 서버가 base.html에 넣어 준다: app/core/locales/{lang}.json) --- */
const I18N = window.I18N_BOOT || { lang: "ko", locale: "ko-KR", cookie: "lang", messages: {} };

function fillMsg(s, args, escape) {
  return s.replace(/\{(\w+)\}/g, (m, k) => {
    if (!args || !(k in args) || args[k] == null) return m;
    const v = args[k];
    const out = Number.isInteger(v) ? v.toLocaleString(I18N.locale) : String(v);
    return escape ? esc(out) : out;
  });
}
/* t: 일반 텍스트용 (textContent, toast, confirm). th: innerHTML용 — 인자만 이스케이프한다. */
function t(key, args) {
  const s = I18N.messages[key];
  return s == null ? key : fillMsg(s, args, false);
}
function th(key, args) {
  const s = I18N.messages[key];
  return s == null ? esc(key) : fillMsg(s, args, true);
}
function hasMsg(key) { return key != null && I18N.messages[key] != null; }
/* 상태 파일의 번역 가능한 필드: obj[field] (기본 언어 문장) + obj[field_key] + obj[field_args] */
function tf(obj, field) {
  if (!obj) return "";
  const k = obj[`${field}_key`];
  return hasMsg(k) ? t(k, obj[`${field}_args`]) : (obj[field] ?? "");
}
/* API 오류 detail: 문자열, {key, args, message}, 또는 pydantic 오류 배열 */
function tdetail(d) {
  if (d == null) return "";
  if (typeof d === "string") return d;
  if (d.key) return hasMsg(d.key) ? t(d.key, d.args) : (d.message || d.key);
  return JSON.stringify(d);
}
const stateLabel = (s) => (hasMsg(`state.${s}`) ? t(`state.${s}`) : s);
const jobLabel = (s) => (hasMsg(`job.${s}`) ? t(`job.${s}`) : s);
const deviceLabel = (d) => (d === "probing" ? t("device.probing") : d);
const datasetName = (d) => (hasMsg(`dataset.${d.id}.name`) ? t(`dataset.${d.id}.name`) : d.display_name);

function setLang(code) {
  try { document.cookie = `${I18N.cookie}=${encodeURIComponent(code)}; path=/; max-age=31536000; SameSite=Lax`; } catch (e) { /* 쿠키 차단 */ }
  location.reload();
}
document.addEventListener("DOMContentLoaded", () => {
  const sel = $("#lang-select");
  if (sel) sel.addEventListener("change", () => setLang(sel.value));
});

async function api(path, { method = "GET", body, form } = {}) {
  const opts = { method, headers: {} };
  if (form) opts.body = form;
  else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) {
    const d = data && data.detail;
    throw new Error(d ? tdetail(d) : `HTTP ${res.status}`);
  }
  return data;
}

const fmt = {
  pct: (v, d = 1) => (v == null ? "–" : (v * 100).toFixed(d) + "%"),
  num: (v, d = 3) => (v == null ? "–" : Number(v).toFixed(d)),
  int: (v) => (v == null ? "–" : Number(v).toLocaleString(I18N.locale)),
  params: (v) => (v == null ? "–" : v >= 1e6 ? (v / 1e6).toFixed(2) + "M" : (v / 1e3).toFixed(0) + "K"),
  dur: (s) => {
    if (s == null) return "–";
    s = Math.round(s);
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
    return h ? t("time.hm", { h, m }) : m ? t("time.ms", { m, s: x }) : t("time.s", { s: x });
  },
  date: (iso) => (iso ? iso.replace("T", " ").slice(0, 16) : "–"),
  ci: (ci) => (ci ? `[${(ci[0] * 100).toFixed(1)}, ${(ci[1] * 100).toFixed(1)}]%` : ""),
};

const FINAL = ["succeeded", "failed", "cancelled"];

function stateBadge(state) {
  return `<span class="pill s-${esc(state)}">${esc(stateLabel(state))}</span>`;
}

let toastTimer = null;
function toast(msg, kind = "info") {
  const el = $("#toast");
  el.textContent = msg;
  el.className = `toast ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), kind === "error" ? 6000 : 3000);
}

function qs(name) {
  return new URLSearchParams(location.search).get(name);
}

/* trainer 상태: 모든 페이지에서 5초마다 갱신 */
const Health = {
  data: null,
  listeners: [],
  on(fn) { this.listeners.push(fn); if (this.data) fn(this.data); },
};

function renderBadge(h) {
  const b = $("#trainer-badge");
  if (!h || h.trainer.status !== "ok") {
    b.className = "badge danger";
    b.textContent = t("badge.down");
    b.title = (h && tdetail(h.trainer.error)) || t("badge.unreachable");
  } else {
    const tr = h.trainer;
    b.className = "badge " + (tr.current_job ? "warn" : "ok");
    b.textContent = t("badge.up", { device: deviceLabel(tr.device), state: tr.current_job ? t("badge.busy") : "idle" });
    b.title = t("badge.detail", { env: tr.versions.env, tf: tr.versions.tensorflow, keras: tr.versions.keras, mem: tr.memory.available_gb });
  }
  const ban = $("#global-banner");
  if (h && h.version_mismatch && h.version_mismatch.length) {
    const list = h.version_mismatch.map((m) => `<b>${esc(m.package)}</b> (web ${esc(m.web)} / trainer ${esc(m.trainer)})`).join(", ");
    ban.innerHTML = "⚠️ " + th("banner.version_mismatch_html").replace("{list}", list);
    ban.classList.remove("hidden");
  } else {
    ban.classList.add("hidden");
  }
}

async function pollHealth() {
  try {
    const h = await api("/api/health");
    Health.data = h;
    renderBadge(h);
    Health.listeners.forEach((f) => f(h));
  } catch (e) {
    renderBadge(null);
  }
  setTimeout(pollHealth, 5000);
}
document.addEventListener("DOMContentLoaded", pollHealth);

/* Job 상태를 2초마다 받아 onUpdate(job)를 호출. 끝나면 멈춘다. 반환값을 호출하면 중지. */
function pollJob(jobId, onUpdate, interval = 2000) {
  let stopped = false;
  async function tick() {
    if (stopped) return;
    try {
      const j = await api(`/api/jobs/${jobId}`);
      onUpdate(j);
      if (FINAL.includes(j.state)) return;
    } catch (e) {
      console.warn(e);
    }
    setTimeout(tick, interval);
  }
  tick();
  return () => { stopped = true; };
}

/* Job 한 줄 설명: 진행 단계(runtime.phase)가 있으면 그것, 없으면 상태 메시지 */
function jobPhase(j) {
  const rt = j.runtime || {};
  return (rt.phase_key || rt.phase) ? tf(rt, "phase") : tf(j, "message");
}

function jobProgress(j) {
  /* 학습 Job의 진행률(0~1)과 설명 */
  const rt = j.runtime || {};
  const epochs = rt.epochs || (j.params && j.params.epochs) || 0;
  const lastEp = (j.epochs || []).length;
  const b = j.last_batch;
  let frac = 0;
  if (epochs) {
    frac = lastEp / epochs;
    if (b && b.epoch > lastEp) frac = (b.epoch - 1 + b.batch / b.steps) / epochs;
  }
  return { frac: Math.min(1, frac), epochs, lastEp, batch: b };
}

function progressBar(frac, label = "") {
  return `<div class="progress"><div class="bar" style="width:${(frac * 100).toFixed(1)}%"></div><span>${esc(label)}</span></div>`;
}

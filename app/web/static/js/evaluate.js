let readyModels = [];
let current = null;           // 현재 표시 중인 평가 결과
let pcSort = { key: "index", dir: 1 };

const modelById = (id) => readyModels.find((m) => m.id === id);

async function loadModels() {
  readyModels = await api("/api/models?status=ready");
  const sel = $("#sel-model");
  if (!readyModels.length) {
    sel.innerHTML = `<option value="">${th("eval.no_ready")}</option>`;
    $("#btn-run").disabled = true;
    return;
  }
  sel.innerHTML = readyModels.map((m) => `<option value="${esc(m.id)}">${esc(m.name)} — ${esc(m.dataset)} · val ${fmt.pct(m.best && m.best.val_acc)}</option>`).join("");
  const want = qs("model");
  if (want && modelById(want)) sel.value = want;
}

async function onModelChange(preferEval) {
  const id = $("#sel-model").value;
  const m = modelById(id);
  if (!m) return;
  $("#eval-data").textContent = t("eval.data_desc", { dataset: m.dataset, n: 10000 });
  const evals = await api(`/api/models/${encodeURIComponent(id)}/evaluations`);
  const sel = $("#sel-eval");
  sel.innerHTML = evals.length
    ? evals.map((e) => `<option value="${esc(e.eval_id)}">${esc(e.eval_id)} · top-1 ${fmt.pct(e.top1)}</option>`).join("")
    : `<option value="">${th("eval.no_results")}</option>`;
  if (preferEval && evals.some((e) => e.eval_id === preferEval)) sel.value = preferEval;
  if (sel.value) await showEval(id, sel.value);
  else { $("#result").classList.add("hidden"); current = null; }
  const active = await api("/api/jobs?type=evaluate&state=queued,running&limit=20");
  const mine = active.find((j) => (j.params || {}).model_id === id);
  if (mine) watch(mine.id);
}

async function run() {
  const id = $("#sel-model").value;
  if (!id) return;
  try {
    const job = await api(`/api/models/${encodeURIComponent(id)}/evaluate`, { method: "POST", body: {} });
    toast(t("eval.toast_queued"), "ok");
    watch(job.id);
  } catch (e) { toast(e.message, "error"); }
}

function watch(jobId) {
  $("#btn-run").disabled = true;
  pollJob(jobId, async (j) => {
    $("#run-status").textContent = `${stateLabel(j.state)} · ${jobPhase(j)}`;
    if (!FINAL.includes(j.state)) return;
    $("#btn-run").disabled = false;
    if (j.state === "succeeded") {
      $("#run-status").textContent = stateLabel("succeeded");
      toast(t("eval.toast_done"), "ok");
      await onModelChange(j.eval_id || (j.result && j.result.eval_id));
    } else {
      toast(t("eval.toast_failed", { state: stateLabel(j.state), msg: tf(j, "message") }), "error");
    }
  }, 1500);
}

function classTable(rows, cols) {
  return `<table><tr>${cols.map((c) => `<th class="${c.num ? "num" : ""}">${c.label}</th>`).join("")}</tr>
    ${rows.map((r) => `<tr>${cols.map((c) => `<td class="${c.num ? "num" : ""}">${c.f(r)}</td>`).join("")}</tr>`).join("")}</table>`;
}

function drawPerClass() {
  if (!current) return;
  const q = $("#pc-filter").value.trim().toLowerCase();
  const rows = current.per_class.filter((r) => !q || r.name.includes(q))
    .sort((a, b) => (a[pcSort.key] > b[pcSort.key] ? 1 : a[pcSort.key] < b[pcSort.key] ? -1 : 0) * pcSort.dir);
  const cols = [["index", "#"], ["name", th("common.class")], ["support", th("eval.support")], ["accuracy", th("eval.acc_recall")], ["precision", "precision"], ["f1", "F1"]];
  $("#r-perclass").innerHTML = `<table><tr>${cols.map(([k, l]) => `<th class="sortable ${k === "name" ? "" : "num"}" data-k="${k}">${l}${pcSort.key === k ? (pcSort.dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("")}</tr>
    ${rows.map((r) => `<tr><td class="num">${r.index}</td><td>${esc(r.name)}</td><td class="num">${r.support}</td>
      <td class="num">${fmt.pct(r.accuracy)}</td><td class="num">${fmt.pct(r.precision)}</td><td class="num">${fmt.num(r.f1)}</td></tr>`).join("")}</table>`;
  $$("#r-perclass th.sortable").forEach((th) => th.addEventListener("click", () => {
    const k = th.dataset.k;
    pcSort = { key: k, dir: pcSort.key === k ? -pcSort.dir : k === "name" || k === "index" ? 1 : -1 };
    drawPerClass();
  }));
}

async function showEval(modelId, evalId) {
  const e = await api(`/api/models/${encodeURIComponent(modelId)}/evaluations/${encodeURIComponent(evalId)}`);
  current = e;
  history.replaceState(null, "", `/evaluate?model=${encodeURIComponent(modelId)}&eval=${encodeURIComponent(evalId)}`);
  $("#result").classList.remove("hidden");
  $("#r-stats").innerHTML = `
    <div class="stat"><div class="k">${th("eval.top1_acc")}</div><div class="v">${fmt.pct(e.top1, 2)}</div><div class="s">95% CI ${fmt.ci(e.top1_ci)}</div></div>
    <div class="stat"><div class="k">${th("eval.top5_acc")}</div><div class="v">${fmt.pct(e.top5, 2)}</div><div class="s">95% CI ${fmt.ci(e.top5_ci)}</div></div>
    <div class="stat"><div class="k">test loss</div><div class="v">${fmt.num(e.loss)}</div><div class="s">cross-entropy</div></div>
    <div class="stat"><div class="k">macro F1</div><div class="v">${fmt.num(e.macro_f1)}</div><div class="s">P ${fmt.num(e.macro_precision)} · R ${fmt.num(e.macro_recall)}</div></div>
    <div class="stat"><div class="k">ECE</div><div class="v">${fmt.num(e.ece)}</div><div class="s">${th("eval.ece_sub")}</div></div>
    <div class="stat"><div class="k">${th("eval.support")}</div><div class="v">${fmt.int(e.n)}</div><div class="s">${esc(e.dataset)} / ${esc(e.split)}</div></div>`;
  $("#r-meta").textContent = t("eval.meta", { id: e.eval_id, device: e.device, sec: e.predict_time_sec, date: fmt.date(e.created_at) });
  $("#r-cm").src = e.base_url + e.files.confusion_matrix;
  $("#r-rel").src = e.base_url + e.files.reliability;
  $("#r-confused").innerHTML = e.confused_pairs.length ? classTable(e.confused_pairs.slice(0, 15), [
    { label: th("eval.col_true"), f: (r) => esc(r.true) }, { label: th("eval.col_pred"), f: (r) => esc(r.pred) },
    { label: th("eval.col_count"), num: true, f: (r) => r.count }, { label: th("eval.col_rate"), num: true, f: (r) => fmt.pct(r.rate) },
  ]) : `<div class="empty">${th("eval.no_mis")}</div>`;
  const ccols = [{ label: th("common.class"), f: (r) => esc(r.name) }, { label: th("common.accuracy"), num: true, f: (r) => fmt.pct(r.accuracy) }, { label: "F1", num: true, f: (r) => fmt.num(r.f1) }];
  $("#r-best").innerHTML = classTable(e.best_classes, ccols);
  $("#r-worst").innerHTML = classTable(e.worst_classes, ccols);
  drawPerClass();
  $("#r-mis").innerHTML = e.misclassified.map((s) => `<figure><img src="${e.base_url + s.file}" alt="">
    <figcaption>${th("eval.cap_true")} <b>${esc(s.true)}</b><br>${th("eval.cap_pred")} <span style="color:var(--danger)">${esc(s.pred)}</span> ${fmt.pct(s.conf, 0)}</figcaption></figure>`).join("") || `<div class="empty">${th("common.none")}</div>`;
  $("#mc-a").innerHTML = `<b>${esc((modelById(modelId) || {}).name || modelId)}</b> <span class="small mono">${esc(evalId)}</span> · top-1 ${fmt.pct(e.top1)}`;
  await loadMcnemarOptions(modelId, evalId, e.dataset);
}

async function loadMcnemarOptions(aModel, aEval, dataset) {
  const opts = [];
  for (const m of readyModels.filter((x) => x.dataset === dataset && x.num_evaluations > 0)) {
    const evals = await api(`/api/models/${encodeURIComponent(m.id)}/evaluations`);
    for (const ev of evals) {
      if (m.id === aModel && ev.eval_id === aEval) continue;
      opts.push(`<option value="${esc(m.id)}|${esc(ev.eval_id)}">${esc(m.name)} · ${esc(ev.eval_id)} · top-1 ${fmt.pct(ev.top1)}</option>`);
    }
  }
  $("#mc-b").innerHTML = opts.join("") || `<option value="">${th("eval.mc_no_other")}</option>`;
  $("#btn-mcnemar").disabled = !opts.length;
  $("#mc-result").innerHTML = "";
}

async function runMcnemar() {
  const [bModel, bEval] = $("#mc-b").value.split("|");
  if (!bModel || !current) return;
  try {
    const p = new URLSearchParams({ a_model: current.model_id, a_eval: current.eval_id, b_model: bModel, b_eval: bEval });
    const r = await api(`/api/compare/mcnemar?${p}`);
    const diff = r.a.top1 - r.b.top1;
    $("#mc-result").innerHTML = `
      <div class="stats">
        <div class="stat"><div class="k">${th("eval.mc_diff")}</div><div class="v">${(diff * 100).toFixed(2)}%p</div><div class="s">A ${fmt.pct(r.a.top1, 2)} / B ${fmt.pct(r.b.top1, 2)}</div></div>
        <div class="stat"><div class="k">${th("eval.mc_only_a")}</div><div class="v">${fmt.int(r.only_a_correct)}</div></div>
        <div class="stat"><div class="k">${th("eval.mc_only_b")}</div><div class="v">${fmt.int(r.only_b_correct)}</div></div>
        <div class="stat"><div class="k">p-value</div><div class="v">${r.p_value < 1e-16 ? "&lt; 1e-16" : r.p_value < 0.0001 ? r.p_value.toExponential(2) : r.p_value.toFixed(4)}</div><div class="s">${esc(tf(r, "method"))}</div></div>
      </div>
      <p class="banner info" style="margin-top:12px">${r.significant_at_0_05
        ? th("eval.mc_sig_html") : th("eval.mc_nonsig_html")}
        ${th("eval.mc_summary", { both: r.both_correct, neither: r.both_wrong, stat: r.statistic.toFixed(3) })}</p>`;
  } catch (e) { toast(e.message, "error"); }
}

$("#sel-model").addEventListener("change", () => onModelChange().catch((e) => toast(e.message, "error")));
$("#sel-eval").addEventListener("change", () => showEval($("#sel-model").value, $("#sel-eval").value).catch((e) => toast(e.message, "error")));
$("#btn-run").addEventListener("click", run);
$("#btn-mcnemar").addEventListener("click", runMcnemar);
$("#pc-filter").addEventListener("input", drawPerClass);

(async function init() {
  await loadModels();
  if (!readyModels.length) return;
  await onModelChange(qs("eval"));
  if (qs("run") === "1") {
    history.replaceState(null, "", `/evaluate?model=${encodeURIComponent($("#sel-model").value)}`);
    if (!$("#btn-run").disabled) run();
  }
})().catch((e) => toast(e.message, "error"));

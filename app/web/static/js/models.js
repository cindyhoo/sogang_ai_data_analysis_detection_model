let models = [];
const selected = new Set();

function row(m) {
  const tags = (m.tags || []).map((t) => `<span class="pill tag">${esc(t)}</span>`).join(" ");
  const enc = encodeURIComponent(m.id);
  return `<tr>
    <td><input type="checkbox" data-sel="${esc(m.id)}" ${selected.has(m.id) ? "checked" : ""} ${m.status === "training" ? "disabled" : ""}></td>
    <td class="wrap"><a href="/models/${enc}"><b>${esc(m.name)}</b></a> ${m.is_default ? `<span class="pill tag">${th("common.default")}</span>` : ""} ${tags}
      <div class="small muted mono">${esc(m.id)}</div></td>
    <td>${esc(m.dataset)}</td>
    <td class="num">${m.hparams ? m.hparams.width : "–"}</td>
    <td class="num">${fmt.params(m.num_params)}</td>
    <td class="num">${m.epochs_completed ?? "–"}${m.hparams ? "/" + m.hparams.epochs : ""}</td>
    <td class="num">${fmt.pct(m.best && m.best.val_acc)}</td>
    <td class="num">${fmt.pct(m.latest_eval && m.latest_eval.top1)}</td>
    <td>${esc(m.device || "–")}</td>
    <td class="num">${m.avg_epoch_sec ? m.avg_epoch_sec.toFixed(1) + "s" : "–"}</td>
    <td>${stateBadge(m.status)}</td>
    <td>${fmt.date(m.created_at)}</td>
    <td><div class="actions">
      <a class="btn small" href="/models/${enc}">${th("models.detail")}</a>
      ${m.status === "ready" ? `<a class="btn small" href="/evaluate?model=${enc}">${th("nav.evaluate")}</a><a class="btn small" href="/test?model=${enc}">${th("nav.test")}</a>` : ""}
      ${m.status === "training" ? "" : `<button class="btn small danger" data-del="${esc(m.id)}">${th("common.delete")}</button>`}
    </div></td>
  </tr>`;
}

function draw() {
  const st = $("#filter-status").value;
  const list = st ? models.filter((m) => m.status === st) : models;
  $("#model-table").innerHTML = list.length ? `<table>
    <tr><th></th><th>${th("common.name")}</th><th>${th("common.dataset")}</th><th class="num">width</th><th class="num">${th("common.params")}</th><th class="num">epoch</th>
      <th class="num">best val acc</th><th class="num">test top-1</th><th>${th("common.device")}</th><th class="num">${th("models.per_epoch")}</th><th>${th("common.status")}</th><th>${th("common.created")}</th><th></th></tr>
    ${list.map(row).join("")}</table>` : `<div class="empty">${th("models.empty")} <a href="/train">${th("common.start_training_link")}</a></div>`;
  $$("[data-sel]").forEach((c) => c.addEventListener("change", () => {
    if (c.checked) {
      if (selected.size >= 3) { c.checked = false; return toast(t("models.max3")); }
      selected.add(c.dataset.sel);
    } else selected.delete(c.dataset.sel);
    $("#btn-compare").disabled = selected.size < 2;
  }));
  $$("[data-del]").forEach((b) => b.addEventListener("click", () => del(b.dataset.del)));
}

async function del(id) {
  const m = models.find((x) => x.id === id);
  if (!confirm(t("common.confirm_delete", { name: m ? m.name : id }))) return;
  try {
    await api(`/api/models/${encodeURIComponent(id)}`, { method: "DELETE" });
    selected.delete(id);
    toast(t("common.toast_deleted"), "ok");
    load();
  } catch (e) { toast(e.message, "error"); }
}

async function compare() {
  const ids = [...selected];
  const details = await Promise.all(ids.map((id) => api(`/api/models/${encodeURIComponent(id)}`)));
  $("#compare-card").classList.remove("hidden");
  const rows = [
    [th("common.dataset"), (d) => esc(d.meta.dataset)],
    ["width", (d) => d.meta.hparams.width],
    [th("common.params"), (d) => fmt.params(d.meta.num_params)],
    [th("models.epochs_done"), (d) => `${d.meta.epochs_completed}/${d.meta.hparams.epochs}`],
    [th("common.augment"), (d) => esc((d.meta.hparams.augment || []).join(", ") || t("common.none"))],
    ["learning rate", (d) => d.meta.hparams.lr],
    ["train subset", (d) => d.meta.hparams.train_subset],
    ["best val acc", (d) => fmt.pct(d.meta.best && d.meta.best.val_acc)],
    ["best val top-5", (d) => fmt.pct(d.meta.best && d.meta.best.val_top5)],
    [th("models.test_top1_latest"), (d) => d.meta.latest_eval ? `${fmt.pct(d.meta.latest_eval.top1)} <span class="muted small">${fmt.ci(d.meta.latest_eval.top1_ci)}</span>` : "–"],
    ["test top-5", (d) => fmt.pct(d.meta.latest_eval && d.meta.latest_eval.top5)],
    ["macro F1", (d) => fmt.num(d.meta.latest_eval && d.meta.latest_eval.macro_f1)],
    ["ECE", (d) => fmt.num(d.meta.latest_eval && d.meta.latest_eval.ece)],
    [th("models.device_epoch"), (d) => `${esc(d.meta.device)} · ${d.meta.avg_epoch_sec ? d.meta.avg_epoch_sec.toFixed(1) + "s" : "–"}`],
    [th("common.train_time"), (d) => fmt.dur(d.meta.train_time_sec)],
  ];
  $("#compare-table").innerHTML = `<table><tr><th></th>${details.map((d) => `<th><a href="/models/${encodeURIComponent(d.meta.id)}">${esc(d.meta.name)}</a></th>`).join("")}</tr>
    ${rows.map(([k, f]) => `<tr><td class="muted">${k}</td>${details.map((d) => `<td>${f(d)}</td>`).join("")}</tr>`).join("")}</table>
    <p class="small muted">${th("models.compare_note_html")}</p>`;
  lineChart($("#compare-acc"), { series: details.map((d) => ({ name: d.meta.name, values: d.history.val_acc || [] })), yFormat: (v) => (v * 100).toFixed(0) + "%" });
  lineChart($("#compare-loss"), { series: details.map((d) => ({ name: d.meta.name, values: d.history.val_loss || [] })) });
  $("#compare-card").scrollIntoView({ behavior: "smooth" });
}

async function load() {
  models = await api("/api/models");
  draw();
  $("#btn-compare").disabled = selected.size < 2;
}

$("#filter-status").addEventListener("change", draw);
$("#btn-compare").addEventListener("click", () => compare().catch((e) => toast(e.message, "error")));
$("#btn-compare-close").addEventListener("click", () => $("#compare-card").classList.add("hidden"));
load().catch((e) => toast(e.message, "error"));
setInterval(() => { if (models.some((m) => m.status === "training")) load(); }, 5000);

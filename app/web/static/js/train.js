const DEFAULTS = {
  name: "", dataset: "cifar100_fine", width: "1.0", epochs: 25, batch_size: "128", lr: 0.0003, dropout: 0.25,
  augment: ["flip", "rotation"], val_split: 0.1, train_subset: 1.0, patience: 5, seed: 1337, device: "auto",
};
const DEMO = { width: "0.5", train_subset: 0.2, epochs: 3, patience: 0 };
const form = $("#train-form");
let stopPoll = null;
let currentJob = null;

function readForm() {
  const f = new FormData(form);
  return {
    name: f.get("name").trim(),
    dataset: f.get("dataset"),
    arch: "xception_mini",
    width: parseFloat(f.get("width")),
    epochs: parseInt(f.get("epochs"), 10),
    batch_size: parseInt(f.get("batch_size"), 10),
    lr: parseFloat(f.get("lr")),
    dropout: parseFloat(f.get("dropout")),
    augment: f.getAll("augment"),
    val_split: parseFloat(f.get("val_split")),
    train_subset: parseFloat(f.get("train_subset")),
    patience: parseInt(f.get("patience"), 10),
    seed: parseInt(f.get("seed"), 10),
    device: f.get("device"),
  };
}

function fillForm(p) {
  for (const [k, v] of Object.entries(p)) {
    if (k === "augment") {
      $$('input[name="augment"]', form).forEach((c) => { c.checked = v.includes(c.value); });
    } else if (k === "width") {
      $$('input[name="width"]', form).forEach((r) => { r.checked = parseFloat(r.value) === parseFloat(v); });
    } else if (form.elements[k]) {
      form.elements[k].value = v;
    }
  }
}

async function start() {
  const btn = $("#btn-start");
  btn.disabled = true;
  try {
    const job = await api("/api/train", { method: "POST", body: readForm() });
    toast(t("train.toast_queued"), "ok");
    history.replaceState(null, "", `/train?job=${job.id}`);
    attach(job.id);
  } catch (e) {
    toast(e.message, "error");
  } finally {
    btn.disabled = false;
  }
}

function attach(jobId) {
  if (stopPoll) stopPoll();
  currentJob = jobId;
  $("#p-empty").classList.add("hidden");
  $("#p-body").classList.remove("hidden");
  stopPoll = pollJob(jobId, render);
}

function render(j) {
  const rt = j.runtime || {};
  const p = jobProgress(j);
  const active = !FINAL.includes(j.state);
  $("#p-title").textContent = active ? t("train.progress") : t("train.recent_result");
  $("#p-state").innerHTML = stateBadge(j.state);
  $("#btn-cancel").classList.toggle("hidden", !active || j.cancel_requested);
  const modelLink = rt.model_id ? ` · ${th("md.model")} <a href="/models/${encodeURIComponent(rt.model_id)}">${esc(rt.model_id)}</a>` : "";
  $("#p-meta").innerHTML = `Job <span class="mono">${esc(j.id)}</span> · ${esc(rt.device || t("train.device_checking"))} · ${esc(jobPhase(j))}${modelLink}` +
    (j.state === "failed" ? `<div style="color:var(--danger);margin-top:4px">${esc(tf(j, "message"))}</div>` : "") +
    (j.cancel_requested && active ? `<div style="color:var(--warn);margin-top:4px">${th("train.cancel_pending")}</div>` : "");
  const b = p.batch;
  const label = p.epochs ? `epoch ${p.lastEp}/${p.epochs}` + (b && active && b.epoch > p.lastEp ? ` · batch ${b.batch}/${b.steps}` : "") : (jobPhase(j) || stateLabel("queued"));
  $("#p-bar").innerHTML = progressBar(j.state === "succeeded" ? 1 : p.frac, label);
  const eps = j.epochs || [];
  const last = eps[eps.length - 1] || {};
  const best = eps.reduce((a, e) => (a == null || (e.val_acc ?? -1) > (a.val_acc ?? -1) ? e : a), null);
  const avg = eps.length ? eps.reduce((s, e) => s + (e.epoch_time || 0), 0) / eps.length : null;
  $("#p-stats").innerHTML = `
    <div class="stat"><div class="k">${th("train.val_acc_last")}</div><div class="v">${fmt.pct(last.val_acc)}</div><div class="s">top-5 ${fmt.pct(last.val_top5)}</div></div>
    <div class="stat"><div class="k">best val acc</div><div class="v">${fmt.pct(best && best.val_acc)}</div><div class="s">${best ? `epoch ${best.epoch}` : ""}</div></div>
    <div class="stat"><div class="k">${th("train.epoch_time")}</div><div class="v">${avg ? avg.toFixed(1) + "s" : "–"}</div><div class="s">${esc(rt.device || "")}</div></div>
    <div class="stat"><div class="k">${active ? th("train.eta") : th("train.elapsed")}</div><div class="v">${active ? fmt.dur(last.eta_sec) : fmt.dur(last.elapsed)}</div><div class="s">train ${fmt.int(rt.train_size)} · val ${fmt.int(rt.val_size)}</div></div>`;
  lineChart($("#chart-loss"), {
    series: [{ name: "loss", values: eps.map((e) => e.loss) }, { name: "val_loss", values: eps.map((e) => e.val_loss), dashed: true }],
    yFormat: (v) => v.toFixed(2),
  });
  lineChart($("#chart-acc"), {
    series: [
      { name: "acc", values: eps.map((e) => e.acc) },
      { name: "val_acc", values: eps.map((e) => e.val_acc), dashed: true },
      { name: "top5", values: eps.map((e) => e.top5), color: "var(--c3)" },
      { name: "val_top5", values: eps.map((e) => e.val_top5), color: "var(--c3)", dashed: true },
    ],
    yFormat: (v) => (v * 100).toFixed(0) + "%",
  });
  const log = $("#p-log");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 10;
  log.textContent = (j.log_tail || []).join("\n") || t("train.no_log");
  if (atBottom) log.scrollTop = log.scrollHeight;
  // 이 화면에서 실행 중인 상태를 본 Job이 끝났을 때만 알린다
  render.seenActive = render.seenActive || {};
  if (active) render.seenActive[j.id] = true;
  else if (render.seenActive[j.id]) {
    delete render.seenActive[j.id];
    if (j.state === "succeeded") toast(t("train.toast_done"), "ok");
    else if (j.state === "failed") toast(t("train.toast_failed", { msg: tf(j, "message") }), "error");
    else toast(t("train.toast_cancelled"));
  }
}

$("#btn-start").addEventListener("click", start);
$("#btn-demo").addEventListener("click", () => { fillForm(DEMO); toast(t("train.toast_demo")); });
$("#btn-reset").addEventListener("click", () => fillForm(DEFAULTS));
$("#btn-cancel").addEventListener("click", async () => {
  if (!currentJob || !confirm(t("train.confirm_cancel"))) return;
  try { await api(`/api/jobs/${currentJob}/cancel`, { method: "POST" }); toast(t("train.toast_cancel_req")); }
  catch (e) { toast(e.message, "error"); }
});

(async function init() {
  const ds = await api("/api/datasets");
  $("#f-dataset").innerHTML = ds.map((d) => `<option value="${esc(d.id)}">${esc(datasetName(d))}${d.prepared ? "" : " " + th("train.not_prepared_opt")}</option>`).join("");
  fillForm(DEFAULTS);
  if (qs("dataset")) $("#f-dataset").value = qs("dataset");
  const jobParam = qs("job");
  if (jobParam) return attach(jobParam);
  const active = await api("/api/jobs?type=train&state=queued,running&limit=1");
  if (active.length) return attach(active[0].id);
  const recent = await api("/api/jobs?type=train&limit=1");
  if (recent.length) attach(recent[0].id);
})().catch((e) => toast(e.message, "error"));

const ENC = encodeURIComponent(MODEL_ID);
let detail = null;

function kv(pairs) {
  return pairs.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

function render(d) {
  detail = d;
  const m = d.meta;
  document.title = `${m.name} · CIFAR-100 CNN PoC`;
  $("#m-name").innerHTML = `${esc(m.name)} ${stateBadge(m.status)} ${m.is_default ? `<span class="pill tag">${th("common.default")}</span>` : ""}`;
  const ready = m.status === "ready";
  $("#m-actions").innerHTML = `
    ${ready ? `<a class="btn primary" href="/evaluate?model=${ENC}&run=1">${th("common.run_eval")}</a><a class="btn" href="/test?model=${ENC}">${th("common.webcam_test")}</a>` : ""}
    ${m.has_file ? `<a class="btn" href="/api/models/${ENC}/download">${th("md.download")}</a>` : ""}
    ${m.status === "training" ? `<a class="btn" href="/train?job=${encodeURIComponent(m.job_id)}">${th("md.view_progress")}</a>` : `<button class="btn danger" id="btn-del">${th("common.delete")}</button>`}`;
  const del = $("#btn-del");
  if (del) del.addEventListener("click", remove);
  if (document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") {
    $("#e-name").value = m.name || "";
    $("#e-tags").value = (m.tags || []).join(", ");
    $("#e-memo").value = m.memo || "";
    $("#e-default").checked = !!m.is_default;
  }
  $("#e-default").disabled = !ready;
  const env = m.train_env || {};
  $("#m-info").innerHTML = kv([
    [th("common.dataset"), esc(m.dataset) + " " + th("md.n_classes", { n: m.num_classes })],
    [th("common.arch"), `${esc(m.arch)} · width ${m.hparams.width}`],
    [th("common.params"), fmt.int(m.num_params)],
    [th("md.data"), `train ${fmt.int(m.data && m.data.train)} · val ${fmt.int(m.data && m.data.val)}`],
    ["epoch", `${m.epochs_completed ?? 0} / ${m.hparams.epochs}${m.early_stopped ? " (early stopping)" : ""}`],
    ["best", m.best ? `epoch ${m.best.epoch} · val acc ${fmt.pct(m.best.val_acc)} · val top-5 ${fmt.pct(m.best.val_top5)}` : "–"],
    [th("common.train_time"), `${fmt.dur(m.train_time_sec)}${m.avg_epoch_sec ? ` · ${th("md.per_epoch", { s: m.avg_epoch_sec.toFixed(1) })}` : ""}`],
    [th("common.device"), esc(m.device || "–")],
    [th("md.train_env"), esc(`${env.env || "?"} · ${env.platform || ""} · Python ${env.python || ""}`)],
    [th("md.versions"), esc(`TF ${env.tensorflow || "?"}${env.tensorflow_metal ? " + metal " + env.tensorflow_metal : ""} · Keras ${env.keras || "?"} · NumPy ${env.numpy || "?"}`)],
    [th("md.created_finished"), `${fmt.date(m.created_at)} / ${fmt.date(m.finished_at)}`],
    ...(m.error ? [[th("md.error"), `<span style="color:var(--danger)">${esc(tf(m, "error"))}</span>`]] : []),
  ]);
  const hp = m.hparams || {};
  $("#m-hparams").innerHTML = kv([
    ["epochs", hp.epochs], ["batch size", hp.batch_size], ["learning rate", hp.lr], ["dropout", hp.dropout],
    [th("common.augment"), esc((hp.augment || []).join(", ") || t("common.none"))], [th("common.val_split"), hp.val_split],
    ["train subset", hp.train_subset], ["early stopping patience", hp.patience], ["seed", hp.seed],
  ]);
  const h = d.history || {};
  lineChart($("#chart-loss"), {
    series: [{ name: "loss", values: h.loss || [] }, { name: "val_loss", values: h.val_loss || [], dashed: true }],
  });
  lineChart($("#chart-acc"), {
    series: [
      { name: "acc", values: h.acc || [] }, { name: "val_acc", values: h.val_acc || [], dashed: true },
      { name: "top5", values: h.top5 || [], color: "var(--c3)" }, { name: "val_top5", values: h.val_top5 || [], color: "var(--c3)", dashed: true },
    ],
    yFormat: (v) => (v * 100).toFixed(0) + "%",
  });
  $("#m-evals").innerHTML = d.evaluations.length ? `<table>
    <tr><th>${th("md.eval_id")}</th><th>${th("md.data")}</th><th class="num">top-1 (95% CI)</th><th class="num">top-5</th><th class="num">macro F1</th><th class="num">ECE</th><th>${th("common.device")}</th><th></th></tr>
    ${d.evaluations.map((e) => `<tr><td class="mono">${esc(e.eval_id)}</td><td>${esc(e.dataset)} / ${esc(e.split)} (${fmt.int(e.n)})</td>
      <td class="num">${fmt.pct(e.top1)} <span class="muted small">${fmt.ci(e.top1_ci)}</span></td><td class="num">${fmt.pct(e.top5)}</td>
      <td class="num">${fmt.num(e.macro_f1)}</td><td class="num">${fmt.num(e.ece)}</td><td>${esc(e.device || "")}</td>
      <td><a class="btn small" href="/evaluate?model=${ENC}&eval=${encodeURIComponent(e.eval_id)}">${th("md.view_result")}</a></td></tr>`).join("")}
  </table>` : `<div class="empty">${th("md.no_evals")}${ready ? ` <a href="/evaluate?model=${ENC}&run=1">${th("common.run_eval")} →</a>` : ""}</div>`;
  $("#m-summary").textContent = d.summary || t("md.no_summary");
}

async function load() {
  render(await api(`/api/models/${ENC}`));
  if (detail.meta.status === "training") setTimeout(load, 4000);
}

async function save() {
  const tags = $("#e-tags").value.split(",").map((t) => t.trim()).filter(Boolean);
  try {
    await api(`/api/models/${ENC}`, {
      method: "PATCH",
      body: { name: $("#e-name").value.trim() || null, memo: $("#e-memo").value, tags, is_default: $("#e-default").checked },
    });
    toast(t("common.toast_saved"), "ok");
    $("#btn-save").blur();
    load();
  } catch (e) { toast(e.message, "error"); }
}

async function remove() {
  if (!confirm(t("common.confirm_delete", { name: detail.meta.name }))) return;
  try {
    await api(`/api/models/${ENC}`, { method: "DELETE" });
    location.href = "/models";
  } catch (e) { toast(e.message, "error"); }
}

$("#btn-save").addEventListener("click", save);
load().catch((e) => toast(e.message, "error"));

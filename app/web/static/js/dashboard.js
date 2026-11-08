function envRow(name, v) {
  return `<tr><td><b>${name}</b></td><td>${esc(v.env)}</td><td>${esc(v.platform)}</td><td>${esc(v.python)}</td>
    <td>${esc(v.tensorflow)}${v.tensorflow_metal ? ` <span class="muted small">+metal ${esc(v.tensorflow_metal)}</span>` : ""}</td>
    <td>${esc(v.keras)}</td><td>${esc(v.numpy)}</td></tr>`;
}

function renderSys(h) {
  const tr = h.trainer, w = h.web, up = tr.status === "ok";
  $("#sys").classList.remove("muted");
  $("#sys").innerHTML = `
    <div class="stats">
      <div class="stat"><div class="k">trainer</div><div class="v">${up ? th("dash.connected") : th("dash.stopped")}</div><div class="s mono">${esc(w.trainer_url)}</div></div>
      <div class="stat"><div class="k">${th("common.train_device")}</div><div class="v">${up ? esc(deviceLabel(tr.device)) : "–"}</div><div class="s">${th("dash.device_pref", { pref: up ? tr.device_pref : "–" })}</div></div>
      <div class="stat"><div class="k">${th("dash.free_mem")}</div><div class="v">${up ? tr.memory.available_gb + "GB" : "–"}</div><div class="s">${th("dash.mem_sub", { total: up ? tr.memory.total_gb : "–", min: up ? tr.memory.min_free_gb : "–" })}</div></div>
      <div class="stat"><div class="k">${th("dash.versions")}</div><div class="v">${!up ? "–" : h.version_mismatch.length ? "⚠️ " + th("dash.ver_diff") : th("dash.ver_same")}</div><div class="s">TensorFlow · Keras · NumPy</div></div>
    </div>
    <h3>${th("dash.env")}</h3>
    <div class="table-wrap"><table>
      <tr><th></th><th>${th("dash.env_col")}</th><th>${th("dash.platform")}</th><th>Python</th><th>TensorFlow</th><th>Keras</th><th>NumPy</th></tr>
      ${envRow("web", w.versions)}
      ${up ? envRow("trainer", tr.versions) : `<tr><td><b>trainer</b></td><td colspan="6" class="wrap muted">${esc(tdetail(tr.error))}</td></tr>`}
    </table></div>
    ${w.loaded_models.length ? `<p class="small muted">${th("dash.loaded", { list: w.loaded_models.join(", ") })}</p>` : ""}`;
}

async function renderActive() {
  try {
    const list = await api("/api/jobs?state=queued,running&limit=10");
    if (!list.length) {
      $("#active-jobs").innerHTML = `<div class="empty">${th("dash.no_active")}</div>`;
    } else {
      const details = await Promise.all(list.map((j) => api(`/api/jobs/${j.id}`).catch(() => j)));
      $("#active-jobs").classList.remove("muted");
      $("#active-jobs").innerHTML = details.map((j) => {
        const rt = j.runtime || {};
        let bar = "";
        if (j.type === "train" && j.state === "running") {
          const p = jobProgress(j);
          bar = progressBar(p.frac, `epoch ${p.lastEp}/${p.epochs || "?"}`);
        }
        const href = j.type === "train" ? `/train?job=${j.id}` : j.type === "evaluate" ? `/evaluate?model=${encodeURIComponent((j.params || {}).model_id || "")}` : "/datasets";
        return `<div style="margin-bottom:12px">
          <div class="row-between"><div><b>${esc(jobLabel(j.type))}</b> ${stateBadge(j.state)} <span class="muted small mono">${esc(j.id)}</span></div><a class="small" href="${href}">${th("common.open")}</a></div>
          <div class="small muted">${esc(jobPhase(j))}${rt.device ? " · " + esc(rt.device) : ""}</div>${bar}</div>`;
      }).join("");
    }
  } catch (e) {
    $("#active-jobs").textContent = e.message;
  }
  setTimeout(renderActive, 3000);
}

async function renderRecent() {
  const [models, jobs] = await Promise.all([api("/api/models"), api("/api/jobs?limit=8")]);
  $("#recent-models").innerHTML = models.length ? `<table>
    <tr><th>${th("common.name")}</th><th>${th("common.dataset")}</th><th class="num">width</th><th class="num">best val acc</th><th class="num">test top-1</th><th>${th("common.device")}</th><th>${th("common.status")}</th><th>${th("common.created")}</th></tr>
    ${models.slice(0, 5).map((m) => `<tr>
      <td><a href="/models/${encodeURIComponent(m.id)}">${esc(m.name)}</a>${m.is_default ? ` <span class="pill tag">${th("common.default")}</span>` : ""}</td>
      <td>${esc(m.dataset)}</td><td class="num">${m.hparams ? m.hparams.width : "–"}</td>
      <td class="num">${fmt.pct(m.best && m.best.val_acc)}</td><td class="num">${fmt.pct(m.latest_eval && m.latest_eval.top1)}</td>
      <td>${esc(m.device || "–")}</td><td>${stateBadge(m.status)}</td><td>${fmt.date(m.created_at)}</td></tr>`).join("")}
  </table>` : `<div class="empty">${th("dash.no_models")} <a href="/train">${th("common.start_training_link")}</a></div>`;
  $("#recent-jobs").innerHTML = jobs.length ? `<table>
    <tr><th>${th("dash.col_job")}</th><th>${th("common.status")}</th><th>${th("dash.col_message")}</th><th>${th("common.device")}</th><th>${th("common.created")}</th><th>${th("dash.col_finished")}</th></tr>
    ${jobs.map((j) => `<tr><td>${esc(jobLabel(j.type))} <span class="muted small mono">${esc(j.id)}</span></td>
      <td>${stateBadge(j.state)}</td><td class="wrap small">${esc(tf(j, "message"))}</td><td>${esc(j.device || (j.runtime || {}).device || "–")}</td>
      <td>${fmt.date(j.created_at)}</td><td>${fmt.date(j.finished_at)}</td></tr>`).join("")}
  </table>` : `<div class="empty">${th("dash.no_jobs")}</div>`;
}

Health.on(renderSys);
renderActive();
renderRecent().catch((e) => toast(e.message, "error"));

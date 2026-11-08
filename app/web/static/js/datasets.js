function dsCard(d) {
  const counts = d.prepared ? `
    <div class="stat"><div class="k">train / test</div><div class="v">${fmt.int(d.train_size)} / ${fmt.int(d.test_size)}</div><div class="s">${th("ds.val_from_train")}</div></div>
    <div class="stat"><div class="k">${th("ds.pixel_mean")}</div><div class="v small">${(d.pixel_mean || []).join(", ")}</div><div class="s">${th("ds.pixel_std", { v: (d.pixel_std || []).join(", ") })}</div></div>` : "";
  const desc = hasMsg(`dataset.${d.id}.description`) ? t(`dataset.${d.id}.description`) : d.description;
  return `<section class="card" data-id="${esc(d.id)}">
    <div class="row-between">
      <div><h2 style="margin-bottom:2px">${esc(datasetName(d))}</h2><div class="muted">${esc(desc)}</div></div>
      <div>${d.prepared ? `<span class="pill s-ready">${th("ds.prepared")}</span> <span class="small muted">${fmt.date(d.prepared_at)}</span>` : `<span class="pill">${th("ds.not_prepared")}</span>`}</div>
    </div>
    <div class="stats" style="margin:14px 0">
      <div class="stat"><div class="k">${th("ds.num_classes")}</div><div class="v">${d.num_classes}</div><div class="s">label_mode="${esc(d.label_mode)}"</div></div>
      <div class="stat"><div class="k">${th("ds.input_size")}</div><div class="v">${d.input_shape.join("×")}</div><div class="s">uint8, 0~255</div></div>
      ${counts}
    </div>
    <div class="actions">
      <button class="btn ${d.prepared ? "" : "primary"}" data-prepare>${d.prepared ? th("ds.reprepare") : th("ds.prepare")}</button>
      <a class="btn" href="/train?dataset=${encodeURIComponent(d.id)}">${th("ds.train_with")}</a>
      <span class="job-status small muted"></span>
    </div>
    ${d.prepared ? `<h3>${th("ds.classes_title", { n: d.samples_per_class })}</h3>
      <input type="text" class="class-filter" placeholder="${esc(t("ds.class_search"))}" style="max-width:320px;margin-bottom:10px">
      <div class="table-wrap classes" style="max-height:520px;overflow:auto"></div>` : ""}
  </section>`;
}

async function renderClasses(card, id) {
  const d = await api(`/api/datasets/${id}`);
  const rows = d.class_names.map((name, i) => ({ i, name, tr: d.class_counts_train[i], te: d.class_counts_test[i] }));
  const draw = (q) => {
    const f = rows.filter((r) => !q || r.name.includes(q.toLowerCase()));
    card.querySelector(".classes").innerHTML = `<table>
      <tr><th class="num">#</th><th>${th("common.class")}</th><th>${th("ds.samples")}</th><th class="num">train</th><th class="num">test</th></tr>
      ${f.map((r) => `<tr><td class="num">${r.i}</td><td>${esc(r.name)}</td>
        <td><img class="strip" loading="lazy" src="/files/datasets/${id}/samples/${String(r.i).padStart(3, "0")}.png" alt="${esc(r.name)}"></td>
        <td class="num">${fmt.int(r.tr)}</td><td class="num">${fmt.int(r.te)}</td></tr>`).join("")}
    </table>`;
  };
  draw("");
  card.querySelector(".class-filter").addEventListener("input", (e) => draw(e.target.value.trim()));
}

function watchJob(card, jobId) {
  const btn = card.querySelector("[data-prepare]");
  const st = card.querySelector(".job-status");
  btn.disabled = true;
  pollJob(jobId, (j) => {
    st.textContent = `${stateLabel(j.state)} · ${jobPhase(j)}`;
    if (j.state === "succeeded") { toast(t("ds.toast_done"), "ok"); load(); }
    else if (FINAL.includes(j.state)) { btn.disabled = false; toast(t("ds.toast_failed", { msg: tf(j, "message") }), "error"); }
  });
}

async function load() {
  const list = await api("/api/datasets");
  $("#ds-list").innerHTML = list.map(dsCard).join("");
  list.forEach((d) => {
    const card = $(`section[data-id="${d.id}"]`);
    card.querySelector("[data-prepare]").addEventListener("click", async () => {
      try {
        const job = await api(`/api/datasets/${d.id}/prepare`, { method: "POST" });
        toast(t("ds.toast_started"));
        watchJob(card, job.id);
      } catch (e) { toast(e.message, "error"); }
    });
    if (d.active_job) watchJob(card, d.active_job);
    if (d.prepared) renderClasses(card, d.id).catch((e) => toast(e.message, "error"));
  });
}

load().catch((e) => toast(e.message, "error"));

/* 웹캠 실시간 테스트 (제안서 §8.2)
 * getUserMedia → 중앙 ROI를 128×128 canvas에 그림 → JPEG로 /api/predict 전송.
 * 이전 응답을 받은 뒤에만 다음 프레임을 보낸다(backpressure). */
const video = $("#video");
const roi = $("#roi");
const canvas = document.createElement("canvas");
canvas.width = canvas.height = 128;
const ctx = canvas.getContext("2d");
const SMOOTH_N = 5;

const S = { stream: null, running: false, timer: null, models: [], history: [], stamps: [], lastTop: null, snaps: 0 };

const ratio = () => parseInt($("#roi-ratio").value, 10) / 100;
const currentModel = () => S.models.find((m) => m.id === $("#sel-model").value);

function roiRect() {
  const vw = video.videoWidth || 640, vh = video.videoHeight || 480;
  const side = Math.min(vw, vh) * ratio();
  return { sx: (vw - side) / 2, sy: (vh - side) / 2, side, vw, vh };
}

function positionRoi() {
  const r = roiRect();
  Object.assign(roi.style, {
    left: (r.sx / r.vw) * 100 + "%", top: (r.sy / r.vh) * 100 + "%",
    width: (r.side / r.vw) * 100 + "%", height: (r.side / r.vh) * 100 + "%",
  });
}

function canvasToJpeg(c, quality = 0.85) {
  /* canvas.toBlob은 브라우저에 따라 인코딩을 유휴 시간으로 미뤄 프레임당 0.5~1초가 걸린다.
     128×128 프레임은 동기식 toDataURL이 1ms 안팎이라 이쪽을 쓴다. */
  const b64 = c.toDataURL("image/jpeg", quality).split(",")[1];
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: "image/jpeg" });
}

function camError(msg) {
  const el = $("#cam-error");
  el.innerHTML = msg;
  el.classList.remove("hidden");
}

async function startCam() {
  $("#cam-error").classList.add("hidden");
  if (!currentModel()) return toast(t("test.pick_model"), "error");
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return camError(th("test.err_unsupported_html"));
  }
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
  } catch (e) {
    const denied = e.name === "NotAllowedError" || e.name === "SecurityError";
    return camError(denied
      ? th("test.err_denied_html")
      : th("test.err_open", { err: `${e.name} ${e.message}` }));
  }
  video.srcObject = S.stream;
  await video.play();
  $("#cam-placeholder").classList.add("hidden");
  roi.classList.remove("hidden");
  positionRoi();
  S.running = true;
  S.history = [];
  S.stamps = [];
  $("#btn-start").disabled = true;
  $("#btn-stop").disabled = false;
  $("#btn-snap").disabled = false;
  loop();
}

function stopCam() {
  S.running = false;
  clearTimeout(S.timer);
  if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
  S.stream = null;
  video.srcObject = null;
  roi.classList.add("hidden");
  $("#cam-placeholder").classList.remove("hidden");
  $("#btn-start").disabled = false;
  $("#btn-stop").disabled = true;
  $("#btn-snap").disabled = true;
  $("#s-fps").textContent = "–";
}

async function loop() {
  if (!S.running) return;
  const m = currentModel();
  if (m && video.readyState >= 2) {
    const r = roiRect();
    ctx.drawImage(video, r.sx, r.sy, r.side, r.side, 0, 0, 128, 128);
    const blob = canvasToJpeg(canvas);
    const fd = new FormData();
    fd.append("model_id", m.id);
    fd.append("file", blob, "frame.jpg");
    fd.append("top_k", $("#opt-smooth").checked ? m.num_classes : 5);
    const t0 = performance.now();
    try {
      const res = await api("/api/predict", { method: "POST", form: fd });
      if (S.running && res.model_id === $("#sel-model").value) render(res, performance.now() - t0);
    } catch (e) {
      $("#pred-note").textContent = t("test.pred_failed", { msg: e.message });
    }
  }
  if (S.running) S.timer = setTimeout(loop, parseInt($("#sel-interval").value, 10));
}

function render(res, rtt) {
  $("#input-preview").src = `data:image/png;base64,${res.input_png}`;
  $("#s-latency").textContent = `${res.latency_ms.toFixed(0)}ms`;
  const now = performance.now();
  S.stamps.push(now);
  S.stamps = S.stamps.filter((t) => now - t < 4000);
  if (S.stamps.length > 1) $("#s-fps").textContent = `${((S.stamps.length - 1) / ((now - S.stamps[0]) / 1000)).toFixed(1)} fps`;
  $("#s-rtt").textContent = t("test.rtt", { ms: rtt.toFixed(0) });

  let top;
  const m = currentModel();
  if ($("#opt-smooth").checked && res.top.length > 5 && m) {
    const vec = new Float64Array(m.num_classes);
    const names = [];
    res.top.forEach((t) => { vec[t.index] = t.prob; names[t.index] = t.class; });
    S.history.push(vec);
    if (S.history.length > SMOOTH_N) S.history.shift();
    S.names = Object.assign(S.names || [], names);
    const avg = new Float64Array(m.num_classes);
    S.history.forEach((v) => v.forEach((p, i) => { avg[i] += p / S.history.length; }));
    top = [...avg].map((p, i) => ({ label: S.names[i], value: p })).sort((a, b) => b.value - a.value).slice(0, 5);
    $("#pred-note").textContent = t("test.note_avg", { n: S.history.length, dataset: m.dataset });
  } else {
    S.history = [];
    top = res.top.slice(0, 5).map((t) => ({ label: t.class, value: t.prob }));
    $("#pred-note").textContent = t("test.note_frame", { dataset: m ? m.dataset : "" });
  }
  S.lastTop = top;
  $("#top5").innerHTML = topBars(top);
}

function snapshot() {
  if (!S.stream) return;
  const r = roiRect();
  const c = document.createElement("canvas");
  c.width = c.height = 224;
  c.getContext("2d").drawImage(video, r.sx, r.sy, r.side, r.side, 0, 0, 224, 224);
  const url = c.toDataURL("image/png");
  const top = S.lastTop && S.lastTop[0];
  const label = top ? `${top.label} ${(top.value * 100).toFixed(1)}%` : t("test.no_pred");
  S.snaps += 1;
  const stamp = new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19);
  const fig = document.createElement("figure");
  fig.innerHTML = `<img src="${url}" alt="" style="image-rendering:auto"><figcaption><b>${esc(label)}</b><br>
    <a href="${url}" download="snapshot-${stamp}-${esc(top ? top.label : "none")}.png">${th("test.download")}</a></figcaption>`;
  const g = $("#snaps");
  if (S.snaps === 1) g.innerHTML = "";
  g.prepend(fig);
}

async function loadModels() {
  const [models, settings] = await Promise.all([api("/api/models?status=ready"), api("/api/settings")]);
  S.models = models;
  const sel = $("#sel-model");
  if (!models.length) {
    sel.innerHTML = `<option value="">${th("test.no_ready")}</option>`;
    $("#btn-start").disabled = true;
    return;
  }
  sel.innerHTML = models.map((m) => `<option value="${esc(m.id)}">${m.is_default ? "★ " : ""}${esc(m.name)} (${esc(m.dataset.replace("cifar100_", ""))}, val ${fmt.pct(m.best && m.best.val_acc)})</option>`).join("");
  const want = qs("model") || settings.default_model;
  if (want && models.some((m) => m.id === want)) sel.value = want;
}

$("#btn-start").addEventListener("click", startCam);
$("#btn-stop").addEventListener("click", stopCam);
$("#btn-snap").addEventListener("click", snapshot);
$("#sel-model").addEventListener("change", () => { S.history = []; S.names = []; });
$("#roi-ratio").addEventListener("input", () => { $("#roi-val").textContent = $("#roi-ratio").value + "%"; positionRoi(); });
$("#opt-mirror").addEventListener("change", (e) => video.classList.toggle("mirror", e.target.checked));
window.addEventListener("beforeunload", stopCam);
Health.on((h) => $("#train-banner").classList.toggle("hidden", !h.training_active));
loadModels().catch((e) => toast(e.message, "error"));

/* 외부 라이브러리 없이 SVG로 그리는 간단한 선 그래프 */
const SERIES_COLORS = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)"];

function lineChart(el, { series, height = 220, yFormat = (v) => v.toFixed(2), yMin = null, yMax = null, xLabel = "epoch" }) {
  const W = 560, H = height, pad = { l: 46, r: 14, t: 12, b: 30 };
  const all = series.flatMap((s) => (s.values || []).filter((v) => v != null && isFinite(v)));
  if (!all.length) {
    el.innerHTML = `<div class="chart-empty">${esc(t("chart.empty"))}</div>`;
    return;
  }
  const n = Math.max(...series.map((s) => (s.values || []).length));
  let lo = yMin ?? Math.min(...all);
  let hi = yMax ?? Math.max(...all);
  if (hi - lo < 1e-9) hi = lo + 1;
  const span = hi - lo;
  if (yMin == null) lo -= span * 0.06;
  if (yMax == null) hi += span * 0.06;
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const x = (i) => pad.l + (n <= 1 ? iw / 2 : (i * iw) / (n - 1));
  const y = (v) => pad.t + ((hi - v) * ih) / (hi - lo);

  let svg = `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" preserveAspectRatio="xMidYMid meet">`;
  for (let k = 0; k <= 4; k++) {
    const v = lo + ((hi - lo) * k) / 4;
    svg += `<line class="grid" x1="${pad.l}" x2="${W - pad.r}" y1="${y(v)}" y2="${y(v)}"/>`;
    svg += `<text class="tick" x="${pad.l - 6}" y="${y(v) + 4}" text-anchor="end">${yFormat(v)}</text>`;
  }
  const step = Math.max(1, Math.ceil(n / 8));
  for (let i = 0; i < n; i += step) {
    svg += `<text class="tick" x="${x(i)}" y="${H - pad.b + 16}" text-anchor="middle">${i + 1}</text>`;
  }
  svg += `<text class="axis" x="${W - pad.r}" y="${H - 4}" text-anchor="end">${xLabel}</text>`;
  series.forEach((s, si) => {
    const color = s.color || SERIES_COLORS[si % SERIES_COLORS.length];
    let d = "", pen = false, last = null;
    (s.values || []).forEach((v, i) => {
      if (v == null || !isFinite(v)) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
      last = [x(i), y(v)];
    });
    svg += `<path d="${d}" fill="none" stroke="${color}" stroke-width="2" ${s.dashed ? 'stroke-dasharray="5 4"' : ""}/>`;
    if (n <= 40) {
      (s.values || []).forEach((v, i) => {
        if (v != null && isFinite(v)) svg += `<circle cx="${x(i)}" cy="${y(v)}" r="2.6" fill="${color}"><title>${esc(s.name)} · ${i + 1}: ${yFormat(v)}</title></circle>`;
      });
    } else if (last) {
      svg += `<circle cx="${last[0]}" cy="${last[1]}" r="3" fill="${color}"/>`;
    }
  });
  svg += `</svg>`;
  const legend = series.map((s, si) => {
    const color = s.color || SERIES_COLORS[si % SERIES_COLORS.length];
    const lastV = (s.values || []).filter((v) => v != null).slice(-1)[0];
    return `<span class="lg"><i style="background:${color}"></i>${esc(s.name)}${lastV != null ? ` <b>${yFormat(lastV)}</b>` : ""}</span>`;
  }).join("");
  el.innerHTML = svg + `<div class="legend">${legend}</div>`;
}

function topBars(items, { max = 5 } = {}) {
  /* items: [{label, value(0~1)}] → 가로 막대 목록 HTML */
  return items.slice(0, max).map((it, i) => `
    <div class="tb-row ${i === 0 ? "first" : ""}">
      <span class="tb-rank">${i + 1}</span>
      <span class="tb-label">${esc(it.label)}</span>
      <span class="tb-bar"><i style="width:${Math.max(0.5, it.value * 100).toFixed(1)}%"></i></span>
      <span class="tb-val">${(it.value * 100).toFixed(1)}%</span>
    </div>`).join("");
}

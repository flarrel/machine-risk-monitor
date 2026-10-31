const form = document.querySelector("#prediction-form");
const result = document.querySelector("#result");
const historyBody = document.querySelector("#history-body");
const chart = document.querySelector("#history-chart");

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function payload(formData) {
  return {
    product_type: formData.get("product_type"),
    air_temperature_k: Number(formData.get("air_temperature_k")),
    process_temperature_k: Number(formData.get("process_temperature_k")),
    rotational_speed_rpm: Number(formData.get("rotational_speed_rpm")),
    torque_nm: Number(formData.get("torque_nm")),
    tool_wear_min: Number(formData.get("tool_wear_min")),
  };
}

function errorMessage(body) {
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail)) return body.detail.map((item) => item.msg).join("; ");
  return "Unable to estimate risk.";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  result.className = "result";
  result.textContent = "Estimating…";

  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload(new FormData(form))),
  });
  const body = await response.json();

  if (!response.ok) {
    result.className = "result error";
    result.textContent = errorMessage(body);
    return;
  }

  result.className = `result ${body.risk_band}`;
  result.replaceChildren();
  const value = document.createElement("strong");
  value.textContent = `${percent(body.probability)} failure risk`;
  const detail = document.createElement("span");
  detail.textContent = `${body.risk_band} band · ${body.threshold_status} cutoff`;
  result.append(value, detail);
  await loadHistory();
});

function drawChart(records) {
  chart.replaceChildren();
  const ns = "http://www.w3.org/2000/svg";
  const axis = document.createElementNS(ns, "line");
  axis.setAttribute("x1", "24");
  axis.setAttribute("y1", "136");
  axis.setAttribute("x2", "580");
  axis.setAttribute("y2", "136");
  axis.setAttribute("class", "chart-axis");
  chart.append(axis);

  const recent = records.slice(0, 30).reverse();
  if (!recent.length) return;

  const points = recent.map((record, index) => {
    const x = recent.length === 1 ? 300 : 24 + (index / (recent.length - 1)) * 556;
    const y = 136 - record.probability * 112;
    return `${x},${y}`;
  });
  const line = document.createElementNS(ns, "polyline");
  line.setAttribute("points", points.join(" "));
  line.setAttribute("class", "chart-line");
  chart.append(line);
}

function renderHistory(records) {
  historyBody.replaceChildren();
  records.slice(0, 10).forEach((record) => {
    const row = document.createElement("tr");
    const values = [record.id, record.product_type, percent(record.probability)];
    values.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    const band = document.createElement("td");
    band.className = `band ${record.risk_band}`;
    band.textContent = record.risk_band;
    row.append(band);
    const source = document.createElement("td");
    source.textContent = record.source;
    row.append(source);
    historyBody.append(row);
  });

  const average = records.length
    ? records.reduce((sum, record) => sum + record.probability, 0) / records.length
    : 0;
  document.querySelector("#prediction-count").textContent = records.length;
  document.querySelector("#average-risk").textContent = percent(average);
  document.querySelector("#flagged-count").textContent = records.filter(
    (record) => record.risk_band === "high",
  ).length;
  drawChart(records);
}

async function loadHistory() {
  const response = await fetch("/api/predictions?limit=50");
  if (response.ok) renderHistory(await response.json());
}

document.querySelector("#refresh").addEventListener("click", loadHistory);
loadHistory();

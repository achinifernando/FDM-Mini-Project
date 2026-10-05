const form = document.querySelector("#predict-form");
const button = document.querySelector("#predict-button");
const message = document.querySelector("#form-message");
const result = document.querySelector("#result");
const locationSelects = {
  state_name: document.querySelector("#state-select"),
  county_name: document.querySelector("#county-select"),
  city_name: document.querySelector("#city-select"),
  cbsa_name: document.querySelector("#metro-select"),
  site_id: document.querySelector("#site-select"),
  year: document.querySelector("#year-select"),
};
const siteDataStatus = document.querySelector("#site-data-status");
const siteModeInputs = document.querySelectorAll('input[name="site_mode"]');
const epaSiteInputs = document.querySelector("#epa-site-inputs");
const manualSiteInputs = document.querySelector("#manual-site-inputs");
const batchFile = document.querySelector("#batch-file");
const batchMessage = document.querySelector("#batch-message");
const batchResults = document.querySelector("#batch-results");
const batchResultRows = document.querySelector("#batch-result-rows");
let lastBatchResults = [];
const measurementFields = [
  "latitude", "longitude", "observation_count", "observation_percent",
  "valid_day_count", "arithmetic_standard_dev", "first_max_value",
  "ninety_nine_percentile", "ninety_eight_percentile",
  "ninety_five_percentile", "ninety_percentile", "seventy_five_percentile",
  "fifty_percentile", "ten_percentile", "pm25_mean",
  "previous_year_pm25_mean", "two_years_ago_pm25_mean",
];
const selectPlaceholders = {
  state_name: "Choose a state",
  county_name: "Choose a county",
  city_name: "Choose a city or site area",
  cbsa_name: "Choose a metro area",
  site_id: "Choose an EPA monitoring site",
  year: "Choose a reading year",
};

function updateSelect(select, options, placeholder, selectedValue = "") {
  select.replaceChildren(new Option(placeholder, ""));
  for (const value of options) {
    select.add(new Option(value, value));
  }
  select.value = options.includes(selectedValue) ? selectedValue : "";
  select.disabled = options.length === 0;
}

async function loadSiteOptions() {
  const params = new URLSearchParams();
  for (const [key, select] of Object.entries(locationSelects)) {
    if (key !== "year" && select.value) params.set(key, select.value);
  }
  const response = await fetch(`/api/options?${params}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Monitoring site options could not be loaded.");

  for (const [key, select] of Object.entries(locationSelects)) {
    const keepSelection = select.value;
    updateSelect(select, data[key] || [], selectPlaceholders[key], keepSelection);
  }
}

function clearMeasurements() {
  for (const fieldName of measurementFields) {
    const field = form.elements.namedItem(fieldName);
    if (field) field.value = "";
  }
  updateMeasurementChart();
}

async function loadSelectedRecord() {
  const siteId = locationSelects.site_id.value;
  const year = locationSelects.year.value;
  clearMeasurements();
  if (!siteId || !year) {
    siteDataStatus.textContent = "";
    return;
  }

  siteDataStatus.textContent = "Loading the recorded site readings…";
  const params = new URLSearchParams({ site_id: siteId, year });
  const response = await fetch(`/api/site-record?${params}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Readings for this site and year could not be loaded.");

  for (const fieldName of measurementFields) {
    const field = form.elements.namedItem(fieldName);
    if (field && data[fieldName] != null) field.value = String(data[fieldName]);
  }
  updateMeasurementChart();
  siteDataStatus.textContent = "Readings filled from the selected EPA record. You can edit the measurements.";
}

function reportSiteOptionError(error) {
  siteDataStatus.textContent = error instanceof Error
    ? error.message
    : "Monitoring site options could not be loaded.";
}

function showMessage(text) {
  message.textContent = text;
  message.hidden = false;
}

function hideMessage() {
  message.textContent = "";
  message.hidden = true;
}

const chartSeries = [
  ["10th", "ten_percentile"],
  ["Median", "fifty_percentile"],
  ["75th", "seventy_five_percentile"],
  ["90th", "ninety_percentile"],
  ["95th", "ninety_five_percentile"],
  ["98th", "ninety_eight_percentile"],
  ["99th", "ninety_nine_percentile"],
  ["Daily max", "first_max_value"],
];
const chartSvg = document.querySelector("#measurement-chart");
const svgNamespace = "http://www.w3.org/2000/svg";

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(svgNamespace, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

function updateMeasurementChart() {
  chartSvg.replaceChildren();
  const values = chartSeries.map(([label, field]) => ({
    label,
    value: form.elements.namedItem(field).value.trim() === ""
      ? Number.NaN
      : Number(form.elements.namedItem(field).value),
  })).filter((item) => Number.isFinite(item.value) && item.value >= 0);

  if (!values.length) {
    chartSvg.setAttribute("aria-label", "Enter PM2.5 percentile readings to see the annual profile.");
    const hint = svgElement("text", { x: 360, y: 100, "text-anchor": "middle", class: "chart-empty" });
    hint.textContent = "Your readings will appear here";
    chartSvg.append(hint);
    return;
  }

  const maxValue = Math.max(...values.map(({ value }) => value), 10);
  const top = 20;
  const baseline = 142;
  const chartHeight = baseline - top;
  const step = 680 / values.length;
  const barWidth = Math.min(42, step * 0.58);
  chartSvg.setAttribute(
    "aria-label",
    `PM2.5 annual profile: ${values.map(({ label, value }) => `${label} ${value} micrograms per cubic meter`).join(", ")}.`,
  );
  chartSvg.append(svgElement("line", { x1: 20, y1: baseline, x2: 700, y2: baseline, class: "chart-axis" }));

  values.forEach(({ label, value }, index) => {
    const x = 22 + step * index + (step - barWidth) / 2;
    const height = Math.max(2, (value / maxValue) * chartHeight);
    const bar = svgElement("rect", {
      x,
      y: baseline - height,
      width: barWidth,
      height,
      rx: 5,
      class: "chart-bar",
      tabindex: 0,
      role: "img",
      "aria-label": `${label}: ${value} micrograms per cubic meter`,
    });
    const number = svgElement("text", { x: x + barWidth / 2, y: Math.max(13, baseline - height - 7), "text-anchor": "middle", class: "chart-value" });
    number.textContent = Number.isInteger(value) ? String(value) : value.toFixed(1);
    const caption = svgElement("text", { x: x + barWidth / 2, y: 164, "text-anchor": "middle", class: "chart-label" });
    caption.textContent = label;
    chartSvg.append(bar, number, caption);
  });
}

function readForm() {
  const data = new FormData(form);
  const payload = {};
  for (const [key, value] of data.entries()) {
    if (value.trim() === "") {
      payload[key] = null;
      continue;
    }
    const field = form.elements.namedItem(key);
    payload[key] = key === "year"
      ? Number(value)
      : field instanceof HTMLInputElement && field.type === "number"
      ? Number(value)
      : value.trim();
  }
  if (payload.site_mode === "manual") {
    for (const [target, source] of Object.entries({
      state_name: "manual_state_name",
      county_name: "manual_county_name",
      city_name: "manual_city_name",
      cbsa_name: "manual_cbsa_name",
      site_id: "manual_site_id",
      year: "manual_year",
      latitude: "manual_latitude",
      longitude: "manual_longitude",
    })) {
      const value = payload[source];
      payload[target] = value ?? null;
      delete payload[source];
    }
  } else {
    payload.year = Number(payload.year);
  }
  delete payload.site_mode;
  return payload;
}

function showResult(data) {
  document.querySelector("#result-category").textContent = data.category;
  document.querySelector("#result-description").textContent = data.description;
  document.querySelector("#result-range").textContent = data.range;
  const confidencePanel = document.querySelector("#confidence-panel");
  const confidenceValue = document.querySelector("#result-confidence");
  const confidenceNote = document.querySelector("#confidence-note");
  if (Number.isFinite(data.confidence)) {
    confidenceValue.textContent = `${data.confidence}%`;
    document.querySelector("#confidence-label").textContent =
      data.confidence_label || "MODEL PROBABILITY ESTIMATE";
    confidenceNote.textContent = "An estimate from this saved model; it is not a guarantee.";
    confidencePanel.hidden = false;
  } else {
    confidenceValue.textContent = "Unavailable";
    document.querySelector("#confidence-label").textContent = "MODEL CONFIDENCE";
    confidenceNote.textContent = "The saved model provides a category only.";
    confidencePanel.hidden = false;
  }
  document.querySelector("#activity-guidance").textContent =
    `${data.sensitivity_label} · ${data.activity_alert ? "Activity alert" : "No activity alert"}: ${data.activity_guidance}`;
  const explanations = document.querySelector("#result-explanation");
  explanations.replaceChildren(...data.explanation.map((text) => {
    const item = document.createElement("li");
    item.textContent = text;
    return item;
  }));
  result.hidden = false;
  result.dataset.style = data.style;
  result.dataset.alert = String(data.activity_alert);
  for (const step of document.querySelectorAll("#risk-band [data-category]")) {
    const selected = step.dataset.category === data.category;
    step.classList.toggle("is-predicted", selected);
    step.setAttribute("aria-current", selected ? "true" : "false");
  }
  document.querySelector("#risk-band").setAttribute(
    "aria-label",
    `Predicted category: ${data.category}.`,
  );
  result.scrollIntoView({ behavior: "smooth", block: "center" });
}

for (const [, fieldName] of chartSeries) {
  form.elements.namedItem(fieldName).addEventListener("input", updateMeasurementChart);
}
updateMeasurementChart();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideMessage();
  result.hidden = true;

  if (!form.reportValidity()) {
    const invalid = form.querySelector(":invalid");
    invalid?.focus();
    return;
  }

  button.disabled = true;
  button.querySelector(".button-label").textContent = "Working on it…";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readForm()),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "We couldn't make a prediction. Please check the form and try again.");
    }
    showResult(data);
  } catch (error) {
    showMessage(error instanceof Error
      ? error.message
      : "We couldn't reach the prediction service. Check that it is running, then try again.");
  } finally {
    button.disabled = false;
    button.querySelector(".button-label").textContent = "Get my outlook";
  }
});

const csvColumns = [
  "site_id", "state_name", "county_name", "city_name", "cbsa_name", "year",
  "latitude", "longitude", "observation_count", "observation_percent",
  "valid_day_count", "arithmetic_standard_dev", "first_max_value",
  "ninety_nine_percentile", "ninety_eight_percentile",
  "ninety_five_percentile", "ninety_percentile", "seventy_five_percentile",
  "fifty_percentile", "ten_percentile", "pm25_mean",
  "previous_year_pm25_mean", "two_years_ago_pm25_mean",
];

function downloadCsv(filename, rows) {
  const csvValue = (value) => {
    const text = String(value ?? "");
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  const contents = rows.map((row) => row.map(csvValue).join(",")).join("\r\n");
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([contents], { type: "text/csv;charset=utf-8" }));
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

function explainBatchErrors(error) {
  batchMessage.textContent = error instanceof Error
    ? error.message
    : "Batch prediction failed. Check the file and try again.";
  batchMessage.hidden = false;
}

document.querySelector("#download-template").addEventListener("click", () => {
  downloadCsv("pm25-batch-template.csv", [csvColumns]);
});

document.querySelector("#run-batch").addEventListener("click", async () => {
  batchMessage.hidden = true;
  batchResults.hidden = true;
  if (!batchFile.files?.length) {
    explainBatchErrors(new Error("Choose a CSV file first, or download the template."));
    return;
  }
  const file = batchFile.files[0];
  if (!file.name.toLowerCase().endsWith(".csv") || file.size > 2_000_000) {
    explainBatchErrors(new Error("Choose a CSV file no larger than 2 MB."));
    return;
  }
  const runButton = document.querySelector("#run-batch");
  runButton.disabled = true;
  runButton.textContent = "Predicting sites…";
  try {
    const response = await fetch("/api/predict/batch", {
      method: "POST",
      headers: {
        "Content-Type": "text/csv",
        "X-Sensitivity": document.querySelector("#batch-sensitivity").value,
      },
      body: await file.text(),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Batch prediction failed.");
    lastBatchResults = data.results;
    renderBatchResults(data.results);
  } catch (error) {
    explainBatchErrors(error);
  } finally {
    runButton.disabled = false;
    runButton.innerHTML = 'Predict all sites <span aria-hidden="true">↗</span>';
  }
});

function renderBatchResults(rows) {
  batchResultRows.replaceChildren();
  let successCount = 0;
  for (const row of rows) {
    const cells = [
      row.site_id || `Row ${row.row}`,
      row.year || "—",
      row.category || row.error,
      row.confidence == null ? "—" : `${row.confidence}%`,
      row.activity_alert == null ? "—" : row.activity_alert ? "Alert" : "No alert",
      row.explanation?.join(" ") || "",
    ];
    const tableRow = document.createElement("tr");
    for (const value of cells) {
      const cell = document.createElement("td");
      cell.textContent = value;
      tableRow.append(cell);
    }
    if (row.category) successCount += 1;
    batchResultRows.append(tableRow);
  }
  document.querySelector("#batch-summary").textContent =
    `${successCount} of ${rows.length} sites predicted${successCount < rows.length ? ` · ${rows.length - successCount} row(s) need attention` : ""}`;
  batchResults.hidden = false;
}

document.querySelector("#download-results").addEventListener("click", () => {
  const headers = ["row", "site_id", "year", "category", "confidence", "sensitivity_label", "activity_alert", "activity_guidance", "explanation", "error"];
  const rows = lastBatchResults.map((result) =>
    headers.map((header) =>
      header === "explanation"
        ? result.explanation?.join("; ")
        : result[header],
    ),
  );
  downloadCsv("pm25-prediction-results.csv", [headers, ...rows]);
});

for (const radio of siteModeInputs) {
  radio.addEventListener("change", () => {
    const manual = radio.checked && radio.value === "manual";
    if (!radio.checked) return;
    epaSiteInputs.hidden = manual;
    manualSiteInputs.hidden = !manual;
    for (const field of epaSiteInputs.querySelectorAll("input, select")) {
      field.disabled = manual;
    }
    for (const field of manualSiteInputs.querySelectorAll("input")) {
      field.disabled = !manual;
      field.required = manual;
    }
    clearMeasurements();
    siteDataStatus.textContent = manual
      ? "Enter annual readings for the site below."
      : "Choose a site and year to load its recorded readings.";
  });
}

const downstreamFields = {
  state_name: ["county_name", "city_name", "cbsa_name", "site_id", "year"],
  county_name: ["city_name", "cbsa_name", "site_id", "year"],
  city_name: ["cbsa_name", "site_id", "year"],
  cbsa_name: ["site_id", "year"],
  site_id: ["year"],
};

for (const [fieldName, select] of Object.entries(locationSelects)) {
  if (fieldName === "year") {
    select.addEventListener("change", () => loadSelectedRecord().catch(reportSiteOptionError));
    continue;
  }
  select.addEventListener("change", async () => {
    for (const downstreamName of downstreamFields[fieldName]) {
      const downstreamSelect = locationSelects[downstreamName];
      updateSelect(downstreamSelect, [], selectPlaceholders[downstreamName]);
    }
    clearMeasurements();
    siteDataStatus.textContent = "";
    try {
      await loadSiteOptions();
    } catch (error) {
      reportSiteOptionError(error);
    }
  });
}

loadSiteOptions().catch(reportSiteOptionError);

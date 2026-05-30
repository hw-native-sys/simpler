const DATA_FILES = {
  benchmarks: "data/benchmarks.json",
  methods: "data/methods.json",
  results: "data/results.json",
};

const state = {};

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`failed to load ${path}: ${response.status}`);
  }
  return response.json();
}

function text(value) {
  return document.createTextNode(String(value));
}

function metric(label, value) {
  const item = document.createElement("div");
  item.className = "metric";
  const span = document.createElement("span");
  span.append(text(label));
  const strong = document.createElement("strong");
  strong.append(text(value));
  item.append(span, strong);
  return item;
}

function renderSnapshot() {
  const snapshot = state.results.snapshot;
  const root = document.getElementById("snapshot");
  root.replaceChildren(
    metric("Commit", snapshot.commit),
    metric("Full samples", snapshot.full_capture.samples),
    metric("Compact samples", snapshot.compact_capture.samples),
    metric("Tensor tile", snapshot.tensor_tile),
    metric("GPU targets", snapshot.gpu_targets.join(", ")),
  );
}

function renderHeadlineResults() {
  const root = document.getElementById("headline-results");
  const rows = state.results.headline_results;
  root.replaceChildren(table(
    ["GPU", "N", "Method", "Host ns", "Device ns", "Status"],
    rows.map((row) => [
      row.gpu,
      row.n,
      row.method,
      row.host_wall_ns,
      row.device_wall_ns,
      row.status,
    ]),
  ));
}

function evidenceList(refs) {
  const list = document.createElement("ul");
  refs.forEach((ref) => {
    const item = document.createElement("li");
    item.append(text(`${ref.path}: ${ref.symbols.join(", ")}`));
    list.append(item);
  });
  return list;
}

function commandBlock(command) {
  const pre = document.createElement("pre");
  const code = document.createElement("code");
  code.append(text(command));
  pre.append(code);
  return pre;
}

function renderBenchmarks() {
  const root = document.getElementById("benchmark-list");
  root.replaceChildren(...state.benchmarks.benchmarks.map((benchmark) => {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.append(text(benchmark.title));
    const desc = document.createElement("p");
    desc.append(text(benchmark.description));
    const math = document.createElement("p");
    math.innerHTML = `<strong>Math:</strong> ${benchmark.math}`;
    const code = document.createElement("pre");
    const codeText = document.createElement("code");
    codeText.append(text(benchmark.code));
    code.append(codeText);
    const run = document.createElement("h3");
    run.append(text("Run"));
    const evidence = document.createElement("h3");
    evidence.append(text("Evidence"));
    details.append(
      summary,
      desc,
      math,
      code,
      run,
      commandBlock(benchmark.run.command),
      evidence,
      evidenceList(benchmark.evidence_refs),
    );
    return details;
  }));
}

function renderMethods() {
  const root = document.getElementById("method-list");
  root.replaceChildren(...state.methods.methods.map((method) => {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.append(text(method.name));
    const runtime = document.createElement("p");
    runtime.innerHTML = `<strong>Runtime flow:</strong> ${method.runtime_flow}`;
    const lifecycle = document.createElement("p");
    lifecycle.innerHTML = `<strong>Lifecycle mapping:</strong> ${method.lifecycle}`;
    details.append(summary, runtime, lifecycle, evidenceList(method.evidence_refs));
    return details;
  }));
}

function table(headers, rows) {
  const tableEl = document.createElement("table");
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  headers.forEach((header) => {
    const th = document.createElement("th");
    th.append(text(header));
    headerRow.append(th);
  });
  thead.append(headerRow);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((cell) => {
      const td = document.createElement("td");
      td.append(text(cell));
      tr.append(td);
    });
    tbody.append(tr);
  });
  tableEl.append(thead, tbody);
  return tableEl;
}

function renderResults() {
  const root = document.getElementById("result-table");
  root.replaceChildren(table(
    ["GPU", "Machine", "Method", "N", "Tasks", "Host ns", "Device ns"],
    state.results.selected_rows.map((row) => [
      row.gpu,
      row.machine,
      row.method,
      row.n,
      row.task_count,
      row.host_wall_ns,
      row.device_wall_ns,
    ]),
  ));
}

function wireTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".view").forEach((item) => item.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(tab.dataset.view).classList.add("active");
    });
  });
}

async function main() {
  wireTabs();
  try {
    const [benchmarks, methods, results] = await Promise.all([
      loadJson(DATA_FILES.benchmarks),
      loadJson(DATA_FILES.methods),
      loadJson(DATA_FILES.results),
    ]);
    Object.assign(state, {benchmarks, methods, results});
    renderSnapshot();
    renderHeadlineResults();
    renderBenchmarks();
    renderMethods();
    renderResults();
  } catch (error) {
    const errorBox = document.getElementById("load-error");
    errorBox.classList.remove("hidden");
    errorBox.textContent = `${error.message}. Serve the repo with python3 -m http.server and open the viewer through http://localhost:8000/.`;
  }
}

main();

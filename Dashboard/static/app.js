
const API = {
  kpis: "/api/kpis",
  module1: "/api/module1",
  module2: "/api/module2",
  module3: "/api/module3"
};

const DATA = {};
const $ = (id) => document.getElementById(id);

const PAGE_INFO = {
  overview: ["Overview", "A consolidated view of available model-scored MPLADS works."],
  works: ["Works Browser", "Browse available model-scored work records."],
  module1: [" Overrun", "Time-overrun and underutilization screening."],
  module2: [" Financial Anomaly" , "Financial anomaly screening and review evidence."],
  module3: [" Duplicates", "Candidate pairs ranked for duplicate-work review."],
  // newwork: ["Score a New Work", "New-work inference is not connected yet."],
  planned: ["Planned Capabilities", "Future modules and integrations."]
};

function esc(value) {
  return String(value ?? "—").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[c]);
}

function num(value, digits = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toLocaleString("en-IN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits
  }) : "—";
}

function first(row, names, fallback = null) {
  for (const name of names) {
    if (row && row[name] !== undefined && row[name] !== null && row[name] !== "") {
      return row[name];
    }
  }
  return fallback;
}

function tile(label, value, note = "") {
  return `<article class="kpi-card">
    <span class="kpi-label">${esc(label)}</span>
    <strong>${esc(value)}</strong>
    <span class="kpi-note">${esc(note)}</span>
  </article>`;
}

function tierClass(value) {
  const t = String(value ?? "").toLowerCase();
  if (t.includes("high") || t.includes("potential duplicate")) return "high";
  if (t.includes("medium") || t.includes("review")) return "medium";
  if (t.includes("low") || t.includes("not flagged") || t.includes("lower likelihood")) return "low";
  return "";
}

function pill(value) {
  const label = value ?? "Unknown";
  return `<span class="pill ${tierClass(label)}">${esc(label)}</span>`;
}

function showMessage(message, isError = false) {
  $("message").hidden = !message;
  $("message").textContent = message || "";
  $("message").classList.toggle("error", isError);
}

async function getJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
  return response.json();
}

async function loadData() {
  $("refreshButton").disabled = true;
  showMessage("");
  $("footerStatus").textContent = "Refreshing data…";
  try {
    const [kpis, m1, m2, m3] = await Promise.all([
      getJSON(API.kpis), getJSON(API.module1),
      getJSON(API.module2), getJSON(API.module3)
    ]);
    DATA.kpis = kpis;
    DATA.module1 = m1;
    DATA.module2 = m2;
    DATA.module3 = m3;
    renderAll();
    $("footerStatus").textContent = "Loaded from generated artifact-backed data";
  } catch (error) {
    console.error(error);
    showMessage(`Could not load dashboard data: ${error.message}. Check that the dashboard server is running and rebuild the data if necessary.`, true);
    $("footerStatus").textContent = "Data unavailable";
  } finally {
    $("refreshButton").disabled = false;
  }
}

function renderOverview() {
  const k = DATA.kpis || {};
  const m1 = k.module1 || {};
  const m2 = k.module2 || {};
  const m3 = k.module3 || {};
  $("overviewKpis").innerHTML = [
  tile("Time & Cost Risk Records", num(m1.works_scored),
    "Time-overrun and underutilization assessment"),
  tile("Financial Anomaly Records", num(m2.works_scored),
    "Financial risk screening"),
  tile("Duplicate Candidate Pairs", num(m3.candidate_pairs_scored),
    "Pair-level similarity analysis"),
  tile("Mean Financial Anomaly Score", num(m2.mean_fused_score, 2),
    "Screening score, not fraud probability")
].join("");

  const items = [
  ["Time & Cost Risk", m1.works_scored || 0],
  ["Financial Anomaly", m2.works_scored || 0],
  ["Duplicate Pairs", m3.candidate_pairs_scored || 0]
];
  const max = Math.max(...items.map(x => Number(x[1]) || 0), 1);
  $("coverage").innerHTML = items.map(([label, value]) => `
    <div class="coverage-row">
      <strong>${esc(label)}</strong>
      <div class="coverage-track"><div class="coverage-fill" style="width:${Math.min(100, Number(value) / max * 100)}%"></div></div>
      <span class="coverage-count">${num(value)}</span>
    </div>`).join("");
}

function renderWorks() {
  const rows = DATA.module2 || [];
  const query = ($("worksSearch").value || "").trim().toLowerCase();
  const filtered = rows.filter(w => JSON.stringify(w).toLowerCase().includes(query)).slice(0, 100);

  $("worksBody").innerHTML = filtered.map(w => {
    const description = first(w, ["work_raw", "work_description", "description", "work"]);
    const state = first(w, ["state", "STATE"]);
    const amount = first(w, ["amount", "sanctioned_amount", "ALLOCATION AMOUNT"]);
    const score = first(w, ["fused_score", "m2_fused_score"]);
    const tier = first(w, ["agreement_tier", "m2_tier", "financial_tier"]);
    const m1 = (DATA.module1 || []).find(x =>
      String(first(x, ["work_raw", "work_description", "work"])) === String(description)
    );
    return `<tr>
      <td>${esc(description)}</td><td>${esc(state)}</td>
      <td class="mono">${amount == null ? "—" : "₹" + num(amount)}</td>
      <td>${m1 ? pill(first(m1, ["time_overrun_tier", "time_tier"], "Available")) : '<span class="muted">No match</span>'}</td>
      <td>${esc(score == null ? "—" : num(score, 2))}<br>${tier ? pill(tier) : ""}</td>
    </tr>`;
  }).join("") || '<tr><td colspan="5">No matching works in the loaded sample.</td></tr>';

  $("worksFootnote").textContent =
    `Showing ${filtered.length} of ${rows.length.toLocaleString("en-IN")} Module 2 sample records. Module 1 matching is by work description only and may not be unique.`;
}


function renderModule1() {
  const k = (DATA.kpis || {}).module1 || {};
  const time = k.time_overrun_tiers || {};
  const under = k.underutilization_tiers || {};

  $("module1Kpis").innerHTML = [
    tile("Works Assessed", num(k.works_scored)),
    tile("High Time-Overrun Risk", num(time.High || 0)),
    tile("Medium Time-Overrun Risk", num(time.Medium || 0)),
    tile("High Underutilization Risk", num(under.High || 0))
  ].join("");

  const rows = [...(DATA.module1 || [])].sort((a, b) => {
    const scoreA = Number(first(a, ["time_overrun_score", "time_score"], 0)) || 0;
    const scoreB = Number(first(b, ["time_overrun_score", "time_score"], 0)) || 0;
    return scoreB - scoreA;
  });

  $("module1Body").innerHTML = rows.slice(0, 100).map(w => {
    const timeScore = first(w, ["time_overrun_score", "time_score"]);
    const underScore = first(w, ["underutilization_score", "underutil_score"]);

    return `<tr>
      <td>${esc(first(w, ["work_raw", "work_description", "work"]))}</td>
      <td>${esc(first(w, ["state"]))}</td>
      <td class="mono"><strong>${num(timeScore, 2)}</strong></td>
      <td>${pill(first(w, ["time_overrun_tier", "time_tier"]))}</td>
      <td class="mono">${num(underScore, 2)}</td>
      <td>${pill(first(w, ["underutilization_tier", "underutil_tier"]))}</td>
    </tr>`;
  }).join("") || '<tr><td colspan="6">No scored works available.</td></tr>';
}




function renderModule2() {
  const k = (DATA.kpis || {}).module2 || {};
  const tiers = k.tier_counts || {};

  $("module2Kpis").innerHTML = [
    tile("Works Assessed", num(k.works_scored)),
    tile(
      "Requires Review",
      num(tiers["Needs review"] || tiers["Needs Review"] || 0)
    ),
    tile(
      "High-Confidence Flags",
      num(tiers["High confidence"] || tiers["High Confidence"] || 0)
    ),
    tile(
      "Mean Fused Score",
      num(k.mean_fused_score, 2),
      "Ranking aid only; not a fraud probability"
    )
  ].join("");

  const rows = [...(DATA.module2 || [])].sort((a, b) => {
    const scoreA = Number(first(a, ["fused_score"], 0)) || 0;
    const scoreB = Number(first(b, ["fused_score"], 0)) || 0;
    return scoreB - scoreA;
  });

  function flagLabel(value) {
    if (value === true || Number(value) === 1) return "Flagged";
    if (value === false || value === 0) return "Not flagged";
    return value == null || value === "" ? "Not available" : String(value);
  }

  function featureEvidence(work) {
    const features = [
      ["Threshold avoidance", "f1_threshold_avoidance"],
      ["Benford excess", "f2_benford_excess"],
      ["Round-number pattern", "f3_round_number"],
      ["Agency cost trend", "f4_agency_cost_trend"],
      ["Approval-gap evidence", "f5_approval_gap"],
      ["Agency fingerprint", "f6_agency_fingerprint"]
    ];

    return features
      .filter(([, key]) => work[key] !== undefined && work[key] !== null && work[key] !== "")
      .map(([label, key]) => `${label}: ${num(work[key], 3)}`)
      .join(" | ") || "Feature evidence unavailable";
  }

  function ruleEvidence(work) {
    const flags = [
      ["Threshold avoidance", "flag_threshold_avoidance"],
      ["Fast approval", "flag_fast_approval"],
      ["Pricing outlier", "flag_pricing_outlier"]
    ];

    return flags.map(([label, key]) => {
      if (work[key] === undefined || work[key] === null || work[key] === "") {
        return `${label}: unavailable`;
      }
      return `${label}: ${flagLabel(work[key])}`;
    }).join(" | ");
  }

  function factorText(value) {
    if (value == null || value === "") return "No factor summary available";

    if (Array.isArray(value)) {
      return value.map(item =>
        typeof item === "object" ? JSON.stringify(item) : String(item)
      ).join(", ");
    }

    // Some CSV exports store this field as text rather than a JSON array.
    return String(value);
  }

  $("module2Body").innerHTML = rows.slice(0, 100).map(w => {
    const description = first(
      w, ["work_raw", "work_description", "description", "work"], "Unnamed work"
    );
    const amount = first(w, ["amount", "sanctioned_amount"]);
    const isolation = first(w, ["isolation_forest_score"]);
    const autoencoder = first(w, ["autoencoder_score"]);
    const tier = first(w, ["agreement_tier"], "Unknown");

    return `<tr>
      <td>
        <strong>${esc(description)}</strong>
        <br><span class="muted">${esc(first(w, ["state"], "State unavailable"))}</span>
      </td>
      <td class="mono">${amount == null ? "—" : "₹" + num(amount)}</td>
      <td class="mono">${isolation == null ? "—" : num(isolation, 2)}</td>
      <td class="mono">${autoencoder == null ? "—" : num(autoencoder, 2)}</td>
      <td>${pill(tier)}</td>
      <td>
        <details>
          <summary>View evidence</summary>
          <p><strong>Rule checks</strong><br>${esc(ruleEvidence(w))}</p>
          <p><strong>Feature values</strong><br>${esc(featureEvidence(w))}</p>
          <p><strong>Model factor summary</strong><br>${esc(factorText(
            first(w, ["top_contributing_factors"])
          ))}</p>
        </details>
      </td>
    </tr>`;
  }).join("") || '<tr><td colspan="6">No scored works available.</td></tr>';

  $("module2Footnote") && (
    $("module2Footnote").textContent =
      `Showing ${Math.min(rows.length, 100)} of ${rows.length.toLocaleString("en-IN")} loaded historical records. Scores are screening signals, not proof of fraud. Rule flags and feature values require contextual review.`
  );
}


function workDescription(work) {
  if (!work) return "Work details unavailable";
  return first(work, ["description", "work_raw", "work"], "Work details unavailable");
}

function workLocation(work) {
  if (!work) return "—";
  return [first(work, ["state"]), first(work, ["village"]), first(work, ["block"])]
    .filter(Boolean).join(" · ") || "—";
}

function renderModule3() {
  const k = (DATA.kpis || {}).module3 || {};
  const bands = k.band_counts || {};
  const potential = bands["Potential Duplicate"] || bands["High"] || 0;
  const review = bands["Needs Review"] || bands["Medium"] || 0;
  const lower = bands["Lower Likelihood"] || bands["Low"] || 0;

  $("module3Kpis").innerHTML = [
    tile("Candidate pairs scored", num(k.candidate_pairs_scored), "Pair-level records in the artifact"),
    tile("Potential duplicate pairs", num(potential), "Candidate band; verify manually"),
    tile("Pairs needing review", num(review), "Review evidence and context"),
    tile("Lower-likelihood pairs", num(lower), "Lower scoring band")
  ].join("");

  const all = DATA.module3 || [];
  const query = ($("pairsSearch").value || "").trim().toLowerCase();
  const filtered = all.filter(p => JSON.stringify(p).toLowerCase().includes(query)).slice(0, 100);

  $("module3Body").innerHTML = filtered.map(p => {
    const a = p.work_a;
    const b = p.work_b;
    const score = p.duplicate_score;
    const band = p.result || "Unknown";
    const evidence = [
      `Text ${num(p.S_text, 3)}`,
      `Geo ${num(p.S_geo, 3)}`,
      `Time ${num(p.S_time, 3)}`,
      `Structure ${num(p.S_struct, 3)}`
    ].join(" · ");
    return `<tr>
      <td>${esc(workDescription(a))}<br><span class="muted">ID ${esc(a?.work_id)}</span></td>
      <td>${esc(workDescription(b))}<br><span class="muted">ID ${esc(b?.work_id)}</span></td>
      <td>${esc(workLocation(a))}<br><span class="muted">${esc(workLocation(b))}</span></td>
      <td class="mono">${num(score, 3)}</td>
      <td>${esc(evidence)}</td>
      <td>${pill(band)}</td>
    </tr>`;
  }).join("") || '<tr><td colspan="6">No matching candidate pairs.</td></tr>';

  $("module3Footnote").textContent =
    `Showing ${filtered.length} of ${all.length.toLocaleString("en-IN")} highest-scoring pairs loaded for review. Summary counts use the full scored CSV. Similarity is not proof of duplication.`;
}

function renderAll() {
  renderOverview();
  renderWorks();
  renderModule1();
  renderModule2();
  renderModule3();
}

function activateView(name) {
  document.querySelectorAll(".nav-link").forEach(button => {
    button.classList.toggle("active", button.dataset.view === name);
  });
  document.querySelectorAll(".view").forEach(view => {
    view.classList.toggle("active", view.id === `view-${name}`);
  });
  const info = PAGE_INFO[name] || PAGE_INFO.overview;
  $("pageTitle").textContent = info[0];
  $("pageDescription").textContent = info[1];
}

document.querySelectorAll(".nav-link").forEach(button => {
  button.addEventListener("click", () => activateView(button.dataset.view));
});
$("refreshButton").addEventListener("click", loadData);
$("worksSearch").addEventListener("input", renderWorks);
$("pairsSearch").addEventListener("input", renderModule3);

// $("newWorkForm").addEventListener("submit", async event => {
//   event.preventDefault();

//   const form = event.currentTarget;
//   const result = $("newWorkResult");
//   const submitButton = form.querySelector('button[type="submit"]');

//   const formData = new FormData(form);
//   const payload = Object.fromEntries(formData.entries());

//   payload.amount = payload.amount ? Number(payload.amount) : null;

//   result.replaceChildren();
//   result.textContent = "Assessing the new work against existing records?";
//   submitButton.disabled = true;

//   try {
//     const response = await fetch("/api/assess", {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify(payload)
//     });

//     const data = await response.json();

//     if (!response.ok) {
//       throw new Error(data.error || "Assessment request failed.");
//     }

//     result.replaceChildren();

//     const heading = document.createElement("h3");
//     heading.textContent = "Duplicate Work Detection Results";
//     result.appendChild(heading);

//     const summary = document.createElement("p");
//     summary.textContent =
//       `Compared against ${Number(data.reference_count || 0).toLocaleString()} existing works. ` +
//       `${Number(data.candidates_scored || 0).toLocaleString()} text-similar candidates were scored.`;
//     result.appendChild(summary);

//     const notice = document.createElement("p");
//     notice.textContent =
//       "These are potential matches, not confirmed duplicates. Human verification is required.";
//     result.appendChild(notice);

//     const matches = Array.isArray(data.matches) ? data.matches : [];

//     if (!matches.length) {
//       const empty = document.createElement("p");
//       empty.textContent = "No candidate matches were returned.";
//       result.appendChild(empty);
//     }

//     matches.forEach((match, index) => {
//       const card = document.createElement("article");
//       card.className = "assessment-match";

//       const title = document.createElement("h4");
//       title.textContent = `${index + 1}. ${match.work || "Untitled work"}`;
//       card.appendChild(title);

//       const location = document.createElement("p");
//       location.textContent = [
//         match.state,
//         match.constituency,
//         match.city,
//         match.block,
//         match.village
//       ].filter(Boolean).join(" ? ") || "Location details unavailable";
//       card.appendChild(location);

//       const category = document.createElement("p");
//       category.textContent =
//         `Category: ${match.category || "Not specified"} | ` +
//         `Recommended date: ${match.recommended_date || "Not available"}`;
//       card.appendChild(category);

//       const score = document.createElement("p");
//       score.textContent =
//         `Combined duplicate score: ${(Number(match.duplicate_score) * 100).toFixed(1)}% ? ${match.tier}`;
//       card.appendChild(score);

//       const components = document.createElement("p");
//       components.textContent =
//         `Text: ${(Number(match.text_similarity) * 100).toFixed(1)}% ? ` +
//         `Geography: ${(Number(match.geo_similarity) * 100).toFixed(1)}% ? ` +
//         `Time: ${(Number(match.time_similarity) * 100).toFixed(1)}% ? ` +
//         `Structured fields: ${(Number(match.structured_similarity) * 100).toFixed(1)}%`;
//       card.appendChild(components);

//       result.appendChild(card);
//     });

//     const status = document.createElement("p");
//     status.textContent =
//       "Time & Cost Overrun Risk Analysis and Financial Anomaly Detection " +
//       "have not yet been connected for new-work inference. No scores were generated for those modules.";
//     result.appendChild(status);

//     if (Array.isArray(data.limitations)) {
//       const details = document.createElement("details");
//       const summaryLabel = document.createElement("summary");
//       summaryLabel.textContent = "Assessment limitations";
//       details.appendChild(summaryLabel);

//       const list = document.createElement("ul");
//       data.limitations.forEach(message => {
//         const item = document.createElement("li");
//         item.textContent = message;
//         list.appendChild(item);
//       });
//       details.appendChild(list);
//       result.appendChild(details);
//     }
//   } catch (error) {
//     result.textContent = `Assessment failed: ${error.message}`;
//   } finally {
//     submitButton.disabled = false;
//   }
// });



$("newWorkForm").addEventListener("submit", async event => {
  event.preventDefault();

  const form = event.currentTarget;
  const result = $("newWorkResult");
  const submitButton = form.querySelector('button[type="submit"]');
  const payload = Object.fromEntries(new FormData(form).entries());

  payload.amount = payload.amount ? Number(payload.amount) : null;

  result.replaceChildren();
  result.textContent = "Assessing work against historical records…";
  submitButton.disabled = true;

  try {
    const response = await fetch("/api/assess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Assessment request failed.");
    }

    const heading = document.createElement("h3");
    heading.textContent = "New Work Assessment";
    result.appendChild(heading);

    const notice = document.createElement("p");
    notice.textContent = data.assessment_notice ||
      "Results are screening evidence and require human verification.";
    result.appendChild(notice);

    // Module summary cards
    const grid = document.createElement("div");
    grid.className = "assessment-grid";

    const summary = data.assessment_summary || {};
    const module1 = data.module1_screening || {};
    const module2 = data.module2_screening || {};
    const cards = [
      // {
      //   title: "Duplicate Work Detection",
      //   value: `${Number(data.reference_count || 0).toLocaleString("en-IN")} reference works`,
      //   note: `${Number(data.candidates_scored || 0).toLocaleString("en-IN")} text-similar candidates ranked. Top review band: ${summary.top_duplicate_tier || "Unavailable"}. Candidate matches require verification.`
      // },
      {
        title: "Financial Anomaly Detection",
        value: module2.status === "completed"
          ? `Reference percentile: ${module2.fused_reference_percentile}`
          : "Unavailable",
        note: module2.status === "completed"
          ? `Isolation Forest: ${module2.isolation_forest_percentile}th percentile; Autoencoder: ${module2.autoencoder_percentile}th percentile. Percentiles measure unusualness, not fraud probability.`
          : (module2.message || "The required financial features could not all be calculated. Check that recommended and sanction dates, amount, state, and agency are supplied.")
      },
      // {
        // title: "Time & Cost Overrun Risk Analysis",
        // value: module1.status === "reference_estimate" ? "Historical estimate" : "Unavailable",
        // note: module1.status === "reference_estimate"
        //   ? `Estimated time-overrun score: ${module1.time_overrun_score ?? "—"}/100; under-utilization score: ${module1.underutilization_score ?? "—"}/100. Based on ${module1.similar_works_used ?? 0} similar historical works; not a fresh ML prediction.`
        //   : (module1.message || "A historical-reference estimate could not be calculated for this work.")

      // }
            
      {
        title: "Time & Cost Overrun Risk Analysis",
        value: module1.status === "completed"
          ? "Live ML prediction"
          : "Unavailable",
        note: module1.status === "completed"
          ? `Time-overrun score: ${module1.time_overrun_score ?? "—"}/100 (${module1.time_overrun_tier || "Unclassified"}); under-utilization score: ${module1.underutilization_score ?? "—"}/100 (${module1.underutilization_tier || "Unclassified"}). Model: ${module1.model || "ML model"}. These are model scores, not guaranteed calibrated probabilities.`
          : (module1.error || module1.message || "The Module 1 prediction could not be calculated.")
      },

    ];

    cards.forEach(item => {
      const card = document.createElement("article");
      card.className = "assessment-card";

      const title = document.createElement("h4");
      title.textContent = item.title;

      const value = document.createElement("strong");
      value.textContent = item.value;

      const note = document.createElement("p");
      note.textContent = item.note;

      card.append(title, value, note);
      grid.appendChild(card);
    });

    result.appendChild(grid);

    // Financial rule checks
    // const financialHeading = document.createElement("h3");
    // financialHeading.textContent = "Financial Rule Checks";
    // result.appendChild(financialHeading);

    // const checks = data.financial_screening?.checks || [];
    // const checkGrid = document.createElement("div");
    // checkGrid.className = "assessment-checks";

    // checks.forEach(check => {
    //   const card = document.createElement("article");
    //   card.className = "assessment-check";

    //   const title = document.createElement("strong");
    //   title.textContent = check.name;

    //   const status = document.createElement("p");
    //   status.className = "assessment-status";
    //   status.textContent = ({
    //     flagged: "Flagged",
    //     not_flagged: "Not flagged",
    //     unavailable: "Unavailable"
    //   })[check.status] || check.status || "Unknown";

    //   const explanation = document.createElement("p");
    //   explanation.textContent = check.explanation || "No explanation provided.";

    //   card.append(title, status, explanation);
    //   checkGrid.appendChild(card);
    // });

    // result.appendChild(checkGrid);

    // const financialNotice = document.createElement("p");
    // financialNotice.className = "muted";
    // financialNotice.textContent = data.financial_screening?.notice || "";
    // result.appendChild(financialNotice);

    // Historical duplicate candidates
    const matchesHeading = document.createElement("h3");
    matchesHeading.textContent = "Closest Historical Works";
    result.appendChild(matchesHeading);

    const matchNote = document.createElement("p");
    matchNote.textContent =
      "Similarity scores rank candidates; they are not probabilities of fraud or proof of duplication.";
    result.appendChild(matchNote);

    const tableWrap = document.createElement("div");
    tableWrap.className = "table-wrap";

    const table = document.createElement("table");
    table.innerHTML = `
      <thead>
        <tr>
          <th>Historical work</th>
          <th>Similarity score</th>
          <th>Review band</th>
          <th>Evidence components</th>
        </tr>
      </thead>
      <tbody></tbody>
    `;

    const tbody = table.querySelector("tbody");
    const matches = data.matches || [];

    if (matches.length === 0) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 4;
      cell.textContent = "No candidate matches returned.";
      row.appendChild(cell);
      tbody.appendChild(row);
    }

    matches.forEach(match => {
      const row = document.createElement("tr");

      const workCell = document.createElement("td");
      const workTitle = document.createElement("strong");
      workTitle.textContent = match.work || "Untitled work";

      const location = document.createElement("p");
      location.className = "muted";
      location.textContent = [
        match.state,
        match.constituency,
        match.block,
        match.village
      ].filter(Boolean).join(" · ");

      workCell.append(workTitle, location);

      const scoreCell = document.createElement("td");
      scoreCell.textContent =
        `${(Number(match.duplicate_score || 0) * 100).toFixed(1)}%`;

      const tierCell = document.createElement("td");
      tierCell.textContent = match.tier || "Unknown";

      const evidenceCell = document.createElement("td");
      evidenceCell.textContent = [
        `Text: ${(Number(match.text_similarity || 0) * 100).toFixed(1)}%`,
        `Location: ${(Number(match.geo_similarity || 0) * 100).toFixed(1)}%`,
        `Date: ${(Number(match.time_similarity || 0) * 100).toFixed(1)}%`,
        `Structured: ${(Number(match.structured_similarity || 0) * 100).toFixed(1)}%`
      ].join(" | ");

      row.append(workCell, scoreCell, tierCell, evidenceCell);
      tbody.appendChild(row);
    });

    tableWrap.appendChild(table);
    result.appendChild(tableWrap);

    // Limitations
    const limitations = (data.limitations || []).filter(message => String(message || "").trim());

    if (limitations.length) {
      const details = document.createElement("details");
      const summaryElement = document.createElement("summary");
      summaryElement.textContent = "Assessment limitations";

      const list = document.createElement("ul");

      limitations.forEach(message => {
        const item = document.createElement("li");
        item.textContent = message;
        list.appendChild(item);
      });

      details.append(summaryElement, list);
      result.appendChild(details);
    }

  } catch (error) {
    result.textContent = `Assessment failed: ${error.message}`;
    console.error("New-work assessment error:", error);
  } finally {
    submitButton.disabled = false;
  }
});

loadData();
